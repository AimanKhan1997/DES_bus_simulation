"""
Post-Simulation MILP Optimizer using Gurobi

Optimizes bus battery capacity per line, MAP battery capacity, and number of MAPs
to minimize total system cost, based on outputs from the DES charging simulation.

Decision Variables:
    B_l   : Battery capacity (kWh) per bus on line l          (continuous)
    B_map : MAP battery capacity (kWh)                        (continuous)
    N_map : Number of MAPs                                    (integer)

Objective: Minimize total system cost
    = bus battery cost  (115 $/kWh × capacity × buses per line)
    + MAP battery cost  (115 $/kWh × capacity × number of MAPs)
    + MAP hardware cost (40,000 $ × number of MAPs)
    + overnight charging cost (0.0005 $/kWh × overnight energy)
    + penalty for SOC violations

Constraints:
    - No bus may go below 20% SOC (with penalty slack)
    - No MAP may go below 10% SOC (with penalty slack)
"""

from collections import defaultdict

# ========================
# COST PARAMETERS
# ========================
BATTERY_COST_PER_KWH = 115.0              # $/kWh for bus and MAP batteries
OVERNIGHT_CHARGE_COST_PER_KWH = 0.0005    # $/kWh for overnight depot charging
MAP_HARDWARE_COST = 40000.0               # $ per MAP unit
BUS_SOC_VIOLATION_PENALTY = 1_000_000.0   # penalty per bus that violates 20% SOC
MAP_SOC_VIOLATION_PENALTY = 1_000_000.0   # penalty per MAP that violates 10% SOC
BUS_MIN_SOC_FRACTION = 0.20               # 20% minimum bus SOC
MAP_MIN_SOC_FRACTION = 0.10               # 10% minimum MAP SOC


# ========================
# DATA EXTRACTION
# ========================

def extract_simulation_data(results, stage2_sim, bus_lines):
    """
    Extract simulation output data needed for the MILP.

    Parameters
    ----------
    results : dict
        Dictionary returned by ``Stage2DESTerminalChargingPreemptive.run_simulation``.
    stage2_sim : Stage2DESTerminalChargingPreemptive
        The simulation object (provides per-bus charging and MAP tracker data).
    bus_lines : list[BusLineData]
        List of ``BusLineData`` objects used by the simulation.

    Returns
    -------
    dict  with keys ``per_line``, ``per_bus``, ``per_map``, ``system``.
    """
    from integration_stage2 import MAP_BATTERY_CAPACITY_WH

    sim_battery_wh = results['battery_capacity_wh']
    sim_num_maps = results['num_maps']
    bus_stats = results['bus_statistics']

    # --- per-bus ---
    per_bus = {}
    for bus_id, stats in bus_stats.items():
        per_bus[bus_id] = {
            'line_id': stats['line_id'],
            'total_energy_consumed_wh': stats['total_energy_consumed_wh'],
            'energy_charged_wh': stage2_sim.bus_energy_charged.get(bus_id, 0.0),
            'min_soc_wh': stats['min_soc_wh'],
            'min_soc_ratio': stats['min_soc_ratio'],
            'num_trips': len(stage2_sim.bus_trips_dict.get(bus_id, [])),
        }

    # --- per-line aggregation ---
    line_data = defaultdict(lambda: {
        'num_buses': 0,
        'max_energy_consumed_wh': 0.0,
        'total_energy_consumed_wh': 0.0,
        'max_energy_deficit_wh': 0.0,
        'total_energy_charged_wh': 0.0,
        'total_num_trips': 0,
        'buses': [],
    })

    for bus_id, bdata in per_bus.items():
        lid = bdata['line_id']
        ld = line_data[lid]
        ld['num_buses'] += 1
        ld['max_energy_consumed_wh'] = max(ld['max_energy_consumed_wh'],
                                           bdata['total_energy_consumed_wh'])
        ld['total_energy_consumed_wh'] += bdata['total_energy_consumed_wh']
        deficit = sim_battery_wh - bdata['min_soc_wh']
        ld['max_energy_deficit_wh'] = max(ld['max_energy_deficit_wh'], deficit)
        ld['total_energy_charged_wh'] += bdata['energy_charged_wh']
        ld['total_num_trips'] += bdata['num_trips']
        ld['buses'].append(bus_id)

    # --- per-MAP ---
    per_map = {}
    sim_map_battery_wh = MAP_BATTERY_CAPACITY_WH

    if sim_num_maps > 0:
        for map_id in range(sim_num_maps):
            total_delivered = stage2_sim.map_tracker.map_total_energy.get(map_id, 0.0)
            soc_history = stage2_sim.map_tracker.map_soc_history.get(map_id, [])
            min_soc = min((s for _, s in soc_history), default=sim_map_battery_wh)
            per_map[map_id] = {
                'total_energy_delivered_wh': total_delivered,
                'min_soc_wh': min_soc,
            }

    # --- system totals ---
    total_energy_consumed = sum(b['total_energy_consumed_wh'] for b in per_bus.values())
    total_energy_charged = results['total_energy_charged_wh']

    return {
        'per_line': dict(line_data),
        'per_bus': per_bus,
        'per_map': per_map,
        'system': {
            'sim_battery_wh': sim_battery_wh,
            'sim_num_maps': sim_num_maps,
            'sim_map_battery_wh': sim_map_battery_wh,
            'total_energy_consumed_wh': total_energy_consumed,
            'total_energy_charged_wh': total_energy_charged,
            'num_buses': results['num_buses'],
        },
    }


# ========================
# MILP FORMULATION
# ========================

def run_milp_optimization(sim_data,
                          max_maps=10,
                          bus_cap_min_kwh=50.0,
                          bus_cap_max_kwh=500.0,
                          map_cap_min_kwh=50.0,
                          map_cap_max_kwh=500.0):
    """
    Formulate and solve the post-simulation MILP using Gurobi.

    Parameters
    ----------
    sim_data : dict
        Output of :func:`extract_simulation_data`.
    max_maps : int
        Upper bound on the number of MAPs.
    bus_cap_min_kwh, bus_cap_max_kwh : float
        Lower / upper bounds for bus battery capacity (kWh).
    map_cap_min_kwh, map_cap_max_kwh : float
        Lower / upper bounds for MAP battery capacity (kWh).

    Returns
    -------
    dict  with optimisation status, optimal values, and cost breakdown.
    """
    import gurobipy as gp
    from gurobipy import GRB

    per_line = sim_data['per_line']
    per_bus = sim_data['per_bus']
    per_map = sim_data['per_map']
    sys_data = sim_data['system']

    sim_battery_wh = sys_data['sim_battery_wh']
    sim_num_maps = max(1, sys_data['sim_num_maps'])
    sim_map_battery_wh = max(1.0, sys_data['sim_map_battery_wh'])
    total_energy_consumed = sys_data['total_energy_consumed_wh']
    total_energy_charged = sys_data['total_energy_charged_wh']

    line_ids = sorted(per_line.keys())
    bus_ids = sorted(per_bus.keys())

    # Maximum MAP energy deficit observed in simulation
    max_map_deficit = 0.0
    for mdata in per_map.values():
        max_map_deficit = max(max_map_deficit,
                              sim_map_battery_wh - mdata['min_soc_wh'])

    # Scaling factor: relates (N_map × B_map) to total energy charged
    #   E_charged ∝ N_map × usable_energy_per_MAP
    #   usable_energy_per_MAP ∝ B_map
    scaling_factor = (total_energy_charged
                      / (sim_num_maps * sim_map_battery_wh))

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = gp.Model("PostSimulation_BusMAP_MILP")
    model.setParam('OutputFlag', 1)

    # --- decision variables ---
    B = {}
    for l in line_ids:
        B[l] = model.addVar(lb=bus_cap_min_kwh, ub=bus_cap_max_kwh,
                            vtype=GRB.CONTINUOUS, name=f"B_{l}")

    B_map = model.addVar(lb=map_cap_min_kwh, ub=map_cap_max_kwh,
                         vtype=GRB.CONTINUOUS, name="B_map")

    N_map = model.addVar(lb=0, ub=max_maps, vtype=GRB.INTEGER, name="N_map")

    # Auxiliary: W ≈ N_map × B_map  (linearised via McCormick envelopes)
    W = model.addVar(lb=0.0, ub=max_maps * map_cap_max_kwh,
                     vtype=GRB.CONTINUOUS, name="W_NmapBmap")

    # SOC violation slacks (bus)
    s_bus = {}
    v_bus = {}
    for b in bus_ids:
        s_bus[b] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                name=f"s_bus_{b}")
        v_bus[b] = model.addVar(vtype=GRB.BINARY, name=f"v_bus_{b}")

    # SOC violation slack (MAP, aggregate across all MAPs)
    s_map = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="s_map")
    v_map = model.addVar(vtype=GRB.BINARY, name="v_map")

    # Overnight energy (Wh)
    E_overnight = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                               name="E_overnight")

    # Total energy charged by MAPs under new sizing (Wh)
    E_charged_scaled = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                    name="E_charged_scaled")

    model.update()

    # ---------------------------------------------------------------
    # McCormick envelope for W = N_map × B_map
    # ---------------------------------------------------------------
    NL, NU = 0, max_maps
    BL, BU = map_cap_min_kwh, map_cap_max_kwh

    model.addConstr(W >= NL * B_map + BL * N_map - NL * BL,
                    "McCormick_LB1")
    model.addConstr(W >= NU * B_map + BU * N_map - NU * BU,
                    "McCormick_LB2")
    model.addConstr(W <= NU * B_map + BL * N_map - NU * BL,
                    "McCormick_UB1")
    model.addConstr(W <= NL * B_map + BU * N_map - NL * BU,
                    "McCormick_UB2")

    # ---------------------------------------------------------------
    # Constraints
    # ---------------------------------------------------------------
    BIG_M = 1e7

    # 1. Bus SOC ≥ 20%  (per bus, with penalty slack)
    #    new_min_soc(b) = B_l·1000 − deficit(b)
    #    deficit(b)     = sim_capacity − min_soc_wh(b)
    #    require: new_min_soc(b) ≥ 0.20 · B_l·1000  ⟹
    #      (1−0.20)·B_l·1000 ≥ deficit(b) − s_bus(b)
    for bus_id in bus_ids:
        bdata = per_bus[bus_id]
        lid = bdata['line_id']
        deficit = sim_battery_wh - bdata['min_soc_wh']

        model.addConstr(
            (1.0 - BUS_MIN_SOC_FRACTION) * B[lid] * 1000
            - deficit + s_bus[bus_id] >= 0,
            name=f"bus_soc_{bus_id}")

        # link slack → violation indicator
        model.addConstr(s_bus[bus_id] <= BIG_M * v_bus[bus_id],
                        name=f"bus_viol_{bus_id}")

    # 2. MAP SOC ≥ 10%  (aggregate worst-case, with penalty slack)
    #    new_min_map_soc = B_map·1000 − max_map_deficit
    #    require: (1−0.10)·B_map·1000 ≥ max_map_deficit − s_map
    model.addConstr(
        (1.0 - MAP_MIN_SOC_FRACTION) * B_map * 1000
        - max_map_deficit + s_map >= 0,
        name="map_soc")
    model.addConstr(s_map <= BIG_M * v_map, name="map_viol")

    # 3. Scaled energy charged by MAPs
    #    E_charged_scaled = scaling_factor × W × 1000   (Wh)
    model.addConstr(E_charged_scaled == scaling_factor * W * 1000,
                    name="charged_scaling")

    # Upper-bound: cannot charge more than consumed
    model.addConstr(E_charged_scaled <= total_energy_consumed,
                    name="charged_cap")

    # 4. Overnight energy = consumed − charged  (Wh, ≥ 0)
    model.addConstr(E_overnight >= total_energy_consumed - E_charged_scaled,
                    name="overnight_energy")

    # ---------------------------------------------------------------
    # Objective
    # ---------------------------------------------------------------
    bus_battery_cost = gp.quicksum(
        BATTERY_COST_PER_KWH * B[l] * per_line[l]['num_buses']
        for l in line_ids
    )
    map_battery_cost = BATTERY_COST_PER_KWH * W
    map_hardware_cost = MAP_HARDWARE_COST * N_map
    overnight_cost = OVERNIGHT_CHARGE_COST_PER_KWH * E_overnight / 1000.0
    bus_penalty = BUS_SOC_VIOLATION_PENALTY * gp.quicksum(
        v_bus[b] for b in bus_ids)
    map_penalty = MAP_SOC_VIOLATION_PENALTY * v_map

    model.setObjective(
        bus_battery_cost
        + map_battery_cost
        + map_hardware_cost
        + overnight_cost
        + bus_penalty
        + map_penalty,
        GRB.MINIMIZE,
    )

    # ---------------------------------------------------------------
    # Solve
    # ---------------------------------------------------------------
    model.optimize()

    # ---------------------------------------------------------------
    # Extract results
    # ---------------------------------------------------------------
    if model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        bus_bat = {l: B[l].X for l in line_ids}
        n_map_val = int(round(N_map.X))
        b_map_val = B_map.X
        e_overnight_val = E_overnight.X

        cost_bus_bat = sum(
            BATTERY_COST_PER_KWH * B[l].X * per_line[l]['num_buses']
            for l in line_ids)
        cost_map_bat = BATTERY_COST_PER_KWH * W.X
        cost_map_hw = MAP_HARDWARE_COST * n_map_val
        cost_overnight = OVERNIGHT_CHARGE_COST_PER_KWH * e_overnight_val / 1000.0
        n_bus_violations = sum(int(round(v_bus[b].X)) for b in bus_ids)
        n_map_violations = int(round(v_map.X))
        cost_penalty = (BUS_SOC_VIOLATION_PENALTY * n_bus_violations
                        + MAP_SOC_VIOLATION_PENALTY * n_map_violations)

        return {
            'status': ('optimal' if model.status == GRB.OPTIMAL
                       else 'suboptimal'),
            'objective_value': model.objVal,
            'bus_battery_kwh': bus_bat,
            'map_battery_kwh': b_map_val,
            'num_maps': n_map_val,
            'overnight_energy_kwh': e_overnight_val / 1000.0,
            'total_cost_breakdown': {
                'bus_battery_cost': cost_bus_bat,
                'map_battery_cost': cost_map_bat,
                'map_hardware_cost': cost_map_hw,
                'overnight_cost': cost_overnight,
                'penalty_cost': cost_penalty,
                'bus_soc_violations': n_bus_violations,
                'map_soc_violations': n_map_violations,
            },
        }

    return {
        'status': ('infeasible' if model.status == GRB.INFEASIBLE
                   else f'gurobi_status_{model.status}'),
        'objective_value': None,
    }


# ========================
# MAIN ENTRY POINT
# ========================

def post_simulation_optimize(results, stage2_sim, bus_lines,
                             max_maps=10,
                             bus_cap_min_kwh=50.0,
                             bus_cap_max_kwh=500.0,
                             map_cap_min_kwh=50.0,
                             map_cap_max_kwh=500.0):
    """
    Run the post-simulation MILP optimisation.

    Parameters
    ----------
    results : dict
        Simulation results from ``run_terminal_charging_simulation``.
    stage2_sim : Stage2DESTerminalChargingPreemptive
        The simulation object.
    bus_lines : list[BusLineData]
        Bus line metadata.
    max_maps : int
        Maximum number of MAPs to consider.
    bus_cap_min_kwh, bus_cap_max_kwh : float
        Bounds on bus battery capacity.
    map_cap_min_kwh, map_cap_max_kwh : float
        Bounds on MAP battery capacity.

    Returns
    -------
    dict  with optimisation results, or ``None`` if Gurobi is unavailable.
    """
    print("\n" + "=" * 70)
    print("POST-SIMULATION MILP OPTIMIZATION (Gurobi)")
    print("=" * 70)

    # --- check for Gurobi ---
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        print("\n⚠  gurobipy is not installed. "
              "Install it with:  pip install gurobipy")
        print("Skipping MILP optimisation.\n")
        return None

    # --- extract data ---
    sim_data = extract_simulation_data(results, stage2_sim, bus_lines)

    print("\nSimulation Data Summary:")
    print(f"  Battery capacity (sim): "
          f"{sim_data['system']['sim_battery_wh'] / 1000:.1f} kWh")
    print(f"  Number of MAPs (sim):   "
          f"{sim_data['system']['sim_num_maps']}")
    print(f"  MAP battery (sim):      "
          f"{sim_data['system']['sim_map_battery_wh'] / 1000:.1f} kWh")
    print(f"  Total energy consumed:  "
          f"{sim_data['system']['total_energy_consumed_wh'] / 1e6:.3f} MWh")
    print(f"  Total energy charged:   "
          f"{sim_data['system']['total_energy_charged_wh'] / 1e6:.3f} MWh")
    print(f"  Number of buses:        "
          f"{sim_data['system']['num_buses']}")

    print("\nPer-Line Summary:")
    for lid, ldata in sorted(sim_data['per_line'].items()):
        print(f"  Line {lid}: {ldata['num_buses']} buses, "
              f"{ldata['total_num_trips']} trips, "
              f"max deficit {ldata['max_energy_deficit_wh'] / 1000:.1f} kWh, "
              f"total energy {ldata['total_energy_consumed_wh'] / 1e6:.3f} MWh")

    if sim_data['per_map']:
        print("\nPer-MAP Summary (simulation):")
        for mid, mdata in sorted(sim_data['per_map'].items()):
            print(f"  MAP {mid}: delivered "
                  f"{mdata['total_energy_delivered_wh'] / 1000:.1f} kWh, "
                  f"min SOC {mdata['min_soc_wh'] / 1000:.1f} kWh")

    # --- solve ---
    opt_results = run_milp_optimization(
        sim_data,
        max_maps=max_maps,
        bus_cap_min_kwh=bus_cap_min_kwh,
        bus_cap_max_kwh=bus_cap_max_kwh,
        map_cap_min_kwh=map_cap_min_kwh,
        map_cap_max_kwh=map_cap_max_kwh,
    )

    # --- report ---
    print("\n" + "=" * 70)
    print("MILP OPTIMIZATION RESULTS")
    print("=" * 70)

    if opt_results['objective_value'] is not None:
        print(f"\nStatus: {opt_results['status']}")
        print(f"Total Objective (Cost): ${opt_results['objective_value']:,.2f}")

        print("\nOptimal Bus Battery Capacities (per line):")
        for lid, cap in sorted(opt_results['bus_battery_kwh'].items()):
            n = sim_data['per_line'][lid]['num_buses']
            print(f"  Line {lid}: {cap:>7.1f} kWh  × {n} buses  "
                  f"= {cap * n:>10.1f} kWh total")

        print(f"\nOptimal MAP Battery Capacity: "
              f"{opt_results['map_battery_kwh']:.1f} kWh")
        print(f"Optimal Number of MAPs:      {opt_results['num_maps']}")
        print(f"Overnight Energy:            "
              f"{opt_results['overnight_energy_kwh']:.1f} kWh")

        cb = opt_results['total_cost_breakdown']
        print("\nCost Breakdown:")
        print(f"  Bus battery cost:      ${cb['bus_battery_cost']:>14,.2f}")
        print(f"  MAP battery cost:      ${cb['map_battery_cost']:>14,.2f}")
        print(f"  MAP hardware cost:     ${cb['map_hardware_cost']:>14,.2f}")
        print(f"  Overnight charge cost: ${cb['overnight_cost']:>14,.2f}")
        print(f"  Penalty cost:          ${cb['penalty_cost']:>14,.2f}")
        print(f"  Bus SOC violations:    {cb['bus_soc_violations']}")
        print(f"  MAP SOC violations:    {cb['map_soc_violations']}")
    else:
        print(f"\nOptimisation failed: {opt_results['status']}")

    print(f"\n{'=' * 70}\n")

    return opt_results
