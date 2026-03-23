"""
Post-Simulation MILP Optimizer using Gurobi

Optimizes bus battery capacity per line, MAP battery capacity, and number of MAPs
to minimize total system cost, based on outputs from the DES charging simulation.

Decision Variables:
    B_l   : Battery capacity (kWh) per bus on line l          (integer, step 10)
    B_map : MAP battery capacity (kWh)                        (integer, step 10)
    N_map : Number of MAPs                                    (integer)

Objective: Minimize total system cost
    = bus battery cost  (115 $/kWh × capacity × buses per line)
    + MAP battery cost  (115 $/kWh × capacity × number of MAPs)
    + MAP hardware cost (40,000 $ × number of MAPs)
    + overnight charging infrastructure cost (cost_per_kw × E_overnight_kwh / 4)
    + penalty for SOC violations

Overnight Charging Cost Tiers (per kW of charger capacity):
    Charger power 0–22 kW   → $250/kW
    Charger power 22–50 kW  → $450/kW
    Charger power 50–150 kW → $550/kW
    Charger power 150–350 kW → $600/kW
    Charger power ≥350 kW   → $600/kW

    The tier is determined from the simulation's charger power level.
    Overnight cost = cost_per_kw × (E_overnight / 1000) / 4
    where E_overnight (Wh) is a MILP decision variable, so the optimizer
    can see the benefit of MAPs reducing overnight energy requirements.

Constraints:
    - No bus may go below 20% SOC (with penalty slack)
    - No MAP may go below 10% SOC (with penalty slack)
    - MAP self-charging energy is accounted for in the overnight energy balance
"""

from collections import defaultdict

# ========================
# COST PARAMETERS
# ========================
BATTERY_COST_PER_KWH = 115.0              # $/kWh for bus and MAP batteries
MAP_HARDWARE_COST = 40000.0               # $ per MAP unit
BUS_SOC_VIOLATION_PENALTY = 1_000_000.0   # penalty per bus that violates 20% SOC
MAP_SOC_VIOLATION_PENALTY = 1_000_000.0   # penalty per MAP that violates 10% SOC
BUS_MIN_SOC_FRACTION = 0.20               # 20% minimum bus SOC
MAP_MIN_SOC_FRACTION = 0.10               # 10% minimum MAP SOC
OVERNIGHT_CHARGING_HOURS = 4.0            # hours available for overnight charging


def get_overnight_charger_cost_per_kw(charger_power_kw):
    """
    Return the overnight charger infrastructure cost ($/kW) based on
    the required charger power level.

    Tiers
    -----
        0 – 22 kW   → $250 / kW
        22 – 50 kW  → $450 / kW
        50 – 150 kW → $550 / kW
        150 – 350 kW → $600 / kW
        ≥ 350 kW    → $600 / kW
    """
    if charger_power_kw <= 22:
        return 250.0
    elif charger_power_kw <= 50:
        return 450.0
    elif charger_power_kw <= 150:
        return 550.0
    elif charger_power_kw <= 350:
        return 600.0
    else:
        return 600.0


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

    # Use the MAP battery capacity from the simulation object if available
    actual_map_cap_wh = getattr(stage2_sim, 'map_battery_capacity_wh', MAP_BATTERY_CAPACITY_WH)

    # Per-line simulation battery capacities (Wh).
    # If the simulation provided per-line capacities, use them;
    # otherwise fall back to the single sim_battery_wh for all lines.
    line_battery_capacities_wh = results.get('line_battery_capacities_wh', {})

    # --- per-bus ---
    per_bus = {}
    for bus_id, stats in bus_stats.items():
        lid = stats['line_id']
        # Use per-line capacity if available, else fall back to uniform sim value
        bus_sim_cap_wh = line_battery_capacities_wh.get(lid, sim_battery_wh)
        per_bus[bus_id] = {
            'line_id': lid,
            'total_energy_consumed_wh': stats['total_energy_consumed_wh'],
            'energy_charged_wh': stage2_sim.bus_energy_charged.get(bus_id, 0.0),
            'min_soc_wh': stats['min_soc_wh'],
            'min_soc_ratio': stats['min_soc_ratio'],
            'num_trips': len(stage2_sim.bus_trips_dict.get(bus_id, [])),
            'sim_battery_wh': bus_sim_cap_wh,
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
        bus_sim_cap = bdata['sim_battery_wh']
        deficit = bus_sim_cap - bdata['min_soc_wh']
        ld['max_energy_deficit_wh'] = max(ld['max_energy_deficit_wh'], deficit)
        ld['total_energy_charged_wh'] += bdata['energy_charged_wh']
        ld['total_num_trips'] += bdata['num_trips']
        ld['buses'].append(bus_id)
        # store per-line sim capacity (same for all buses on the line)
        ld['sim_battery_wh'] = bus_sim_cap

    # --- per-MAP ---
    per_map = {}
    sim_map_battery_wh = actual_map_cap_wh
    total_map_self_charge_wh = 0.0

    if sim_num_maps > 0:
        for map_id in range(sim_num_maps):
            total_delivered = stage2_sim.map_tracker.map_total_energy.get(map_id, 0.0)
            soc_history = stage2_sim.map_tracker.map_soc_history.get(map_id, [])
            min_soc = min((s for _, s in soc_history), default=sim_map_battery_wh)

            # Calculate self-charge energy from SOC history:
            # Whenever SOC increases between consecutive entries, that is self-charging
            map_self_charge = 0.0
            for i in range(1, len(soc_history)):
                delta = soc_history[i][1] - soc_history[i - 1][1]
                if delta > 0:
                    map_self_charge += delta
            total_map_self_charge_wh += map_self_charge

            per_map[map_id] = {
                'total_energy_delivered_wh': total_delivered,
                'min_soc_wh': min_soc,
                'self_charge_energy_wh': map_self_charge,
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
            'total_map_self_charge_wh': total_map_self_charge_wh,
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
                          map_cap_max_kwh=500.0,
                          feedback_constraints=None):
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
    feedback_constraints : list[dict], optional
        List of constraint dicts from the MILP–simulation feedback loop.
        Each dict has keys:
            'type'    – one of 'bus_min_cap', 'map_min_cap', 'min_maps'
            'line_id' – (for bus_min_cap) which line to constrain
            'value'   – the minimum value to enforce

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
    sim_num_maps = sys_data['sim_num_maps']
    sim_map_battery_wh = sys_data['sim_map_battery_wh']
    total_energy_consumed = sys_data['total_energy_consumed_wh']
    total_energy_charged = sys_data['total_energy_charged_wh']
    total_map_self_charge = sys_data.get('total_map_self_charge_wh', 0.0)

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
    # When the simulation had no MAPs, no historical charging data exists
    # to calibrate the scaling; set to 0 so the MILP can still add MAPs
    # but overnight energy stays at total consumption.
    if sim_num_maps > 0 and sim_map_battery_wh > 0:
        scaling_factor = (total_energy_charged
                          / (sim_num_maps * sim_map_battery_wh))
    else:
        scaling_factor = 0.0

    # Self-charge scaling factor: relates self-charge energy to MAP sizing
    if sim_num_maps > 0 and sim_map_battery_wh > 0:
        self_charge_scaling = (total_map_self_charge
                               / (sim_num_maps * sim_map_battery_wh))
    else:
        self_charge_scaling = 0.0

    # ---------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------
    model = gp.Model("PostSimulation_BusMAP_MILP")
    model.setParam('OutputFlag', 1)

    import math

    # --- decision variables ---
    # Bus battery capacity per line: discrete in steps of 10 kWh.
    # We introduce an integer helper K_l so that B_l = K_l × 10.
    K = {}     # integer helper: K_l ∈ {ceil(min/10) .. floor(max/10)}
    B = {}     # derived continuous: B_l = K_l × 10  (kWh)
    k_lb = math.ceil(bus_cap_min_kwh / 10)
    k_ub = math.floor(bus_cap_max_kwh / 10)
    for l in line_ids:
        K[l] = model.addVar(lb=k_lb, ub=k_ub,
                            vtype=GRB.INTEGER, name=f"K_{l}")
        B[l] = model.addVar(lb=bus_cap_min_kwh, ub=bus_cap_max_kwh,
                            vtype=GRB.CONTINUOUS, name=f"B_{l}")

    # MAP battery capacity: discrete in steps of 10 kWh (K_map × 10).
    # Like bus batteries, we use an integer helper K_map and link B_map = K_map × 10.
    k_map_lb = math.ceil(map_cap_min_kwh / 10)
    k_map_ub = math.floor(map_cap_max_kwh / 10)
    K_map = model.addVar(lb=k_map_lb, ub=k_map_ub,
                         vtype=GRB.INTEGER, name="K_map")
    B_map = model.addVar(lb=k_map_lb * 10, ub=k_map_ub * 10,
                         vtype=GRB.CONTINUOUS, name="B_map")

    N_map = model.addVar(lb=0, ub=max_maps, vtype=GRB.INTEGER, name="N_map")

    # Auxiliary: W ≈ N_map × B_map  (linearized via McCormick envelopes)
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
    # Link B_l = K_l × 10   (discrete steps of 10 kWh)
    # ---------------------------------------------------------------
    for l in line_ids:
        model.addConstr(B[l] == 10.0 * K[l], name=f"step10_{l}")

    # Link B_map = K_map × 10  (discrete steps of 10 kWh)
    model.addConstr(B_map == 10.0 * K_map, name="step10_map")

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
    # Big-M for linking slack to violation indicator.
    # The maximum possible slack equals the energy deficit (~500 kWh × 1000 = 5e5 Wh),
    # so 1e7 provides a safe margin without causing numerical issues.
    BIG_M = 1e7

    # 1. Bus SOC ≥ 20%  (per bus, with penalty slack)
    #    deficit(b) = sim_capacity_for_line − min_soc_wh(b)
    #    new_min_soc(b) = B_l·1000 − deficit(b) + map_charging_credit
    #    require: new_min_soc(b) ≥ 0.20 · B_l·1000  ⟹
    #      (1−0.20)·B_l·1000 ≥ deficit(b) − map_credit − s_bus(b)
    #
    # The MAP charging credit makes the deficit a function of MAP capacity
    # rather than frozen from one simulation.  For each bus, the credit is
    # proportional to how the total MAP capacity (W) changes relative to
    # the simulation baseline.
    sim_total_map_cap_wh = sim_num_maps * sim_map_battery_wh

    for bus_id in bus_ids:
        bdata = per_bus[bus_id]
        lid = bdata['line_id']
        bus_sim_cap = bdata['sim_battery_wh']
        deficit = bus_sim_cap - bdata['min_soc_wh']

        # Per-bus MAP charging credit rate:
        #   How much energy this bus received from MAPs per Wh of total MAP
        #   capacity.  As MAP capacity grows, the bus receives proportionally
        #   more charging, reducing its effective deficit.
        per_bus_charged = bdata.get('energy_charged_wh', 0.0)
        if sim_total_map_cap_wh > 0:
            charging_rate = per_bus_charged / sim_total_map_cap_wh
        else:
            charging_rate = 0.0

        # map_credit = charging_rate × (W×1000 − baseline)
        #   positive when MAP capacity increases → deficit shrinks
        #   negative when MAP capacity decreases → deficit grows
        map_credit = charging_rate * (W * 1000 - sim_total_map_cap_wh)

        model.addConstr(
            (1.0 - BUS_MIN_SOC_FRACTION) * B[lid] * 1000
            - deficit + map_credit + s_bus[bus_id] >= 0,
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

    # 4. Scaled self-charge energy supplied to MAPs during operation
    #    E_self_charge_scaled = self_charge_scaling × W × 1000   (Wh)
    E_self_charge_scaled = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS,
                                        name="E_self_charge_scaled")
    model.addConstr(E_self_charge_scaled == self_charge_scaling * W * 1000,
                    name="self_charge_scaling")

    # 5. Overnight energy = consumed − charged + self_charge  (Wh, ≥ 0)
    #    Self-charge energy represents grid power consumed during daytime
    #    operations to recharge MAPs.  This is added to the total grid energy
    #    requirement alongside the overnight depot charging.
    model.addConstr(E_overnight >= total_energy_consumed - E_charged_scaled
                    + E_self_charge_scaled,
                    name="overnight_energy")

    # ---------------------------------------------------------------
    # Feedback constraints from MILP–simulation loop
    # ---------------------------------------------------------------
    if feedback_constraints:
        for i, fc in enumerate(feedback_constraints):
            fc_type = fc.get('type', '')
            fc_val = fc.get('value', 0)
            if fc_type == 'bus_min_cap':
                lid = fc.get('line_id', '')
                if lid in B:
                    model.addConstr(B[lid] >= fc_val,
                                    name=f"fb_bus_min_{lid}_{i}")
            elif fc_type == 'map_min_cap':
                model.addConstr(B_map >= fc_val,
                                name=f"fb_map_min_{i}")
            elif fc_type == 'min_maps':
                model.addConstr(N_map >= fc_val,
                                name=f"fb_min_maps_{i}")
            elif fc_type == 'optimality_cut':
                # Tradeoff cut:  B_l + γ · N_map ≥ threshold
                # Allows the MILP to trade smaller batteries for more MAPs.
                lid = fc.get('line_id', '')
                gamma = fc.get('gamma', 0.0)
                threshold = fc.get('threshold', 0.0)
                if lid in B and gamma > 0:
                    model.addConstr(
                        B[lid] + gamma * N_map >= threshold,
                        name=f"fb_opt_cut_{lid}_{i}")

    # ---------------------------------------------------------------
    # Objective
    # ---------------------------------------------------------------
    # Calculate overnight charger power tier from simulation data
    # to determine the appropriate charging infrastructure cost.
    num_buses = sys_data['num_buses']
    sim_overnight_energy_kwh = max(0.0,
        (total_energy_consumed - total_energy_charged) / 1000.0)
    if num_buses > 0:
        energy_per_bus_kwh = sim_overnight_energy_kwh / num_buses
    else:
        energy_per_bus_kwh = 0.0
    charger_power_kw = energy_per_bus_kwh / OVERNIGHT_CHARGING_HOURS
    charger_cost_per_kw = get_overnight_charger_cost_per_kw(charger_power_kw)

    per_bus_charger_cost = charger_cost_per_kw * charger_power_kw
    total_overnight_cost = per_bus_charger_cost * num_buses

    print(f"\n  Overnight charging cost calculation:")
    print(f"    Sim overnight energy:  {sim_overnight_energy_kwh:.1f} kWh")
    print(f"    Number of buses:       {num_buses}")
    print(f"    Energy per bus:        {energy_per_bus_kwh:.1f} kWh")
    print(f"    Charger power level:   {charger_power_kw:.1f} kW")
    print(f"    Charger cost tier:     ${charger_cost_per_kw:.0f}/kW")
    print(f"    Per-bus charger cost:  ${per_bus_charger_cost:,.2f}")
    print(f"    Total overnight cost:  ${total_overnight_cost:,.2f}"
          f"  ({num_buses} buses × ${per_bus_charger_cost:,.2f})")

    bus_battery_cost = gp.quicksum(
        BATTERY_COST_PER_KWH * B[l] * per_line[l]['num_buses']
        for l in line_ids
    )
    map_battery_cost = BATTERY_COST_PER_KWH * W
    map_hardware_cost = MAP_HARDWARE_COST * N_map
    # Overnight cost is a linear function of E_overnight so that the MILP
    # can see the benefit of adding MAPs (which increase E_charged_scaled
    # and thus reduce E_overnight).
    # Total overnight cost = cost_per_kw × E_overnight_kwh / 4
    #   = charger_cost_per_kw × (E_overnight / 1000) / OVERNIGHT_CHARGING_HOURS
    overnight_cost_expr = (charger_cost_per_kw
                           * E_overnight / 1000.0
                           / OVERNIGHT_CHARGING_HOURS)
    bus_penalty = BUS_SOC_VIOLATION_PENALTY * gp.quicksum(
        v_bus[b] for b in bus_ids)
    map_penalty = MAP_SOC_VIOLATION_PENALTY * v_map

    total_cost_expr = (bus_battery_cost
                       + map_battery_cost
                       + map_hardware_cost
                       + overnight_cost_expr
                       + bus_penalty
                       + map_penalty)

    model.setObjective(total_cost_expr, GRB.MINIMIZE)

    # Cost upper-bound constraint (from feedback loop to find cheaper solutions)
    if feedback_constraints:
        for i, fc in enumerate(feedback_constraints):
            if fc.get('type') == 'cost_upper_bound':
                model.addConstr(total_cost_expr <= fc['value'],
                                name=f"fb_cost_ub_{i}")

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
        cost_overnight = (charger_cost_per_kw
                         * (e_overnight_val / 1000.0)
                         / OVERNIGHT_CHARGING_HOURS)
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
            'charger_power_kw': charger_power_kw,
            'charger_cost_per_kw': charger_cost_per_kw,
            'per_bus_charger_cost': per_bus_charger_cost,
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
                             map_cap_max_kwh=500.0,
                             feedback_constraints=None):
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
    feedback_constraints : list[dict], optional
        Extra constraints from the MILP–simulation feedback loop.

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
    print(f"  MAP self-charge energy: "
          f"{sim_data['system']['total_map_self_charge_wh'] / 1e6:.3f} MWh")
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
                  f"min SOC {mdata['min_soc_wh'] / 1000:.1f} kWh, "
                  f"self-charge {mdata['self_charge_energy_wh'] / 1000:.1f} kWh")

    # --- solve ---
    opt_results = run_milp_optimization(
        sim_data,
        max_maps=max_maps,
        bus_cap_min_kwh=bus_cap_min_kwh,
        bus_cap_max_kwh=bus_cap_max_kwh,
        map_cap_min_kwh=map_cap_min_kwh,
        map_cap_max_kwh=map_cap_max_kwh,
        feedback_constraints=feedback_constraints,
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
        print(f"Charger Power Level:         "
              f"{opt_results['charger_power_kw']:.1f} kW")
        print(f"Charger Cost Tier:           "
              f"${opt_results['charger_cost_per_kw']:.0f}/kW")
        print(f"Per-Bus Charger Cost:        "
              f"${opt_results['per_bus_charger_cost']:,.2f}")

        cb = opt_results['total_cost_breakdown']
        num_b = sim_data['system']['num_buses']
        print("\nCost Breakdown:")
        print(f"  Bus battery cost:      ${cb['bus_battery_cost']:>14,.2f}")
        print(f"  MAP battery cost:      ${cb['map_battery_cost']:>14,.2f}")
        print(f"  MAP hardware cost:     ${cb['map_hardware_cost']:>14,.2f}")
        print(f"  Overnight charge cost: ${cb['overnight_cost']:>14,.2f}"
              f"  ({num_b} buses × "
              f"${opt_results['per_bus_charger_cost']:,.2f})")
        print(f"  Penalty cost:          ${cb['penalty_cost']:>14,.2f}")
        print(f"  Bus SOC violations:    {cb['bus_soc_violations']}")
        print(f"  MAP SOC violations:    {cb['map_soc_violations']}")
    else:
        print(f"\nOptimisation failed: {opt_results['status']}")

    print(f"\n{'=' * 70}\n")

    return opt_results
