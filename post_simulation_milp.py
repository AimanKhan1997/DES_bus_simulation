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

# ========================
# ENERGY MODEL CONSTANTS
# (from DES_model.energy_per_meter_for_capacity:
#   rate = 2.7 - (470 - capacity_kwh) * 0.0005
#        = 2.465 + 0.0005 * capacity_kwh  Wh/m)
# ========================
_EPM_A = 2.465   # Wh/m  – constant term of energy-per-metre formula
_EPM_B = 0.0005  # Wh/m per kWh of battery capacity – linear term

# Conservative physical fraction of MAP fleet energy that can be
# delivered to buses in one operating day (accounts for movement
# overhead, self-charging, and scheduling gaps).
MAP_DELIVERY_EFFICIENCY = 0.45

# Minimum dwell time (seconds) between consecutive trips that counts
# as a charging opportunity when computing the worst-case inter-charge
# trip-chain distance.
_LAYOVER_THRESHOLD_S = 600  # 10 minutes


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
# STATIC DEMAND HELPERS
# ========================

def compute_static_line_demand(stage2_sim):
    """Compute worst-case inter-charge trip-chain distance (metres) per line.

    This is independent of any battery capacity used in the simulation — it
    is derived purely from the GTFS schedule geometry.  The result is used
    to build bus SOC feasibility constraints that do not depend on the
    starting values of the optimisation loop.

    Algorithm
    ---------
    For each bus the trip chain is walked in chronological order.  When the
    dwell time between two consecutive trips is at least ``_LAYOVER_THRESHOLD_S``
    seconds the chain is broken (charging opportunity at the terminal).  The
    worst-case distance across all chains and all buses on a line is returned.

    Parameters
    ----------
    stage2_sim : Stage2DESTerminalChargingPreemptive
        The simulation object (provides ``bus_trips_dict`` and ``sim``).

    Returns
    -------
    dict  ``{line_id: worst_case_chain_distance_m}``
        Returns an empty dict if GTFS data is unavailable.
    """
    result: dict = {}
    try:
        bus_trips = stage2_sim.bus_trips_dict
        gtfs = stage2_sim.sim.gtfs
        geod = stage2_sim.sim.geod
    except AttributeError:
        return result

    # Group buses by line_id
    line_buses: dict = defaultdict(list)
    for bus_id in bus_trips:
        try:
            line_id = bus_id.split('_')[0].replace('line', '')
        except Exception:
            continue
        line_buses[line_id].append(bus_id)

    for line_id, bus_ids in line_buses.items():
        max_chain_dist = 0.0

        for bus_id in bus_ids:
            trip_ids = bus_trips.get(bus_id, [])

            # Sort trips chronologically
            def _start_time(t):
                seq = gtfs.stop_sequence(t)
                return seq[0]["arrival"] if seq else float('inf')

            try:
                sorted_trips = sorted(trip_ids, key=_start_time)
            except Exception:
                continue

            # Build per-trip (distance_m, start_time_s, end_time_s) records
            trip_records = []
            for trip_id in sorted_trips:
                try:
                    seq = gtfs.stop_sequence(trip_id)
                    if not seq or len(seq) < 2:
                        continue
                    dist = 0.0
                    for i in range(1, len(seq)):
                        prev, curr = seq[i - 1], seq[i]
                        try:
                            _, _, d = geod.inv(
                                prev["lon"], prev["lat"],
                                curr["lon"], curr["lat"],
                            )
                            dist += abs(d)
                        except Exception:
                            pass
                    trip_records.append(
                        (dist, seq[0]["arrival"], seq[-1]["arrival"])
                    )
                except Exception:
                    continue

            if not trip_records:
                continue

            # Walk the trip chain; break on layover ≥ _LAYOVER_THRESHOLD_S
            chain_dist = trip_records[0][0]
            worst_bus = chain_dist

            for i in range(1, len(trip_records)):
                d_i, t_start_i, _ = trip_records[i]
                _, _, t_end_prev = trip_records[i - 1]
                layover = t_start_i - t_end_prev
                if layover >= _LAYOVER_THRESHOLD_S:
                    # charging opportunity → restart chain
                    worst_bus = max(worst_bus, chain_dist)
                    chain_dist = d_i
                else:
                    chain_dist += d_i

            worst_bus = max(worst_bus, chain_dist)
            max_chain_dist = max(max_chain_dist, worst_bus)

        if max_chain_dist > 0.0:
            result[line_id] = max_chain_dist

    return result


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
    total_map_movement_energy_wh = 0.0

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
            
            # Get movement energy for this MAP
            map_movement_energy = stage2_sim.map_tracker.map_movement_energy_wh.get(map_id, 0.0)
            total_map_movement_energy_wh += map_movement_energy

            per_map[map_id] = {
                'total_energy_delivered_wh': total_delivered,
                'min_soc_wh': min_soc,
                'self_charge_energy_wh': map_self_charge,
                'movement_energy_wh': map_movement_energy,
            }

    # --- system totals ---
    total_energy_consumed = sum(b['total_energy_consumed_wh'] for b in per_bus.values())
    total_energy_charged = results['total_energy_charged_wh']

    # --- static line demands (initialisation-independent) ---
    # Compute worst-case inter-charge trip-chain distances from GTFS geometry.
    # These are independent of any battery capacity and are used in the MILP
    # to build SOC feasibility constraints that do not depend on the starting
    # values of the optimisation loop.
    static_line_distances = compute_static_line_demand(stage2_sim)
    for lid, ld in line_data.items():
        ld['worst_case_distance_m'] = static_line_distances.get(lid, 0.0)

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
            'total_map_movement_energy_wh': total_map_movement_energy_wh,
            'num_buses': results['num_buses'],
        },
    }


# ========================
# MILP FORMULATION
# ========================

def run_milp_optimization(sim_data,
                          max_maps=20,
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

    # ------------------------------------------------------------------
    # MAP delivery scaling
    # ------------------------------------------------------------------
    # We express MAP benefit as an UPPER BOUND on E_charged_scaled rather
    # than as an equality.  This has two advantages:
    #
    #   a) When the previous simulation had no MAPs (sim_num_maps == 0) the
    #      old code forced scaling_factor = 0 and E_charged_scaled ≡ 0,
    #      making the MILP believe MAPs are useless.  The new approach uses a
    #      conservative physical bound so MAPs always get credit.
    #
    #   b) The upper bound is the MINIMUM of the simulation-calibrated factor
    #      (when available) and the physical delivery bound, which avoids both
    #      over-optimism and the start-dependence caused by relying solely on
    #      calibration from a single run.
    #
    # Physical upper bound: MAP fleet can deliver at most
    #   MAP_DELIVERY_EFFICIENCY × (1 − MAP_MIN_SOC) × W × 1000  Wh
    # where MAP_DELIVERY_EFFICIENCY is a conservative operational fraction
    # that accounts for MAP movement overhead, scheduling gaps, and the
    # energy MAPs need for self-charging.
    physical_delivery_factor = MAP_DELIVERY_EFFICIENCY * (1.0 - MAP_MIN_SOC_FRACTION)

    if sim_num_maps > 0 and sim_map_battery_wh > 0:
        sim_calibrated_factor = total_energy_charged / (sim_num_maps * sim_map_battery_wh)
        # Take minimum: use simulation evidence but never exceed physical bound
        effective_delivery_factor = min(sim_calibrated_factor, physical_delivery_factor)
    else:
        # No historical calibration data — rely on the physical bound alone.
        # This avoids the previous forced-zero that prevented MAPs from being
        # credited when the prior run happened to use no MAPs.
        effective_delivery_factor = physical_delivery_factor

    # Self-charge scaling factor: relates self-charge energy to MAP sizing.
    # This adds to overnight cost, so keeping it at 0 when sim had no MAPs
    # is conservative (we never under-count overnight energy).
    if sim_num_maps > 0 and sim_map_battery_wh > 0:
        self_charge_scaling = (total_map_self_charge
                               / (sim_num_maps * sim_map_battery_wh))
    else:
        self_charge_scaling = 0.0

    # Total MAP fleet capacity in the PREVIOUS simulation (Wh).
    # Used as the reference baseline for the per-bus MAP charging credit.
    sim_total_map_cap_wh = sim_num_maps * sim_map_battery_wh

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
    E_overnight = model.addVar(lb=0.0, ub=total_energy_consumed, vtype=GRB.CONTINUOUS,
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

    # Total number of buses (used for physical MAP credit fallback below)
    total_num_buses = max(1, len(bus_ids))

    # 1. Bus SOC ≥ 20%  (per bus, with penalty slack)
    #
    # STATIC DEMAND FORMULATION (initialisation-independent)
    # -------------------------------------------------------
    # Instead of using the simulation-observed deficit
    #   deficit(b) = bus_sim_cap − min_soc_wh(b)
    # which depends on the battery size used in the last simulation run,
    # we use the worst-case inter-charge trip-chain demand computed purely
    # from GTFS route geometry:
    #
    #   demand_wh(B_l) = chain_dist_m × energy_per_metre(B_l)
    #                  = chain_dist_m × (_EPM_A + _EPM_B × B_l)
    #
    # where _EPM_A = 2.465 Wh/m and _EPM_B = 0.0005 Wh/(m·kWh).
    # This is linear in B_l (the decision variable), so the constraint
    # remains an LP/MIP constraint:
    #
    #   (1 − 0.20) × B_l × 1000 ≥ demand_wh(B_l) − map_credit − s_bus(b)
    #   ⟺  [(1−0.20)×1000 − chain_dist_m × _EPM_B] × B_l
    #       ≥ chain_dist_m × _EPM_A − map_credit − s_bus(b)
    #
    # When no GTFS distance is available, we fall back to the simulation
    # deficit (backward compatibility).
    #
    # MAP CREDIT (robust to sim_num_maps = 0)
    # ----------------------------------------
    # The per-bus MAP charging credit is:
    #   map_credit = charging_rate × (W×1000 − baseline)
    # where charging_rate captures how much of the total MAP capacity
    # was delivered to this bus in the prior run.
    #
    # When the prior simulation had NO MAPs (sim_total_map_cap_wh = 0),
    # the old code set charging_rate = 0, making MAPs appear useless.
    # We now fall back to a conservative physical estimate:
    #   charging_rate_physical = physical_delivery_factor / total_num_buses
    # This ensures that even with no prior MAP data, the MILP can explore
    # configurations where MAPs reduce individual bus battery requirements.

    for bus_id in bus_ids:
        bdata = per_bus[bus_id]
        lid = bdata['line_id']

        # --- demand term (static, from GTFS geometry) ---
        chain_dist_m = per_line[lid].get('worst_case_distance_m', 0.0)

        if chain_dist_m > 0.0:
            # Static demand as a linear function of B_l (kWh):
            #   demand_wh = chain_dist_m * _EPM_A  +  chain_dist_m * _EPM_B * B_l
            static_demand_const_wh = chain_dist_m * _EPM_A
            static_demand_coeff = chain_dist_m * _EPM_B  # multiplies B_l (kWh) → Wh
            use_static = True
        else:
            # Fallback: use simulation-based deficit (capacity-dependent but
            # kept for lines where GTFS geometry is unavailable)
            bus_sim_cap = bdata['sim_battery_wh']
            static_demand_const_wh = bus_sim_cap - bdata['min_soc_wh']
            static_demand_coeff = 0.0
            use_static = False

        # --- MAP credit rate ---
        per_bus_charged = bdata.get('energy_charged_wh', 0.0)
        if sim_total_map_cap_wh > 0:
            # Simulation-calibrated rate: energy delivered to this bus per
            # unit of total MAP fleet capacity.
            charging_rate = per_bus_charged / sim_total_map_cap_wh
        else:
            # Physical fallback: assume MAP fleet delivers its effective
            # usable capacity uniformly across all buses.
            charging_rate = physical_delivery_factor / total_num_buses

        # map_credit = charging_rate × (W×1000 − baseline)
        #   > 0 when MAP capacity grows → demand falls
        #   < 0 when MAP capacity shrinks → demand rises
        map_credit = charging_rate * (W * 1000 - sim_total_map_cap_wh)

        # SOC feasibility constraint (linear in B_l):
        #   (1 − min_soc_fraction) × B_l × 1000
        #   − (static_demand_const_wh + static_demand_coeff × B_l)
        #   + map_credit + s_bus ≥ 0
        model.addConstr(
            (1.0 - BUS_MIN_SOC_FRACTION) * B[lid] * 1000
            - static_demand_const_wh
            - static_demand_coeff * B[lid]
            + map_credit
            + s_bus[bus_id] >= 0,
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

    # 3. MAP delivery energy — physical upper bound (not equality)
    #
    # OLD:  E_charged_scaled == scaling_factor × W × 1000
    #   Problem: when sim_num_maps == 0, scaling_factor was forced to 0 so
    #   E_charged_scaled ≡ 0 regardless of the MAP fleet the MILP proposes.
    #
    # NEW:  E_charged_scaled ≤ effective_delivery_factor × W × 1000
    #   The MILP maximises E_charged_scaled (reduces overnight cost), so it
    #   will naturally push E_charged_scaled to its upper bound.  The bound
    #   is the MINIMUM of the simulation-calibrated factor and the physical
    #   delivery limit, so it is never over-optimistic AND is always > 0
    #   even when the prior run had no MAPs.
    model.addConstr(E_charged_scaled <= effective_delivery_factor * W * 1000,
                    name="charged_scaling_ub")

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
    map_battery_cost = BATTERY_COST_PER_KWH * N_map * B_map
    map_hardware_cost = MAP_HARDWARE_COST * N_map
    # Use fixed overnight cost based on simulation's charger power tier
    # The E_overnight variable is still constrained in the SOC constraint,
    # but the cost is fixed to the simulation-based per-bus charger cost
    # to ensure consistency between objective and cost breakdown
    overnight_cost_expr = per_bus_charger_cost * num_buses
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
        cost_map_bat = BATTERY_COST_PER_KWH * n_map_val * b_map_val
        cost_map_hw = MAP_HARDWARE_COST * n_map_val
        cost_overnight = per_bus_charger_cost * num_buses
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
                             max_maps=20,
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
    print(f"  MAP movement energy:    "
          f"{sim_data['system']['total_map_movement_energy_wh'] / 1e6:.3f} MWh")
    print(f"  Number of buses:        "
          f"{sim_data['system']['num_buses']}")

    print("\nPer-Line Summary:")
    for lid, ldata in sorted(sim_data['per_line'].items()):
        wc_dist_km = ldata.get('worst_case_distance_m', 0.0) / 1000.0
        print(f"  Line {lid}: {ldata['num_buses']} buses, "
              f"{ldata['total_num_trips']} trips, "
              f"max deficit {ldata['max_energy_deficit_wh'] / 1000:.1f} kWh, "
              f"total energy {ldata['total_energy_consumed_wh'] / 1e6:.3f} MWh, "
              f"worst-chain dist {wc_dist_km:.1f} km"
              + (" (static)" if wc_dist_km > 0 else " (fallback: sim deficit)"))

    if sim_data['per_map']:
        print("\nPer-MAP Summary (simulation):")
        for mid, mdata in sorted(sim_data['per_map'].items()):
            print(f"  MAP {mid}: delivered "
                  f"{mdata['total_energy_delivered_wh'] / 1000:.1f} kWh, "
                  f"min SOC {mdata['min_soc_wh'] / 1000:.1f} kWh, "
                  f"self-charge {mdata['self_charge_energy_wh'] / 1000:.1f} kWh, "
                  f"movement {mdata['movement_energy_wh'] / 1000:.1f} kWh")

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

