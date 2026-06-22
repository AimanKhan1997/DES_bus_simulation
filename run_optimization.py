"""
Run Terminal-Only Charging with INTEGRATED Optimized Preemption Strategy

Modes:
    python run_optimization.py --sim-only   Run a single simulation without the feedback loop (validation mode)
    (default)    Run the constraint-driven feasibility search
"""

from integration_stage2 import run_terminal_charging_simulation
#from post_simulation_milp import post_simulation_optimize
from dataclasses import dataclass
import math
import sys
import random


@dataclass
class BusLineData:
    line_id: str
    num_buses: int
    trips_df: object


def calculate_system_cost(bus_caps_kwh, num_maps, map_battery_kwh, sim_results, bus_lines):
    """
    Calculate total system cost for given capacities WITHOUT re-optimizing.

    This function uses the simulation results to determine charger tier and
    calculates all cost components directly.

    Parameters
    ----------
    bus_caps_kwh : dict
        {line_id: capacity_kwh} for buses on each line
    num_maps : int
        Number of MAPs
    map_battery_kwh : float
        MAP battery capacity in kWh
    sim_results : dict
        Simulation results including num_buses, total_energy_consumed_wh,
        total_energy_charged_wh, bus_statistics, min_soc_overall_ratio
    bus_lines : list[BusLineData]
        Bus line metadata (used to get num_buses per line)

    Returns
    -------
    dict with keys: 'total_cost', 'cost_breakdown' (dict with cost components)
    """
    # Cost parameters
    BATTERY_COST_PER_KWH = 115.0
    MAP_HARDWARE_COST = 40000.0
    BUS_SOC_VIOLATION_PENALTY = 1_000_000.0
    MAP_SOC_VIOLATION_PENALTY = 1_000_000.0
    OVERNIGHT_CHARGING_HOURS = 4.0

    def get_charger_cost_per_kw(charger_power_kw):
        """Determine charger cost tier based on power level."""
        if charger_power_kw <= 22:
            return 250.0
        elif charger_power_kw <= 50:
            return 450.0
        elif charger_power_kw <= 150:
            return 550.0
        else:
            return 600.0

    # Resolve bus counts per line from simulation first, then metadata fallback.
    bus_stats = sim_results.get('bus_statistics', {})
    num_buses_per_line = {}
    for stats in bus_stats.values():
        lid = stats.get('line_id')
        if lid is not None:
            num_buses_per_line[lid] = num_buses_per_line.get(lid, 0) + 1

    if not num_buses_per_line:
        for line in bus_lines:
            num_buses_per_line[line.line_id] = line.num_buses

    num_buses = sim_results.get('num_buses', sum(num_buses_per_line.values()))

    # ---- Bus battery cost ----
    cost_bus_battery = 0.0
    for line_id, cap_kwh in bus_caps_kwh.items():
        n_buses_on_line = num_buses_per_line.get(line_id, 0)
        cost_bus_battery += BATTERY_COST_PER_KWH * (cap_kwh * n_buses_on_line)

    # ---- MAP battery and hardware cost ----
    cost_map_battery = BATTERY_COST_PER_KWH * num_maps * map_battery_kwh
    cost_map_hardware = MAP_HARDWARE_COST * num_maps

    # ---- Overnight charger cost ----
    # Determine charger power requirement from simulation
    total_energy_consumed = sim_results.get('total_energy_consumed_wh')
    if total_energy_consumed is None:
        total_energy_consumed = sum(
            stats.get('total_energy_consumed_wh', 0.0) for stats in bus_stats.values())
    total_energy_charged = sim_results.get('total_energy_charged_wh', 0.0)
    overnight_energy_wh = max(0.0, total_energy_consumed - total_energy_charged)
    overnight_energy_kwh = overnight_energy_wh / 1000.0

    if num_buses > 0:
        energy_per_bus_kwh = overnight_energy_kwh / num_buses
    else:
        energy_per_bus_kwh = 0.0

    charger_power_kw = energy_per_bus_kwh / OVERNIGHT_CHARGING_HOURS
    charger_cost_per_kw = get_charger_cost_per_kw(charger_power_kw)
    per_bus_charger_cost = charger_cost_per_kw * charger_power_kw
    cost_overnight = per_bus_charger_cost * num_buses

    # ---- SOC violation penalties ----
    # For a feasible solution, this should be 0
    cost_penalties = 0.0
    n_bus_violations = 0
    n_map_violations = 0

    # Check for bus SOC violations
    for bus_id, stats in bus_stats.items():
        if stats.get('min_soc_ratio', 1.0) < 0.20:  # 20% threshold
            n_bus_violations += 1
    cost_penalties += BUS_SOC_VIOLATION_PENALTY * n_bus_violations

    # Check for MAP violations (if any MAPs have min_soc < 10%)
    # This is harder to determine without simulation details, so we assume 0
    # for feasible solutions
    if sim_results.get('min_soc_overall_ratio', 1.0) < 0.10 and num_maps > 0:
        n_map_violations = 1
    cost_penalties += MAP_SOC_VIOLATION_PENALTY * n_map_violations

    # ---- Total cost ----
    total_cost = (cost_bus_battery + cost_map_battery + cost_map_hardware +
                  cost_overnight + cost_penalties)

    cost_breakdown = {
        'bus_battery_cost': cost_bus_battery,
        'map_battery_cost': cost_map_battery,
        'map_hardware_cost': cost_map_hardware,
        'overnight_cost': cost_overnight,
        'penalty_cost': cost_penalties,
        'bus_soc_violations': n_bus_violations,
        'map_soc_violations': n_map_violations,
    }

    return {
        'total_cost': total_cost,
        'cost_breakdown': cost_breakdown,
        'charger_power_kw': charger_power_kw,
        'charger_cost_per_kw': charger_cost_per_kw,
    }


def calculate_and_print_cost(bus_caps_kwh, num_maps, map_battery_kwh,
                             sim_results, bus_lines, label="Configuration"):
    """
    Calculate system cost and print a formatted breakdown.

    Wraps :func:`calculate_system_cost` so the same reporting is used in both
    simulation-only mode and after a feasible solution is found in the
    constraint-driven search.

    Returns the dict produced by :func:`calculate_system_cost`.
    """
    cost_result = calculate_system_cost(
        bus_caps_kwh=bus_caps_kwh,
        num_maps=num_maps,
        map_battery_kwh=map_battery_kwh,
        sim_results=sim_results,
        bus_lines=bus_lines,
    )

    print(f"\n" + "=" * 70)
    print(f"COST BREAKDOWN ({label})")
    print("=" * 70)
    print(f"\nConfiguration:")
    for lid, cap in sorted(bus_caps_kwh.items()):
        print(f"  Line {lid}: {cap:.0f} kWh")
    if map_battery_kwh is not None:
        print(f"  MAP battery: {map_battery_kwh:.0f} kWh")
    else:
        print(f"  MAP battery: (unset)")
    print(f"  Number of MAPs: {num_maps}")

    cost_br = cost_result['cost_breakdown']
    print(f"\nCost Breakdown:")
    print(f"  Bus battery cost:      ${cost_br['bus_battery_cost']:>12,.2f}")
    print(f"  MAP battery cost:      ${cost_br['map_battery_cost']:>12,.2f}")
    print(f"  MAP hardware cost:     ${cost_br['map_hardware_cost']:>12,.2f}")
    print(f"  Overnight charger:     ${cost_br['overnight_cost']:>12,.2f}")
    print(f"  Penalty cost:          ${cost_br['penalty_cost']:>12,.2f}")
    print(f"  {'-' * 45}")
    print(f"  TOTAL COST:            ${cost_result['total_cost']:>12,.2f}")
    print(f"\nCharger power level: {cost_result['charger_power_kw']:.2f} kW "
          f"(${cost_result['charger_cost_per_kw']:.0f}/kW tier)")

    return cost_result


def resolve_map_battery_kwh(results, stage2_sim, requested_map_battery_kwh=None):
    """Resolve actual MAP battery capacity used by simulation (kWh)."""
    map_wh = results.get('map_battery_capacity_wh')
    if map_wh is None:
        map_wh = getattr(stage2_sim, 'map_battery_capacity_wh', None)
    if map_wh is not None:
        actual_kwh = map_wh / 1000.0
    else:
        actual_kwh = requested_map_battery_kwh

    if (requested_map_battery_kwh is not None and actual_kwh is not None
            and abs(actual_kwh - requested_map_battery_kwh) > 1e-6):
        print(f"  [WARN] Requested MAP battery {requested_map_battery_kwh:.1f} kWh, "
              f"but simulation used {actual_kwh:.1f} kWh")

    return actual_kwh



def load_stockholm_data():
    from DES_model import GTFSBusSim
    from Trip_assign import load_trips

    GTFS_FOLDER = "gtfs_data"
    OSM_XML = "map.xml"
    DATE_STR = "20231108"
    LINES = ["1", "2", "3", "4", "6"]

    print("Loading Stockholm transit data...")
    sim = GTFSBusSim(GTFS_FOLDER, OSM_XML, date_str=DATE_STR)
    trips_df = load_trips(GTFS_FOLDER, DATE_STR)

    bus_lines = []
    for line_id in LINES:
        line_trips = trips_df[trips_df["route_short_name"] == line_id].reset_index(drop=True)
        if not line_trips.empty:
            num_buses = max(1, int(len(line_trips) / 15))
            bus_lines.append(BusLineData(
                line_id=line_id,
                num_buses=num_buses,
                trips_df=line_trips
            ))

    return sim, bus_lines


def compute_trip_change_stops(sim, bus_trips_dict):
    """Compute trip-change stops"""
    trip_change_stops = set()

    for bus_id, trip_ids in bus_trips_dict.items():
        sorted_trips = sorted(
            trip_ids,
            key=lambda t: (
                sim.gtfs.stop_sequence(t)[0]["arrival"]
                if sim.gtfs.stop_sequence(t)
                else float('inf')
            )
        )

        prev_end_stop = None
        prev_end_time = None

        for trip_id in sorted_trips:
            try:
                seq = sim.gtfs.stop_sequence(trip_id)
                if not seq:
                    continue

                start_time = seq[0]["arrival"]

                if prev_end_time is not None and prev_end_stop is not None:
                    dwell = start_time - prev_end_time
                    if dwell > 0:
                        trip_change_stops.add(prev_end_stop)

                prev_end_time = seq[-1]["arrival"]
                prev_end_stop = seq[-1]["stop_id"]

            except Exception:
                continue

    return trip_change_stops


def generate_random_initial_values(bus_lines, min_bus_cap_kwh=40, max_bus_cap_kwh=500,
                                   min_maps=2, max_maps=20, min_map_cap_kwh=90,
                                   max_map_cap_kwh=500):
    """
    Generate random initial values for LINE_BATTERY_CAPACITIES_KWH, number of MAPs,
    and MAP battery capacity.

    Parameters
    ----------
    bus_lines : list[BusLineData]
        Bus lines to generate capacities for
    min_bus_cap_kwh, max_bus_cap_kwh : float
        Range for per-line bus battery capacities
    min_maps, max_maps : int
        Range for number of MAPs
    min_map_cap_kwh, max_map_cap_kwh : float
        Range for MAP battery capacity

    Returns
    -------
    tuple (line_battery_capacities_wh, initial_capacity_wh, initial_num_maps,
           map_battery_capacity_wh)
    """
    # Random per-line bus battery capacities (in steps of 10 kWh)
    line_battery_capacities_wh = {}
    for line in bus_lines:
        cap_kwh = random.randint(
            int(min_bus_cap_kwh / 10),
            int(max_bus_cap_kwh / 10)
        ) * 10
        line_battery_capacities_wh[line.line_id] = cap_kwh * 1000

    # Random default capacity (fallback)
    initial_capacity_wh = random.randint(
        int(min_bus_cap_kwh / 10),
        int(max_bus_cap_kwh / 10)
    ) * 10 * 1000

    # Random number of MAPs
    initial_num_maps = random.randint(min_maps, max_maps)

    # Random MAP battery capacity (in steps of 10 kWh)
    map_cap_kwh = random.randint(
        int(min_map_cap_kwh / 10),
        int(max_map_cap_kwh / 10)
    ) * 10
    map_battery_capacity_wh = map_cap_kwh * 1000

    return line_battery_capacities_wh, initial_capacity_wh, initial_num_maps, map_battery_capacity_wh


def diagnose_infeasibility(results, bus_stats):
    """
    Analyze simulation results to determine why the scenario was infeasible.

    IMPROVED: Includes fallback logic to ensure a constraint is always generated
    when the simulation is infeasible.

    Returns a list of feedback constraint dicts that should be added to the
    MILP on the next iteration.  Each dict has:
        'type'    – 'bus_min_cap', 'map_min_cap', or 'min_maps'
        'line_id' – (for bus_min_cap only) which line
        'value'   – minimum value to enforce
        'reason'  – human-readable explanation
    """
    constraints = []
    bus_min_soc = 0.20  # 20%

    # --- Check per-line bus SOC violations ---
    line_worst = {}  # {line_id: worst min_soc_ratio}
    for bus_id, stats in bus_stats.items():
        lid = stats['line_id']
        ratio = stats['min_soc_ratio']
        if lid not in line_worst or ratio < line_worst[lid]:
            line_worst[lid] = ratio

    for lid, ratio in line_worst.items():
        if ratio < bus_min_soc:
            # Bus ran out of charge on this line.  Require a 10 kWh step increase.
            # Round current capacity UP to next step of 10.
            line_cap_kwh = results.get('line_battery_capacities_wh', {}).get(lid)
            if line_cap_kwh is not None:
                line_cap_kwh = line_cap_kwh / 1000
            else:
                line_cap_kwh = results['battery_capacity_wh'] / 1000
            new_min = math.ceil((line_cap_kwh + 10) / 10) * 10
            constraints.append({
                'type': 'bus_min_cap',
                'line_id': lid,
                'value': new_min,
                'num_maps_at_creation': results.get('num_maps', 0),
                'reason': (f"Line {lid}: buses reached {ratio * 100:.1f}% SOC "
                           f"(< 20%).  Increasing min battery to {new_min} kWh."),
            })

    # --- Check MAP SOC violations / insufficient MAPs ---
    num_maps_cur = results.get('num_maps', 0)
    if num_maps_cur == 0 and results['min_soc_overall_ratio'] < bus_min_soc:
        # No MAPs at all → charging disabled → buses can't recharge during
        # the day.  Force the MILP to use at least 1 MAP.
        constraints.append({
            'type': 'min_maps',
            'value': 1,
            'reason': (f"Overall min SOC {results['min_soc_overall_ratio'] * 100:.1f}% "
                       f"< {bus_min_soc * 100:.0f}% with 0 MAPs (charging disabled).  "
                       f"Requiring at least 1 MAP."),
        })
    elif num_maps_cur > 0 and results['min_soc_overall_ratio'] < bus_min_soc:
        # MAPs are present but not enough → request one more
        constraints.append({
            'type': 'min_maps',
            'value': num_maps_cur + 1,
            'reason': (f"Overall min SOC {results['min_soc_overall_ratio'] * 100:.1f}% "
                       f"< {bus_min_soc * 100:.0f}% despite {num_maps_cur} MAPs.  "
                       f"Requiring at least {num_maps_cur + 1} MAPs."),
        })

    # --- FALLBACK: If no constraints were generated but simulation is infeasible ---
    # This handles edge cases where infeasibility wasn't caught above
    if not constraints and not results.get('feasible', True):
        # Default: increase bus batteries across all lines
        print(f"  [DEBUG] Infeasibility diagnosed but no specific cause found.")
        print(f"  [DEBUG] Min SOC overall: {results['min_soc_overall_ratio'] * 100:.1f}%")
        print(f"  [DEBUG] Number of MAPs: {num_maps_cur}")
        print(f"  [DEBUG] Applying fallback: increasing all bus batteries by 20 kWh")

        for lid, ratio in line_worst.items():
            line_cap_kwh = results.get('line_battery_capacities_wh', {}).get(lid)
            if line_cap_kwh is not None:
                line_cap_kwh = line_cap_kwh / 1000
            else:
                line_cap_kwh = results['battery_capacity_wh'] / 1000
            # Fallback: increase by 20 kWh (two steps of 10)
            new_min = math.ceil((line_cap_kwh + 20) / 10) * 10
            constraints.append({
                'type': 'bus_min_cap',
                'line_id': lid,
                'value': new_min,
                'num_maps_at_creation': results.get('num_maps', 0),
                'reason': (f"Line {lid}: infeasibility detected but cause unclear. "
                           f"Fallback: increasing min battery to {new_min} kWh."),
            })

        # If no line-specific data, add a general MAP constraint
        if not constraints:
            new_num_maps = max(1, num_maps_cur + 1)
            constraints.append({
                'type': 'min_maps',
                'value': new_num_maps,
                'reason': (f"Infeasibility detected with no per-line SOC data. "
                           f"Fallback: requiring at least {new_num_maps} MAPs."),
            })

    return constraints


def select_single_constraint_for_alternation(new_constraints, last_action=None):
    """
    Select constraints to apply based on alternation pattern.

    When infeasible, apply constraints in alternating phases:
    - Bus phase: apply ALL bus battery increases together
    - MAP phase: apply MAP-count increase
    - Repeat until feasible

    Parameters:
    -----------
    new_constraints : list[dict]
        All suggested constraints from diagnosis
    last_action : str or None
        'bus', 'map', or None (first action)

    Returns:
    --------
    tuple (constraints_to_apply, remaining_constraints, action_type)
        constraints_to_apply: list of dicts to apply this iteration
        remaining_constraints: list of not-yet-applied constraints
        action_type: 'bus' or 'map' indicating what was applied
    """
    bus_constraints = [fc for fc in new_constraints if fc.get('type') == 'bus_min_cap']
    min_maps_constraints = [fc for fc in new_constraints if fc.get('type') == 'min_maps']
    map_cap_constraints = [fc for fc in new_constraints if fc.get('type') == 'map_min_cap']

    # If last action was MAP (or first action), apply ALL bus increases together.
    if last_action in ('map', None) and bus_constraints:
        remaining = [fc for fc in new_constraints if fc.get('type') != 'bus_min_cap']
        return bus_constraints, remaining, 'bus'

    # If last action was bus, apply MAP increase.
    if last_action == 'bus' and min_maps_constraints:
        strongest_min_maps = max(min_maps_constraints, key=lambda fc: fc.get('value', 0))
        remaining = [fc for fc in new_constraints if fc != strongest_min_maps]
        return [strongest_min_maps], remaining, 'map'

    # Fallback: if no alternation applies, try all MAP if available.
    if min_maps_constraints:
        strongest_min_maps = max(min_maps_constraints, key=lambda fc: fc.get('value', 0))
        remaining = [fc for fc in new_constraints if fc != strongest_min_maps]
        return [strongest_min_maps], remaining, 'map'

    # Last resort: MAP battery.
    if map_cap_constraints:
        strongest_map_cap = max(map_cap_constraints, key=lambda fc: fc.get('value', 0))
        remaining = [fc for fc in new_constraints if fc != strongest_map_cap]
        return [strongest_map_cap], remaining, 'map'

    return [], [], None

def _refine_bus_capacities(sim, bus_trips, bus_lines, trip_change_stops,
                           bus_caps_kwh, num_maps, default_bus_capacity_kwh,
                           requested_map_battery_kwh, min_kwh=30, step_kwh=10):
    """
    Greedy per-line reduction of bus battery capacities while feasible.

    Iterates over each line, decreasing its capacity in ``step_kwh`` steps
    (down to ``min_kwh``) and keeps the reduction whenever the resulting
    simulation is feasible.

    Returns a new dict of optimized bus capacities (kWh).
    """
    bus_caps_kwh = bus_caps_kwh.copy()
    for line_id in sorted(bus_caps_kwh.keys()):
        print(f"\nOptimizing Line {line_id} bus battery "
              f"(current: {bus_caps_kwh[line_id]:.0f} kWh)...")

        while bus_caps_kwh[line_id] > min_kwh:
            test_bus_caps = bus_caps_kwh.copy()
            test_bus_caps[line_id] = max(min_kwh, test_bus_caps[line_id] - step_kwh)

            test_results, _ = run_terminal_charging_simulation(
                sim=sim,
                bus_trips_dict=bus_trips,
                bus_lines=bus_lines,
                trip_change_stops=trip_change_stops,
                battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
                num_maps=num_maps,
                optimize_threshold=True,
                preemption_threshold=None,
                simulation_duration_s=86400,
                line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in test_bus_caps.items()},
                map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                         else int(requested_map_battery_kwh * 1000)),
                skip_plots=True,
            )

            if test_results['feasible']:
                bus_caps_kwh[line_id] = test_bus_caps[line_id]
                print(f"  ✓ Reduced to {bus_caps_kwh[line_id]:.0f} kWh - still feasible")
            else:
                print(f"  ✗ Cannot reduce below {bus_caps_kwh[line_id]:.0f} kWh "
                      f"- infeasible at {test_bus_caps[line_id]:.0f} kWh")
                break

    return bus_caps_kwh


def _evaluate_config_cost(sim, bus_trips, bus_lines, trip_change_stops,
                          bus_caps_kwh, num_maps, default_bus_capacity_kwh,
                          requested_map_battery_kwh):
    """
    Run a single simulation for the given config and compute its total cost.

    Returns a dict {'total_cost', 'sim_results', 'stage2_sim',
    'map_battery_kwh'}, or ``None`` if the simulation is infeasible.
    """
    results, stage2_sim = run_terminal_charging_simulation(
        sim=sim,
        bus_trips_dict=bus_trips,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
        num_maps=num_maps,
        optimize_threshold=True,
        preemption_threshold=None,
        simulation_duration_s=86400,
        line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in bus_caps_kwh.items()},
        map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                 else int(requested_map_battery_kwh * 1000)),
        skip_plots=True,
    )

    if not results['feasible']:
        return None

    map_kwh = resolve_map_battery_kwh(
        results, stage2_sim, requested_map_battery_kwh=requested_map_battery_kwh)
    cost = calculate_system_cost(
        bus_caps_kwh=bus_caps_kwh,
        num_maps=num_maps,
        map_battery_kwh=map_kwh,
        sim_results=results,
        bus_lines=bus_lines,
    )
    return {
        'total_cost': cost['total_cost'],
        'sim_results': results,
        'stage2_sim': stage2_sim,
        'map_battery_kwh': map_kwh,
    }


def _run_refinement_pass(sim, bus_trips, bus_lines, trip_change_stops,
                         bus_caps_kwh, num_maps, map_battery_kwh,
                         requested_map_battery_kwh, default_bus_capacity_kwh):
    """
    Run a single Phase 1 + Phase 2 (2a + 2b) + Phase 3 refinement pass.

    All phases are greedy and non-increasing in total cost, so the returned
    configuration is at least as cheap as the input.

    Returns
    -------
    tuple (bus_caps_kwh, num_maps, map_battery_kwh, requested_map_battery_kwh)
    """
    bus_caps_kwh = bus_caps_kwh.copy()

    # --- Phase 1: Reduce bus battery capacities per line ---
    print("\n--- Phase 1: Reducing bus battery capacities ---")
    bus_caps_kwh = _refine_bus_capacities(
        sim=sim,
        bus_trips=bus_trips,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        bus_caps_kwh=bus_caps_kwh,
        num_maps=num_maps,
        default_bus_capacity_kwh=default_bus_capacity_kwh,
        requested_map_battery_kwh=requested_map_battery_kwh,
    )

    print(f"\nBus batteries after optimization:")
    for lid, cap in sorted(bus_caps_kwh.items()):
        print(f"  Line {lid}: {cap:.0f} kWh")

    # --- Phase 2: Tune MAP count ---
    print("\n--- Phase 2: Tuning MAP count ---")

    baseline = _evaluate_config_cost(
        sim=sim,
        bus_trips=bus_trips,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        bus_caps_kwh=bus_caps_kwh,
        num_maps=num_maps,
        default_bus_capacity_kwh=default_bus_capacity_kwh,
        requested_map_battery_kwh=requested_map_battery_kwh,
    )
    if baseline is None:
        print(" [WARN] Current configuration became infeasible - skipping Phase 2a.")
        current_cost = float('inf')
    else:
        current_cost = baseline['total_cost']
        print(f"Baseline cost at MAPs={num_maps}: ${current_cost:,.2f}")

        # --- Phase 2a: Try MAPs+1 and re-run Phase 1 (accept if cheaper) ---
        print("\n--- Phase 2a: Trying MAP increases (compensated by smaller bus batteries) ---")
        while True:
            trial_num_maps = num_maps + 1
            test_map_battery = max(50, map_battery_kwh + 10)
            print(f"\n Trying MAPs={trial_num_maps}: re-running Phase 1 to lower bus batteries...")

            trial_bus_caps = _refine_bus_capacities(
                sim=sim,
                bus_trips=bus_trips,
                bus_lines=bus_lines,
                trip_change_stops=trip_change_stops,
                bus_caps_kwh=bus_caps_kwh,
                num_maps=trial_num_maps,
                default_bus_capacity_kwh=default_bus_capacity_kwh,
                requested_map_battery_kwh=test_map_battery,
            )

            trial = _evaluate_config_cost(
                sim=sim,
                bus_trips=bus_trips,
                bus_lines=bus_lines,
                trip_change_stops=trip_change_stops,
                bus_caps_kwh=trial_bus_caps,
                num_maps=trial_num_maps,
                default_bus_capacity_kwh=default_bus_capacity_kwh,
                requested_map_battery_kwh=test_map_battery,
            )

            if trial is None:
                print(f"✗ MAPs={trial_num_maps} infeasible at the reduced bus batteries; "
                      f"abandoning Phase 2a.")
                break

            trial_cost = trial['total_cost']
            print(f"Trial cost (MAPs={trial_num_maps}): ${trial_cost:,.2f} "
                  f"vs current ${current_cost:,.2f}")

            if trial_cost < current_cost:
                print(f"✓ Cheaper - accepting MAPs={trial_num_maps} with reduced bus batteries.")
                num_maps = trial_num_maps
                bus_caps_kwh = trial_bus_caps
                current_cost = trial_cost
                # Try another +1
            else:
                print(f"✗ Not cheaper - falling back to decreasing-MAP logic.")
                break

    # --- Phase 2b: Reduce MAP count ---
    print("\n--- Phase 2b: Reducing MAP count ---")
    print(f"Current MAP count: {num_maps}")

    while num_maps > 1:  # Minimum 1 MAP
        test_num_maps = num_maps - 1
        test_results, _ = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
            num_maps=test_num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in bus_caps_kwh.items()},
            map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                     else int(requested_map_battery_kwh * 1000)),
            skip_plots=True,
        )

        if test_results['feasible']:
            num_maps = test_num_maps
            print(f"  ✓ Reduced to {num_maps} MAPs - still feasible")
        else:
            print(f"  ✗ Cannot reduce below {num_maps} MAPs - infeasible at {test_num_maps} MAPs")
            break

    print(f"MAP count after optimization: {num_maps}")

    # --- Phase 3: Reduce MAP battery capacity ---
    print("\n--- Phase 3: Reducing MAP battery capacity ---")
    print(f"Current MAP battery: {map_battery_kwh:.0f} kWh")

    while map_battery_kwh > 50:  # Minimum 50 kWh
        test_map_battery = max(50, map_battery_kwh - 10)
        test_results, _ = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
            num_maps=num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in bus_caps_kwh.items()},
            map_battery_capacity_wh=int(test_map_battery * 1000),
            skip_plots=True,
        )

        if test_results['feasible']:
            map_battery_kwh = test_map_battery
            requested_map_battery_kwh = test_map_battery
            print(f"  ✓ Reduced to {map_battery_kwh:.0f} kWh - still feasible")
        else:
            print(f"  ✗ Cannot reduce below {map_battery_kwh:.0f} kWh - infeasible at {test_map_battery:.0f} kWh")
            break

    print(f"MAP battery after optimization: {map_battery_kwh:.0f} kWh")

    return bus_caps_kwh, num_maps, map_battery_kwh, requested_map_battery_kwh


def run_milp_simulation_loop(sim, bus_trips, bus_lines, trip_change_stops,
                             line_battery_capacities_wh,
                             initial_capacity_wh, initial_num_maps,
                             max_iterations=100):
    """
    Iterative constraint-driven feasibility search (no MILP optimization).

    1. Run initial simulation with starting parameters.
    2. If feasible -> record as solution.
    3. If infeasible -> diagnose and generate feedback constraints.
    4. Apply constraint values directly: use suggested B_l, N_map, B_map.
    5. Re-run simulation with updated values.
    6. Repeat until feasible or max_iterations reached.

    This approach uses the diagnostic feedback directly instead of solving
    an optimization problem. Constraints suggest minimum values (e.g.,
    "B_l ≥ 80 kWh"), and we apply those values directly.

    Returns (solution_dict, sim_results, stage2_sim, iteration_log).
    """
    feedback_constraints = []
    iteration_log = []
    best_feasible = None

    # Track current capacities (start with user-provided values)
    current_bus_caps_kwh = {lid: wh / 1000.0 for lid, wh in line_battery_capacities_wh.items()}
    default_bus_capacity_kwh = initial_capacity_wh / 1000.0
    # MAP battery is simulation-driven; None means use integration default
    requested_map_battery_kwh = None
    current_map_battery_kwh = None
    current_num_maps = initial_num_maps
    last_action = None  # Track last action to enforce alternation: 'bus' or 'map'
    pending_constraints = []  # Constraints deferred from diagnosis

    # --- Iteration 0: simulate with user-provided start values ---
    print("\n" + "=" * 70)
    print("CONSTRAINT-DRIVEN SEARCH: ITERATION 0 (User-Provided Start Values)")
    print("=" * 70)

    print(f"  User-provided start values:")
    for lid, cap in sorted(current_bus_caps_kwh.items()):
        print(f"    Line {lid} bus battery: {cap:.0f} kWh")
    print(f"    Default bus battery: {default_bus_capacity_kwh:.0f} kWh")
    print("    MAP battery: SIMULATION DEFAULT")
    print(f"    Number of MAPs: {current_num_maps}")

    results, stage2_sim = run_terminal_charging_simulation(
        sim=sim,
        bus_trips_dict=bus_trips,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
        num_maps=current_num_maps,
        optimize_threshold=True,
        preemption_threshold=None,
        simulation_duration_s=86400,
        line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in current_bus_caps_kwh.items()},
        map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                 else int(requested_map_battery_kwh * 1000)),
        skip_plots=True,
    )
    current_map_battery_kwh = resolve_map_battery_kwh(
        results, stage2_sim, requested_map_battery_kwh=requested_map_battery_kwh)

    print(f"\n  Simulation feasibility: {'FEASIBLE [OK]' if results['feasible'] else 'INFEASIBLE [X]'}")
    print(f"  Min SOC ratio: {results['min_soc_overall_ratio'] * 100:.1f}%")

    if results['feasible']:
        best_feasible = {
            'sim_results': results,
            'stage2_sim': stage2_sim,
            'iteration': 0,
            'bus_battery_kwh': current_bus_caps_kwh.copy(),
            'map_battery_kwh': current_map_battery_kwh,
            'num_maps': current_num_maps,
        }
        print(f"\n[OK] FEASIBLE solution found at iteration 0!")

    iteration_log.append({
        'iteration': 0,
        'method': 'user_initial',
        'sim_feasible': results['feasible'],
        'bus_battery_kwh': current_bus_caps_kwh.copy(),
        'map_battery_kwh': current_map_battery_kwh,
        'num_maps': current_num_maps,
        'constraints_applied': [],
        'constraints_deferred': [],
        'last_action': last_action,
    })

    # --- Main constraint-driven loop ---
    for iteration in range(1, max_iterations + 1):
        print("\n" + "=" * 70)
        print(f"CONSTRAINT-DRIVEN SEARCH: ITERATION {iteration}")
        print("=" * 70)

        # Check if already feasible - DO NOT BREAK, continue for refinement
        if best_feasible is not None:
            print(f"\n[OK] Found a feasible solution at iteration {best_feasible['iteration']}.")
            print(f"  Bus batteries: {best_feasible['bus_battery_kwh']}")
            print(f"  MAP battery: {best_feasible['map_battery_kwh']:.0f} kWh")
            print(f"  Number of MAPs: {best_feasible['num_maps']}")
            print(f"\n  Entering REFINEMENT PHASE to minimize capacities...")
            break  # Exit the loop to enter refinement phase below

        # If infeasible, diagnose and apply ONE constraint at a time (alternating bus/MAP)
        if not results['feasible']:
            # First, add any newly diagnosed constraints to pending.
            new_constraints = diagnose_infeasibility(results, results['bus_statistics'])
            if not new_constraints:
                print("  Could not diagnose infeasibility - continuing to next iteration.")
                iteration_log.append({
                    'iteration': iteration,
                    'method': 'no_diagnosis',
                    'sim_feasible': False,
                    'bus_battery_kwh': current_bus_caps_kwh.copy(),
                    'map_battery_kwh': current_map_battery_kwh,
                    'num_maps': current_num_maps,
                    'constraints_applied': [],
                })
                continue

            # Apply the constraints: extract the suggested values
            print(f"\n  Applying feedback constraints:")
            constraints_applied = []
            for fc in new_constraints:
                print(f"    {fc['reason']}")
                constraints_applied.append(fc['reason'])

                if fc['type'] == 'bus_min_cap':
                    lid = fc['line_id']
                    current_bus_caps_kwh[lid] = fc['value']
                elif fc['type'] == 'min_maps':
                    current_num_maps = fc['value']
                elif fc['type'] == 'map_min_cap':
                    requested_map_battery_kwh = fc['value']

            feedback_constraints.extend(new_constraints)
        else:
            constraints_applied = []

            # Run simulation with current values
        print(f"\n  Current configuration:")
        for lid, cap in sorted(current_bus_caps_kwh.items()):
            print(f"    Line {lid} bus battery: {cap:.0f} kWh")
        if requested_map_battery_kwh is None:
            print("    MAP battery: SIMULATION DEFAULT")
        else:
            print(f"    MAP battery (requested): {requested_map_battery_kwh:.0f} kWh")
        print(f"    Number of MAPs: {current_num_maps}")

        results, stage2_sim = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
            num_maps=current_num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in current_bus_caps_kwh.items()},
            map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                     else int(requested_map_battery_kwh * 1000)),
            skip_plots=True,
        )
        current_map_battery_kwh = resolve_map_battery_kwh(
            results, stage2_sim, requested_map_battery_kwh=requested_map_battery_kwh)

        print(f"\n  Simulation feasibility: {'FEASIBLE [OK]' if results['feasible'] else 'INFEASIBLE [X]'}")
        print(f"  Min SOC ratio: {results['min_soc_overall_ratio'] * 100:.1f}%")

        if results['feasible']:
            best_feasible = {
                'sim_results': results,
                'stage2_sim': stage2_sim,
                'iteration': iteration,
                'bus_battery_kwh': current_bus_caps_kwh.copy(),
                'map_battery_kwh': current_map_battery_kwh,
                'num_maps': current_num_maps,
            }
            print(f"\n[OK] FEASIBLE solution found at iteration {iteration}!")

        iteration_log.append({
            'iteration': iteration,
            'method': 'constraint_driven',
            'sim_feasible': results['feasible'],
            'bus_battery_kwh': current_bus_caps_kwh.copy(),
            'map_battery_kwh': current_map_battery_kwh,
            'num_maps': current_num_maps,
            'constraints_applied': constraints_applied,
        })

    else:
        print(f"\n[!] Max iterations ({max_iterations}) reached."
              + (" Best feasible solution found."
                 if best_feasible else " No feasible solution found."))

    # --- REFINEMENT PHASE: Minimize capacities while maintaining feasibility ---
    if best_feasible is not None:
        print("\n" + "=" * 70)
        print("REFINEMENT PHASE: Minimizing capacities")
        print("=" * 70)

        # Start from the best feasible solution
        current_bus_caps_kwh = best_feasible['bus_battery_kwh'].copy()
        current_map_battery_kwh = best_feasible['map_battery_kwh']
        requested_map_battery_kwh = current_map_battery_kwh
        current_num_maps = best_feasible['num_maps']

        # --- Phase 1: Reduce bus battery capacities per line ---
        print("\n--- Phase 1: Reducing bus battery capacities ---")
        current_bus_caps_kwh = _refine_bus_capacities(
            sim=sim,
            bus_trips=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            bus_caps_kwh=current_bus_caps_kwh,
            num_maps=current_num_maps,
            default_bus_capacity_kwh=default_bus_capacity_kwh,
            requested_map_battery_kwh=requested_map_battery_kwh,
        )

        print(f"\nBus batteries after optimization:")
        for lid, cap in sorted(current_bus_caps_kwh.items()):
            print(f"  Line {lid}: {cap:.0f} kWh")

        # --- Phase 2: Tune MAP count ---
        # First check whether ADDING a MAP and re-running Phase 1 yields a
        # cheaper configuration (more MAPs can compensate for smaller bus
        # batteries).  If not, fall through to decreasing the MAP count.
        print("\n--- Phase 2: Tuning MAP count ---")

        baseline = _evaluate_config_cost(
            sim=sim,
            bus_trips=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            bus_caps_kwh=current_bus_caps_kwh,
            num_maps=current_num_maps,
            default_bus_capacity_kwh=default_bus_capacity_kwh,
            requested_map_battery_kwh=requested_map_battery_kwh,
        )
        if baseline is None:
            print(" [WARN] Current configuration became infeasible - skipping Phase 2a.")
            current_cost = float('inf')
        else:
            current_cost = baseline['total_cost']
            print(f"Baseline cost at MAPs={current_num_maps}: ${current_cost:,.2f}")

            # --- Phase 2a: Try MAPs+1 and re-run Phase 1 (accept if cheaper) ---
            print("\n--- Phase 2a: Trying MAP increases (compensated by smaller bus batteries) ---")
            while True:
                trial_num_maps = current_num_maps + 1
                test_map_battery = max(50, current_map_battery_kwh + 10)
                print(f"\n Trying MAPs={trial_num_maps}: re-running Phase 1 to lower bus batteries...")

                trial_bus_caps = _refine_bus_capacities(
                    sim=sim,
                    bus_trips=bus_trips,
                    bus_lines=bus_lines,
                    trip_change_stops=trip_change_stops,
                    bus_caps_kwh=current_bus_caps_kwh,
                    num_maps=trial_num_maps,
                    default_bus_capacity_kwh=default_bus_capacity_kwh,
                    requested_map_battery_kwh=test_map_battery,
                )

                trial = _evaluate_config_cost(
                    sim=sim,
                    bus_trips=bus_trips,
                    bus_lines=bus_lines,
                    trip_change_stops=trip_change_stops,
                    bus_caps_kwh=trial_bus_caps,
                    num_maps=trial_num_maps,
                    default_bus_capacity_kwh=default_bus_capacity_kwh,
                    requested_map_battery_kwh=test_map_battery,
                )

                if trial is None:
                    print(f"✗ MAPs={trial_num_maps} infeasible at the reduced bus batteries; "
                          f"abandoning Phase 2a.")
                    break

                trial_cost = trial['total_cost']
                print(f"Trial cost (MAPs={trial_num_maps}): ${trial_cost:,.2f} "
                      f"vs current ${current_cost:,.2f}")

                if trial_cost < current_cost:
                    print(f"✓ Cheaper - accepting MAPs={trial_num_maps} with reduced bus batteries.")
                    current_num_maps = trial_num_maps
                    current_bus_caps_kwh = trial_bus_caps
                    current_cost = trial_cost
                    # Try another +1
                else:
                    print(f"✗ Not cheaper - falling back to decreasing-MAP logic.")
                    break

        # --- Phase 2b: Reduce MAP count ---
        print("\n--- Phase 2b: Reducing MAP count ---")
        print(f"Current MAP count: {current_num_maps}")

        while current_num_maps > 1:  # Minimum 1 MAP
            # Try reducing by 1 MAP
            test_num_maps = current_num_maps - 1

            # Run simulation with reduced MAP count
            test_results, _ = run_terminal_charging_simulation(
                sim=sim,
                bus_trips_dict=bus_trips,
                bus_lines=bus_lines,
                trip_change_stops=trip_change_stops,
                battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
                num_maps=test_num_maps,
                optimize_threshold=True,
                preemption_threshold=None,
                simulation_duration_s=86400,
                line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in current_bus_caps_kwh.items()},
                map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                         else int(requested_map_battery_kwh * 1000)),
                skip_plots=True,
            )

            if test_results['feasible']:
                current_num_maps = test_num_maps
                print(f"  ✓ Reduced to {current_num_maps} MAPs - still feasible")
            else:
                print(f"  ✗ Cannot reduce below {current_num_maps} MAPs - infeasible at {test_num_maps} MAPs")
                break

        print(f"MAP count after optimization: {current_num_maps}")

        # --- Phase 3: Reduce MAP battery capacity ---
        print("\n--- Phase 3: Reducing MAP battery capacity ---")
        print(f"Current MAP battery: {current_map_battery_kwh:.0f} kWh")

        while current_map_battery_kwh > 50:  # Minimum 50 kWh
            # Try reducing by 10 kWh
            test_map_battery = max(50, current_map_battery_kwh - 10)

            # Run simulation with reduced MAP battery
            test_results, _ = run_terminal_charging_simulation(
                sim=sim,
                bus_trips_dict=bus_trips,
                bus_lines=bus_lines,
                trip_change_stops=trip_change_stops,
                battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
                num_maps=current_num_maps,
                optimize_threshold=True,
                preemption_threshold=None,
                simulation_duration_s=86400,
                line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in current_bus_caps_kwh.items()},
                map_battery_capacity_wh=int(test_map_battery * 1000),
                skip_plots=True,
            )

            if test_results['feasible']:
                current_map_battery_kwh = test_map_battery
                requested_map_battery_kwh = test_map_battery
                print(f"  ✓ Reduced to {current_map_battery_kwh:.0f} kWh - still feasible")
            else:
                print(f"  ✗ Cannot reduce below {current_map_battery_kwh:.0f} kWh - infeasible at {test_map_battery:.0f} kWh")
                break

        print(f"MAP battery after optimization: {current_map_battery_kwh:.0f} kWh")

        # --- Update best_feasible with refined values ---
        best_feasible['bus_battery_kwh'] = current_bus_caps_kwh
        best_feasible['map_battery_kwh'] = current_map_battery_kwh
        best_feasible['num_maps'] = current_num_maps

        # Run final simulation to get updated results
        final_results, final_stage2_sim = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=int(default_bus_capacity_kwh * 1000),
            num_maps=current_num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh={lid: int(cap * 1000) for lid, cap in current_bus_caps_kwh.items()},
            map_battery_capacity_wh=(None if requested_map_battery_kwh is None
                                     else int(requested_map_battery_kwh * 1000)),
            skip_plots=True,
        )
        current_map_battery_kwh = resolve_map_battery_kwh(
            final_results, final_stage2_sim, requested_map_battery_kwh=requested_map_battery_kwh)
        best_feasible['map_battery_kwh'] = current_map_battery_kwh
        best_feasible['sim_results'] = final_results
        best_feasible['stage2_sim'] = final_stage2_sim

        print("\n" + "=" * 70)
        print("REFINEMENT PHASE COMPLETE")
        print("=" * 70)
        print(f"\nOptimized configuration:")
        for lid, cap in sorted(current_bus_caps_kwh.items()):
            print(f"  Line {lid}: {cap:.0f} kWh")
        print(f"  MAP battery: {current_map_battery_kwh:.0f} kWh")
        print(f"  Number of MAPs: {current_num_maps}")
        print(f"  Min SOC: {final_results['min_soc_overall_ratio'] * 100:.1f}%")

    # --- Print iteration summary ---
    print("\n" + "=" * 70)
    print("CONSTRAINT-DRIVEN SEARCH SUMMARY")
    print("=" * 70)
    print(f"\n{'Iter':<6} {'Method':<18} {'Feasible':<10} {'Action':<8} {'MAPs':>6} {'MAP kWh':>9}")
    print("-" * 70)
    for entry in iteration_log:
        feas = "Y" if entry.get('sim_feasible') else "N"
        maps = str(entry.get('num_maps', '-')) if entry.get('num_maps') is not None else '-'
        map_kwh = f"{entry['map_battery_kwh']:.0f}" if entry.get('map_battery_kwh') is not None else '-'
        action = entry.get('last_action') or ''
        if action == 'bus':
            action = 'BUS↑'
        elif action == 'map':
            action = 'MAP↑'
        else:
            action = ''
        best_mark = " <-- BEST" if (best_feasible
                                    and entry.get('sim_feasible')
                                    and entry.get('iteration') == best_feasible['iteration']) else ""
        print(
            f"{entry['iteration']:<6} {entry.get('method', ''):<18} {feas:<10} {action:<8} {maps:>6} {map_kwh:>9}{best_mark}")

        # Print bus battery capacities for this iteration
        bus_caps = entry.get('bus_battery_kwh')
        if bus_caps:
            caps_str = ", ".join([f"L{lid}:{cap:.0f}" for lid, cap in sorted(bus_caps.items())])
            print(f"  Bus: {caps_str}")

        # Print constraints applied in this iteration
        constraints = entry.get('constraints_applied', [])
        if constraints:
            for constraint_reason in constraints:
                print(f"  → {constraint_reason}")

        deferred_constraints = entry.get('constraints_deferred', [])
        if deferred_constraints:
            for constraint_reason in deferred_constraints:
                print(f"  (pending) {constraint_reason}")
    print()

    # --- Generate plots only for the best feasible solution ---
    if best_feasible:
        print("Generating plots for the feasible solution "
              f"(iteration {best_feasible['iteration']})...")
        optimal_sim = best_feasible['stage2_sim']
        optimal_sim.plot_soc(save_path="bus_soc_terminal_charging_optimized.png")
        optimal_sim.plot_map_energy_delivery(save_path="map_energy_delivery.png")
        optimal_sim.plot_cumulative_energy_delivery(save_path="cumulative_energy_delivery.png")
        optimal_sim.plot_map_movement(save_path="map_movement_distance.png")
        optimal_sim.plot_map_soc(save_path="map_soc_over_time.png")
        optimal_sim.plot_map_self_charge_heatmap(save_path="map_self_charge_heatmap.png")

    if best_feasible:
        return best_feasible, best_feasible['sim_results'], best_feasible['stage2_sim'], iteration_log
    else:
        return None, results, stage2_sim, iteration_log


def main():
    # --- Parse mode flag ---
    simulation_only = "--sim-only" in sys.argv

    if simulation_only:
        print("\n" + "=" * 70)
        print("STAGE-2: SIMULATION-ONLY MODE (no MILP loop)")
        print("  Use this to validate parameters proposed by the MILP.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("STAGE-2: TERMINAL CHARGING WITH INTEGRATED OPTIMIZED PREEMPTION")
        print("  + MILP <-> SIMULATION FEEDBACK LOOP")
        print("=" * 70)

    try:
        sim, bus_lines = load_stockholm_data()
        print(f"Data loaded: {len(bus_lines)} lines")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    try:
        print("\nAssigning trips to buses...")
        LINES = [line.line_id for line in bus_lines]
        trip_to_bus, unassigned, bus_trips = sim.assign_trips_by_lines(
            turnover_time=300,
            lines=LINES
        )
        print(f"Trip assignment complete: {len(bus_trips)} buses")

        trip_change_stops = compute_trip_change_stops(sim, bus_trips)

    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Per-line battery capacities (kWh)
    LINE_BATTERY_CAPACITIES_KWH = {
        "1": 90,
        "2": 60,
        "3": 90,
        "4": 120,
        "6": 70,
    }
    # Convert to Wh for the simulation
    line_battery_capacities_wh = {
        lid: cap * 1000 for lid, cap in LINE_BATTERY_CAPACITIES_KWH.items()
    }

    # Default fallback capacity and initial MAP count
    initial_capacity_wh = 140 * 1000
    initial_num_maps = 9

    if simulation_only:
        # ---- Simulation-only mode ----
        print("\n" + "=" * 70)
        print("RUNNING SINGLE SIMULATION (no MILP)")
        print("=" * 70)

        results, stage2_sim = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=initial_capacity_wh,
            num_maps=initial_num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh=line_battery_capacities_wh,
            map_battery_capacity_wh=None,
            skip_plots=False,  # always generate plots in sim-only mode
        )

        # Final summary
        print("\n" + "=" * 70)
        print("SIMULATION-ONLY RESULTS")
        print("=" * 70)
        print(f"\nSimulation feasible: {'YES' if results['feasible'] else 'NO'}")
        print(f"Min SOC: {results['min_soc_overall_ratio'] * 100:.1f}%")
        print(f"Total energy charged: {results['total_energy_charged_wh'] / 1e6:,.3f} MWh")

        # Calculate and print costs for the proposed values
        bus_caps_kwh = {lid: cap / 1000.0 for lid, cap in line_battery_capacities_wh.items()}
        map_battery_kwh = resolve_map_battery_kwh(results, stage2_sim)

        calculate_and_print_cost(
            bus_caps_kwh=bus_caps_kwh,
            num_maps=initial_num_maps,
            map_battery_kwh=map_battery_kwh,
            sim_results=results,
            bus_lines=bus_lines,
            label="Proposed Values",
        )

        print(f"\n{'=' * 70}\n")

    else:
        # ---- Constraint-driven search mode ----
        print("\n" + "=" * 70)
        print("CONSTRAINT-DRIVEN FEASIBILITY SEARCH")
        print("=" * 70)

        solution, sim_results, stage2_sim, log = run_milp_simulation_loop(
            sim=sim,
            bus_trips=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            line_battery_capacities_wh=line_battery_capacities_wh,
            initial_capacity_wh=initial_capacity_wh,
            initial_num_maps=initial_num_maps,
            max_iterations=50,
        )

        # Final summary
        print("\n" + "=" * 70)
        print("FINAL RESULTS")
        print("=" * 70)

        if solution:
            print(f"\nFeasible Solution Found!")
            print(f"Number of MAPs: {solution['num_maps']}")
            print(f"MAP battery: {solution['map_battery_kwh']:.0f} kWh")
            print("\nBus battery capacities (per line):")
            for lid, cap in sorted(solution['bus_battery_kwh'].items()):
                print(f"  Line {lid}: {cap:.0f} kWh")
            print(f"\nSimulation feasible: {'YES' if sim_results['feasible'] else 'NO'}")
            print(f"Min SOC: {sim_results['min_soc_overall_ratio'] * 100:.1f}%")

            # Calculate and print costs for the feasible solution
            calculate_and_print_cost(
                bus_caps_kwh=solution['bus_battery_kwh'],
                num_maps=solution['num_maps'],
                map_battery_kwh=solution['map_battery_kwh'],
                sim_results=sim_results,
                bus_lines=bus_lines,
                label=f"Feasible Solution (iteration {solution['iteration']})",
            )
        else:
            print("\nNo feasible solution found within maximum iterations.")

        print(f"\n{'=' * 70}\n")

if __name__ == "__main__":
    main()
