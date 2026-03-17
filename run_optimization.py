"""
Run Terminal-Only Charging with INTEGRATED Optimized Preemption Strategy
WITH MILP <-> SIMULATION FEEDBACK LOOP for iterative feasibility search

Modes:
    --sim-only   Run a single simulation without the MILP loop (validation mode)
    (default)    Run the MILP <-> simulation feedback loop
"""

from integration_stage2 import run_terminal_charging_simulation
from post_simulation_milp import post_simulation_optimize
from dataclasses import dataclass
import math
import sys

@dataclass
class BusLineData:
    line_id: str
    num_buses: int
    trips_df: object

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


def diagnose_infeasibility(results, bus_stats):
    """
    Analyze simulation results to determine why the scenario was infeasible.

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
    line_worst = {}   # {line_id: worst min_soc_ratio}
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
                'reason': (f"Line {lid}: buses reached {ratio*100:.1f}% SOC "
                           f"(< 20%).  Increasing min battery to {new_min} kWh."),
            })

    # --- Check MAP SOC violations / insufficient MAPs ---
    if results.get('charging_enabled') and results['num_maps'] > 0:
        # If min overall bus SOC is still very low despite charging,
        # the system may need more MAPs
        if results['min_soc_overall_ratio'] < bus_min_soc:
            num_maps_cur = results['num_maps']
            constraints.append({
                'type': 'min_maps',
                'value': num_maps_cur + 1,
                'reason': (f"Overall min SOC {results['min_soc_overall_ratio']*100:.1f}% "
                           f"< 20% despite {num_maps_cur} MAPs.  "
                           f"Requiring at least {num_maps_cur + 1} MAPs."),
            })

    return constraints


def run_milp_simulation_loop(sim, bus_trips, bus_lines, trip_change_stops,
                              line_battery_capacities_wh,
                              initial_capacity_wh, initial_num_maps,
                              max_iterations=50):
    """
    Iterative MILP <-> simulation feedback loop.

    1. Run initial simulation with starting parameters.
    2. Run MILP to find optimal battery/MAP sizing.
    3. Re-run simulation with MILP-recommended values.
    4. If simulation is feasible -> record as best if lowest cost, then
       add a cost upper-bound constraint and continue searching.
    5. If infeasible -> diagnose, add constraints, re-run MILP, repeat.
    6. Always runs all iterations up to max_iterations and then returns
       the best feasible solution found across all iterations.

    The default of 50 iterations allows the MILP sufficient room to
    explore the search space.  The loop never stops early — it always
    runs to max_iterations so the MILP has maximum opportunity to find
    cheaper solutions.

    Graphs are generated only for the best feasible (optimal) iteration,
    not for intermediate ones.

    Returns (milp_results, sim_results, stage2_sim, iteration_log).
    """
    feedback_constraints = []
    iteration_log = []
    best_feasible = None

    # --- Iteration 0: simulate with user-provided start values ---
    print("\n" + "=" * 70)
    print("FEEDBACK LOOP: ITERATION 0 (User-Provided Start Values)")
    print("=" * 70)

    # Show the user-provided start values used for this iteration
    user_line_caps_kwh = {lid: wh / 1000.0
                          for lid, wh in line_battery_capacities_wh.items()}
    print(f"  User-provided start values:")
    for lid, cap in sorted(user_line_caps_kwh.items()):
        print(f"    Line {lid} bus battery: {cap:.0f} kWh")
    print(f"    Default bus battery: {initial_capacity_wh / 1000.0:.0f} kWh")
    print(f"    Number of MAPs: {initial_num_maps}")

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
        skip_plots=True,  # plots only for optimal iteration
    )

    milp_results = post_simulation_optimize(
        results, stage2_sim, bus_lines,
        feedback_constraints=feedback_constraints,
    )

    if milp_results is None:
        print("MILP solver not available - cannot run feedback loop.")
        return None, results, stage2_sim, []

    # Log iteration 0 with user-provided values (not MILP outputs)
    iteration_log.append({
        'iteration': 0,
        'milp_status': 'user_initial',
        'sim_feasible': results['feasible'],
        'bus_battery_kwh': user_line_caps_kwh,
        'map_battery_kwh': None,
        'num_maps': initial_num_maps,
        'objective': None,
    })

    # Track consecutive MILP failures to avoid infinite spinning
    consecutive_milp_failures = 0
    MAX_CONSECUTIVE_MILP_FAILURES = 3

    # --- Feedback iterations ---
    for iteration in range(1, max_iterations + 1):
        if milp_results is None or milp_results.get('objective_value') is None:
            consecutive_milp_failures += 1
            if consecutive_milp_failures >= MAX_CONSECUTIVE_MILP_FAILURES:
                print(f"\n[!] MILP returned no solution {MAX_CONSECUTIVE_MILP_FAILURES} times in a row - resetting and continuing.")
                consecutive_milp_failures = 0
            # Remove the cost_upper_bound so MILP can find any feasible
            # solution on the next pass.
            feedback_constraints = [
                fc for fc in feedback_constraints
                if fc.get('type') != 'cost_upper_bound'
            ]
            try:
                milp_results = post_simulation_optimize(
                    results, stage2_sim, bus_lines,
                    feedback_constraints=feedback_constraints,
                )
            except Exception as e:
                print(f"\n[!] MILP error during retry: {e}")
                milp_results = None
            continue

        consecutive_milp_failures = 0  # reset on success

        # Extract MILP-recommended values
        milp_bus_caps_kwh = milp_results['bus_battery_kwh']     # {line_id: kWh}
        milp_map_cap_kwh = milp_results['map_battery_kwh']      # kWh
        milp_num_maps = milp_results['num_maps']

        # Round MAP battery to nearest step of 10 (it should already be,
        # but ensure integer arithmetic).
        milp_map_cap_kwh = round(milp_map_cap_kwh / 10) * 10

        # Convert to Wh
        milp_line_caps_wh = {lid: cap * 1000 for lid, cap in milp_bus_caps_kwh.items()}
        milp_map_cap_wh = milp_map_cap_kwh * 1000

        print("\n" + "=" * 70)
        print(f"FEEDBACK LOOP: ITERATION {iteration}")
        print("=" * 70)
        print(f"  MILP-recommended values:")
        for lid, cap in sorted(milp_bus_caps_kwh.items()):
            print(f"    Line {lid} bus battery: {cap:.0f} kWh")
        print(f"    MAP battery: {milp_map_cap_kwh:.0f} kWh")
        print(f"    Number of MAPs: {milp_num_maps}")

        # Re-run simulation with MILP values (skip plots for intermediate)
        results, stage2_sim = run_terminal_charging_simulation(
            sim=sim,
            bus_trips_dict=bus_trips,
            bus_lines=bus_lines,
            trip_change_stops=trip_change_stops,
            battery_capacity_wh=initial_capacity_wh,
            num_maps=milp_num_maps,
            optimize_threshold=True,
            preemption_threshold=None,
            simulation_duration_s=86400,
            line_battery_capacities_wh=milp_line_caps_wh,
            map_battery_capacity_wh=milp_map_cap_wh,
            skip_plots=True,  # plots only for optimal iteration
        )

        print(f"\n  Simulation feasibility: {'FEASIBLE [OK]' if results['feasible'] else 'INFEASIBLE [X]'}")
        print(f"  Min SOC ratio: {results['min_soc_overall_ratio']*100:.1f}%")

        if results['feasible']:
            current_obj = milp_results.get('objective_value')
            best_obj = (best_feasible['milp_results']['objective_value']
                        if best_feasible else None)

            if best_obj is None or current_obj < best_obj:
                print(f"\n[OK] NEW BEST feasible solution at iteration {iteration}!"
                      f"  Cost: ${current_obj:,.0f}"
                      + (f" (prev best: ${best_obj:,.0f})" if best_obj else ""))
                best_feasible = {
                    'milp_results': milp_results,
                    'sim_results': results,
                    'stage2_sim': stage2_sim,
                    'iteration': iteration,
                }
            else:
                print(f"\n  Feasible but not cheaper (${current_obj:,.0f}"
                      f" >= best ${best_obj:,.0f}).")

            iteration_log.append({
                'iteration': iteration,
                'milp_status': milp_results.get('status'),
                'sim_feasible': True,
                'bus_battery_kwh': milp_bus_caps_kwh,
                'map_battery_kwh': milp_map_cap_kwh,
                'num_maps': milp_num_maps,
                'objective': current_obj,
            })

            # Add cost upper-bound to search for a cheaper feasible solution.
            # Use the best known cost minus a small tolerance (1$) so the MILP
            # is forced to find a strictly cheaper configuration.
            best_cost = best_feasible['milp_results']['objective_value']
            # Remove any previous cost_upper_bound constraint
            feedback_constraints = [
                fc for fc in feedback_constraints
                if fc.get('type') != 'cost_upper_bound'
            ]
            feedback_constraints.append({
                'type': 'cost_upper_bound',
                'value': best_cost - 1.0,
                'reason': (f"Searching for solution cheaper than "
                           f"${best_cost:,.0f}."),
            })

            # Re-run MILP with tighter cost bound
            try:
                milp_results = post_simulation_optimize(
                    results, stage2_sim, bus_lines,
                    feedback_constraints=feedback_constraints,
                )
            except Exception as e:
                print(f"\n[!] MILP error during cost-bound search: {e}")
                milp_results = None

            # Do NOT break here - let the loop continue so the MILP has
            # a chance to find cheaper solutions with different constraints.
            continue

        # --- Infeasible: diagnose and add constraints ---
        new_constraints = diagnose_infeasibility(results, results['bus_statistics'])
        if not new_constraints:
            print("  Could not diagnose infeasibility - continuing to next iteration.")
            continue

        for fc in new_constraints:
            print(f"  FEEDBACK: {fc['reason']}")
            feedback_constraints.append(fc)

        # Re-run MILP with updated constraints
        try:
            milp_results = post_simulation_optimize(
                results, stage2_sim, bus_lines,
                feedback_constraints=feedback_constraints,
            )
        except Exception as e:
            print(f"\n[!] MILP error with feedback constraints: {e}")
            milp_results = None

        iteration_log.append({
            'iteration': iteration,
            'milp_status': milp_results.get('status') if milp_results else None,
            'sim_feasible': False,
            'bus_battery_kwh': milp_results.get('bus_battery_kwh') if milp_results else None,
            'map_battery_kwh': milp_results.get('map_battery_kwh') if milp_results else None,
            'num_maps': milp_results.get('num_maps') if milp_results else None,
            'objective': milp_results.get('objective_value') if milp_results else None,
            'feedback_constraints': [fc['reason'] for fc in new_constraints],
        })

    else:
        print(f"\n[!] Max iterations ({max_iterations}) reached."
              + (" Best feasible solution retained."
                 if best_feasible else " No feasible solution found."))

    # --- Print iteration summary ---
    print("\n" + "=" * 70)
    print("FEEDBACK LOOP SUMMARY")
    print("=" * 70)
    print(f"\n{'Iter':<6} {'Status':<12} {'Feasible':<10} {'Objective':>15} {'MAPs':>6} {'MAP kWh':>9}")
    print("-" * 65)
    for entry in iteration_log:
        feas = "Y" if entry.get('sim_feasible') else "N"
        obj = f"${entry['objective']:,.0f}" if entry.get('objective') is not None else "N/A"
        maps = str(entry.get('num_maps', '-')) if entry.get('num_maps') is not None else '-'
        map_kwh = f"{entry['map_battery_kwh']:.0f}" if entry.get('map_battery_kwh') is not None else '-'
        best_mark = " <-- BEST" if (best_feasible
                                   and entry.get('sim_feasible')
                                   and entry.get('iteration') == best_feasible['iteration']) else ""
        print(f"{entry['iteration']:<6} {entry.get('milp_status',''):<12} {feas:<10} {obj:>15} {maps:>6} {map_kwh:>9}{best_mark}")
    print()

    # --- Generate plots only for the best feasible (optimal) iteration ---
    if best_feasible:
        print("Generating plots for the optimal iteration "
              f"(iteration {best_feasible['iteration']})...")
        optimal_sim = best_feasible['stage2_sim']
        optimal_sim.plot_soc(save_path="bus_soc_terminal_charging_optimized.png")
        optimal_sim.plot_map_energy_delivery(save_path="map_energy_delivery.png")
        optimal_sim.plot_cumulative_energy_delivery(save_path="cumulative_energy_delivery.png")
        optimal_sim.plot_map_movement(save_path="map_movement_distance.png")
        optimal_sim.plot_map_soc(save_path="map_soc_over_time.png")
        optimal_sim.plot_map_self_charge_heatmap(save_path="map_self_charge_heatmap.png")

    if best_feasible:
        return (best_feasible['milp_results'], best_feasible['sim_results'],
                best_feasible['stage2_sim'], iteration_log)
    else:
        return milp_results, results, stage2_sim, iteration_log


def main():
    # --- Parse mode flag ---
    simulation_only = "--sim-only" in sys.argv

    if simulation_only:
        print("\n" + "="*70)
        print("STAGE-2: SIMULATION-ONLY MODE (no MILP loop)")
        print("  Use this to validate parameters proposed by the MILP.")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("STAGE-2: TERMINAL CHARGING WITH INTEGRATED OPTIMIZED PREEMPTION")
        print("  + MILP <-> SIMULATION FEEDBACK LOOP")
        print("="*70)

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
        "1": 270,
        "2": 130,
        "3": 280,
        "4": 200,
        "6": 50,
    }
    # Convert to Wh for the simulation
    line_battery_capacities_wh = {
        lid: cap * 1000 for lid, cap in LINE_BATTERY_CAPACITIES_KWH.items()
    }

    # Default fallback capacity and initial MAP count
    initial_capacity_wh = 140 * 1000
    initial_num_maps = 2

    if simulation_only:
        # ---- Simulation-only mode ----
        print("\n" + "="*70)
        print("RUNNING SINGLE SIMULATION (no MILP)")
        print("="*70)

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
            skip_plots=False,  # always generate plots in sim-only mode
        )

        # Final summary
        print("\n" + "="*70)
        print("SIMULATION-ONLY RESULTS")
        print("="*70)
        print(f"\nSimulation feasible: {'YES' if results['feasible'] else 'NO'}")
        print(f"Min SOC: {results['min_soc_overall_ratio']*100:.1f}%")
        print(f"Total energy charged: {results['total_energy_charged_wh']/1e6:,.3f} MWh")
        print(f"\n{'='*70}\n")

    else:
        # ---- MILP feedback loop mode ----
        print("\n" + "="*70)
        print("MILP <-> SIMULATION FEEDBACK LOOP")
        print("="*70)

        milp_results, sim_results, stage2_sim, log = run_milp_simulation_loop(
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
        print("\n" + "="*70)
        print("FINAL RESULTS")
        print("="*70)

        if milp_results and milp_results.get('objective_value') is not None:
            print(f"\nOptimal Cost: ${milp_results['objective_value']:,.2f}")
            print(f"Number of MAPs: {milp_results['num_maps']}")
            print(f"MAP battery: {milp_results['map_battery_kwh']:.0f} kWh")
            print("\nBus battery capacities (per line):")
            for lid, cap in sorted(milp_results['bus_battery_kwh'].items()):
                print(f"  Line {lid}: {cap:.0f} kWh")
            print(f"\nSimulation feasible: {'YES' if sim_results['feasible'] else 'NO'}")
            print(f"Min SOC: {sim_results['min_soc_overall_ratio']*100:.1f}%")
        else:
            print("\nNo feasible MILP solution found.")

        print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()
