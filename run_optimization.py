"""
Run Terminal-Only Charging with INTEGRATED Optimized Preemption Strategy
FIXED: Properly handles num_maps=0 (no charging allowed)
"""

from integration_stage2 import run_terminal_charging_simulation
from post_simulation_milp import post_simulation_optimize
from dataclasses import dataclass

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

def main():
    print("\n" + "="*70)
    print("STAGE-2: TERMINAL CHARGING WITH INTEGRATED OPTIMIZED PREEMPTION")
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

    print("\n" + "="*70)
    print("TESTING BATTERY CAPACITIES WITH AUTOMATIC THRESHOLD OPTIMIZATION")
    print("="*70)

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

    # Default fallback capacity (used if a line is not in the per-line map)
    battery_capacities_kwh = [140]
    num_maps_options = [2]
    results_summary = []

    for capacity_kwh in battery_capacities_kwh:
        capacity_wh = capacity_kwh * 1000

        for num_maps in num_maps_options:
            print(f"\n{'='*70}")
            print(f"Default Battery: {capacity_kwh} kWh | MAPs: {num_maps}")
            for lid, cap in sorted(LINE_BATTERY_CAPACITIES_KWH.items()):
                print(f"  Line {lid}: {cap} kWh")
            print(f"{'='*70}")

            try:
                results, stage2_sim = run_terminal_charging_simulation(
                    sim=sim,
                    bus_trips_dict=bus_trips,
                    bus_lines=bus_lines,
                    trip_change_stops=trip_change_stops,
                    battery_capacity_wh=capacity_wh,
                    num_maps=num_maps,
                    optimize_threshold=True,
                    preemption_threshold=None,
                    simulation_duration_s=86400,
                    line_battery_capacities_wh=line_battery_capacities_wh
                )

                results_summary.append({
                    'battery_kwh': capacity_kwh,
                    'num_maps': num_maps,
                    'charging_enabled': results['charging_enabled'],
                    'optimal_threshold': results['preemption_threshold'],
                    'feasible': results['feasible'],
                    'min_soc_ratio': results['min_soc_overall_ratio'],
                    'num_preemptions': results['num_preemptions']
                })

                # Run post-simulation MILP optimization
                milp_results = post_simulation_optimize(
                    results, stage2_sim, bus_lines
                )

            except Exception as e:
                print(f"ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - OPTIMAL PREEMPTION THRESHOLDS")
    print("="*70)
    print(f"\n{'Battery':<12} {'MAPs':<8} {'Charging':<12} {'Optimal Threshold':<20} {'Feasible':<12} {'Min SOC %':<15}")
    print("-"*95)

    for result in results_summary:
        feasible_str = "✓ YES" if result['feasible'] else "✗ NO"
        charging_str = "YES" if result['charging_enabled'] else "NO"

        if result['optimal_threshold'] is not None:
            threshold_str = f"{result['optimal_threshold']*100:.1f}%"
        else:
            threshold_str = "DISABLED"

        print(f"{result['battery_kwh']} kWh   {result['num_maps']:<8} "
              f"{charging_str:<12} {threshold_str:<20} {feasible_str:<12} "
              f"{result['min_soc_ratio']*100:<14.1f}%")

    print(f"\n{'='*70}\n")

if __name__ == "__main__":
    main()