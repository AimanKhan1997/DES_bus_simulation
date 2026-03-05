"""
Charging Pipeline Runner
========================
Supports three operating modes, selected via the ``--mode`` flag:

  baseline   Run the stationary-MAP baseline simulation (one MAP per terminal,
             terminal-only charging with SOC-based preemption).  After the
             simulation the ChargingStrategyOptimizer performs an offline
             analysis and prints a comparison report that indicates whether a
             dynamic (en-route / hybrid) strategy would improve outcomes.

  dynamic    Run the full dynamic simulation with en-route MAP following,
             pre-simulation strategy optimisation and post-simulation depot
             placement analysis.  This is the original behaviour.

  full       Run the baseline first, then always run the dynamic simulation so
             that both sets of outputs can be compared side by side.

Usage
-----
  python run_optimization.py                  # defaults to --mode dynamic
  python run_optimization.py --mode baseline
  python run_optimization.py --mode dynamic
  python run_optimization.py --mode full

Optional flags
--------------
  --battery-kwh   INTEGER   Bus battery capacity in kWh  (default: 140)
  --num-maps      INTEGER   Number of MAPs for dynamic mode  (default: 2)
  --duration-h    FLOAT     Simulation duration in hours  (default: 24)
  --lines         SPACE-SEPARATED list of route IDs  (default: 1 2 3 4 6)
"""

import argparse
import traceback
from dataclasses import dataclass

from integration_stage2 import (
    run_terminal_charging_simulation,
    run_baseline_simulation,
)

# Minimum SOC improvement (percentage points) for the dynamic strategy to be
# declared strictly better when both scenarios are already feasible.
_MIN_SOC_IMPROVEMENT_PCT = 1.0


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

@dataclass
class BusLineData:
    line_id: str
    num_buses: int
    trips_df: object


def load_stockholm_data(lines=None, gtfs_folder="gtfs_data",
                        osm_xml="map.xml", date_str="20231108"):
    """Load GTFS + OSM data and return (sim, bus_lines)."""
    from DES_model import GTFSBusSim
    from Trip_assign import load_trips

    if lines is None:
        lines = ["1", "2", "3", "4", "6"]

    print("Loading Stockholm transit data...")
    sim = GTFSBusSim(gtfs_folder, osm_xml, date_str=date_str)
    trips_df = load_trips(gtfs_folder, date_str)

    bus_lines = []
    for line_id in lines:
        line_trips = trips_df[
            trips_df["route_short_name"] == line_id
        ].reset_index(drop=True)
        if not line_trips.empty:
            num_buses = max(1, int(len(line_trips) / 15))
            bus_lines.append(BusLineData(
                line_id=line_id,
                num_buses=num_buses,
                trips_df=line_trips,
            ))

    return sim, bus_lines


def compute_trip_change_stops(sim, bus_trips_dict):
    """Return the set of stop IDs where buses turn around between trips."""
    trip_change_stops = set()

    for bus_id, trip_ids in bus_trips_dict.items():
        sorted_trips = sorted(
            trip_ids,
            key=lambda t: (
                sim.gtfs.stop_sequence(t)[0]["arrival"]
                if sim.gtfs.stop_sequence(t)
                else float("inf")
            ),
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
                    if start_time - prev_end_time > 0:
                        trip_change_stops.add(prev_end_stop)
                prev_end_time = seq[-1]["arrival"]
                prev_end_stop = seq[-1]["stop_id"]
            except Exception:
                continue

    return trip_change_stops


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------

def run_baseline_mode(sim, bus_trips, bus_lines, trip_change_stops,
                      battery_kwh, duration_s, preemption_threshold=None):
    """Run the stationary-MAP baseline + post-simulation strategy comparison."""
    print("\n" + "=" * 70)
    print("MODE: BASELINE — STATIONARY MAPs + STRATEGY COMPARISON")
    print("=" * 70)

    out = run_baseline_simulation(
        sim=sim,
        bus_trips_dict=bus_trips,
        trip_change_stops=trip_change_stops,
        battery_capacity_wh=battery_kwh * 1000,
        preemption_threshold=preemption_threshold,
        simulation_duration_s=duration_s,
    )

    br = out.get("baseline_results", {})
    cmp = out.get("comparison", {})

    print("\n" + "=" * 70)
    print("BASELINE PIPELINE COMPLETE")
    print("=" * 70)
    thr = br.get("preemption_threshold")
    if thr is not None:
        print(f"  Preemption threshold:       {thr*100:.1f}% SOC")
    print(f"  Baseline feasible:          {'✓ YES' if br['feasible'] else '✗ NO'}")
    print(f"  Minimum bus SOC:            {br['min_soc_overall_ratio']*100:.1f}%")
    print(f"  Buses below 20% floor:      {len(br['buses_below_floor'])}")
    print(f"  Preemptions:                {br['preemption_count']}")
    print(f"  Total energy charged:       {br['total_energy_charged_wh']/1e6:,.3f} MWh")
    if cmp.get("dynamic_strategy_improves"):
        print("\n  ⚡ Recommendation: run with --mode dynamic (or --mode full)")
        print("     to deploy en-route / hybrid MAP charging.")
    else:
        print("\n  ✓ Baseline is sufficient — no additional strategy needed.")
    print("=" * 70 + "\n")

    return out


def run_dynamic_mode(sim, bus_trips, bus_lines, trip_change_stops,
                     battery_kwh, num_maps, duration_s):
    """Run the dynamic en-route MAP simulation."""
    print("\n" + "=" * 70)
    print("MODE: DYNAMIC — EN-ROUTE MAP CHARGING")
    print("=" * 70)

    results, stage2_sim = run_terminal_charging_simulation(
        sim=sim,
        bus_trips_dict=bus_trips,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        battery_capacity_wh=battery_kwh * 1000,
        num_maps=num_maps,
        optimize_threshold=True,
        preemption_threshold=None,
        simulation_duration_s=duration_s,
    )

    print("\n" + "=" * 70)
    print("DYNAMIC PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Feasible:                   {'✓ YES' if results['feasible'] else '✗ NO'}")
    print(f"  Minimum bus SOC:            {results['min_soc_overall_ratio']*100:.1f}%")
    print(f"  Total energy charged:       {results['total_energy_charged_wh']/1e6:,.3f} MWh")
    print("=" * 70 + "\n")

    return results, stage2_sim


def run_full_mode(sim, bus_trips, bus_lines, trip_change_stops,
                  battery_kwh, num_maps, duration_s, preemption_threshold=None):
    """Run baseline then dynamic simulation and print a side-by-side summary."""
    print("\n" + "=" * 70)
    print("MODE: FULL — BASELINE THEN DYNAMIC (SIDE-BY-SIDE COMPARISON)")
    print("=" * 70)

    # Phase 1: Baseline
    baseline_out = run_baseline_mode(
        sim, bus_trips, bus_lines, trip_change_stops,
        battery_kwh, duration_s,
        preemption_threshold=preemption_threshold,
    )

    # Phase 2: Dynamic
    dynamic_results, _ = run_dynamic_mode(
        sim, bus_trips, bus_lines, trip_change_stops,
        battery_kwh, num_maps, duration_s,
    )

    # Side-by-side comparison table
    br = baseline_out["baseline_results"]
    dr = dynamic_results

    print("\n" + "=" * 70)
    print("SIDE-BY-SIDE COMPARISON: BASELINE vs DYNAMIC")
    print("=" * 70)
    thr = br.get("preemption_threshold")
    if thr is not None:
        print(f"  (Baseline preemption threshold: {thr*100:.1f}% SOC)\n")
    print(f"\n{'Metric':<35} {'Baseline':>15} {'Dynamic':>15}")
    print("-" * 70)
    print(f"{'Feasible':<35} "
          f"{'✓ YES' if br['feasible'] else '✗ NO':>15} "
          f"{'✓ YES' if dr['feasible'] else '✗ NO':>15}")
    print(f"{'Min bus SOC (%)':<35} "
          f"{br['min_soc_overall_ratio']*100:>14.1f}% "
          f"{dr['min_soc_overall_ratio']*100:>14.1f}%")
    print(f"{'Total energy charged (MWh)':<35} "
          f"{br['total_energy_charged_wh']/1e6:>14.3f}  "
          f"{dr['total_energy_charged_wh']/1e6:>14.3f} ")
    print(f"{'Buses below 20% SOC floor':<35} "
          f"{len(br['buses_below_floor']):>15} "
          f"{'N/A':>15}")
    print(f"{'Preemptions':<35} "
          f"{br['preemption_count']:>15} "
          f"{dr['num_preemptions']:>15}")
    print("-" * 70)

    b_feasible = br["feasible"]
    d_feasible = dr["feasible"]
    if not b_feasible and d_feasible:
        verdict = "✓ Dynamic strategy RESOLVES baseline infeasibility"
    elif b_feasible and d_feasible:
        soc_gain = (dr["min_soc_overall_ratio"] - br["min_soc_overall_ratio"]) * 100
        if soc_gain > _MIN_SOC_IMPROVEMENT_PCT:
            verdict = f"✓ Dynamic improves min SOC by {soc_gain:.1f} percentage points"
        else:
            verdict = "  Both strategies are feasible; baseline is sufficient"
    else:
        verdict = "  Review individual simulation outputs for details"
    print(f"\n  Verdict: {verdict}")
    print("=" * 70 + "\n")

    return baseline_out, dynamic_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Bus MAP charging pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "dynamic", "full"],
        default="dynamic",
        help="Simulation mode (default: dynamic)",
    )
    parser.add_argument(
        "--battery-kwh",
        type=int,
        default=140,
        metavar="KWH",
        help="Bus battery capacity in kWh (default: 140)",
    )
    parser.add_argument(
        "--num-maps",
        type=int,
        default=2,
        metavar="N",
        help="Number of MAPs for dynamic mode (default: 2)",
    )
    parser.add_argument(
        "--duration-h",
        type=float,
        default=24.0,
        metavar="HOURS",
        help="Simulation duration in hours (default: 24)",
    )
    parser.add_argument(
        "--lines",
        nargs="+",
        default=["1", "2", "3", "4", "6"],
        metavar="LINE",
        help="Route IDs to simulate (default: 1 2 3 4 6)",
    )
    parser.add_argument(
        "--preemption-threshold",
        type=float,
        default=None,
        metavar="SOC",
        help=(
            "SOC fraction (0–1) at which a bus requests the terminal charger "
            "in baseline/full modes.  When omitted the threshold is derived "
            "automatically by PreemptionStrategyAnalyzer (recommended)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    duration_s = args.duration_h * 3600

    print("\n" + "=" * 70)
    print("BUS MAP CHARGING PIPELINE")
    print("=" * 70)
    print(f"  Mode:           {args.mode}")
    print(f"  Battery:        {args.battery_kwh} kWh")
    if args.mode in ("dynamic", "full"):
        print(f"  MAPs (dynamic): {args.num_maps}")
    if args.mode in ("baseline", "full"):
        if args.preemption_threshold is not None:
            print(f"  Preemption thr: {args.preemption_threshold*100:.1f}% SOC (override)")
        else:
            print(f"  Preemption thr: auto (PreemptionStrategyAnalyzer)")
    print(f"  Duration:       {args.duration_h:.1f} h")
    print(f"  Lines:          {', '.join(args.lines)}")
    print("=" * 70)

    # --- Load data ---
    try:
        sim, bus_lines = load_stockholm_data(lines=args.lines)
        print(f"Data loaded: {len(bus_lines)} line(s)")
    except Exception as e:
        print(f"ERROR loading data: {e}")
        traceback.print_exc()
        return

    # --- Assign trips ---
    try:
        print("\nAssigning trips to buses...")
        trip_to_bus, unassigned, bus_trips = sim.assign_trips_by_lines(
            turnover_time=300,
            lines=args.lines,
        )
        print(f"Trip assignment complete: {len(bus_trips)} buses")
        trip_change_stops = compute_trip_change_stops(sim, bus_trips)
        print(f"Terminal (trip-change) stops: {len(trip_change_stops)}")
    except Exception as e:
        print(f"ERROR assigning trips: {e}")
        traceback.print_exc()
        return

    # --- Run selected mode ---
    try:
        if args.mode == "baseline":
            run_baseline_mode(
                sim, bus_trips, bus_lines, trip_change_stops,
                args.battery_kwh, duration_s,
                preemption_threshold=args.preemption_threshold,
            )
        elif args.mode == "dynamic":
            run_dynamic_mode(
                sim, bus_trips, bus_lines, trip_change_stops,
                args.battery_kwh, args.num_maps, duration_s,
            )
        elif args.mode == "full":
            run_full_mode(
                sim, bus_trips, bus_lines, trip_change_stops,
                args.battery_kwh, args.num_maps, duration_s,
                preemption_threshold=args.preemption_threshold,
            )
    except Exception as e:
        print(f"ERROR during simulation: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()