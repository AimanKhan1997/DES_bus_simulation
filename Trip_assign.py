# python
import pandas as pd
from collections import defaultdict
import datetime

def hhmmss_to_sec(ts):
    h, m, s = map(int, ts.split(":"))
    return h * 3600 + m * 60 + s


def get_active_service_ids(gtfs_folder, date_str):
    """Return a set of service_ids active on the given date (YYYYMMDD)."""
    target_date = datetime.datetime.strptime(date_str, "%Y%m%d").date()
    services = set()

    cal_path = f"{gtfs_folder}/calendar.txt"
    cal_dates_path = f"{gtfs_folder}/calendar_dates.txt"

    try:
        cal = pd.read_csv(cal_path, dtype=str)
        cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d").dt.date
        cal["end_date"] = pd.to_datetime(cal["end_date"], format="%Y%m%d").dt.date

        weekday_name = target_date.strftime("%A").lower()
        for _, row in cal.iterrows():
            if row["start_date"] <= target_date <= row["end_date"]:
                if str(row.get(weekday_name, "0")) == "1":
                    services.add(row["service_id"])
    except FileNotFoundError:
        pass

    try:
        cal_dates = pd.read_csv(cal_dates_path, dtype=str)
        adds = cal_dates[cal_dates["date"] == date_str]
        for _, row in adds.iterrows():
            service_id = row["service_id"]
            ex_type = row.get("exception_type", "1")
            if ex_type == "1":
                services.add(service_id)
            elif ex_type == "2" and service_id in services:
                services.discard(service_id)
    except FileNotFoundError:
        pass

    return services


def load_trips(gtfs_folder, date_str=None):
    """
    Load trips; include route information when available.
    If date_str provided, filter trips to active service_ids.
    Returned DataFrame columns include: trip_id, start, end, headsign, route_id, route_short_name
    """
    stop_times = pd.read_csv(f"{gtfs_folder}/stop_times.txt", dtype=str)
    stop_times["arrival_secs"] = stop_times["arrival_time"].apply(hhmmss_to_sec)
    stop_times["stop_sequence"] = stop_times["stop_sequence"].astype(int)

    # read trips metadata (route_id, service_id) if available
    try:
        trips_meta = pd.read_csv(f"{gtfs_folder}/trips.txt", dtype=str)
    except FileNotFoundError:
        trips_meta = pd.DataFrame(columns=["trip_id", "route_id", "service_id"])

    # merge route_id and service_id into stop_times for easy filtering
    if not trips_meta.empty:
        stop_times = stop_times.merge(
            trips_meta[["trip_id", "route_id", "service_id"]],
            on="trip_id",
            how="left"
        )

    # try to add route_short_name from routes.txt
    try:
        routes = pd.read_csv(f"{gtfs_folder}/routes.txt", dtype=str)
        if "route_short_name" in routes.columns and "route_id" in routes.columns:
            stop_times = stop_times.merge(
                routes[["route_id", "route_short_name"]],
                on="route_id",
                how="left"
            )
    except FileNotFoundError:
        pass

    # date filtering using service_id if requested
    if date_str is not None:
        if trips_meta.empty:
            print(f"Warning: trips.txt not found — cannot filter by date {date_str}; returning no trips.")
            return pd.DataFrame(columns=["trip_id", "start", "end", "headsign", "route_id", "route_short_name"]).astype({"start":int, "end":int})
        active_services = get_active_service_ids(gtfs_folder, date_str)
        if len(active_services) == 0:
            print(f"Warning: no active services found for {date_str} — returning no trips.")
            return pd.DataFrame(columns=["trip_id", "start", "end", "headsign", "route_id", "route_short_name"]).astype({"start":int, "end":int})
        valid_trip_ids = set(trips_meta[trips_meta["service_id"].isin(active_services)]["trip_id"])
        stop_times = stop_times[stop_times["trip_id"].isin(valid_trip_ids)]

    trips = []
    for trip_id, grp in stop_times.groupby("trip_id"):
        grp = grp.sort_values("stop_sequence")
        trips.append({
            "trip_id": trip_id,
            "start": int(grp.iloc[0]["arrival_secs"]),
            "end": int(grp.iloc[-1]["arrival_secs"]),
            "headsign": grp.iloc[0].get("stop_headsign"),
            "route_id": grp.iloc[0].get("route_id"),
            "route_short_name": grp.iloc[0].get("route_short_name")
        })

    cols = ["trip_id", "start", "end", "headsign", "route_id", "route_short_name"]
    return pd.DataFrame(trips).sort_values("start").reset_index(drop=True)[cols]


def assign_trips(turnover, trips_df):
    """
    Minimise buses first, then maximise chain length.
    """

    buses = []
    trip_to_bus = {}
    bus_trips = defaultdict(list)

    for _, trip in trips_df.iterrows():
        candidates = []

        for bus in buses:
            if (
                bus["available"] + turnover <= trip.start and
                bus["headsign"] != trip.headsign
            ):
                candidates.append(bus)

        if candidates:
            # choose bus with largest chain
            bus = max(candidates, key=lambda b: len(bus_trips[b["id"]]))
        else:
            bus = {
                "id": f"bus_{len(buses)+1}",
                "available": 0,
                "headsign": None
            }
            buses.append(bus)

        bus["available"] = trip.end
        bus["headsign"] = trip.headsign
        trip_to_bus[trip.trip_id] = bus["id"]
        bus_trips[bus["id"]].append(trip.trip_id)

    return trip_to_bus, bus_trips


def evaluate_solution(bus_trips):
    num_buses = len(bus_trips)
    trips_per_bus = [len(v) for v in bus_trips.values()] if num_buses > 0 else [0]
    return {
        "buses": num_buses,
        "avg_trips": sum(trips_per_bus) / num_buses if num_buses > 0 else 0.0,
        "max_trips": max(trips_per_bus) if num_buses > 0 else 0
    }


def optimise_turnover(gtfs_folder, turnovers, date_str, lines=None):
    """
    If lines is None: behave as before (global optimisation).
    If lines is a list of identifiers (route_short_name or route_id), run optimisation per line and return a dict.
    """
    trips_df = load_trips(gtfs_folder, date_str)

    if lines is None:
        # original behaviour: find single best across all trips
        best = None
        for t in turnovers:
            trip_to_bus, bus_trips = assign_trips(t, trips_df)
            metrics = evaluate_solution(bus_trips)

            all_trips = set(trips_df["trip_id"]) if not trips_df.empty else set()
            assigned = set(trip_to_bus.keys())
            unassigned = sorted(list(all_trips - assigned))

            print(
                f"Turnover={t:4d}s | "
                f"Buses={metrics['buses']:3d} | "
                f"Avg trips={metrics['avg_trips']:.2f} | "
                f"Max trips={metrics['max_trips']} | "
                f"Unassigned={len(unassigned)}"
            )

            if len(unassigned) > 0:
                sample = unassigned[:10]
                print(f"  Unassigned trip IDs (sample up to 10): {sample}")

            if best is None:
                best = (t, metrics, bus_trips, unassigned)
            else:
                if (
                    metrics["buses"] < best[1]["buses"] or
                    (
                        metrics["buses"] == best[1]["buses"] and
                        metrics["avg_trips"] > best[1]["avg_trips"]
                    )
                ):
                    best = (t, metrics, bus_trips, unassigned)

        return best

    # per-line optimisation
    per_line_results = {}
    for line in lines:
        # prefer matching route_short_name, else route_id
        if "route_short_name" in trips_df.columns and trips_df["route_short_name"].notna().any() and (trips_df["route_short_name"] == line).any():
            df_line = trips_df[trips_df["route_short_name"] == line].reset_index(drop=True)
        else:
            df_line = trips_df[trips_df["route_id"] == line].reset_index(drop=True)

        if df_line.empty:
            print(f"Warning: no trips found for line {line}")
            per_line_results[line] = None
            continue

        best = None
        for t in turnovers:
            trip_to_bus, bus_trips = assign_trips(t, df_line)
            metrics = evaluate_solution(bus_trips)

            all_trips = set(df_line["trip_id"])
            assigned = set(trip_to_bus.keys())
            unassigned = sorted(list(all_trips - assigned))

            print(
                f"Line={line} | Turnover={t:4d}s | "
                f"Buses={metrics['buses']:3d} | "
                f"Avg trips={metrics['avg_trips']:.2f} | "
                f"Max trips={metrics['max_trips']} | "
                f"Unassigned={len(unassigned)}"
            )

            if len(unassigned) > 0:
                sample = unassigned[:10]
                print(f"  Line {line} unassigned (sample up to 10): {sample}")

            if best is None:
                best = (t, metrics, bus_trips, unassigned)
            else:
                if (
                    metrics["buses"] < best[1]["buses"] or
                    (
                        metrics["buses"] == best[1]["buses"] and
                        metrics["avg_trips"] > best[1]["avg_trips"]
                    )
                ):
                    best = (t, metrics, bus_trips, unassigned)

        per_line_results[line] = best

    return per_line_results


if __name__ == "__main__":
    GTFS_FOLDER = "gtfs_data"
    TURNOVERS = [300]
    DATE_STR = "20231108"

    # examine specific lines (route_short_name or route_id)
    LINES = ["1", "2", "3", "4", "6"]

    results = optimise_turnover(GTFS_FOLDER, TURNOVERS, DATE_STR, lines=LINES)

    print("\n================ PER-LINE RESULTS ================")
    for line, res in results.items():
        if res is None:
            print(f"Line {line}: no trips")
            continue
        best_turnover, metrics, bus_trips, unassigned = res
        print(f"Line {line}: Best turnover={best_turnover}s | Buses={metrics['buses']} | Avg trips={metrics['avg_trips']:.2f} | Max trips={metrics['max_trips']}")
        print(f"  {len(bus_trips)} buses assigned; unassigned trips: {len(unassigned)}")

