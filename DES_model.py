import simpy
import time
from dataclasses import dataclass
from collections import defaultdict
from pyproj import Geod
import shapely.geometry as geom
import shapely.ops as ops
import matplotlib.pyplot as plt
import Trip_assign as trip_assign
from gtfs_loader import GTFSLoader
from osm_graph import OSMGraph
# ---------------------------
# Configuration
# ---------------------------
ENERGY_PER_METER_WH = 2.7  # Wh per meter (default for 470 kWh battery)
BATTERY_CAPACITY_WH = 470000


def energy_per_meter_for_capacity(capacity_kwh: float) -> float:
    """Return energy consumption rate (Wh/m) adjusted for battery capacity.

    Lighter batteries (smaller capacity) reduce bus weight and thus energy
    consumption.  The formula linearly reduces consumption from the baseline
    2.7 Wh/m at 470 kWh:

        rate = 2.7 - (470 - capacity_kwh) * 0.0005
    """
    return 2.7 - (470.0 - capacity_kwh) * 0.0005


def energy_consumption_wh(distance_m: float, capacity_kwh: float = 470.0) -> float:
    """Energy consumed (Wh) over *distance_m* metres.

    When *capacity_kwh* is supplied the per-line rate is used; otherwise the
    default 470 kWh rate (2.7 Wh/m) is applied for backward compatibility.
    """
    return energy_per_meter_for_capacity(capacity_kwh) * distance_m

@dataclass
class BusState:
    bus_id: str
    trip_id: str
    soc_wh: float
    location_node: int

# ---------------------------
# Charging policy abstraction
# ---------------------------
class ChargingPolicy:
    """Encapsulates charging logic and thresholds.

    Uses fractional thresholds (start_pct, stop_pct) relative to capacity and a rate in Wh/s.
    """
    def __init__(self, start_pct: float = 0.7, stop_pct: float = 0.8, rate_wh_per_s: float = 97.22):
        self.start_pct = float(start_pct)
        self.stop_pct = float(stop_pct)
        self.rate_wh_per_s = float(rate_wh_per_s)

    def wants_charge(self, soc_wh: float, capacity_wh: float) -> bool:
        if capacity_wh is None or capacity_wh <= 0:
            return False
        return soc_wh < (self.start_pct * capacity_wh)

    def max_target_wh(self, capacity_wh: float) -> float:
        return self.stop_pct * capacity_wh

    def charge_amount_for_duration(self, soc_wh: float, capacity_wh: float, duration_s: float) -> float:
        """Return Wh that can be delivered in duration_s without exceeding stop threshold."""
        if duration_s <= 0:
            return 0.0
        target = self.max_target_wh(capacity_wh)
        remaining = max(0.0, target - soc_wh)
        possible = self.rate_wh_per_s * duration_s
        return min(remaining, possible)

# ---------------------------
# Stop Charging Manager (per-stop exclusive / queued)
# ---------------------------
class StopChargingManager:
    """Manage exclusive charging slots per GTFS stop_id with a waiting queue prioritized by lowest SOC.

    Behavior:
    - capacity_per_stop: number of concurrent chargers at a given stop (default 1).
    - request_stop_charging(stop_id, bus_id, soc_wh, capacity_wh, desired_duration_s)
      attempts to start charging immediately; if a slot is free it will start and invoke the on_charge callback.
      If no slot is free, the request is enqueued. When a slot frees, the manager assigns the queued bus with the
      lowest SOC whose remaining desired duration is still positive.
    - on_charge: optional callback (bus_id, amount_wh, duration_s) invoked when charging actually starts.
    - get_stop_logs() returns per-stop session logs.
    """
    def __init__(self, env: simpy.Environment, capacity_per_stop: int = 1, policy: ChargingPolicy = None, on_charge=None):
        self.env = env
        self.capacity = max(1, int(capacity_per_stop))
        self.policy = policy or ChargingPolicy()
        self.on_charge = on_charge

        # active counts per stop
        self._active_counts = defaultdict(int)
        # per-stop queue: stop_id -> list of requests {bus_id, soc, capacity, desired_s, requested_time}
        self._queues = defaultdict(list)
        # per-stop logs
        self._logs = defaultdict(list)

    def _can_start(self, stop_id: str) -> bool:
        return self._active_counts.get(stop_id, 0) < self.capacity

    def _start_session(self, stop_id: str, req: dict):
        """Start a charging session for given request dict and schedule its release."""
        bus_id = req["bus_id"]
        soc = req["soc"]
        cap = req["capacity"]
        desired = req["desired"]

        amount = self.policy.charge_amount_for_duration(soc, cap, desired)
        if amount <= 0 or desired <= 0:
            return (False, 0.0, 0.0)

        duration = min(amount / self.policy.rate_wh_per_s, desired)

        # mark active and log
        self._active_counts[stop_id] += 1
        entry = {"bus_id": bus_id, "start_time": float(self.env.now), "duration_s": duration, "amount_wh": amount}
        self._logs[stop_id].append(entry)

        # call callback immediately so simulation-level SOC is updated
        try:
            if callable(self.on_charge):
                self.on_charge(bus_id, amount, duration)
        except Exception:
            pass

        # schedule release and attempt to assign next when done
        def _release(sid, dur):
            yield self.env.timeout(dur)
            self._active_counts[sid] = max(0, self._active_counts.get(sid, 1) - 1)
            # try to assign next queued request at this stop
            self._assign_next_for_stop(sid)

        self.env.process(_release(stop_id, duration))
        return (True, duration, amount)

    def _assign_next_for_stop(self, stop_id: str):
        q = self._queues.get(stop_id)
        if not q:
            return
        # compute remaining desired for queued entries and filter out expired ones
        now = float(self.env.now)
        alive = []
        for r in q:
            elapsed = now - r["requested_time"]
            remaining = r["desired"] - elapsed
            if remaining > 0:
                newr = r.copy()
                newr["desired"] = remaining
                alive.append(newr)
        self._queues[stop_id] = alive
        if not alive:
            return
        # choose lowest SOC first (then earliest request time)
        alive.sort(key=lambda x: (x["soc"], x["requested_time"]))
        if self._can_start(stop_id):
            req = alive.pop(0)
            self._queues[stop_id] = alive
            self._start_session(stop_id, req)

    def request_stop_charging(self, stop_id: str, bus_id: str, soc_wh: float, capacity_wh: float, desired_duration_s: float):
        """Request charging at stop. Returns (started, actual_dur, amount_wh). If not started the request is queued."""
        if not stop_id:
            return (False, 0.0, 0.0)
        # only attempt if bus needs charging
        if not self.policy.wants_charge(soc_wh, capacity_wh):
            return (False, 0.0, 0.0)

        req = {"bus_id": bus_id, "soc": soc_wh, "capacity": capacity_wh, "desired": desired_duration_s, "requested_time": float(self.env.now)}

        if self._can_start(stop_id):
            return self._start_session(stop_id, req)

        # otherwise enqueue and return not started
        self._queues[stop_id].append(req)
        return (False, 0.0, 0.0)

    def get_stop_logs(self):
        return {sid: list(logs) for sid, logs in self._logs.items()}

# ---------------------------
# Simulation class
# ---------------------------
class GTFSBusSim:
    def __init__(self, gtfs_folder: str, osm_xml: str, date_str: str = None):
        self.env = simpy.Environment()
        # Pass optional date filter to GTFSLoader
        self.gtfs = GTFSLoader(gtfs_folder, date_str)
        # store folder/date so we can reuse the Trip_assign helper which reads GTFS directly
        self.gtfs_folder = gtfs_folder
        self.date_str = date_str
        self.osm = OSMGraph(osm_xml)
        self.geod = Geod(ellps="WGS84")

        self.stops = self.osm.snap_stops(self.gtfs.stops)
        self.stop_node = dict(zip(self.stops.stop_id, self.stops.node_id))

        # Metrics
        self.bus_soc_history = defaultdict(list)
        self.bus_distance = defaultdict(float)
        self.bus_energy = defaultdict(float)
        # per-bus charged energy and distance while charging (tracked via charging manager responses)
        self.bus_energy_charged = defaultdict(float)
        self.bus_charge_distance = defaultdict(float)
        # per-bus maximum dwell: maps bus_id -> {"stop_id": str, "dwell_s": float}
        self.bus_max_dwell = {}
        self.total_distance = 0.0
        self.total_energy = 0.0

        # low-SOC event log: list of dicts {bus_id, time_s, node_id, lat, lon, soc_wh, hour}
        self.low_soc_events = []
        # last low-SOC logged per bus to avoid duplicate logging at the same node/time
        self._last_low_soc_logged = {}

        # Per-line battery capacities (Wh). Modify values here to change per-line capacities.
        # Keys should match the line identifiers used when assigning trips (route_short_name or route_id).
        self.line_battery_map = {
            "1": 270000,
            "2": 130000,
            "3": 280000,
            "4": 200000,
            "6": 50000,
        }

        # create charging policy and stop-charging manager (start 70%, stop 80%, 97.22 Wh/s)
        policy = ChargingPolicy(start_pct=0.7, stop_pct=0.8, rate_wh_per_s=97.22)
        # stop_charging_manager enforces per-stop exclusivity and queues by lowest SOC
        # pass on_charge callback so manager updates central SOC when a charging session starts
        self.stop_charging_manager = StopChargingManager(self.env, capacity_per_stop=1, policy=policy, on_charge=self._on_stop_charge)

        # Central SOC store (Wh) used so the StopChargingManager can update SOC asynchronously
        self.bus_soc = defaultdict(float)

        # Set of GTFS stop_ids where trip-change charging is allowed (only one bus can charge at such a stop at a time)
        # Populate this set with known trip-change stops (example: line 4 uses the two stop_ids provided)
        # You can modify this externally before running the sim, e.g. `sim.trip_change_stops.add('9022001010098003')`.
        self.trip_change_stops = set()

        # internal cache for shape distances
        self._shape_cache = {}

    def _on_stop_charge(self, bus_id: str, amount_wh: float, duration_s: float):
        """Callback from StopChargingManager invoked when a charging session starts.

        Update the central SOC and per-bus charged metrics immediately so the simulation
        processes observe the increased SOC on subsequent steps.
        """
        try:
            cap = self.get_bus_capacity(bus_id)
            prev = self.bus_soc.get(bus_id, float(cap))
            new = min(float(cap), prev + amount_wh)
            self.bus_soc[bus_id] = new
            # track delivered energy
            self.bus_energy_charged[bus_id] += amount_wh
            # this is end-station charging; distance charged remains zero or unspecified
            self.bus_charge_distance[bus_id] += 0.0
            # record SOC history point
            self.bus_soc_history[bus_id].append((float(self.env.now), new))
        except Exception:
            pass

    # ---------------------------
    # Shape distance
    # ---------------------------
    def shape_distance(self, shape_id, a, b):
        if shape_id is None or a is None or b is None:
            return 0.0

        key = (shape_id, a, b)
        if key in self._shape_cache:
            return self._shape_cache[key]

        entry = self.gtfs.shape_polylines.get(shape_id)
        if entry is None:
            return 0.0

        # Extract substring of the shape between the projected points and compute
        # geodetic distance along the substring coordinates for exact metric length.
        line = entry["line"]
        pa = geom.Point(a[1], a[0])
        pb = geom.Point(b[1], b[0])

        try:
            da = line.project(pa)
            db = line.project(pb)
            if da > db:
                da, db = db, da

            sub = ops.substring(line, da, db)
            coords = list(sub.coords)
            # if substring is a single point, fallback to straight geod
            if len(coords) < 2:
                _, _, direct = self.geod.inv(a[1], a[0], b[1], b[0])
                self._shape_cache[key] = abs(direct)
                return abs(direct)

            dist = 0.0
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i + 1]
                _, _, d = self.geod.inv(lon1, lat1, lon2, lat2)
                dist += abs(d)

            self._shape_cache[key] = dist
            return dist
        except Exception:
            # fallback: straight-line geod distance
            _, _, direct = self.geod.inv(a[1], a[0], b[1], b[0])
            self._shape_cache[key] = abs(direct)
            return abs(direct)

    def trip_headsign(self, trip_id: str):
        df = self.gtfs.stop_times[self.gtfs.stop_times.trip_id == trip_id]
        if df.empty:
            return None
        return df.sort_values("stop_sequence").iloc[0].stop_headsign

    def assign_trips_by_lines(self, turnover_time: int = 300, lines=None):
        if lines is None:
            raise ValueError("lines must be provided for per-line assignment")
        per_line_results = trip_assign.optimise_turnover(self.gtfs_folder, [turnover_time], self.date_str, lines=lines)

        aggregated_trip_to_bus = {}
        aggregated_bus_trips = {}
        aggregated_unassigned = []

        for line, res in per_line_results.items():
            if res is None:
                continue
            _, _, bus_trips_line, unassigned_line = res
            for bus_id, trips in bus_trips_line.items():
                prefixed_id = f"line{line}_{bus_id}"
                aggregated_bus_trips[prefixed_id] = list(trips)
                for t in trips:
                    aggregated_trip_to_bus[t] = prefixed_id
            aggregated_unassigned.extend(unassigned_line)

        aggregated_unassigned = sorted(set(aggregated_unassigned))

        print(f"\nAssigned trips across lines {lines}: total buses={len(aggregated_bus_trips)}; unassigned={len(aggregated_unassigned)}")
        return aggregated_trip_to_bus, aggregated_unassigned, aggregated_bus_trips

    def bus_process(self, bus_id, trips):
        capacity = self.get_bus_capacity(bus_id)
        # Per-line energy consumption rate (Wh/m) based on battery capacity
        bus_energy_per_meter = energy_per_meter_for_capacity(capacity / 1000.0)
        # initialize central SOC for this bus to full capacity at start
        self.bus_soc[bus_id] = capacity
        # local reference will be read from central store as needed
        current_time = 0.0
        prev_end_node = None
        prev_end_time = None
        # previous trip end stop id (GTFS stop_id) for reporting dwell between trips
        prev_end_stop_id = None

        self.bus_soc_history[bus_id].append((0.0, capacity))

        for trip_id in trips:
            seq = self.gtfs.stop_sequence(trip_id)
            if not seq:
                continue
            shape_id = self.gtfs.shape_for_trip(trip_id)

            # Jump to trip start time (idle time between end of previous trip and this trip)
            trip_start = seq[0]["arrival"]
            idle_dt = max(0.0, trip_start - current_time)
            if idle_dt > 0:
                # wait the idle time (the bus continues schedule regardless of charging availability)
                yield self.env.timeout(idle_dt)
                current_time = trip_start

                # Record the inter-trip idle (dwell between trips) at the previous end stop (GTFS stop_id)
                if prev_end_stop_id is not None:
                    dwell = idle_dt
                    prev = self.bus_max_dwell.get(bus_id)
                    if prev is None or dwell > prev.get("dwell_s", 0.0):
                        self.bus_max_dwell[bus_id] = {"stop_id": prev_end_stop_id, "dwell_s": dwell}

                # End-station charging (unchanged logic) – only attempt if bus ended at same node where trip begins
                try:
                    start_node = self.stop_node.get(seq[0].get("stop_id"))
                except Exception:
                    start_node = None

                # Only allow stop-charging when bus is changing trips at a designated trip-change stop
                if prev_end_node is not None and start_node is not None and prev_end_node == start_node and prev_end_time is not None:
                    duration = trip_start - prev_end_time
                    # check that this GTFS stop is in the configured trip-change stops set
                    start_stop_id = seq[0].get("stop_id")
                    if duration > 0 and start_stop_id in self.trip_change_stops:
                        # obtain current SOC from central store
                        soc_now = self.bus_soc.get(bus_id, capacity)
                        started, actual_dur, amount = self.stop_charging_manager.request_stop_charging(
                            stop_id=start_stop_id,
                            bus_id=bus_id,
                            soc_wh=soc_now,
                            capacity_wh=capacity,
                            desired_duration_s=duration,
                        )
                        # Do not update SOC here; StopChargingManager will call the on_charge callback
                        # when the charging session actually starts (immediately or later from queue),
                        # which updates self.bus_soc and self.bus_energy_charged.

            # Move through stops in the trip
            for i in range(1, len(seq)):
                prev = seq[i - 1]
                curr = seq[i]

                dt = curr["arrival"] - prev["arrival"]
                # compute both OSM network distance (if available) and shape distance BEFORE the timeout
                osm_dist = None
                shape_dist = None

                node_a = self.stop_node.get(prev.get("stop_id"))
                node_b = self.stop_node.get(curr.get("stop_id"))
                if node_a is not None and node_b is not None:
                    try:
                        osm_dist = self.osm.shortest_path_distance(node_a, node_b)
                    except Exception:
                        osm_dist = None

                try:
                    shape_dist = self.shape_distance(
                        shape_id,
                        (prev["lat"], prev["lon"]),
                        (curr["lat"], curr["lon"])
                    )
                except Exception:
                    shape_dist = None

                try:
                    _, _, straight_dist = self.geod.inv(prev["lon"], prev["lat"], curr["lon"], curr["lat"])
                    straight_dist = abs(straight_dist)
                except Exception:
                    straight_dist = None

                if straight_dist is not None:
                    dist = straight_dist
                else:
                    if shape_dist is not None:
                        dist = shape_dist
                    elif osm_dist is not None:
                        dist = osm_dist
                    else:
                        dist = 0.0


                # No mobile charging while moving: advance simulation time for the movement/dwell interval
                yield self.env.timeout(max(0.0, dt))
                current_time = curr["arrival"]

                energy = dist * bus_energy_per_meter
                # subtract consumption from central SOC store
                prev_soc = self.bus_soc.get(bus_id, capacity)
                new_soc = max(0.0, prev_soc - energy)
                self.bus_soc[bus_id] = new_soc

                # update per-bus aggregates only; global totals are computed from these
                self.bus_distance[bus_id] += dist
                self.bus_energy[bus_id] += energy

                # record low-SOC event if SOC below threshold at stop arrival
                try:
                    node_id = self.stop_node.get(curr.get("stop_id"))
                    threshold_wh = 0.2 * capacity
                    curr_soc_for_bus = self.bus_soc.get(bus_id, capacity)
                    if curr_soc_for_bus < threshold_wh and node_id is not None:
                        last = self._last_low_soc_logged.get(bus_id)
                        now = float(self.env.now)
                        last_ok = last is None or (last.get("node") != node_id) or (now - last.get("time", 0) > 60)
                        if last_ok:
                            self.record_low_soc_event(bus_id=bus_id, time_s=now, node_id=node_id, soc_wh=curr_soc_for_bus, capacity_wh=capacity)
                            self._last_low_soc_logged[bus_id] = {"node": node_id, "time": now}
                except Exception:
                    pass

                # append SOC history after applying consumption (and possibly earlier charging via on_charge)
                self.bus_soc_history[bus_id].append((self.env.now, self.bus_soc.get(bus_id, capacity)))

            # update previous trip end info (node id for charging comparisons, and GTFS stop_id for dwell reporting)
            prev_end_node = self.stop_node.get(seq[-1].get("stop_id"))
            prev_end_stop_id = seq[-1].get("stop_id")
            prev_end_time = seq[-1]["arrival"]

    def record_low_soc_event(self, bus_id, time_s, node_id, soc_wh, capacity_wh=None):
        lat = None
        lon = None
        try:
            if node_id in self.osm.nodes.index:
                row = self.osm.nodes.loc[node_id]
                lat = float(row.get("y", row.get("lat", None)))
                lon = float(row.get("x", row.get("lon", None)))
            else:
                try:
                    stop_row = self.stops[self.stops.node_id == node_id].iloc[0]
                    lat = float(stop_row.stop_lat)
                    lon = float(stop_row.stop_lon)
                except Exception:
                    lat = None
                    lon = None
        except Exception:
            lat = None
            lon = None

        hour = int((time_s % 86400) // 3600) if time_s is not None else None

        ev = {
            "bus_id": bus_id,
            "time_s": time_s,
            "hour": hour,
            "node_id": node_id,
            "lat": lat,
            "lon": lon,
            "soc_wh": soc_wh,
        }
        try:
            if capacity_wh is not None:
                ev["capacity_wh"] = capacity_wh
                ev["soc_pct"] = (soc_wh / capacity_wh) * 100 if capacity_wh and capacity_wh > 0 else None
            else:
                cap = self.get_bus_capacity(bus_id)
                ev["capacity_wh"] = cap
                ev["soc_pct"] = (soc_wh / cap) * 100 if cap and cap > 0 else None
        except Exception:
            pass
        self.low_soc_events.append(ev)

    def get_bus_capacity(self, bus_id: str):
        try:
            if not bus_id:
                return BATTERY_CAPACITY_WH
            if bus_id.startswith("line"):
                rest = bus_id[4:]
                parts = rest.split("_", 1)
                line_key = parts[0]
                if line_key in self.line_battery_map:
                    return self.line_battery_map[line_key]
            return BATTERY_CAPACITY_WH
        except Exception:
            return BATTERY_CAPACITY_WH

    def run(self, until: float):
        t0 = time.time()
        self.env.run(until=until)
        print(f"Simulation finished in {time.time() - t0:.2f}s")

    def print_statistics(self):
        print("\n=== BUS STATISTICS ===")
        for b, d in self.bus_distance.items():
            print(f"{b}: {d:.1f} m, {self.bus_energy[b]:.1f} Wh")

        # Compute totals from per-bus aggregates to avoid accidental double-counting
        computed_total_distance = sum(self.bus_distance.values())
        computed_total_energy = sum(self.bus_energy.values())

        if abs(self.total_distance - computed_total_distance) > 1e-6:
            self.total_distance = computed_total_distance
        if abs(self.total_energy - computed_total_energy) > 1e-6:
            self.total_energy = computed_total_energy
        below_count = 0
        total_buses = 0
        for b, hist in self.bus_soc_history.items():
            if not hist:
                continue
            total_buses += 1
            last_soc = hist[-1][1]
            cap = self.get_bus_capacity(b)
            threshold_wh_bus = 0.2 * cap
            if last_soc < threshold_wh_bus:
                below_count += 1

        print(f"\nTOTAL distance: {self.total_distance:.1f} m")
        print(f"TOTAL energy:   {self.total_energy:.1f} Wh")
        print(f"Buses below 20% SOC: {below_count} / {total_buses}")

        # Print per-bus longest dwell and stop id
        print("\n=== BUS MAX DWELL ===")
        # consider buses that have SOC history or dwell records
        all_buses = sorted(set(list(self.bus_soc_history.keys()) + list(self.bus_max_dwell.keys())))
        for b in all_buses:
            md = self.bus_max_dwell.get(b)
            if md is None:
                print(f"{b}: no dwell records")
            else:
                print(f"{b}: longest dwell {md['dwell_s']:.0f} s at stop {md['stop_id']}")

    def plot_soc(self):
        plt.figure(figsize=(10, 6))
        lines_seen = {}
        cmap = plt.get_cmap("tab10")
        next_color_idx = 0

        for b, series in self.bus_soc_history.items():
            filtered = [x for x in series if x[0] > 0]
            if not filtered:
                continue
            t = [x[0] for x in filtered]
            s = [x[1] for x in filtered]

            line_key = "default"
            try:
                if b and b.startswith("line"):
                    rest = b[4:]
                    line_key = rest.split("_", 1)[0]
            except Exception:
                line_key = "default"

            if line_key not in lines_seen:
                lines_seen[line_key] = cmap(next_color_idx % cmap.N)
                next_color_idx += 1
            color = lines_seen[line_key]

            plt.plot(t, s, label="_nolegend_", color=color)

        if lines_seen:
            try:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=col, lw=3) for col in lines_seen.values()]
                labels = [f"Line {lk}" for lk in lines_seen.keys()]
                plt.legend(handles=handles, labels=labels, loc="best", fontsize="small", ncol=2, title="Line")
            except Exception:
                pass

        plt.xlabel("Time [s]")
        plt.ylabel("Battery SOC [Wh]")
        plt.title("Bus SOC trajectories (Wh)")
        plt.grid(True)
        plt.show()

    def simulation_end_time(self, trip_ids, buffer: int = 0):
        end_times = []
        for trip_id in trip_ids:
            seq = self.gtfs.stop_sequence(trip_id)
            if not seq:
                continue
            end_times.append(seq[-1]["arrival"])
        if not end_times:
            return 0.0
        return max(end_times) + buffer

    def compute_trip_distance_methods(self, trip_id):
        seq = self.gtfs.stop_sequence(trip_id)
        if not seq or len(seq) < 2:
            return {"straight_sum": 0.0, "shape_sum": 0.0, "osm_sum": None}

        shape_id = self.gtfs.shape_for_trip(trip_id)
        straight_sum = 0.0

        for i in range(1, len(seq)):
            a = seq[i-1]
            b = seq[i]
            _, _, d = self.geod.inv(a["lon"], a["lat"], b["lon"], b["lat"])
            straight_sum += abs(d)

        return {
            "straight_sum": straight_sum,
        }

    def compute_bus_max_dwell_from_assignments(self, bus_trips: dict):
        """Compute per-bus maximum inter-trip dwell from assigned bus_trips without running the sim.

        bus_trips: dict mapping bus_id -> [trip_id,...] where each trip's stop_sequence can be read from self.gtfs.
        This fills self.bus_max_dwell (bus_id -> {"stop_id", "dwell_s"}) and sets self.trip_change_stops
        to the set of stop_ids where these maximum dwells occur.
        """
        results = {}
        stops_set = set()
        for bus_id, trips in bus_trips.items():
            # ensure trips are chronological
            ordered = sorted(trips, key=lambda t: self.gtfs.stop_sequence(t)[0]["arrival"] if self.gtfs.stop_sequence(t) else 0)
            prev_end_time = None
            prev_end_stop = None
            max_dwell = 0.0
            max_stop = None
            for trip_id in ordered:
                seq = self.gtfs.stop_sequence(trip_id)
                if not seq:
                    continue
                start_time = seq[0]["arrival"]
                if prev_end_time is not None:
                    idle = max(0.0, start_time - prev_end_time)
                    if idle > max_dwell and prev_end_stop is not None:
                        max_dwell = idle
                        max_stop = prev_end_stop
                # update prev_end for next iteration
                prev_end_time = seq[-1]["arrival"]
                prev_end_stop = seq[-1]["stop_id"]

            if max_stop is not None:
                results[bus_id] = {"stop_id": max_stop, "dwell_s": max_dwell}
                stops_set.add(max_stop)

        # store into sim
        for b, v in results.items():
            self.bus_max_dwell[b] = v

        # auto-populate trip_change_stops with all detected max-dwell stops
        self.trip_change_stops = set(stops_set)
        return results

if __name__ == "__main__":

     GTFS_FOLDER = "gtfs_data"
     OSM_XML = "map.xml"

     DATE_STR = "20231108"  # Only consider trips on this date

     sim = GTFSBusSim(GTFS_FOLDER, OSM_XML, date_str=DATE_STR)

     # Automatically detect trip-change stops from assigned bus trips (uses per-bus max inter-trip dwell)

     # Assign trips per line using Trip_assign logic for Lines 1,2,3,4,6
     LINES = ["1", "2", "3", "4", "6"]
     TURNOVER = 300
     trip_to_bus, unassigned_trips, bus_trips = sim.assign_trips_by_lines(turnover_time=TURNOVER, lines=LINES)

     # 3. Sort trips per bus chronologically
     for bus_id in bus_trips:
         bus_trips[bus_id].sort(
             key=lambda t: sim.gtfs.stop_sequence(t)[0]["arrival"]
         )

     # Auto-compute per-bus maximum inter-trip dwell and use these stops as trip-change stops
     detected = sim.compute_bus_max_dwell_from_assignments(bus_trips)
     print(f"Detected {len(sim.trip_change_stops)} trip-change stops from assignments")

     # 4. Start SimPy processes
     for bus_id, trips in bus_trips.items():
         sim.env.process(sim.bus_process(bus_id, trips))
     # 5. Compute correct simulation horizon
     sim_end = sim.simulation_end_time(
         trip_ids=trip_to_bus.keys(),
         buffer=300  # optional safety buffer
     )
     # 5. Run simulation
     sim.run(until=sim_end)
     # 6. Output
     sim.print_statistics()
     sim.plot_soc()
     stop_logs = sim.stop_charging_manager.get_stop_logs()
     for stop_id, sessions in stop_logs.items():
         print(stop_id)
         for s in sessions:
             print("  ", s)  # {bus_id, start_time, duration_s, amount_wh}

# import simpy
# import time
# from dataclasses import dataclass
# from collections import defaultdict
# from pyproj import Geod
# import shapely.geometry as geom
# import shapely.ops as ops
# import matplotlib.pyplot as plt
# import Trip_assign as trip_assign
# from gtfs_loader import GTFSLoader
# from osm_graph import OSMGraph
# # ---------------------------
# # Configuration
# # ---------------------------
# ENERGY_PER_METER_WH = 2.7  # Wh per meter
# BATTERY_CAPACITY_WH = 470000
# def energy_consumption_wh(distance_m: float) -> float:
#     return ENERGY_PER_METER_WH * distance_m
#
# @dataclass
# class BusState:
#     bus_id: str
#     trip_id: str
#     soc_wh: float
#     location_node: int
#
# class GTFSBusSim:
#     def __init__(self, gtfs_folder: str, osm_xml: str, date_str: str = None):
#         self.env = simpy.Environment()
#         # Pass optional date filter to GTFSLoader
#         self.gtfs = GTFSLoader(gtfs_folder, date_str)
#         # store folder/date so we can reuse the Trip_assign helper which reads GTFS directly
#         self.gtfs_folder = gtfs_folder
#         self.date_str = date_str
#         self.osm = OSMGraph(osm_xml)
#         self.geod = Geod(ellps="WGS84")
#
#         self.stops = self.osm.snap_stops(self.gtfs.stops)
#         self.stop_node = dict(zip(self.stops.stop_id, self.stops.node_id))
#
#         # Metrics
#         self.bus_soc_history = defaultdict(list)
#         self.bus_distance = defaultdict(float)
#         self.bus_energy = defaultdict(float)
#         self.total_distance = 0.0
#         self.total_energy = 0.0
#
#         # low-SOC event log: list of dicts {bus_id, time_s, node_id, lat, lon, soc_wh, hour}
#         self.low_soc_events = []
#         # last low-SOC logged per bus to avoid duplicate logging at the same node/time
#         self._last_low_soc_logged = {}
#
#         # Per-line battery capacities (Wh). Modify values here to change per-line capacities.
#         # Keys should match the line identifiers used when assigning trips (route_short_name or route_id).
#         self.line_battery_map = {
#             "1": 270000,
#             "2": 130000,
#             "3": 280000,
#             "4": 200000,
#             "6": 50000,
#         }
#
#     # ---------------------------
#     # Shape distance
#     # ---------------------------
#     def shape_distance(self, shape_id, a, b):
#         if shape_id is None or a is None or b is None:
#             return 0.0
#
#         key = (shape_id, a, b)
#         if key in self._shape_cache:
#             return self._shape_cache[key]
#
#         entry = self.gtfs.shape_polylines.get(shape_id)
#         if entry is None:
#             return 0.0
#
#         # Extract substring of the shape between the projected points and compute
#         # geodetic distance along the substring coordinates for exact metric length.
#         line = entry["line"]
#         pa = geom.Point(a[1], a[0])
#         pb = geom.Point(b[1], b[0])
#
#         try:
#             da = line.project(pa)
#             db = line.project(pb)
#             if da > db:
#                 da, db = db, da
#
#             sub = ops.substring(line, da, db)
#             coords = list(sub.coords)
#             # if substring is a single point, fallback to straight geod
#             if len(coords) < 2:
#                 _, _, direct = self.geod.inv(a[1], a[0], b[1], b[0])
#                 self._shape_cache[key] = abs(direct)
#                 return abs(direct)
#
#             dist = 0.0
#             for i in range(len(coords) - 1):
#                 lon1, lat1 = coords[i]
#                 lon2, lat2 = coords[i + 1]
#                 _, _, d = self.geod.inv(lon1, lat1, lon2, lat2)
#                 dist += abs(d)
#
#             self._shape_cache[key] = dist
#             return dist
#         except Exception:
#             # fallback: straight-line geod distance
#             _, _, direct = self.geod.inv(a[1], a[0], b[1], b[0])
#             self._shape_cache[key] = abs(direct)
#             return abs(direct)
#
#     def trip_headsign(self, trip_id: str):
#         df = self.gtfs.stop_times[self.gtfs.stop_times.trip_id == trip_id]
#         if df.empty:
#             return None
#         return df.sort_values("stop_sequence").iloc[0].stop_headsign
#
#     def assign_trips_by_lines(self, turnover_time: int = 300, lines=None):
#         """Assign trips per specified lines using Trip_assign.optimise_turnover.
#
#         Returns aggregated trip_to_bus (trip_id->bus_id), unassigned_trips list, and
#         aggregated bus_trips dict (bus_id -> [trip_id,...]) where bus_ids are prefixed
#         with the line identifier to ensure uniqueness across lines.
#         """
#         if lines is None:
#             raise ValueError("lines must be provided for per-line assignment")
#
#         # Use Trip_assign.optimise_turnover to run per-line assignment (with single turnover)
#         per_line_results = trip_assign.optimise_turnover(self.gtfs_folder, [turnover_time], self.date_str, lines=lines)
#
#         aggregated_trip_to_bus = {}
#         aggregated_bus_trips = {}
#         aggregated_unassigned = []
#
#         for line, res in per_line_results.items():
#             if res is None:
#                 continue
#             # res is (best_turnover, metrics, bus_trips, unassigned)
#             _, _, bus_trips_line, unassigned_line = res
#             # prefix bus ids to avoid collisions across lines
#             for bus_id, trips in bus_trips_line.items():
#                 prefixed_id = f"line{line}_{bus_id}"
#                 aggregated_bus_trips[prefixed_id] = list(trips)
#                 for t in trips:
#                     aggregated_trip_to_bus[t] = prefixed_id
#             aggregated_unassigned.extend(unassigned_line)
#
#         # remove duplicates in unassigned
#         aggregated_unassigned = sorted(set(aggregated_unassigned))
#
#         print(f"\nAssigned trips across lines {lines}: total buses={len(aggregated_bus_trips)}; unassigned={len(aggregated_unassigned)}")
#         return aggregated_trip_to_bus, aggregated_unassigned, aggregated_bus_trips
#
#     def bus_process(self, bus_id, trips):
#         # Determine bus capacity based on its line prefix (e.g. 'line1_bus_1') or fallback
#         capacity = self.get_bus_capacity(bus_id)
#         soc = capacity
#         current_time = 0.0
#
#         self.bus_soc_history[bus_id].append((0.0, soc))
#
#         for trip_id in trips:
#             seq = self.gtfs.stop_sequence(trip_id)
#             shape_id = self.gtfs.shape_for_trip(trip_id)
#
#             # Jump to trip start time
#             trip_start = seq[0]["arrival"]
#             yield self.env.timeout(max(0.0, trip_start - current_time))
#             current_time = trip_start
#
#             for i in range(1, len(seq)):
#                 prev = seq[i - 1]
#                 curr = seq[i]
#
#                 dt = curr["arrival"] - prev["arrival"]
#                 yield self.env.timeout(max(0.0, dt))
#                 current_time = curr["arrival"]
#
#                 # compute both OSM network distance (if available) and shape distance
#                 osm_dist = None
#                 shape_dist = None
#
#                 # OSM shortest-path between snapped nodes
#                 node_a = self.stop_node.get(prev.get("stop_id"))
#                 node_b = self.stop_node.get(curr.get("stop_id"))
#                 if node_a is not None and node_b is not None:
#                     try:
#                         osm_dist = self.osm.shortest_path_distance(node_a, node_b)
#                     except Exception:
#                         osm_dist = None
#
#                 # shape-based precise distance
#                 try:
#                     shape_dist = self.shape_distance(
#                         shape_id,
#                         (prev["lat"], prev["lon"]),
#                         (curr["lat"], curr["lon"])
#                     )
#                 except Exception:
#                     shape_dist = None
#
#                 # straight-line geod distance between stops (fallback)
#                 try:
#                     _, _, straight_dist = self.geod.inv(prev["lon"], prev["lat"], curr["lon"], curr["lat"])
#                     straight_dist = abs(straight_dist)
#                 except Exception:
#                     straight_dist = None
#
#                 # Use straight-line (geodetic) distance as the authoritative measure
#                 if straight_dist is not None:
#                     dist = straight_dist
#                 else:
#                     # fallback to shape or osm if straight not available
#                     if shape_dist is not None:
#                         dist = shape_dist
#                     elif osm_dist is not None:
#                         dist = osm_dist
#                     else:
#                         dist = 0.0
#
#                 energy = energy_consumption_wh(dist)
#                 soc = max(0.0, soc - energy)
#
#                 # update per-bus aggregates only; global totals are computed from these
#                 self.bus_distance[bus_id] += dist
#                 self.bus_energy[bus_id] += energy
#
#                 # record low-SOC event if SOC below threshold at stop arrival
#                 try:
#                     # determine node for current stop
#                     node_id = self.stop_node.get(curr.get("stop_id"))
#                     # threshold in Wh based on this bus's capacity
#                     threshold_wh = 0.2 * capacity
#                     if soc < threshold_wh and node_id is not None:
#                         # avoid duplicate logs for same bus at same node within 60s
#                         last = self._last_low_soc_logged.get(bus_id)
#                         now = float(self.env.now)
#                         last_ok = last is None or (last.get("node") != node_id) or (now - last.get("time", 0) > 60)
#                         if last_ok:
#                             self.record_low_soc_event(bus_id=bus_id, time_s=now, node_id=node_id, soc_wh=soc, capacity_wh=capacity)
#                             self._last_low_soc_logged[bus_id] = {"node": node_id, "time": now}
#                 except Exception:
#                     pass
#
#                 self.bus_soc_history[bus_id].append((self.env.now, soc))
#
#     def record_low_soc_event(self, bus_id, time_s, node_id, soc_wh, capacity_wh=None):
#         """Record a low-SOC event with location and time.
#
#         time_s: simulation seconds since midnight (or since simulation start).
#         node_id: OSM node id where event occurred.
#         soc_wh: battery SOC in Wh.
#         capacity_wh: optional battery capacity of this bus (Wh).
#         """
#         lat = None
#         lon = None
#         try:
#             # self.osm.nodes is a GeoDataFrame where index is node id
#             if node_id in self.osm.nodes.index:
#                 row = self.osm.nodes.loc[node_id]
#                 # nodes gdf might have 'y' and 'x' columns
#                 lat = float(row.get("y", row.get("lat", None)))
#                 lon = float(row.get("x", row.get("lon", None)))
#             else:
#                 # fallback: try to get coordinates from stops if node came from a stop
#                 # reverse map: find stop with that node
#                 try:
#                     stop_row = self.stops[self.stops.node_id == node_id].iloc[0]
#                     lat = float(stop_row.stop_lat)
#                     lon = float(stop_row.stop_lon)
#                 except Exception:
#                     lat = None
#                     lon = None
#         except Exception:
#             lat = None
#             lon = None
#
#         hour = int((time_s % 86400) // 3600) if time_s is not None else None
#
#         ev = {
#             "bus_id": bus_id,
#             "time_s": time_s,
#             "hour": hour,
#             "node_id": node_id,
#             "lat": lat,
#             "lon": lon,
#             "soc_wh": soc_wh,
#         }
#         # include capacity if provided, else attempt to infer it
#         try:
#             if capacity_wh is not None:
#                 ev["capacity_wh"] = capacity_wh
#                 ev["soc_pct"] = (soc_wh / capacity_wh) * 100 if capacity_wh and capacity_wh > 0 else None
#             else:
#                 cap = self.get_bus_capacity(bus_id)
#                 ev["capacity_wh"] = cap
#                 ev["soc_pct"] = (soc_wh / cap) * 100 if cap and cap > 0 else None
#         except Exception:
#             pass
#         self.low_soc_events.append(ev)
#
#     def get_bus_capacity(self, bus_id: str):
#         """Return battery capacity (Wh) for a bus based on its bus_id prefix.
#
#         If bus_id starts with 'line{line}_', the line part is used to lookup capacity
#         in self.line_battery_map. Otherwise, fall back to BATTERY_CAPACITY_WH.
#         """
#         try:
#             if not bus_id:
#                 return BATTERY_CAPACITY_WH
#             if bus_id.startswith("line"):
#                 rest = bus_id[4:]
#                 # rest like '1_bus_3' -> split at '_' to get '1'
#                 parts = rest.split("_", 1)
#                 line_key = parts[0]
#                 if line_key in self.line_battery_map:
#                     return self.line_battery_map[line_key]
#             # fallback: try to infer line from bus_id if contains 'line'
#             # otherwise return default
#             return BATTERY_CAPACITY_WH
#         except Exception:
#             return BATTERY_CAPACITY_WH
#
#     def run(self, until: float):
#         t0 = time.time()
#         self.env.run(until=until)
#         print(f"Simulation finished in {time.time() - t0:.2f}s")
#
#     def print_statistics(self):
#         print("\n=== BUS STATISTICS ===")
#         for b, d in self.bus_distance.items():
#             print(f"{b}: {d:.1f} m, {self.bus_energy[b]:.1f} Wh")
#
#         # Compute totals from per-bus aggregates to avoid accidental double-counting
#         computed_total_distance = sum(self.bus_distance.values())
#         computed_total_energy = sum(self.bus_energy.values())
#
#         # If running tally differs from computed totals, report a warning and prefer computed totals
#         if abs(self.total_distance - computed_total_distance) > 1e-6:
#             self.total_distance = computed_total_distance
#         if abs(self.total_energy - computed_total_energy) > 1e-6:
#             self.total_energy = computed_total_energy
#         # Count buses below 20% SOC using the last recorded SOC in history, per-bus capacities
#         below_count = 0
#         total_buses = 0
#         for b, hist in self.bus_soc_history.items():
#             if not hist:
#                 continue
#             total_buses += 1
#             last_soc = hist[-1][1]  # SOC in Wh
#             cap = self.get_bus_capacity(b)
#             threshold_wh_bus = 0.2 * cap
#             if last_soc < threshold_wh_bus:
#                 below_count += 1
#
#         print(f"\nTOTAL distance: {self.total_distance:.1f} m")
#         print(f"TOTAL energy:   {self.total_energy:.1f} Wh")
#         print(f"Buses below 20% SOC: {below_count} / {total_buses}")
#
#     def plot_soc(self):
#         plt.figure(figsize=(10, 6))
#         # assign consistent colors per line using a categorical colormap
#         lines_seen = {}
#         cmap = plt.get_cmap("tab10")
#         next_color_idx = 0
#
#         for b, series in self.bus_soc_history.items():
#             filtered = [x for x in series if x[0] > 0]
#             if not filtered:
#                 continue
#             t = [x[0] for x in filtered]
#             # plot battery energy in Wh (raw SOC values), not percent
#             s = [x[1] for x in filtered]
#
#             # derive line key from bus id (e.g. 'line1_bus_3' -> '1')
#             line_key = "default"
#             try:
#                 if b and b.startswith("line"):
#                     rest = b[4:]
#                     line_key = rest.split("_", 1)[0]
#             except Exception:
#                 line_key = "default"
#
#             if line_key not in lines_seen:
#                 lines_seen[line_key] = cmap(next_color_idx % cmap.N)
#                 next_color_idx += 1
#             color = lines_seen[line_key]
#
#             # plot without adding a legend entry for each bus
#             plt.plot(t, s, label="_nolegend_", color=color)
#
#         # create a legend that maps one entry per line_key -> color
#         if lines_seen:
#             try:
#                 from matplotlib.lines import Line2D
#                 handles = [Line2D([0], [0], color=col, lw=3) for col in lines_seen.values()]
#                 labels = [f"Line {lk}" for lk in lines_seen.keys()]
#                 plt.legend(handles=handles, labels=labels, loc="best", fontsize="small", ncol=2, title="Line")
#             except Exception:
#                 # fallback: no custom legend if Line2D import fails
#                 pass
#
#         plt.xlabel("Time [s]")
#         plt.ylabel("Battery SOC [Wh]")
#         plt.title("Bus SOC trajectories (Wh)")
#         plt.grid(True)
#         plt.show()
#
#     def simulation_end_time(self, trip_ids, buffer: int = 0):
#         """
#         Compute the simulation horizon so that all trips fully execute.s
#         """
#         end_times = []
#
#         for trip_id in trip_ids:
#             seq = self.gtfs.stop_sequence(trip_id)
#             if not seq:
#                 continue
#             end_times.append(seq[-1]["arrival"])
#
#         if not end_times:
#             return 0.0
#
#         return max(end_times) + buffer
#
#     def compute_trip_distance_methods(self, trip_id):
#         """Return three distance estimates (meters) for a given trip_id:
#            - straight_sum: sum of geodetic distances between consecutive stop coords
#         """
#         seq = self.gtfs.stop_sequence(trip_id)
#         if not seq or len(seq) < 2:
#             return {"straight_sum": 0.0, "shape_sum": 0.0, "osm_sum": None}
#
#         shape_id = self.gtfs.shape_for_trip(trip_id)
#         straight_sum = 0.0
#
#         for i in range(1, len(seq)):
#             a = seq[i-1]
#             b = seq[i]
#             # straight-line geod distance between stops
#             _, _, d = self.geod.inv(a["lon"], a["lat"], b["lon"], b["lat"])
#             straight_sum += abs(d)
#
#
#         return {
#             "straight_sum": straight_sum,
#         }
#
# if __name__ == "__main__":
#
#      GTFS_FOLDER = "gtfs_data"
#      OSM_XML = "map.xml"
#
#      DATE_STR = "20231108"  # Only consider trips on this date
#
#      sim = GTFSBusSim(GTFS_FOLDER, OSM_XML, date_str=DATE_STR)
#
#      # Assign trips per line using Trip_assign logic for Lines 1,2,3,4,6
#      LINES = ["1", "2", "3", "4", "6"]
#      TURNOVER = 300
#      trip_to_bus, unassigned_trips, bus_trips = sim.assign_trips_by_lines(turnover_time=TURNOVER, lines=LINES)
#
#      # 3. Sort trips per bus chronologically
#      for bus_id in bus_trips:
#          bus_trips[bus_id].sort(
#              key=lambda t: sim.gtfs.stop_sequence(t)[0]["arrival"]
#          )
#
#      # 4. Start SimPy processes
#      for bus_id, trips in bus_trips.items():
#          sim.env.process(sim.bus_process(bus_id, trips))
#      # 5. Compute correct simulation horizon
#      sim_end = sim.simulation_end_time(
#          trip_ids=trip_to_bus.keys(),
#          buffer=300  # optional safety buffer
#      )
#      # 5. Run simulation
#      sim.run(until=sim_end)
#      # 6. Output
#      sim.print_statistics()
#      sim.plot_soc()
#
#      # Optional debug: print distance comparisons for first few trips
#      debug_trip_ids = list(trip_to_bus.keys())[:3]
#      print("\n=== Distance Comparison (first 10 trips) ===")
#      for trip_id in debug_trip_ids:
#          distances = sim.compute_trip_distance_methods(trip_id)
#          print(f"{trip_id}: {distances}")