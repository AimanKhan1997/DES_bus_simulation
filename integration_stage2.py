"""
Stage 2: DES Charging Simulation - Terminal + Dynamic MAP Movement Charging
WITH MAP MOVEMENT, ROUTE-BASED CHARGING, AND ADVANCED TRACKING
"""

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import heapq
from dataclasses import dataclass
from pyproj import Geod

# ========================
# PARAMETERS & CONSTANTS
# ========================

BUS_BATTERY_MAX_WH = 470000
MAP_CHARGING_RATE_WH_S = 97.22

BUS_CHARGE_THRESHOLD_SOC = 0.70
BUS_CHARGE_CUTOFF_SOC = 0.80
BUS_MIN_SOC = 0.20

MAX_CONCURRENT_CHARGERS = 2
CHARGER_SPEED_MS = 37.78
ENERGY_PER_METER_WH = 2.7

MAP_BATTERY_CAPACITY_WH = 150000   # 150 kWh per MAP
MAP_MIN_SOC = 0.10                 # MAP cannot go below 10% SOC
MAP_SELF_CHARGE_RATE_WH_S = 233.0  # MAP self-charges at 233 Wh/s

# ========================
# MAP USAGE TRACKER (NEW)
# ========================

@dataclass
class MAPChargingRecord:
    """Record of a MAP charging event"""
    map_id: int
    bus_id: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    energy_delivered_wh: float
    location: str
    soc_before_wh: float
    soc_after_wh: float

@dataclass
class MAPMovementRecord:
    """Record of a MAP movement/location event"""
    map_id: int
    start_time_s: float
    end_time_s: float
    from_location: str  # stop_id, segment_id, or "depot"
    to_location: str
    distance_m: float
    duration_s: float
    associated_bus_id: Optional[str] = None

@dataclass
class MAPState:
    """Current state of a MAP"""
    map_id: int
    current_location: str  # stop_id, segment_id, or "depot"
    current_time_s: float
    current_soc_wh: float
    battery_capacity_wh: float
    is_charging: bool = False
    assigned_bus_id: Optional[str] = None
    target_location: Optional[str] = None
    arrival_time_at_target_s: Optional[float] = None
    distance_traveled_m: float = 0.0
    total_charging_time_s: float = 0.0
    num_charging_events: int = 0

class MAPUsageTracker:
    """Tracks MAP usage and charging statistics"""

    def __init__(self, num_maps: int):
        self.num_maps = num_maps
        self.charging_records = []  # List of MAPChargingRecord
        self.movement_records = []  # List of MAPMovementRecord (NEW)
        self.map_bus_assignments = defaultdict(set)  # {map_id: set(bus_ids)}
        self.map_total_energy = defaultdict(float)  # {map_id: total_energy}
        self.map_total_time = defaultdict(float)  # {map_id: total_time}
        self.map_num_events = defaultdict(int)  # {map_id: num_events}
        self.bus_charging_history = defaultdict(list)  # {bus_id: [(time, energy, map_id)]}
        # Movement tracking (NEW)
        self.map_total_distance_m = defaultdict(float)  # {map_id: total_distance}
        self.map_movement_events = defaultdict(int)  # {map_id: num_movements}
        # MAP SOC history for plotting {map_id: [(time_s, soc_wh)]}
        self.map_soc_history = defaultdict(list)

    def record_charge(self, map_id: int, bus_id: str, start_time: float,
                     end_time: float, energy_wh: float, location: str,
                     soc_before: float, soc_after: float):
        """Record a charging event"""

        duration = end_time - start_time
        record = MAPChargingRecord(
            map_id=map_id,
            bus_id=bus_id,
            start_time_s=start_time,
            end_time_s=end_time,
            duration_s=duration,
            energy_delivered_wh=energy_wh,
            location=location,
            soc_before_wh=soc_before,
            soc_after_wh=soc_after
        )

        self.charging_records.append(record)
        self.map_bus_assignments[map_id].add(bus_id)
        self.map_total_energy[map_id] += energy_wh
        self.map_total_time[map_id] += duration
        self.map_num_events[map_id] += 1
        self.bus_charging_history[bus_id].append((start_time, energy_wh, map_id))

    def record_movement(self, map_id: int, start_time: float, end_time: float,
                       from_location: str, to_location: str, distance_m: float,
                       associated_bus_id: Optional[str] = None):
        """Record a MAP movement event"""
        duration = end_time - start_time
        record = MAPMovementRecord(
            map_id=map_id,
            start_time_s=start_time,
            end_time_s=end_time,
            from_location=from_location,
            to_location=to_location,
            distance_m=distance_m,
            duration_s=duration,
            associated_bus_id=associated_bus_id
        )
        self.movement_records.append(record)
        self.map_total_distance_m[map_id] += distance_m
        self.map_movement_events[map_id] += 1

    def get_summary(self) -> Dict:
        """Get summary of MAP usage"""

        total_energy = sum(self.map_total_energy.values())
        total_time = sum(self.map_total_time.values())
        total_distance = sum(self.map_total_distance_m.values())

        map_summaries = {}
        for map_id in range(self.num_maps):
            map_summaries[map_id] = {
                'buses_charged': len(self.map_bus_assignments[map_id]),
                'total_energy_wh': self.map_total_energy[map_id],
                'total_time_s': self.map_total_time[map_id],
                'num_events': self.map_num_events[map_id],
                'avg_energy_per_event': (self.map_total_energy[map_id] /
                                        max(1, self.map_num_events[map_id])),
                'bus_list': sorted(list(self.map_bus_assignments[map_id])),
                'total_distance_m': self.map_total_distance_m[map_id],
                'num_movements': self.map_movement_events[map_id]
            }

        return {
            'total_energy_wh': total_energy,
            'total_time_s': total_time,
            'num_events': sum(self.map_num_events.values()),
            'map_summaries': map_summaries,
            'total_distance_m': total_distance,
            'total_movements': sum(self.map_movement_events.values())
        }

    def print_summary(self, battery_capacity_wh: float):
        """Print MAP usage summary"""

        summary = self.get_summary()

        print("\n" + "="*70)
        print("MAP USAGE & MOVEMENT STATISTICS")
        print("="*70)

        print(f"\nOverall Statistics:")
        print(f"  Total energy delivered: {summary['total_energy_wh']/1e6:,.3f} MWh")
        print(f"  Total charging time: {summary['total_time_s']/3600:,.1f} hours")
        print(f"  Total charging events: {summary['num_events']}")
        print(f"  Total distance traveled: {summary['total_distance_m']/1000:,.1f} km")
        print(f"  Total movement events: {summary['total_movements']}")

        if summary['num_events'] > 0:
            print(f"  Average energy per event: {summary['total_energy_wh']/summary['num_events']/1000:,.0f} kWh")

        print(f"\nPer-MAP Statistics:")
        print("-"*100)
        print(f"{'MAP ID':<8} {'Buses':<12} {'Energy (MWh)':<15} {'Time (hrs)':<12} {'Events':<8} {'Distance (km)':<15} {'Movements':<10}")
        print("-"*100)

        for map_id, map_summary in summary['map_summaries'].items():
            energy_mwh = map_summary['total_energy_wh'] / 1e6
            time_hrs = map_summary['total_time_s'] / 3600
            distance_km = map_summary['total_distance_m'] / 1000

            print(f"{map_id:<8} {map_summary['buses_charged']:<12} "
                  f"{energy_mwh:<15.3f} {time_hrs:<12.1f} {map_summary['num_events']:<8} "
                  f"{distance_km:<15.1f} {map_summary['num_movements']:<10}")

        print(f"\n{'='*70}\n")

    def print_detailed_assignments(self):
        """Print which buses are charged by each MAP"""

        print("\n" + "="*70)
        print("MAP TO BUS ASSIGNMENTS")
        print("="*70)

        for map_id in range(self.num_maps):
            buses = sorted(list(self.map_bus_assignments[map_id]))
            print(f"\nMAP {map_id}:")
            print(f"  Total buses charged: {len(buses)}")

            if buses:
                print(f"  Bus list: {', '.join(buses)}")
            else:
                print(f"  No buses charged")

        print(f"\n{'='*70}\n")

    def print_bus_charging_schedule(self, num_to_print: int = 10):
        """Print charging schedule for buses"""

        print("\n" + "="*70)
        print(f"BUS CHARGING SCHEDULE (First {num_to_print} buses)")
        print("="*70)

        count = 0
        for bus_id, events in sorted(self.bus_charging_history.items()):
            if count >= num_to_print:
                break

            print(f"\n{bus_id}:")
            print(f"  Total charging events: {len(events)}")
            print(f"  Charging by MAP:")

            # Group by MAP
            by_map = defaultdict(list)
            for time_s, energy_wh, map_id in events:
                by_map[map_id].append((time_s, energy_wh))

            for map_id in sorted(by_map.keys()):
                events_for_map = by_map[map_id]
                total_energy = sum(e[1] for e in events_for_map)
                print(f"    MAP {map_id}: {len(events_for_map)} events, {total_energy/1000:,.0f} kWh")

                for time_s, energy_wh in events_for_map[:3]:
                    hour = int(time_s // 3600)
                    minute = int((time_s % 3600) // 60)
                    print(f"      - {hour:02d}:{minute:02d} - {energy_wh/1000:,.0f} kWh")

                if len(events_for_map) > 3:
                    print(f"      ... and {len(events_for_map) - 3} more")

            count += 1

        if len(self.bus_charging_history) > num_to_print:
            print(f"\n... and {len(self.bus_charging_history) - num_to_print} more buses\n")
        else:
            print()

# ========================
# PREEMPTION STRATEGY ANALYZER
# ========================

class PreemptionStrategyAnalyzer:
    """Analyzes bus energy consumption patterns and recommends optimal preemption strategy"""

    def __init__(self, sim, bus_trips_dict: Dict[str, List[str]], battery_capacity_wh: float, num_maps: int = 1):
        self.sim = sim
        self.bus_trips_dict = bus_trips_dict
        self.battery_capacity_wh = battery_capacity_wh
        self.num_maps = num_maps
        self.bus_metrics = {}
        self._analyze_bus_patterns()

    def _analyze_bus_patterns(self):
        """Analyze energy consumption and SOC patterns for all buses"""

        for bus_id, trip_ids in self.bus_trips_dict.items():
            energy_consumptions = []
            soc_drops = []
            max_soc_drop = 0.0
            total_energy = 0.0

            sorted_trips = sorted(
                trip_ids,
                key=lambda t: (
                    self.sim.gtfs.stop_sequence(t)[0]["arrival"]
                    if self.sim.gtfs.stop_sequence(t)
                    else float('inf')
                )
            )

            for trip_id in sorted_trips:
                try:
                    seq = self.sim.gtfs.stop_sequence(trip_id)
                    if not seq or len(seq) < 2:
                        continue

                    trip_energy = 0.0

                    for i in range(1, len(seq)):
                        prev_stop = seq[i - 1]
                        curr_stop = seq[i]

                        try:
                            _, _, dist = self.sim.geod.inv(
                                prev_stop["lon"], prev_stop["lat"],
                                curr_stop["lon"], curr_stop["lat"]
                            )
                            dist = abs(dist)
                        except Exception:
                            dist = 0.0

                        energy = dist * ENERGY_PER_METER_WH
                        trip_energy += energy
                        total_energy += energy
                        energy_consumptions.append(energy)

                    trip_soc_drop = trip_energy / self.battery_capacity_wh
                    soc_drops.append(trip_soc_drop)
                    max_soc_drop = max(max_soc_drop, trip_soc_drop)

                except Exception:
                    continue

            if energy_consumptions:
                avg_energy = np.mean(energy_consumptions)
                std_energy = np.std(energy_consumptions)
                max_energy = np.max(energy_consumptions)
            else:
                avg_energy = std_energy = max_energy = 0.0

            if soc_drops:
                avg_soc_drop = np.mean(soc_drops)
                max_soc_drop_pct = max_soc_drop
            else:
                avg_soc_drop = max_soc_drop_pct = 0.0

            self.bus_metrics[bus_id] = {
                'total_energy_wh': total_energy,
                'avg_segment_energy_wh': avg_energy,
                'std_segment_energy_wh': std_energy,
                'max_segment_energy_wh': max_energy,
                'avg_soc_drop_per_trip': avg_soc_drop,
                'max_soc_drop_per_trip': max_soc_drop_pct,
                'num_trips': len(sorted_trips)
            }

    def recommend_preemption_threshold(self) -> float:
        """Recommend optimal preemption threshold"""

        if self.num_maps <= 0:
            return None

        if not self.bus_metrics:
            return 0.40

        all_max_drops = [m['max_soc_drop_per_trip'] for m in self.bus_metrics.values()]
        all_avg_drops = [m['avg_soc_drop_per_trip'] for m in self.bus_metrics.values()]

        max_drop = np.max(all_max_drops) if all_max_drops else 0.0
        avg_drop = np.mean(all_avg_drops) if all_avg_drops else 0.0
        std_drop = np.std(all_max_drops) if len(all_max_drops) > 1 else 0.0

        num_buses = len(self.bus_trips_dict)

        base_threshold = BUS_CHARGE_THRESHOLD_SOC - (avg_drop + std_drop)
        map_factor = 1.0 - (self.num_maps / num_buses) * 0.2

        if max_drop > 0.5:
            variability_factor = 0.9
        elif max_drop > 0.3:
            variability_factor = 0.95
        else:
            variability_factor = 1.0

        recommended = base_threshold * map_factor * variability_factor
        recommended = np.clip(recommended, 0.25, 0.60)

        return recommended

    def get_analysis_report(self) -> Dict:
        """Get detailed analysis report"""

        if self.num_maps <= 0:
            return {
                'total_buses': len(self.bus_metrics),
                'num_maps': self.num_maps,
                'charging_available': False,
                'recommended_threshold': None,
                'message': 'No MAPs available - charging disabled'
            }

        if not self.bus_metrics:
            return {}

        all_max_drops = [m['max_soc_drop_per_trip'] for m in self.bus_metrics.values()]
        all_avg_drops = [m['avg_soc_drop_per_trip'] for m in self.bus_metrics.values()]

        max_drop = np.max(all_max_drops)
        avg_drop = np.mean(all_avg_drops)
        std_drop = np.std(all_max_drops)

        total_system_energy = np.sum([m['total_energy_wh'] for m in self.bus_metrics.values()])
        avg_bus_energy = np.mean([m['total_energy_wh'] for m in self.bus_metrics.values()])

        critical_buses = {
            bid: m for bid, m in self.bus_metrics.items()
            if m['max_soc_drop_per_trip'] > avg_drop + std_drop
        }

        return {
            'total_buses': len(self.bus_metrics),
            'num_maps': self.num_maps,
            'charging_available': True,
            'critical_buses': len(critical_buses),
            'total_system_energy_wh': total_system_energy,
            'avg_bus_energy_wh': avg_bus_energy,
            'recommended_threshold': self.recommend_preemption_threshold()
        }

    def print_analysis_report(self):
        """Print detailed analysis report"""

        report = self.get_analysis_report()

        if not report or not report.get('charging_available', False):
            print("\n" + "="*70)
            print("PREEMPTION STRATEGY ANALYSIS")
            print("="*70)
            print("\n⚠️  NO MAPs AVAILABLE - CHARGING DISABLED\n")
            return

        print("\n" + "="*70)
        print("PREEMPTION STRATEGY ANALYSIS")
        print("="*70)
        print(f"\nSystem Metrics:")
        print(f"  Total buses: {report['total_buses']}")
        print(f"  Available MAPs: {report['num_maps']}")
        print(f"  Critical buses: {report['critical_buses']}")
        print(f"  Total system energy demand: {report['total_system_energy_wh']/1e6:,.2f} MWh")
        print(f"  Average per bus: {report['avg_bus_energy_wh']/1000:,.0f} kWh")
        print(f"\nRecommended Preemption Threshold: {report['recommended_threshold']*100:.1f}% SOC")
        print(f"\n{'='*70}\n")

# ========================
# CHARGING POLICY
# ========================

class ChargingPolicy:
    """Encapsulates charging logic and thresholds."""
    def __init__(self, start_pct: float = 0.7, stop_pct: float = 0.8, rate_wh_per_s: float = 97.22):
        self.start_pct = float(start_pct)
        self.stop_pct = float(stop_pct)
        self.rate_wh_per_s = float(rate_wh_per_s)

    def wants_charge(self, soc_wh: float, capacity_wh: float) -> bool:
        """Allow partial recharge: any bus below stop_pct (default 80%) can charge."""
        if capacity_wh is None or capacity_wh <= 0:
            return False
        return soc_wh < (self.stop_pct * capacity_wh)

    def max_target_wh(self, capacity_wh: float) -> float:
        return self.stop_pct * capacity_wh

    def charge_amount_for_duration(self, soc_wh: float, capacity_wh: float, duration_s: float) -> float:
        if duration_s <= 0:
            return 0.0
        target = self.max_target_wh(capacity_wh)
        remaining = max(0.0, target - soc_wh)
        possible = self.rate_wh_per_s * duration_s
        return min(remaining, possible)

# ========================
# STOP CHARGING MANAGER WITH MAP TRACKING
# ========================

@dataclass
class ChargingSession:
    """Represents an active charging session"""
    bus_id: str
    stop_id: str
    start_time: float
    soc_at_start: float
    capacity: float
    target_soc: float
    process_handle: any
    interrupted: bool = False
    map_id: int = -1

class PreemptiveStopChargingManager:
    """Manage exclusive charging slots with MAP tracking"""

    def __init__(self, env: simpy.Environment, capacity_per_stop: int = 1,
                 policy: ChargingPolicy = None, on_charge=None,
                 preemption_threshold: Optional[float] = 0.40,
                 num_maps: int = 1,
                 map_tracker: Optional[MAPUsageTracker] = None,
                 map_movement_scheduler: Optional['MAPMovementScheduler'] = None):
        self.env = env
        self.capacity = max(1, int(capacity_per_stop))
        self.policy = policy or ChargingPolicy()
        self.on_charge = on_charge
        self.preemption_threshold = preemption_threshold
        self.num_maps = num_maps
        self.map_tracker = map_tracker
        self.map_movement_scheduler = map_movement_scheduler

        self.charging_enabled = (num_maps > 0)

        self._active_counts = defaultdict(int)
        self._queues = defaultdict(list)
        self._high_priority_queues = defaultdict(list)
        self._logs = defaultdict(list)
        self._charging_events = []
        self._active_sessions = {}
        self._preemption_events = []
        self._current_map_id = 0
        self._global_active_count = 0  # Track total buses charging across all stops
        self._buses_currently_charging = set()  # Track which buses are actively charging
        self._map_locations = {}  # Track MAP locations {map_id: stop_id}
        self._line_charge_counts = defaultdict(int)  # Per-line charging counts for fairness

    def _can_start(self, stop_id: str) -> bool:
        if not self.charging_enabled:
            return False
        # Check global limit: only allow as many buses charging as there are MAPs
        if self._global_active_count >= self.num_maps:
            return False
        # Check that at least one MAP is available (has energy and not self-charging)
        if self.map_movement_scheduler:
            any_available = any(
                self.map_movement_scheduler.is_map_available(mid)
                for mid in range(self.num_maps)
            )
            if not any_available:
                return False
        return True

    def _get_next_map_id(self, bus_id=None, stop_id=None) -> int:
        """Greedy MAP selection based on distance, energy, and line fairness.

        Considers:
        - MAP availability (not self-charging, has energy)
        - Distance from MAP to bus stop (closer is better)
        - MAP remaining energy (more energy is better)
        - Line fairness (underserved lines get priority)
        Falls back to round-robin if no MAP scores positively.
        """
        if self.map_movement_scheduler:
            # Extract line_id from bus_id (e.g. 'line1_bus_0' → '1')
            line_id = "unknown"
            if bus_id:
                try:
                    line_id = bus_id.split('_')[0].replace('line', '')
                except Exception:
                    pass

            best_map = None
            best_score = -1.0

            for mid in range(self.num_maps):
                if not self.map_movement_scheduler.is_map_available(mid):
                    continue

                avail = self.map_movement_scheduler.available_energy_wh(mid)
                if avail <= 0:
                    continue

                # Distance score (prefer closer MAPs)
                dist_score = 1.0
                if stop_id:
                    map_loc = self.map_movement_scheduler.get_map_location(mid)
                    if map_loc:
                        dist_m, _ = self.map_movement_scheduler.calculate_travel_time(
                            map_loc, stop_id)
                        dist_score = 1.0 / (1.0 + dist_m / 1000.0)

                # Energy score (prefer MAPs with more energy)
                cap = self.map_movement_scheduler.map_states[mid].battery_capacity_wh
                energy_score = avail / cap if cap > 0 else 0.0

                # Line fairness score (boost underserved lines)
                line_count = self._line_charge_counts.get(line_id, 0)
                fairness_score = 1.0 / (1.0 + line_count)

                score = dist_score * energy_score * fairness_score
                if score > best_score:
                    best_score = score
                    best_map = mid

            if best_map is not None:
                return best_map

            # Fallback: round-robin if no MAP scored positively
            for _ in range(self.num_maps):
                map_id = self._current_map_id % self.num_maps
                self._current_map_id += 1
                if self.map_movement_scheduler.is_map_available(map_id):
                    return map_id
            map_id = self._current_map_id % self.num_maps
            self._current_map_id += 1
            return map_id
        else:
            map_id = self._current_map_id % self.num_maps
            self._current_map_id += 1
            return map_id

    def _start_session(self, stop_id: str, req: dict):
        """Start a charging session with MAP tracking and location verification"""
        if not self.charging_enabled:
            return (False, 0.0, 0.0)

        bus_id = req["bus_id"]
        soc = req["soc"]
        cap = req["capacity"]
        desired = req["desired"]

        # Check if bus is already being charged
        if bus_id in self._buses_currently_charging:
            return (False, 0.0, 0.0)

        # Use preferred MAP if provided and available, else greedy selection
        preferred = req.get("preferred_map_id")
        if (preferred is not None and self.map_movement_scheduler
                and self.map_movement_scheduler.is_map_available(preferred)):
            map_id = preferred
        else:
            map_id = self._get_next_map_id(bus_id=bus_id, stop_id=stop_id)

        amount = self.policy.charge_amount_for_duration(soc, cap, desired)
        if amount <= 0 or desired <= 0:
            return (False, 0.0, 0.0)

        # Cap by MAP available energy
        if self.map_movement_scheduler:
            avail = self.map_movement_scheduler.available_energy_wh(map_id)
            if avail <= 0:
                return (False, 0.0, 0.0)
            amount = min(amount, avail)

        duration = min(amount / self.policy.rate_wh_per_s, desired)

        # Set MAP location
        self._map_locations[map_id] = stop_id

        # Mark bus as currently charging
        self._buses_currently_charging.add(bus_id)
        self._global_active_count += 1
        self._active_counts[stop_id] += 1
        start_time = float(self.env.now)

        entry = {
            "bus_id": bus_id,
            "start_time": start_time,
            "duration_s": duration,
            "amount_wh": amount,
            "map_id": map_id
        }
        self._logs[stop_id].append(entry)

        charging_event = {
            "bus_id": bus_id,
            "stop_id": stop_id,
            "start_time_s": start_time,
            "duration_s": duration,
            "energy_wh": amount,
            "preempted": False,
            "map_id": map_id
        }
        self._charging_events.append(charging_event)

        # Track per-line charging counts for fairness
        try:
            _line_id = bus_id.split('_')[0].replace('line', '')
        except Exception:
            _line_id = "unknown"
        self._line_charge_counts[_line_id] += 1

        if callable(self.on_charge):
            try:
                self.on_charge(bus_id, amount, duration)
            except Exception:
                pass

        session = ChargingSession(
            bus_id=bus_id,
            stop_id=stop_id,
            start_time=start_time,
            soc_at_start=soc,
            capacity=cap,
            target_soc=min(cap, soc + amount),
            process_handle=None,
            map_id=map_id
        )

        # Eagerly deduct energy from MAP so concurrent checks see correct SOC
        if self.map_movement_scheduler:
            self.map_movement_scheduler.update_map_soc(map_id, amount)

        def _release(sid, dur, sess, delivered_amount):
            try:
                yield self.env.timeout(dur)
                if not sess.interrupted:
                    if self.map_tracker:
                        self.map_tracker.record_charge(
                            map_id=sess.map_id,
                            bus_id=sess.bus_id,
                            start_time=sess.start_time,
                            end_time=float(self.env.now),
                            energy_wh=delivered_amount,
                            location=f"stop_{sid}",
                            soc_before=sess.soc_at_start,
                            soc_after=min(cap, sess.soc_at_start + delivered_amount)
                        )

                    # If MAP SOC is at floor, trigger self-charging
                    if self.map_movement_scheduler:
                        if self.map_movement_scheduler.map_needs_self_charge(sess.map_id):
                            self.env.process(
                                self.map_movement_scheduler.self_charge_process(sess.map_id))

                    # Release resources
                    self._global_active_count = max(0, self._global_active_count - 1)
                    self._active_counts[sid] = max(0, self._active_counts.get(sid, 1) - 1)
                    self._buses_currently_charging.discard(sess.bus_id)
                    self._assign_next_for_stop(sid)
            except Exception:
                pass

        process = self.env.process(_release(stop_id, duration, session, amount))
        session.process_handle = process
        self._active_sessions[(stop_id, bus_id)] = session

        return (True, duration, amount)

    def _assign_next_for_stop(self, stop_id: str):
        """Assign next queued request"""

        if not self.charging_enabled:
            return

        hp_queue = self._high_priority_queues.get(stop_id, [])
        if hp_queue:
            now = float(self.env.now)
            alive = []
            for r in hp_queue:
                elapsed = now - r["requested_time"]
                remaining = r["desired"] - elapsed
                if remaining > 0:
                    newr = r.copy()
                    newr["desired"] = remaining
                    alive.append(newr)

            self._high_priority_queues[stop_id] = alive
            if alive and self._can_start(stop_id):
                req = alive.pop(0)
                self._high_priority_queues[stop_id] = alive
                self._start_session(stop_id, req)
                return

        q = self._queues.get(stop_id, [])
        if not q:
            return

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

        alive.sort(key=lambda x: (x["soc"], x["requested_time"]))

        if self._can_start(stop_id):
            req = alive.pop(0)
            self._queues[stop_id] = alive
            self._start_session(stop_id, req)

    def request_stop_charging(self, stop_id: str, bus_id: str, soc_wh: float,
                             capacity_wh: float, desired_duration_s: float,
                             preferred_map_id: int = None):
        """Request charging"""

        if not self.charging_enabled:
            return (False, 0.0, 0.0)

        if not stop_id:
            return (False, 0.0, 0.0)

        if not self.policy.wants_charge(soc_wh, capacity_wh):
            return (False, 0.0, 0.0)

        req = {
            "bus_id": bus_id,
            "soc": soc_wh,
            "capacity": capacity_wh,
            "desired": desired_duration_s,
            "requested_time": float(self.env.now),
            "preferred_map_id": preferred_map_id
        }

        soc_ratio = soc_wh / capacity_wh if capacity_wh > 0 else 0
        is_high_priority = self.preemption_threshold is not None and soc_ratio < self.preemption_threshold

        if is_high_priority:
            for key, session in list(self._active_sessions.items()):
                if key[0] == stop_id and not session.interrupted:
                    current_soc_ratio = session.soc_at_start / session.capacity if session.capacity > 0 else 0

                    if self.preemption_threshold is not None and current_soc_ratio >= self.preemption_threshold:
                        self._preempt_session(stop_id, session, bus_id, soc_wh)
                        if self._can_start(stop_id):
                            return self._start_session(stop_id, req)

        if self._can_start(stop_id):
            return self._start_session(stop_id, req)

        if is_high_priority:
            self._high_priority_queues[stop_id].append(req)
        else:
            self._queues[stop_id].append(req)

        return (False, 0.0, 0.0)

    def _preempt_session(self, stop_id: str, session: ChargingSession,
                        new_bus_id: str, new_bus_soc: float):
        """Preempt a charging session and refund undelivered energy to MAP"""

        session.interrupted = True

        # Calculate how much energy was actually delivered before preemption
        elapsed = float(self.env.now) - session.start_time
        # Find the original planned amount from the charging event
        planned_amount = 0.0
        planned_duration = 0.0
        for event in reversed(self._charging_events):
            if event["bus_id"] == session.bus_id and event["stop_id"] == stop_id:
                planned_amount = event["energy_wh"]
                planned_duration = event["duration_s"]
                event["preempted"] = True
                break

        if planned_duration > 0:
            fraction_delivered = min(1.0, elapsed / planned_duration)
        else:
            fraction_delivered = 1.0
        actual_delivered = planned_amount * fraction_delivered
        undelivered = planned_amount - actual_delivered

        # Refund undelivered energy to the MAP
        if undelivered > 0 and self.map_movement_scheduler and session.map_id >= 0:
            state = self.map_movement_scheduler.map_states.get(session.map_id)
            if state:
                state.current_soc_wh = min(state.battery_capacity_wh,
                                           state.current_soc_wh + undelivered)
                if self.map_tracker:
                    self.map_tracker.map_soc_history[session.map_id].append(
                        (float(self.env.now), state.current_soc_wh))

        preemption_event = {
            "stop_id": stop_id,
            "interrupted_bus": session.bus_id,
            "new_bus": new_bus_id,
            "time_s": float(self.env.now),
            "interrupted_bus_soc": session.soc_at_start,
            "new_bus_soc": new_bus_soc,
            "soc_difference": (new_bus_soc - session.soc_at_start) / session.capacity if session.capacity > 0 else 0,
            "map_id": session.map_id
        }
        self._preemption_events.append(preemption_event)

        if session.process_handle:
            try:
                session.process_handle.interrupt()
            except Exception:
                pass

        self._global_active_count = max(0, self._global_active_count - 1)
        self._active_counts[stop_id] = max(0, self._active_counts.get(stop_id, 1) - 1)
        self._buses_currently_charging.discard(session.bus_id)

        for key in list(self._active_sessions.keys()):
            if key == (stop_id, session.bus_id):
                del self._active_sessions[key]

    def get_stop_logs(self):
        return {sid: list(logs) for sid, logs in self._logs.items()}

    def get_charging_events(self):
        return list(self._charging_events)

    def get_preemption_events(self):
        return list(self._preemption_events)

    def get_line_charge_counts(self):
        """Return per-line charging counts (for greedy MAP selection fairness)."""
        return dict(self._line_charge_counts)

# ========================
# MAP MOVEMENT SCHEDULER (NEW)
# ========================

class MAPMovementScheduler:
    """Manages MAP movement, routing, location tracking, and MAP energy"""

    def __init__(self, env: simpy.Environment, num_maps: int,
                 map_battery_capacity_wh: float = MAP_BATTERY_CAPACITY_WH,
                 map_speed_ms: float = 27.78,
                 map_tracker: Optional[MAPUsageTracker] = None):
        self.env = env
        self.num_maps = num_maps
        self.map_speed_ms = map_speed_ms
        self.map_tracker = map_tracker

        # MAP states indexed by map_id
        self.map_states = {}
        for map_id in range(num_maps):
            self.map_states[map_id] = MAPState(
                map_id=map_id,
                current_location="depot",
                current_time_s=0.0,
                current_soc_wh=map_battery_capacity_wh,
                battery_capacity_wh=map_battery_capacity_wh,
            )
            # Record initial SOC
            if map_tracker:
                map_tracker.map_soc_history[map_id].append((0.0, map_battery_capacity_wh))

        # Per-MAP self-charging flag
        self._is_self_charging = {mid: False for mid in range(num_maps)}

        # Request queue for MAP assignments
        self.charging_requests = defaultdict(list)  # {map_id: [requests]}

        # Location cache for stop coordinates {stop_id: (lat, lon)}
        self.stop_locations = {}
        self.geod = Geod(ellps="WGS84")

    def set_stop_locations(self, stops_df):
        """Set stop coordinate information for distance calculations"""
        try:
            for _, row in stops_df.iterrows():
                stop_id = row.get('stop_id')
                lat = row.get('stop_lat')
                lon = row.get('stop_lon')
                if stop_id and lat is not None and lon is not None:
                    self.stop_locations[stop_id] = (float(lat), float(lon))
        except Exception as e:
            print(f"Warning: Could not set stop locations: {e}")

    def calculate_travel_time(self, from_location: str, to_location: str) -> Tuple[float, float]:
        """Calculate travel time and distance between two locations (in meters and seconds)

        Returns: (distance_m, time_s)
        """
        if from_location not in self.stop_locations or to_location not in self.stop_locations:
            # If locations unknown, return default (0 distance, 0 time)
            return (0.0, 0.0)

        lat1, lon1 = self.stop_locations[from_location]
        lat2, lon2 = self.stop_locations[to_location]

        try:
            _, _, distance_m = self.geod.inv(lon1, lat1, lon2, lat2)
            distance_m = abs(distance_m)
            time_s = distance_m / self.map_speed_ms if self.map_speed_ms > 0 else 0.0
            return (distance_m, time_s)
        except Exception:
            return (0.0, 0.0)

    def assign_map_to_bus(self, map_id: int, bus_id: str, current_location: str,
                         target_location: str) -> Optional[Tuple[float, float]]:
        """Assign a MAP to move to a bus location

        Returns: (travel_time_s, distance_m) or None if invalid
        """
        if map_id < 0 or map_id >= self.num_maps:
            return None

        map_state = self.map_states.get(map_id)
        if map_state is None:
            return None

        distance_m, travel_time_s = self.calculate_travel_time(map_state.current_location, target_location)

        if travel_time_s > 0:
            # Schedule movement
            def move_process():
                yield self.env.timeout(travel_time_s)

                # Update MAP state
                map_state.current_location = target_location
                map_state.assigned_bus_id = bus_id
                map_state.arrival_time_at_target_s = float(self.env.now)
                map_state.distance_traveled_m += distance_m

                # Record movement
                if self.map_tracker:
                    self.map_tracker.record_movement(
                        map_id=map_id,
                        start_time=float(self.env.now) - travel_time_s,
                        end_time=float(self.env.now),
                        from_location=map_state.current_location,
                        to_location=target_location,
                        distance_m=distance_m,
                        associated_bus_id=bus_id
                    )

            self.env.process(move_process())
            return (travel_time_s, distance_m)
        else:
            # Already at location
            map_state.current_location = target_location
            map_state.assigned_bus_id = bus_id
            map_state.arrival_time_at_target_s = float(self.env.now)
            return (0.0, distance_m)

    def get_map_location(self, map_id: int) -> str:
        """Get current location of a MAP"""
        if map_id in self.map_states:
            return self.map_states[map_id].current_location
        return None

    def get_map_soc(self, map_id: int) -> float:
        """Get current SOC of a MAP"""
        if map_id in self.map_states:
            return self.map_states[map_id].current_soc_wh
        return 0.0

    def update_map_soc(self, map_id: int, energy_consumed_wh: float):
        """Update MAP SOC after charging a bus (energy is deducted from MAP battery).
        SOC is clamped so it never drops below MAP_MIN_SOC floor."""
        if map_id in self.map_states:
            state = self.map_states[map_id]
            floor = MAP_MIN_SOC * state.battery_capacity_wh
            state.current_soc_wh = max(floor, state.current_soc_wh - energy_consumed_wh)
            state.num_charging_events += 1
            if self.map_tracker:
                self.map_tracker.map_soc_history[map_id].append(
                    (float(self.env.now), state.current_soc_wh))

    def available_energy_wh(self, map_id: int) -> float:
        """Return energy available for charging (above MAP_MIN_SOC floor).
        Returns 0 if remaining is negligible (< 1 Wh) to avoid rounding issues."""
        if map_id not in self.map_states:
            return 0.0
        state = self.map_states[map_id]
        floor = MAP_MIN_SOC * state.battery_capacity_wh
        avail = state.current_soc_wh - floor
        return avail if avail >= 1.0 else 0.0

    def map_needs_self_charge(self, map_id: int) -> bool:
        """Check if MAP needs to self-charge (available energy is 0)"""
        return self.available_energy_wh(map_id) <= 0.0

    def is_map_available(self, map_id: int) -> bool:
        """Check if MAP is available to charge a bus (not self-charging and has energy)"""
        if self._is_self_charging.get(map_id, False):
            return False
        return self.available_energy_wh(map_id) > 0.0

    def self_charge_process(self, map_id: int):
        """SimPy process: self-charge a MAP back to full capacity at MAP_SELF_CHARGE_RATE_WH_S.
        The MAP cannot charge buses while this process is running."""
        if map_id not in self.map_states:
            return
        state = self.map_states[map_id]
        if state.current_soc_wh >= state.battery_capacity_wh:
            return  # already full

        self._is_self_charging[map_id] = True
        deficit = state.battery_capacity_wh - state.current_soc_wh
        duration = deficit / MAP_SELF_CHARGE_RATE_WH_S

        print(f"  [MAP {map_id}] Self-charging started at t={self.env.now:.0f}s "
              f"(SOC {state.current_soc_wh/1000:.1f} kWh → {state.battery_capacity_wh/1000:.1f} kWh, "
              f"duration {duration:.0f}s)")

        yield self.env.timeout(duration)

        state.current_soc_wh = state.battery_capacity_wh
        self._is_self_charging[map_id] = False

        if self.map_tracker:
            self.map_tracker.map_soc_history[map_id].append(
                (float(self.env.now), state.current_soc_wh))

        print(f"  [MAP {map_id}] Self-charging complete at t={self.env.now:.0f}s "
              f"(SOC {state.current_soc_wh/1000:.1f} kWh)")

    def map_is_at_location(self, map_id: int, location: str) -> bool:
        """Check if a MAP is at a specific location"""
        if map_id in self.map_states:
            return self.map_states[map_id].current_location == location
        return False

    def get_summary(self) -> Dict:
        """Get summary of MAP states"""
        summary = {}
        for map_id, state in self.map_states.items():
            summary[map_id] = {
                'current_location': state.current_location,
                'current_soc_wh': state.current_soc_wh,
                'distance_traveled_m': state.distance_traveled_m,
                'num_charging_events': state.num_charging_events,
                'assigned_bus': state.assigned_bus_id
            }
        return summary

# ========================
# DATA CLASSES
# ========================

@dataclass
class LayoverRecord:
    """Record of a bus layover at terminal"""
    bus_id: str
    line_id: str
    current_trip_id: str
    next_trip_id: str
    arrival_at_terminal_s: float
    departure_from_terminal_s: float
    layover_duration_s: float
    location: str
    stop_id: str
    soc_at_arrival_wh: float
    soc_at_departure_wh: float
    was_charged: bool = False
    energy_charged_wh: float = 0.0
    was_preempted: bool = False

# ========================
# STAGE 2: DES SIMULATION
# ========================

class Stage2DESTerminalChargingPreemptive:
    """DES simulation with MAP usage tracking"""

    def __init__(self, sim, bus_trips_dict: Dict[str, List[str]], bus_lines,
                 trip_change_stops: set,
                 initial_battery_capacity_wh: float = 250000,
                 num_maps: int = 1,
                 optimize_threshold: bool = True,
                 preemption_threshold: Optional[float] = None,
                 line_battery_capacities_wh: Optional[Dict[str, float]] = None,
                 map_battery_capacity_wh: Optional[float] = None):

        self.sim = sim
        self.bus_trips_dict = bus_trips_dict
        self.bus_lines = bus_lines
        self.trip_change_stops = trip_change_stops
        self.battery_capacity_wh = initial_battery_capacity_wh
        self.num_maps = num_maps
        # Per-line battery capacities (Wh); falls back to initial_battery_capacity_wh
        self.line_battery_capacities_wh = line_battery_capacities_wh or {}
        # MAP battery capacity; falls back to module constant
        self.map_battery_capacity_wh = map_battery_capacity_wh if map_battery_capacity_wh is not None else MAP_BATTERY_CAPACITY_WH

        # Initialize MAP tracker
        self.map_tracker = MAPUsageTracker(num_maps)

        if num_maps <= 0:
            self.preemption_threshold = None
            self.threshold_analysis = None
            print(f"\n⚠️  WARNING: num_maps = {num_maps}")
            print(f"  Charging is DISABLED for this simulation\n")
        elif preemption_threshold is not None:
            self.preemption_threshold = preemption_threshold
            self.threshold_analysis = None
            print(f"\nUsing provided preemption threshold: {self.preemption_threshold*100:.1f}%")
        elif optimize_threshold:
            print(f"\nAnalyzing system to determine optimal preemption threshold...")
            analyzer = PreemptionStrategyAnalyzer(sim, bus_trips_dict, initial_battery_capacity_wh, num_maps)
            self.preemption_threshold = analyzer.recommend_preemption_threshold()
            self.threshold_analysis = analyzer.get_analysis_report()

            if self.preemption_threshold is not None:
                print(f"Recommended preemption threshold: {self.preemption_threshold*100:.1f}% SOC")

            analyzer.print_analysis_report()
        else:
            self.preemption_threshold = 0.40 if num_maps > 0 else None
            self.threshold_analysis = None

        self.env = simpy.Environment()

        # Initialize MAP Movement Scheduler (NEW)
        self.map_movement_scheduler = MAPMovementScheduler(
            self.env,
            num_maps=num_maps,
            map_battery_capacity_wh=self.map_battery_capacity_wh,
            map_speed_ms=CHARGER_SPEED_MS,
            map_tracker=self.map_tracker
        )
        # Set stop locations for movement calculations
        self.map_movement_scheduler.set_stop_locations(sim.stops)

        self.buses = {}
        for bus_id, trip_ids in bus_trips_dict.items():
            try:
                line_id = bus_id.split('_')[0].replace('line', '')
            except:
                line_id = "unknown"

            bus_cap = self._get_bus_capacity(line_id)
            self.buses[bus_id] = {
                'line_id': line_id,
                'trip_ids': trip_ids,
                'soc_wh': bus_cap,
                'min_soc_wh': bus_cap,
                'total_energy_consumed_wh': 0.0
            }

        self.layovers = []
        self.bus_soc_history = defaultdict(list)

        policy = ChargingPolicy(start_pct=0.7, stop_pct=0.8, rate_wh_per_s=97.22)
        self.stop_charging_manager = PreemptiveStopChargingManager(
            self.env,
            capacity_per_stop=1,
            policy=policy,
            on_charge=self._on_charge,
            preemption_threshold=self.preemption_threshold,
            num_maps=num_maps,
            map_tracker=self.map_tracker,
            map_movement_scheduler=self.map_movement_scheduler
        )

        self.bus_soc = defaultdict(float)
        self.bus_energy_charged = defaultdict(float)

    def _get_bus_capacity(self, line_id: str) -> float:
        """Return battery capacity (Wh) for a given line_id.
        Uses per-line map if available, otherwise falls back to default."""
        return self.line_battery_capacities_wh.get(line_id, self.battery_capacity_wh)

    def _get_capacity_for_bus(self, bus_id: str) -> float:
        """Return battery capacity (Wh) for a given bus_id."""
        try:
            line_id = bus_id.split('_')[0].replace('line', '')
        except Exception:
            line_id = "unknown"
        return self._get_bus_capacity(line_id)

    def _on_charge(self, bus_id: str, amount_wh: float, duration_s: float):
        """Callback when charging starts"""
        try:
            cap = self._get_capacity_for_bus(bus_id)
            prev = self.bus_soc.get(bus_id, float(cap))
            new = min(float(cap), prev + amount_wh)
            self.bus_soc[bus_id] = new
            self.bus_energy_charged[bus_id] += amount_wh
            self.bus_soc_history[bus_id].append((float(self.env.now), new))
        except Exception:
            pass

    def _select_best_map(self, bus_id: str, line_id: str, stop_id: str,
                         soc_wh: float, capacity_wh: float) -> Optional[int]:
        """Greedy heuristic: select the best available MAP for a bus.

        Scoring considers:
        - MAP availability (not self-charging, has energy above 10%)
        - Distance from MAP to bus stop (closer is better)
        - MAP remaining energy (more energy is better)
        - Line fairness (underserved lines get priority)
        - Bus SOC urgency (lower SOC = higher priority)

        Returns map_id or None if no MAP is available.
        """
        if self.num_maps <= 0:
            return None

        best_map = None
        best_score = -1.0

        # Get line charge counts from the charging manager for fairness
        line_counts = self.stop_charging_manager.get_line_charge_counts()

        for map_id in range(self.num_maps):
            if not self.map_movement_scheduler.is_map_available(map_id):
                continue

            avail = self.map_movement_scheduler.available_energy_wh(map_id)
            if avail <= 0:
                continue

            # Distance score (prefer closer MAPs; 1.0 when at same location)
            map_loc = self.map_movement_scheduler.get_map_location(map_id)
            dist_m, _ = self.map_movement_scheduler.calculate_travel_time(
                map_loc or "depot", stop_id)
            distance_score = 1.0 / (1.0 + dist_m / 1000.0)

            # Energy score (prefer MAPs with more available energy)
            cap_map = self.map_movement_scheduler.map_states[map_id].battery_capacity_wh
            energy_score = avail / cap_map if cap_map > 0 else 0.0

            # Line fairness score (boost lines that have been charged less)
            line_count = line_counts.get(line_id, 0)
            fairness_score = 1.0 / (1.0 + line_count)

            # Bus urgency (lower SOC → higher urgency).
            # Floor at 0.01 to prevent zero score for fully charged buses.
            soc_ratio = soc_wh / capacity_wh if capacity_wh > 0 else 1.0
            urgency = max(0.01, 1.0 - soc_ratio)

            score = distance_score * energy_score * fairness_score * urgency
            if score > best_score:
                best_score = score
                best_map = map_id

        return best_map

    def run_simulation(self, duration_s: float = 86400) -> Dict:
        """Run DES simulation"""

        print("\n" + "="*70)
        print("STAGE-2: DES TERMINAL CHARGING WITH MAP USAGE TRACKING")
        print("="*70)
        print(f"Battery capacity (default): {self.battery_capacity_wh/1000:,.1f} kWh")
        if self.line_battery_capacities_wh:
            print(f"Per-line battery capacities:")
            for lid, cap in sorted(self.line_battery_capacities_wh.items()):
                print(f"  Line {lid}: {cap/1000:,.1f} kWh")
        print(f"Trip-change stops: {len(self.trip_change_stops)}")
        print(f"Available MAPs: {self.num_maps}")
        print(f"MAP battery: {self.map_battery_capacity_wh/1000:.0f} kWh | Min SOC: {MAP_MIN_SOC*100:.0f}% | Self-charge: {MAP_SELF_CHARGE_RATE_WH_S} Wh/s")

        if self.preemption_threshold is not None:
            print(f"Preemption threshold (OPTIMIZED): {self.preemption_threshold*100:.1f}% SOC")
        else:
            print(f"Preemption threshold: DISABLED (no MAPs)")

        print(f"Simulation duration: {duration_s/3600:,.1f} hours\n")

        for bus_id, trip_ids in self.bus_trips_dict.items():
            self.env.process(self._simulate_bus(bus_id, trip_ids))

        print(f"Running simulation until t={duration_s}s...")
        self.env.run(until=duration_s)
        print(f"Simulation complete!\n")

        stats = self._collect_statistics()

        return stats

    def _simulate_bus(self, bus_id: str, trip_ids: List[str]):
        """Simulate a single bus"""

        try:
            line_id = bus_id.split('_')[0].replace('line', '')
        except:
            line_id = "unknown"

        capacity = self._get_bus_capacity(line_id)
        self.bus_soc[bus_id] = capacity

        current_time = 0.0
        prev_end_node = None
        prev_end_time = None
        prev_end_stop_id = None

        self.bus_soc_history[bus_id].append((0.0, capacity))

        for trip_idx, trip_id in enumerate(trip_ids):
            try:
                seq = self.sim.gtfs.stop_sequence(trip_id)
            except Exception:
                continue

            if not seq:
                continue

            trip_start = seq[0]["arrival"]
            idle_dt = max(0.0, trip_start - current_time)

            if idle_dt > 0:
                yield self.env.timeout(idle_dt)
                current_time = trip_start

                try:
                    start_node = self.sim.stop_node.get(seq[0].get("stop_id"))
                except Exception:
                    start_node = None

                if (prev_end_node is not None and start_node is not None and
                    prev_end_node == start_node and prev_end_time is not None):

                    duration = trip_start - prev_end_time
                    start_stop_id = seq[0].get("stop_id")

                    if duration > 0 and start_stop_id in self.trip_change_stops:
                        soc_now = self.bus_soc.get(bus_id, capacity)

                        next_trip_id = trip_ids[trip_idx + 1] if trip_idx + 1 < len(trip_ids) else None
                        layover = LayoverRecord(
                            bus_id=bus_id,
                            line_id=line_id,
                            current_trip_id=trip_id,
                            next_trip_id=next_trip_id,
                            arrival_at_terminal_s=prev_end_time,
                            departure_from_terminal_s=trip_start,
                            layover_duration_s=duration,
                            location=f"stop_{start_stop_id}",
                            stop_id=start_stop_id,
                            soc_at_arrival_wh=soc_now,
                            soc_at_departure_wh=soc_now
                        )

                        # Greedy MAP assignment: charge any bus below BUS_CHARGE_CUTOFF_SOC
                        if self.stop_charging_manager.charging_enabled and soc_now < BUS_CHARGE_CUTOFF_SOC * capacity:
                            # Greedy heuristic selects best MAP (by distance, energy, line fairness, urgency)
                            map_id = self._select_best_map(bus_id, line_id, start_stop_id, soc_now, capacity)

                            if map_id is not None:
                                # Move MAP to this location
                                travel_time_distance = self.map_movement_scheduler.assign_map_to_bus(
                                    map_id=map_id,
                                    bus_id=bus_id,
                                    current_location=self.map_movement_scheduler.get_map_location(map_id),
                                    target_location=start_stop_id
                                )

                                if travel_time_distance:
                                    travel_time_s, distance_m = travel_time_distance
                                    if travel_time_s > 0:
                                        yield self.env.timeout(travel_time_s)

                                    remaining_duration = max(0, duration - travel_time_s)
                                else:
                                    remaining_duration = duration

                                if remaining_duration > 0:
                                    started, actual_dur, amount = self.stop_charging_manager.request_stop_charging(
                                        stop_id=start_stop_id,
                                        bus_id=bus_id,
                                        soc_wh=soc_now,
                                        capacity_wh=capacity,
                                        desired_duration_s=remaining_duration,
                                        preferred_map_id=map_id
                                    )

                                    if started and actual_dur > 0:
                                        yield self.env.timeout(actual_dur)

                        # Wait for the full layover duration
                        yield self.env.timeout(duration)

                        recent_events = [e for e in self.stop_charging_manager.get_charging_events()
                                       if e['bus_id'] == bus_id and
                                       prev_end_time <= e['start_time_s'] <= trip_start]

                        if recent_events:
                            total_charged = sum(e['energy_wh'] for e in recent_events)
                            layover.was_charged = True
                            layover.energy_charged_wh = total_charged

                        preemptions = [p for p in self.stop_charging_manager.get_preemption_events()
                                     if p['interrupted_bus'] == bus_id and p['stop_id'] == start_stop_id]
                        if preemptions:
                            layover.was_preempted = True

                        layover.soc_at_departure_wh = self.bus_soc.get(bus_id, capacity)
                        self.layovers.append(layover)

                        self.bus_soc_history[bus_id].append((self.env.now, self.bus_soc.get(bus_id, capacity)))

            for i in range(1, len(seq)):
                prev_stop = seq[i - 1]
                curr_stop = seq[i]

                dt = curr_stop["arrival"] - prev_stop["arrival"]

                try:
                    _, _, dist = self.sim.geod.inv(
                        prev_stop["lon"], prev_stop["lat"],
                        curr_stop["lon"], curr_stop["lat"]
                    )
                    dist = abs(dist)
                except Exception:
                    dist = 0.0

                yield self.env.timeout(max(0.0, dt))
                current_time = curr_stop["arrival"]

                energy = dist * ENERGY_PER_METER_WH
                prev_soc = self.bus_soc.get(bus_id, capacity)
                new_soc = max(0.0, prev_soc - energy)
                self.bus_soc[bus_id] = new_soc

                self.buses[bus_id]['total_energy_consumed_wh'] += energy

                if new_soc < self.buses[bus_id]['min_soc_wh']:
                    self.buses[bus_id]['min_soc_wh'] = new_soc

                self.bus_soc_history[bus_id].append((self.env.now, self.bus_soc.get(bus_id, capacity)))

            prev_end_node = self.sim.stop_node.get(seq[-1].get("stop_id"))
            prev_end_stop_id = seq[-1].get("stop_id")
            prev_end_time = seq[-1]["arrival"]

    def _collect_statistics(self) -> Dict:
        """Collect statistics"""

        bus_stats = {}
        min_soc_overall = float('inf')

        for bus_id, info in self.buses.items():
            bus_cap = self._get_bus_capacity(info['line_id'])
            soc_ratio = info['soc_wh'] / bus_cap
            min_soc_ratio = info['min_soc_wh'] / bus_cap

            bus_stats[bus_id] = {
                'line_id': info['line_id'],
                'final_soc_wh': info['soc_wh'],
                'final_soc_ratio': soc_ratio,
                'min_soc_wh': info['min_soc_wh'],
                'min_soc_ratio': min_soc_ratio,
                'total_energy_consumed_wh': info['total_energy_consumed_wh']
            }

            min_soc_overall = min(min_soc_overall, min_soc_ratio)

        total_energy_charged = sum(self.bus_energy_charged.values())
        num_preemptions = len(self.stop_charging_manager.get_preemption_events())

        return {
            'battery_capacity_wh': self.battery_capacity_wh,
            'line_battery_capacities_wh': dict(self.line_battery_capacities_wh),
            'num_maps': self.num_maps,
            'charging_enabled': self.stop_charging_manager.charging_enabled,
            'preemption_threshold': self.preemption_threshold,
            'trip_change_stops': len(self.trip_change_stops),
            'num_buses': len(self.buses),
            'num_layovers': len(self.layovers),
            'num_preemptions': num_preemptions,
            'min_soc_overall_ratio': min_soc_overall,
            'min_soc_overall_wh': min_soc_overall * self.battery_capacity_wh,
            'min_soc_threshold': BUS_MIN_SOC,
            'feasible': min_soc_overall >= BUS_MIN_SOC,
            'total_energy_charged_wh': total_energy_charged,
            'bus_statistics': bus_stats
        }

    def print_first_layovers(self, num_to_print: int = 5):
        """Print first N layover records"""

        print("\n" + "="*70)
        print(f"FIRST {num_to_print} LAYOVER RECORDS")
        print("="*70)
        print(f"\n{'Bus ID':<20} {'Stop ID':<15} {'Arrival (s)':<15} {'Departure (s)':<15} {'Layover (s)':<12} {'Charged':<12}")
        print("-"*110)

        for layover in self.layovers[:num_to_print]:
            charged_str = f"Yes ({layover.energy_charged_wh:.0f}Wh)" if layover.was_charged else "No"
            print(f"{layover.bus_id:<20} {layover.stop_id:<15} "
                  f"{layover.arrival_at_terminal_s:<15.1f} "
                  f"{layover.departure_from_terminal_s:<15.1f} {layover.layover_duration_s:<12.1f} {charged_str:<12}")

        print(f"\nTotal layovers recorded: {len(self.layovers)}")

    def print_charging_events(self, num_to_print: int = 10):
        """Print charging events"""

        events = self.stop_charging_manager.get_charging_events()

        if not events:
            print("\n" + "="*70)
            print("NO CHARGING EVENTS")
            print("="*70 + "\n")
            return

        print("\n" + "="*70)
        print(f"CHARGING EVENTS (First {num_to_print})")
        print("="*70)
        print(f"\n{'Bus ID':<20} {'MAP ID':<8} {'Start Time (s)':<18} {'Duration (s)':<15} {'Energy (Wh)':<15}")
        print("-"*95)

        for event in events[:num_to_print]:
            print(f"{event['bus_id']:<20} {event['map_id']:<8} "
                  f"{event['start_time_s']:<18.1f} {event['duration_s']:<15.1f} {event['energy_wh']:<15.0f}")

        if len(events) > num_to_print:
            print(f"\n... and {len(events) - num_to_print} more events")

        print(f"\nTotal charging events: {len(events)}")

    def print_preemption_events(self):
        """Print all preemption events"""

        preemptions = self.stop_charging_manager.get_preemption_events()

        if not preemptions:
            print("\n" + "="*70)
            print("NO PREEMPTION EVENTS")
            print("="*70 + "\n")
            return

        print("\n" + "="*70)
        print(f"PREEMPTION EVENTS ({len(preemptions)} total)")
        print("="*70)
        print(f"\nTotal preemptions: {len(preemptions)}\n")

    def print_map_usage(self):
        """Print MAP usage statistics"""
        self.map_tracker.print_summary(self.battery_capacity_wh)

    def print_map_assignments(self):
        """Print which buses are charged by each MAP"""
        self.map_tracker.print_detailed_assignments()

    def print_map_movement(self):
        """Print MAP movement and battery statistics"""
        summary = self.map_movement_scheduler.get_summary()

        print("\n" + "="*70)
        print("MAP MOVEMENT & BATTERY STATISTICS")
        print("="*70)
        print(f"\n{'MAP ID':<8} {'Location':<20} {'SOC (kWh)':<12} {'SOC %':<8} {'Dist (km)':<12} {'Events':<8}")
        print("-"*80)

        for map_id, state_summary in summary.items():
            distance_km = state_summary['distance_traveled_m'] / 1000
            soc_kwh = state_summary['current_soc_wh'] / 1000
            cap = self.map_movement_scheduler.map_states[map_id].battery_capacity_wh
            soc_pct = (state_summary['current_soc_wh'] / cap * 100) if cap > 0 else 0
            print(f"{map_id:<8} {state_summary['current_location']:<20} "
                  f"{soc_kwh:<12.1f} {soc_pct:<8.1f} {distance_km:<12.1f} "
                  f"{state_summary['num_charging_events']:<8}")

        print(f"\nMAP battery capacity: {MAP_BATTERY_CAPACITY_WH/1000:.0f} kWh | "
              f"Min SOC: {MAP_MIN_SOC*100:.0f}% ({MAP_MIN_SOC*MAP_BATTERY_CAPACITY_WH/1000:.0f} kWh) | "
              f"Self-charge rate: {MAP_SELF_CHARGE_RATE_WH_S} Wh/s")
        print(f"{'='*70}\n")

    def print_bus_charging_schedule(self, num_to_print: int = 10):
        """Print bus charging schedule"""
        self.map_tracker.print_bus_charging_schedule(num_to_print)

    def plot_soc(self, save_path: str = "bus_soc_trajectories.png"):
        """Plot bus SOC trajectories"""

        plt.figure(figsize=(14, 8))

        lines_seen = {}
        cmap = plt.get_cmap("tab10")
        next_color_idx = 0

        for bus_id, series in self.bus_soc_history.items():
            filtered = [x for x in series if x[0] > 0]
            if not filtered:
                continue

            t = [x[0] for x in filtered]
            s = [x[1] for x in filtered]

            line_key = "default"
            try:
                if bus_id and bus_id.startswith("line"):
                    rest = bus_id[4:]
                    line_key = rest.split("_", 1)[0]
            except Exception:
                line_key = "default"

            if line_key not in lines_seen:
                lines_seen[line_key] = cmap(next_color_idx % cmap.N)
                next_color_idx += 1

            color = lines_seen[line_key]
            plt.plot(t, s, label="_nolegend_", color=color, linewidth=1.5, alpha=0.8)

        if lines_seen:
            try:
                from matplotlib.lines import Line2D
                handles = [Line2D([0], [0], color=col, lw=3) for col in lines_seen.values()]
                labels = [f"Line {lk}" for lk in sorted(lines_seen.keys())]
                plt.legend(handles=handles, labels=labels, loc="best", fontsize="small", ncol=2, title="Line")
            except Exception:
                pass

        battery_cap = self.battery_capacity_wh
        plt.axhline(y=BUS_CHARGE_CUTOFF_SOC * battery_cap, color='green', linestyle='--',
                   linewidth=2, alpha=0.5, label='80%')
        plt.axhline(y=BUS_CHARGE_THRESHOLD_SOC * battery_cap, color='orange', linestyle='--',
                   linewidth=2, alpha=0.5, label='70%')
        plt.axhline(y=BUS_MIN_SOC * battery_cap, color='red', linestyle='--',
                   linewidth=2, alpha=0.5, label='20%')

        plt.ylim(bottom=0)
        plt.xlabel("Time [s]", fontweight='bold')
        plt.ylabel("Battery SOC [Wh]", fontweight='bold')
        plt.title(f"Bus SOC Trajectories ({self.num_maps} MAPs)", fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()

    def plot_map_energy_delivery(self, save_path: str = "map_energy_delivery.png"):
        """Plot energy delivered by each MAP"""

        records = self.map_tracker.charging_records

        if not records:
            print("No charging records to plot")
            return

        by_map = defaultdict(lambda: {'energies': [], 'buses': []})

        for record in records:
            by_map[record.map_id]['energies'].append(record.energy_delivered_wh / 1000)
            by_map[record.map_id]['buses'].append(record.bus_id)

        num_maps = self.num_maps
        fig, axes = plt.subplots(num_maps, 1, figsize=(14, 4*num_maps))

        if num_maps == 1:
            axes = [axes]

        colors = plt.cm.Set3(np.linspace(0, 1, num_maps))

        for map_id in range(num_maps):
            ax = axes[map_id]

            if map_id in by_map:
                energies = by_map[map_id]['energies']
                buses = by_map[map_id]['buses']

                ax.bar(range(len(energies)), energies, color=colors[map_id], alpha=0.7, edgecolor='black')

                for i, (energy, bus) in enumerate(zip(energies, buses)):
                    ax.text(i, energy + 1, bus.split('_')[-1], ha='center', va='bottom', fontsize=8)

                total_energy = sum(energies)
                ax.set_title(f"MAP {map_id}: {total_energy:.1f} kWh ({len(energies)} events)", fontweight='bold')
            else:
                ax.text(0.5, 0.5, f'MAP {map_id}: No events', ha='center', va='center', transform=ax.transAxes)

            ax.set_xlabel('Event', fontweight='bold')
            ax.set_ylabel('Energy (kWh)', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()

    def plot_cumulative_energy_delivery(self, save_path: str = "cumulative_energy_delivery.png"):
        """Plot cumulative energy delivery over time for each MAP"""

        records = self.map_tracker.charging_records

        if not records:
            print("No charging records to plot")
            return

        records_sorted = sorted(records, key=lambda r: r.start_time_s)

        plt.figure(figsize=(14, 7))

        by_map = defaultdict(list)
        for record in records_sorted:
            by_map[record.map_id].append(record)

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_maps))

        for map_id in range(self.num_maps):
            if map_id in by_map:
                map_records = by_map[map_id]
                times = [r.start_time_s / 3600 for r in map_records]
                cumulative_energy = np.cumsum([r.energy_delivered_wh / 1000 for r in map_records])

                plt.plot(times, cumulative_energy, marker='o', linewidth=2.5,
                        label=f'MAP {map_id}', color=colors[map_id], markersize=6)

        plt.xlabel('Time (hours)', fontweight='bold', fontsize=12)
        plt.ylabel('Cumulative Energy (kWh)', fontweight='bold', fontsize=12)
        plt.title('Cumulative Energy Delivery by MAPs', fontweight='bold', fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()

    def plot_map_movement(self, save_path: str = "map_movement_distance.png"):
        """Plot MAP distance traveled over time"""

        records = self.map_tracker.movement_records

        if not records:
            print("No MAP movement records to plot")
            return

        records_sorted = sorted(records, key=lambda r: r.start_time_s)

        plt.figure(figsize=(14, 7))

        by_map = defaultdict(list)
        for record in records_sorted:
            by_map[record.map_id].append(record)

        colors = plt.cm.tab10(np.linspace(0, 1, self.num_maps))

        for map_id in range(self.num_maps):
            if map_id in by_map:
                map_records = by_map[map_id]
                times = [r.end_time_s / 3600 for r in map_records]
                cumulative_distance = np.cumsum([r.distance_m / 1000 for r in map_records])

                plt.plot(times, cumulative_distance, marker='o', linewidth=2.5,
                        label=f'MAP {map_id}', color=colors[map_id], markersize=6)

        plt.xlabel('Time (hours)', fontweight='bold', fontsize=12)
        plt.ylabel('Cumulative Distance (km)', fontweight='bold', fontsize=12)
        plt.title('Cumulative Distance Traveled by MAPs', fontweight='bold', fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()

    def plot_map_soc(self, save_path: str = "map_soc_over_time.png"):
        """Plot MAP battery SOC levels over time"""

        plt.figure(figsize=(14, 7))

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, self.num_maps)))
        has_data = False

        for map_id in range(self.num_maps):
            series = self.map_tracker.map_soc_history.get(map_id, [])
            if not series:
                continue
            has_data = True
            t = [x[0] / 3600 for x in series]      # hours
            s = [x[1] / 1000 for x in series]       # kWh
            plt.plot(t, s, linewidth=2, label=f'MAP {map_id}',
                     color=colors[map_id], alpha=0.9)

        if not has_data:
            print("No MAP SOC history to plot")
            plt.close()
            return

        cap_kwh = MAP_BATTERY_CAPACITY_WH / 1000
        plt.axhline(y=cap_kwh, color='green', linestyle='--',
                     linewidth=1.5, alpha=0.6, label=f'Full ({cap_kwh:.0f} kWh)')
        plt.axhline(y=MAP_MIN_SOC * cap_kwh, color='red', linestyle='--',
                     linewidth=1.5, alpha=0.6,
                     label=f'Min SOC {MAP_MIN_SOC*100:.0f}% ({MAP_MIN_SOC*cap_kwh:.0f} kWh)')

        plt.ylim(bottom=0, top=cap_kwh * 1.05)
        plt.xlabel('Time (hours)', fontweight='bold', fontsize=12)
        plt.ylabel('MAP Battery SOC (kWh)', fontweight='bold', fontsize=12)
        plt.title('MAP Battery SOC Over Time', fontweight='bold', fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
        plt.show()

# ========================
# SIMULATION RUNNER
# ========================

def run_terminal_charging_simulation(sim, bus_trips_dict, bus_lines, trip_change_stops,
                                    battery_capacity_wh: float = 250000,
                                    num_maps: int = 1,
                                    optimize_threshold: bool = True,
                                    preemption_threshold: Optional[float] = None,
                                    simulation_duration_s: float = 86400,
                                    line_battery_capacities_wh: Optional[Dict[str, float]] = None,
                                    map_battery_capacity_wh: Optional[float] = None):
    """Execute terminal charging simulation with MAP tracking"""

    stage2 = Stage2DESTerminalChargingPreemptive(
        sim=sim,
        bus_trips_dict=bus_trips_dict,
        bus_lines=bus_lines,
        trip_change_stops=trip_change_stops,
        initial_battery_capacity_wh=battery_capacity_wh,
        num_maps=num_maps,
        optimize_threshold=optimize_threshold,
        preemption_threshold=preemption_threshold,
        line_battery_capacities_wh=line_battery_capacities_wh,
        map_battery_capacity_wh=map_battery_capacity_wh
    )

    results = stage2.run_simulation(simulation_duration_s)

    # Print all outputs
    stage2.print_first_layovers(num_to_print=5)
    stage2.print_charging_events(num_to_print=10)
    stage2.print_preemption_events()

    # Print MAP usage
    stage2.print_map_usage()
    stage2.print_map_assignments()
    stage2.print_map_movement()  # NEW: Print MAP movement statistics
    stage2.print_bus_charging_schedule(num_to_print=5)

    # Print simulation results
    print("\n" + "="*70)
    print("STAGE-2 SIMULATION RESULTS")
    print("="*70)

    print("\nFEASIBILITY CHECK:")
    print("-"*70)
    print(f"Battery capacity: {results['battery_capacity_wh']/1000:,.1f} kWh")
    print(f"Trip-change stops: {results['trip_change_stops']}")
    print(f"Available MAPs: {results['num_maps']}")
    print(f"Charging enabled: {'YES' if results['charging_enabled'] else 'NO'}")

    if results['preemption_threshold'] is not None:
        print(f"Preemption threshold: {results['preemption_threshold']*100:.1f}% SOC")
    else:
        print(f"Preemption threshold: DISABLED")

    print(f"Minimum SOC reached: {results['min_soc_overall_ratio']*100:.1f}%")

    if results['feasible']:
        print(f"✓ FEASIBLE: Min SOC stays above 20%")
    else:
        print(f"✗ INFEASIBLE")

    print("\nCHARGING STATISTICS:")
    print("-"*70)
    print(f"Total layovers: {results['num_layovers']}")
    print(f"Total preemptions: {results['num_preemptions']}")
    print(f"Total energy charged: {results['total_energy_charged_wh']/1e6:,.3f} MWh")

    print("\n" + "="*70)

    # Plots
    stage2.plot_soc(save_path="bus_soc_terminal_charging_optimized.png")
    stage2.plot_map_energy_delivery(save_path="map_energy_delivery.png")
    stage2.plot_cumulative_energy_delivery(save_path="cumulative_energy_delivery.png")
    stage2.plot_map_movement(save_path="map_movement_distance.png")
    stage2.plot_map_soc(save_path="map_soc_over_time.png")

    return results, stage2
