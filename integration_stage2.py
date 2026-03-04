"""
Stage 2: DES Charging Simulation - Terminal + Dynamic MAP Movement Charging
WITH MAP MOVEMENT, ROUTE-BASED CHARGING, AND ADVANCED TRACKING
"""

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
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
CHARGER_SPEED_MS = 27.78
ENERGY_PER_METER_WH = 2.7

MAP_ENERGY_PER_METER_WH = 0.3       # MAP energy consumption per metre of travel (Wh/m)
MAP_RECHARGE_RATE_WH_S = 233.33     # Rate at which idle MAPs self-recharge (Wh/s)
MAP_BATTERY_CAPACITY_WH = 100000    # Default MAP battery capacity (100 kWh)

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
    charging_type: str = "terminal"  # "terminal", "en_route_stop", "en_route_segment"

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
    movement_type: str = "independent"  # "independent", "following_bus"

@dataclass
class MAPState:
    """Current state of a MAP"""
    map_id: int
    current_location: Optional[str]  # stop_id, or None when unspawned
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
    spawned: bool = False          # False = unspawned (not yet deployed)
    is_following_bus: bool = False  # True = attached to a bus, moving with it
    is_recharging: bool = False    # True = MAP is self-recharging (unavailable for bus charging)
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None

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

    def record_charge(self, map_id: int, bus_id: str, start_time: float,
                     end_time: float, energy_wh: float, location: str,
                     soc_before: float, soc_after: float,
                     charging_type: str = "terminal"):
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
            soc_after_wh=soc_after,
            charging_type=charging_type
        )

        self.charging_records.append(record)
        self.map_bus_assignments[map_id].add(bus_id)
        self.map_total_energy[map_id] += energy_wh
        self.map_total_time[map_id] += duration
        self.map_num_events[map_id] += 1
        self.bus_charging_history[bus_id].append((start_time, energy_wh, map_id))

    def record_movement(self, map_id: int, start_time: float, end_time: float,
                       from_location: str, to_location: str, distance_m: float,
                       associated_bus_id: Optional[str] = None,
                       movement_type: str = "independent"):
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
            associated_bus_id=associated_bus_id,
            movement_type=movement_type
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

        # Breakdown by charging type
        charging_type_breakdown = defaultdict(lambda: {'events': 0, 'energy_wh': 0.0})
        for record in self.charging_records:
            charging_type_breakdown[record.charging_type]['events'] += 1
            charging_type_breakdown[record.charging_type]['energy_wh'] += record.energy_delivered_wh

        # Breakdown by movement type
        movement_type_breakdown = defaultdict(lambda: {'events': 0, 'distance_m': 0.0})
        for record in self.movement_records:
            movement_type_breakdown[record.movement_type]['events'] += 1
            movement_type_breakdown[record.movement_type]['distance_m'] += record.distance_m

        return {
            'total_energy_wh': total_energy,
            'total_time_s': total_time,
            'num_events': sum(self.map_num_events.values()),
            'map_summaries': map_summaries,
            'total_distance_m': total_distance,
            'total_movements': sum(self.map_movement_events.values()),
            'charging_type_breakdown': dict(charging_type_breakdown),
            'movement_type_breakdown': dict(movement_type_breakdown),
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

        # Charging type breakdown
        ct_breakdown = summary.get('charging_type_breakdown', {})
        if ct_breakdown:
            print(f"\nCharging Type Breakdown:")
            for ctype, stats in sorted(ct_breakdown.items()):
                print(f"  {ctype:<20}: {stats['events']:>5} events, "
                      f"{stats['energy_wh']/1000:,.1f} kWh")

        # Movement type breakdown
        mv_breakdown = summary.get('movement_type_breakdown', {})
        if mv_breakdown:
            print(f"\nMovement Type Breakdown:")
            for mtype, stats in sorted(mv_breakdown.items()):
                print(f"  {mtype:<20}: {stats['events']:>5} events, "
                      f"{stats['distance_m']/1000:,.1f} km")

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
        if capacity_wh is None or capacity_wh <= 0:
            return False
        return soc_wh < (self.start_pct * capacity_wh)

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
# MAP MOVEMENT SCHEDULER (NEW)
# ========================

class MAPMovementScheduler:
    """Manages MAP movement, routing, and location tracking"""

    def __init__(self, env: simpy.Environment, num_maps: int,
                 map_battery_capacity_wh: Optional[Union[float, List[float]]] = None,
                 map_speed_ms: float = 27.78,
                 map_tracker: Optional[MAPUsageTracker] = None):
        self.env = env
        self.num_maps = num_maps
        self.map_speed_ms = map_speed_ms
        self.map_tracker = map_tracker

        # Resolve per-MAP battery capacities (float → same for all; list → per-MAP)
        if map_battery_capacity_wh is None:
            capacities = [MAP_BATTERY_CAPACITY_WH] * num_maps
        elif isinstance(map_battery_capacity_wh, (list, tuple)):
            capacities = list(map_battery_capacity_wh)
            # Pad or trim to num_maps
            while len(capacities) < num_maps:
                capacities.append(capacities[-1] if capacities else MAP_BATTERY_CAPACITY_WH)
            capacities = capacities[:num_maps]
        else:
            capacities = [float(map_battery_capacity_wh)] * num_maps

        # MAP states indexed by map_id — start unspawned (current_location=None)
        self.map_states = {}
        for map_id in range(num_maps):
            cap = capacities[map_id]
            self.map_states[map_id] = MAPState(
                map_id=map_id,
                current_location=None,  # unspawned
                current_time_s=0.0,
                current_soc_wh=cap,
                battery_capacity_wh=cap,
                spawned=False,
            )

        # MAP SOC history for plotting: {map_id: [(time_s, soc_wh), ...]}
        self.map_soc_history: Dict[int, List] = defaultdict(list)
        # Record initial SOC for each MAP
        for map_id in range(num_maps):
            self.map_soc_history[map_id].append((0.0, self.map_states[map_id].current_soc_wh))

        # SimPy process handles for self-recharging: {map_id: simpy.Process}
        self.map_recharge_processes: Dict[int, object] = {}

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

    def record_map_soc(self, map_id: int, time_s: float):
        """Record the current MAP SOC at the given simulation time."""
        state = self.map_states.get(map_id)
        if state is not None:
            self.map_soc_history[map_id].append((time_s, state.current_soc_wh))

    def calculate_travel_time(self, from_location: str, to_location: str) -> Tuple[float, float]:
        """Calculate travel time and distance between two locations (in meters and seconds)

        Returns: (distance_m, time_s)
        """
        if from_location not in self.stop_locations or to_location not in self.stop_locations:
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

    def spawn_or_travel_to(self, map_id: int, target_stop_id: str,
                           target_lat: Optional[float] = None,
                           target_lon: Optional[float] = None) -> Tuple[float, float]:
        """Spawn MAP at target on first use (zero travel time), or travel there if already spawned.

        Returns: (travel_time_s, distance_m)
        """
        state = self.map_states.get(map_id)
        if state is None:
            return (0.0, 0.0)

        if not state.spawned:
            # First use: spawn instantly at target
            state.current_location = target_stop_id
            state.current_lat = target_lat
            state.current_lon = target_lon
            state.spawned = True
            state.arrival_time_at_target_s = float(self.env.now)
            return (0.0, 0.0)

        # Already spawned: calculate travel from current location
        from_loc = state.current_location or "unknown"
        distance_m, travel_time_s = self.calculate_travel_time(from_loc, target_stop_id)

        if self.map_tracker and from_loc != target_stop_id:
            self.map_tracker.record_movement(
                map_id=map_id,
                start_time=float(self.env.now),
                end_time=float(self.env.now) + travel_time_s,
                from_location=from_loc,
                to_location=target_stop_id,
                distance_m=distance_m,
                associated_bus_id=None,
                movement_type="independent"
            )

        # Deduct travel energy from MAP battery
        travel_energy = distance_m * MAP_ENERGY_PER_METER_WH
        state.current_soc_wh = max(0.0, state.current_soc_wh - travel_energy)

        state.current_location = target_stop_id
        if target_lat is not None:
            state.current_lat = target_lat
        if target_lon is not None:
            state.current_lon = target_lon
        state.distance_traveled_m += distance_m
        state.arrival_time_at_target_s = float(self.env.now) + travel_time_s
        self.record_map_soc(map_id, float(self.env.now) + travel_time_s)
        return (travel_time_s, distance_m)

    def start_following(self, map_id: int, bus_id: str):
        """Mark a MAP as attached to a bus and following it"""
        state = self.map_states.get(map_id)
        if state:
            # Cancel any running self-recharge before the MAP starts working
            if state.is_recharging:
                proc = self.map_recharge_processes.get(map_id)
                if proc is not None and proc.is_alive:
                    try:
                        proc.interrupt()
                    except RuntimeError:
                        pass
                state.is_recharging = False
            state.is_following_bus = True
            state.assigned_bus_id = bus_id
            state.is_charging = True

    def stop_following(self, map_id: int):
        """Detach a MAP from its bus; MAP becomes available again and self-recharges if needed"""
        state = self.map_states.get(map_id)
        if state:
            state.is_following_bus = False
            state.assigned_bus_id = None
            state.is_charging = False
            self._schedule_recharge(map_id)

    def update_location_following_bus(self, map_id: int, stop_id: str,
                                      lat: Optional[float] = None,
                                      lon: Optional[float] = None):
        """Sync MAP location with bus after each segment when following"""
        state = self.map_states.get(map_id)
        if state is None:
            return
        from_loc = state.current_location
        if from_loc != stop_id:
            distance_m, _ = self.calculate_travel_time(from_loc, stop_id) if from_loc else (0.0, 0.0)
            if self.map_tracker:
                self.map_tracker.record_movement(
                    map_id=map_id,
                    start_time=float(self.env.now),
                    end_time=float(self.env.now),
                    from_location=from_loc or "unknown",
                    to_location=stop_id,
                    distance_m=distance_m,
                    associated_bus_id=state.assigned_bus_id,
                    movement_type="following_bus"
                )
            # Deduct travel energy from MAP battery
            travel_energy = distance_m * MAP_ENERGY_PER_METER_WH
            state.current_soc_wh = max(0.0, state.current_soc_wh - travel_energy)
            state.current_location = stop_id
            state.distance_traveled_m += distance_m
            self.record_map_soc(map_id, float(self.env.now))
        if lat is not None:
            state.current_lat = lat
        if lon is not None:
            state.current_lon = lon

    def _schedule_recharge(self, map_id: int):
        """Start a self-recharge SimPy process for a MAP if its battery is not full."""
        state = self.map_states.get(map_id)
        if state is None or state.current_soc_wh >= state.battery_capacity_wh:
            return
        # Don't start a new process if one is already running
        proc = self.map_recharge_processes.get(map_id)
        if proc is not None and proc.is_alive:
            return
        # Set is_recharging synchronously so it takes effect immediately
        state.is_recharging = True
        self.record_map_soc(map_id, float(self.env.now))
        process = self.env.process(self._recharge_map_process(map_id))
        self.map_recharge_processes[map_id] = process

    def _recharge_map_process(self, map_id: int):
        """SimPy generator: self-recharge a MAP at MAP_RECHARGE_RATE_WH_S until full."""
        state = self.map_states[map_id]
        try:
            while state.current_soc_wh < state.battery_capacity_wh:
                energy_needed = state.battery_capacity_wh - state.current_soc_wh
                time_to_full = energy_needed / MAP_RECHARGE_RATE_WH_S
                yield self.env.timeout(time_to_full)
                state.current_soc_wh = state.battery_capacity_wh
        except simpy.Interrupt:
            pass
        finally:
            state.is_recharging = False
            self.map_recharge_processes.pop(map_id, None)
            self.record_map_soc(map_id, float(self.env.now))

    def get_available_map(self, current_stop_id: Optional[str] = None,
                          current_lat: Optional[float] = None,
                          current_lon: Optional[float] = None) -> Optional[int]:
        """Return best available MAP: unspawned preferred, then nearest idle spawned.

        A MAP is considered unavailable if it is following a bus, self-recharging,
        or has an empty battery.

        Returns map_id or None if no MAP is available.
        """
        # Prefer unspawned (spawns at target for free), but only if it has energy
        for map_id, state in self.map_states.items():
            if not state.spawned and state.current_soc_wh > 0:
                return map_id

        # Then nearest idle spawned MAP (not following, not recharging, has energy)
        best_map = None
        best_dist = float('inf')
        for map_id, state in self.map_states.items():
            if state.is_following_bus or state.is_charging or state.is_recharging:
                continue
            if state.current_soc_wh <= 0:
                continue
            if current_stop_id and state.current_location:
                dist, _ = self.calculate_travel_time(state.current_location, current_stop_id)
            else:
                dist = 0.0
            if dist < best_dist:
                best_dist = dist
                best_map = map_id
        return best_map

    def get_following_map_for_bus(self, bus_id: str) -> Optional[int]:
        """Return the MAP currently following the given bus, or None"""
        for map_id, state in self.map_states.items():
            if state.is_following_bus and state.assigned_bus_id == bus_id:
                return map_id
        return None

    def assign_map_to_bus(self, map_id: int, bus_id: str, current_location: str,
                         target_location: str) -> Optional[Tuple[float, float]]:
        """Assign a MAP to move to a bus location (legacy helper kept for compatibility)

        Returns: (travel_time_s, distance_m) or None if invalid
        """
        if map_id < 0 or map_id >= self.num_maps:
            return None

        map_state = self.map_states.get(map_id)
        if map_state is None:
            return None

        distance_m, travel_time_s = self.calculate_travel_time(
            map_state.current_location or "", target_location)

        map_state.current_location = target_location
        map_state.assigned_bus_id = bus_id
        map_state.arrival_time_at_target_s = float(self.env.now) + travel_time_s
        map_state.distance_traveled_m += distance_m
        return (travel_time_s, distance_m)

    def get_map_location(self, map_id: int) -> Optional[str]:
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
        """Update MAP SOC after charging a bus"""
        if map_id in self.map_states:
            state = self.map_states[map_id]
            state.current_soc_wh = max(0.0, state.current_soc_wh - energy_consumed_wh)
            state.num_charging_events += 1

    def map_is_at_location(self, map_id: int, location: str) -> bool:
        """Check if a MAP is at a specific location"""
        if map_id in self.map_states:
            return self.map_states[map_id].current_location == location
        return False

    def get_summary(self) -> Dict:
        """Get summary of MAP states"""
        summary = {}
        for map_id, state in self.map_states.items():
            soc_pct = (state.current_soc_wh / state.battery_capacity_wh * 100.0
                       if state.battery_capacity_wh > 0 else 0.0)
            summary[map_id] = {
                'current_location': state.current_location,
                'spawned': state.spawned,
                'is_following_bus': state.is_following_bus,
                'is_recharging': state.is_recharging,
                'current_soc_wh': state.current_soc_wh,
                'battery_capacity_wh': state.battery_capacity_wh,
                'current_soc_pct': soc_pct,
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
                 map_battery_capacity_wh: Optional[Union[float, List[float]]] = None):

        self.sim = sim
        self.bus_trips_dict = bus_trips_dict
        self.bus_lines = bus_lines
        self.trip_change_stops = trip_change_stops
        self.battery_capacity_wh = initial_battery_capacity_wh
        self.num_maps = num_maps

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
            map_battery_capacity_wh=map_battery_capacity_wh,
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

            self.buses[bus_id] = {
                'line_id': line_id,
                'trip_ids': trip_ids,
                'soc_wh': initial_battery_capacity_wh,
                'min_soc_wh': initial_battery_capacity_wh,
                'total_energy_consumed_wh': 0.0
            }

        self.layovers = []
        self.bus_soc_history = defaultdict(list)

        self.bus_soc = defaultdict(float)
        self.bus_energy_charged = defaultdict(float)

    def _terminal_charge(self, bus_id: str, stop_id: str, layover_duration_s: float,
                         capacity: float):
        """Handle terminal charging as a SimPy sub-process (generator).

        Called with ``yield from`` inside ``_simulate_bus``.  The generator
        yields SimPy timeouts so the simulated clock advances appropriately.
        MAPs that are already following persist through the layover.
        """
        if self.num_maps <= 0:
            return

        soc = self.bus_soc.get(bus_id, capacity)
        target_wh = BUS_CHARGE_CUTOFF_SOC * capacity

        # --- Case 1: a MAP is already following this bus ---
        following_map = self.map_movement_scheduler.get_following_map_for_bus(bus_id)
        if following_map is not None:
            if soc < target_wh:
                map_state = self.map_movement_scheduler.map_states[following_map]
                available_map_energy = map_state.current_soc_wh
                t_start = float(self.env.now)
                energy = min(MAP_CHARGING_RATE_WH_S * layover_duration_s,
                             target_wh - soc,
                             available_map_energy)
                charge_dur = energy / MAP_CHARGING_RATE_WH_S if MAP_CHARGING_RATE_WH_S > 0 else 0.0
                if charge_dur > 0:
                    yield self.env.timeout(charge_dur)
                    soc_before = self.bus_soc.get(bus_id, capacity)
                    new_soc = min(capacity, soc_before + energy)
                    self.bus_soc[bus_id] = new_soc
                    self.bus_energy_charged[bus_id] += energy
                    # Deduct energy from MAP battery
                    map_state.current_soc_wh = max(0.0, map_state.current_soc_wh - energy)
                    self.map_movement_scheduler.record_map_soc(following_map, float(self.env.now))
                    self.map_tracker.record_charge(
                        map_id=following_map, bus_id=bus_id,
                        start_time=t_start, end_time=float(self.env.now),
                        energy_wh=energy, location=f"stop_{stop_id}",
                        soc_before=soc_before, soc_after=new_soc,
                        charging_type="terminal"
                    )
                    if new_soc >= target_wh or map_state.current_soc_wh <= 0:
                        self.map_movement_scheduler.stop_following(following_map)
            return

        # --- Case 2: no MAP following; request one if bus needs charge ---
        if soc >= BUS_CHARGE_THRESHOLD_SOC * capacity:
            return

        map_id = self.map_movement_scheduler.get_available_map(stop_id)
        if map_id is None:
            return

        travel_time_s, _ = self.map_movement_scheduler.spawn_or_travel_to(map_id, stop_id)

        remaining = layover_duration_s - travel_time_s
        if remaining <= 0:
            return

        if travel_time_s > 0:
            yield self.env.timeout(travel_time_s)

        self.map_movement_scheduler.start_following(map_id, bus_id)
        map_state = self.map_movement_scheduler.map_states[map_id]
        soc = self.bus_soc.get(bus_id, capacity)
        t_start = float(self.env.now)
        available_map_energy = map_state.current_soc_wh
        energy = min(MAP_CHARGING_RATE_WH_S * remaining,
                     max(0.0, target_wh - soc),
                     available_map_energy)
        charge_dur = energy / MAP_CHARGING_RATE_WH_S if (MAP_CHARGING_RATE_WH_S > 0 and energy > 0) else 0.0
        if charge_dur > 0:
            yield self.env.timeout(charge_dur)
            soc_before = self.bus_soc.get(bus_id, capacity)
            new_soc = min(capacity, soc_before + energy)
            self.bus_soc[bus_id] = new_soc
            self.bus_energy_charged[bus_id] += energy
            # Deduct energy from MAP battery
            map_state.current_soc_wh = max(0.0, map_state.current_soc_wh - energy)
            self.map_movement_scheduler.record_map_soc(map_id, float(self.env.now))
            self.map_tracker.record_charge(
                map_id=map_id, bus_id=bus_id,
                start_time=t_start, end_time=float(self.env.now),
                energy_wh=energy, location=f"stop_{stop_id}",
                soc_before=soc_before, soc_after=new_soc,
                charging_type="terminal"
            )
            if new_soc >= target_wh or map_state.current_soc_wh <= 0:
                self.map_movement_scheduler.stop_following(map_id)

    def run_simulation(self, duration_s: float = 86400) -> Dict:
        """Run DES simulation"""

        print("\n" + "="*70)
        print("STAGE-2: DES TERMINAL CHARGING WITH MAP USAGE TRACKING")
        print("="*70)
        print(f"Battery capacity: {self.battery_capacity_wh/1000:,.1f} kWh")
        print(f"Trip-change stops: {len(self.trip_change_stops)}")
        print(f"Available MAPs: {self.num_maps}")

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
        """Simulate a single bus with dynamic en-route MAP charging."""

        try:
            line_id = bus_id.split('_')[0].replace('line', '')
        except Exception:
            line_id = "unknown"

        capacity = self.battery_capacity_wh
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

            # --- Inter-trip idle / layover phase ---
            if idle_dt > 0:
                try:
                    start_node = self.sim.stop_node.get(seq[0].get("stop_id"))
                except Exception:
                    start_node = None

                is_terminal = (
                    prev_end_node is not None and start_node is not None
                    and prev_end_node == start_node and prev_end_time is not None
                )
                start_stop_id = seq[0].get("stop_id")

                if is_terminal and start_stop_id in self.trip_change_stops:
                    duration = trip_start - prev_end_time
                    soc_now = self.bus_soc.get(bus_id, capacity)

                    next_trip_id = (trip_ids[trip_idx + 1]
                                    if trip_idx + 1 < len(trip_ids) else None)
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

                    # Handle terminal charging (generator sub-process)
                    energy_before = sum(self.bus_energy_charged.values())
                    yield from self._terminal_charge(bus_id, start_stop_id, duration, capacity)
                    energy_after = sum(self.bus_energy_charged.values())

                    # Wait for the remainder of the layover
                    time_used = float(self.env.now) - (trip_start - duration)
                    remaining_idle = max(0.0, duration - time_used)
                    if remaining_idle > 0:
                        yield self.env.timeout(remaining_idle)

                    energy_charged = energy_after - energy_before
                    if energy_charged > 0:
                        layover.was_charged = True
                        layover.energy_charged_wh = energy_charged

                    layover.soc_at_departure_wh = self.bus_soc.get(bus_id, capacity)
                    self.layovers.append(layover)
                    self.bus_soc_history[bus_id].append(
                        (self.env.now, self.bus_soc.get(bus_id, capacity)))
                else:
                    yield self.env.timeout(idle_dt)

                current_time = trip_start

            # --- Trip segment execution ---
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

                # 1. Apply energy consumption for this segment
                energy_consumed = dist * ENERGY_PER_METER_WH
                prev_soc = self.bus_soc.get(bus_id, capacity)
                new_soc = max(0.0, prev_soc - energy_consumed)
                self.bus_soc[bus_id] = new_soc

                self.buses[bus_id]['total_energy_consumed_wh'] += energy_consumed
                if new_soc < self.buses[bus_id]['min_soc_wh']:
                    self.buses[bus_id]['min_soc_wh'] = new_soc

                curr_stop_id = curr_stop.get("stop_id")
                curr_lat = curr_stop.get("lat")
                curr_lon = curr_stop.get("lon")

                # 2. If a MAP is following, deliver en-route segment charge
                following_map = self.map_movement_scheduler.get_following_map_for_bus(bus_id)
                if following_map is not None:
                    was_following_map = following_map  # remember for location update
                    soc = self.bus_soc[bus_id]
                    target_wh = BUS_CHARGE_CUTOFF_SOC * capacity
                    map_state = self.map_movement_scheduler.map_states[following_map]
                    available_map_energy = map_state.current_soc_wh
                    if soc < target_wh and dt > 0 and available_map_energy > 0:
                        seg_energy = min(
                            MAP_CHARGING_RATE_WH_S * max(0.0, dt),
                            target_wh - soc,
                            available_map_energy
                        )
                        if seg_energy > 0:
                            t_start = float(self.env.now) - max(0.0, dt)
                            soc_before = soc
                            new_charged_soc = min(capacity, soc + seg_energy)
                            self.bus_soc[bus_id] = new_charged_soc
                            self.bus_energy_charged[bus_id] += seg_energy
                            # Deduct delivered energy from MAP battery
                            map_state.current_soc_wh = max(0.0, map_state.current_soc_wh - seg_energy)
                            self.map_movement_scheduler.record_map_soc(following_map, float(self.env.now))
                            self.map_tracker.record_charge(
                                map_id=following_map, bus_id=bus_id,
                                start_time=t_start, end_time=float(self.env.now),
                                energy_wh=seg_energy,
                                location=f"stop_{curr_stop_id}",
                                soc_before=soc_before, soc_after=new_charged_soc,
                                charging_type="en_route_segment"
                            )
                            if new_charged_soc >= target_wh or map_state.current_soc_wh <= 0:
                                self.map_movement_scheduler.stop_following(following_map)
                                following_map = None

                    # Always update MAP location to current stop (even if just detached)
                    self.map_movement_scheduler.update_location_following_bus(
                        was_following_map, curr_stop_id, curr_lat, curr_lon)

                # 3. If no MAP following and SOC is below threshold, request one
                if (self.map_movement_scheduler.get_following_map_for_bus(bus_id) is None
                        and self.num_maps > 0
                        and self.bus_soc[bus_id] < BUS_CHARGE_THRESHOLD_SOC * capacity):
                    available_map = self.map_movement_scheduler.get_available_map(
                        curr_stop_id, curr_lat, curr_lon)
                    if available_map is not None:
                        # Spawn at bus location (0 time) or travel (recorded but non-blocking)
                        self.map_movement_scheduler.spawn_or_travel_to(
                            available_map, curr_stop_id, curr_lat, curr_lon)
                        self.map_movement_scheduler.start_following(available_map, bus_id)

                self.bus_soc_history[bus_id].append(
                    (self.env.now, self.bus_soc.get(bus_id, capacity)))

            prev_end_node = self.sim.stop_node.get(seq[-1].get("stop_id"))
            prev_end_stop_id = seq[-1].get("stop_id")
            prev_end_time = seq[-1]["arrival"]

    def _collect_statistics(self) -> Dict:
        """Collect statistics"""

        bus_stats = {}
        min_soc_overall = float('inf')

        for bus_id, info in self.buses.items():
            soc_ratio = info['soc_wh'] / self.battery_capacity_wh
            min_soc_ratio = info['min_soc_wh'] / self.battery_capacity_wh

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

        return {
            'battery_capacity_wh': self.battery_capacity_wh,
            'num_maps': self.num_maps,
            'charging_enabled': self.num_maps > 0,
            'preemption_threshold': self.preemption_threshold,
            'trip_change_stops': len(self.trip_change_stops),
            'num_buses': len(self.buses),
            'num_layovers': len(self.layovers),
            'num_preemptions': 0,  # No preemption in dynamic en-route charging model
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

        records = self.map_tracker.charging_records

        if not records:
            print("\n" + "="*70)
            print("NO CHARGING EVENTS")
            print("="*70 + "\n")
            return

        print("\n" + "="*70)
        print(f"CHARGING EVENTS (First {num_to_print})")
        print("="*70)
        print(f"\n{'Bus ID':<20} {'MAP ID':<8} {'Start Time (s)':<18} {'Duration (s)':<15} {'Energy (Wh)':<15} {'Type':<20}")
        print("-"*115)

        for record in records[:num_to_print]:
            print(f"{record.bus_id:<20} {record.map_id:<8} "
                  f"{record.start_time_s:<18.1f} {record.duration_s:<15.1f} "
                  f"{record.energy_delivered_wh:<15.0f} {record.charging_type:<20}")

        if len(records) > num_to_print:
            print(f"\n... and {len(records) - num_to_print} more events")

        print(f"\nTotal charging events: {len(records)}")

    def print_preemption_events(self):
        """Print all preemption events (none in dynamic en-route charging model)"""

        print("\n" + "="*70)
        print("NO PREEMPTION EVENTS")
        print("  (Dynamic en-route charging model does not use preemption)")
        print("="*70 + "\n")

    def print_map_usage(self):
        """Print MAP usage statistics"""
        self.map_tracker.print_summary(self.battery_capacity_wh)

    def print_map_assignments(self):
        """Print which buses are charged by each MAP"""
        self.map_tracker.print_detailed_assignments()

    def print_map_movement(self):
        """Print MAP movement statistics"""
        summary = self.map_movement_scheduler.get_summary()

        # Column widths: 8+10+12+15+20+12+15+15 = 107 chars + 7 spaces = 114
        _SEP_WIDTH = 114

        print("\n" + "="*70)
        print("MAP MOVEMENT STATISTICS")
        print("="*70)
        print(f"\n{'MAP ID':<8} {'Spawned':<10} {'Recharging':<12} {'Following Bus':<15} "
              f"{'Current Location':<20} {'SOC':<12} {'Distance (km)':<15} {'Charging Events':<15}")
        print("-"*_SEP_WIDTH)

        for map_id, state_summary in summary.items():
            distance_km = state_summary['distance_traveled_m'] / 1000
            loc = state_summary['current_location'] or "unspawned"
            spawned_str = "Yes" if state_summary.get('spawned') else "No"
            recharging_str = "Yes" if state_summary.get('is_recharging') else "No"
            following_str = str(state_summary.get('assigned_bus') or "-")
            soc_str = f"{state_summary['current_soc_pct']:.1f}%"
            print(f"{map_id:<8} {spawned_str:<10} {recharging_str:<12} {following_str:<15} "
                  f"{loc:<20} {soc_str:<12} {distance_km:<15.1f} "
                  f"{state_summary['num_charging_events']:<15}")

        print(f"\n{'='*70}\n")

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

    def plot_map_soc(self, save_path: str = "map_soc_trajectories.png"):
        """Plot MAP battery SOC (%) versus simulation time for each MAP."""

        has_data = any(
            len(self.map_movement_scheduler.map_soc_history.get(mid, [])) > 1
            for mid in range(self.num_maps)
        )
        if not has_data:
            print("No MAP SOC history to plot")
            return

        plt.figure(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, max(self.num_maps, 1)))

        for map_id in range(self.num_maps):
            history = self.map_movement_scheduler.map_soc_history.get(map_id, [])
            if len(history) < 2:
                continue
            cap_wh = self.map_movement_scheduler.map_states[map_id].battery_capacity_wh
            times = [h[0] / 3600 for h in history]
            soc_pct = [h[1] / cap_wh * 100.0 for h in history]
            plt.plot(times, soc_pct, linewidth=2, label=f'MAP {map_id} (cap {cap_wh/1000:.0f} kWh)',
                     color=colors[map_id])

        plt.axhline(y=100.0, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='100 %')
        plt.axhline(y=0.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='0 %')
        plt.ylim(-5, 110)
        plt.xlabel('Time (hours)', fontweight='bold', fontsize=12)
        plt.ylabel('MAP Battery SOC (%)', fontweight='bold', fontsize=12)
        plt.title('MAP Battery State of Charge vs Time', fontweight='bold', fontsize=13)
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
                                    map_battery_capacity_wh: Optional[Union[float, List[float]]] = None):
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

    # Show en-route vs terminal breakdown
    ct_breakdown = stage2.map_tracker.get_summary().get('charging_type_breakdown', {})
    if ct_breakdown:
        print(f"\nCharging breakdown by type:")
        for ctype, stats in sorted(ct_breakdown.items()):
            print(f"  {ctype:<20}: {stats['events']:>5} events, "
                  f"{stats['energy_wh']/1000:,.1f} kWh")

    print("\n" + "="*70)

    # Plots
    stage2.plot_soc(save_path="bus_soc_terminal_charging_optimized.png")
    stage2.plot_map_energy_delivery(save_path="map_energy_delivery.png")
    stage2.plot_cumulative_energy_delivery(save_path="cumulative_energy_delivery.png")
    stage2.plot_map_movement(save_path="map_movement_distance.png")  # NEW: Plot MAP movement
    stage2.plot_map_soc(save_path="map_soc_trajectories.png")       # NEW: Plot MAP SOC vs time

    return results, stage2