"""
Advanced MAP Charging Heuristics Module

Replaces the greedy heuristic with a state-of-the-art rolling-horizon
scheduling algorithm that considers:
  - Number of buses and MAPs
  - Current SOC of every bus and MAP
  - Battery capacities of buses and MAPs
  - Energy required by each bus for remaining trips
  - Schedule and location of each bus
  - Dynamic start/stop charging thresholds (not fixed 70%/80%)

Objectives:
  - No bus drops below 20% SOC
  - No MAP drops below 10% SOC
  - Minimize total energy waste and MAP travel distance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

# Constraints (same as integration_stage2.py)
BUS_MIN_SOC = 0.20
MAP_MIN_SOC = 0.10
from integration_stage2 import energy_per_meter_for_capacity

# Default fallback thresholds (used only when capacity data is unavailable)
DEFAULT_CHARGE_START_PCT = 0.70
DEFAULT_CHARGE_TARGET_PCT = 0.80

# Priority scoring weights (SOC urgency, energy shortfall, fairness, contention)
W_URGENCY = 0.40
W_SHORTFALL = 0.30
W_FAIRNESS = 0.15
W_CONTENTION = 0.15

# MAP selection scoring weights (distance, deliverable, energy, fairness, urgency, conservation)
W_MAP_DISTANCE = 0.25
W_MAP_DELIVERABLE = 0.25
W_MAP_ENERGY = 0.20
W_MAP_FAIRNESS = 0.10
W_MAP_URGENCY = 0.10
W_MAP_CONSERVATION = 0.10


@dataclass
class ChargingDecision:
    """Result of an advanced charging decision."""
    should_charge: bool
    map_id: Optional[int]         # Which MAP to use
    target_soc_wh: float          # How much to charge the bus to (Wh)
    priority: float               # Priority score (higher = more urgent)
    reason: str                   # Human-readable reason for the decision


class AdvancedMAPScheduler:
    """Rolling-horizon MAP charging scheduler.

    At every layover decision point the scheduler:
      1. Estimates remaining energy needs for each bus.
      2. Computes an urgency score per bus based on predicted future SOC.
      3. Determines a dynamic charging threshold instead of fixed 70%/80%.
      4. Selects the best MAP considering distance, energy, and system-wide
         resource contention.
      5. Computes a per-bus target SOC that balances charging depth with MAP
         energy conservation.
    """

    def __init__(
        self,
        sim,
        bus_trips_dict: Dict[str, List[str]],
        line_battery_capacities_wh: Dict[str, float],
        default_battery_capacity_wh: float,
        num_maps: int,
        map_battery_capacity_wh: float,
        map_movement_scheduler,            # MAPMovementScheduler instance
        charging_rate_wh_s: float = 97.22,
    ):
        self.sim = sim
        self.bus_trips_dict = bus_trips_dict
        self.line_battery_capacities_wh = line_battery_capacities_wh
        self.default_battery_capacity_wh = default_battery_capacity_wh
        self.num_maps = num_maps
        self.map_battery_capacity_wh = map_battery_capacity_wh
        self.map_sched = map_movement_scheduler
        self.charging_rate_wh_s = charging_rate_wh_s

        # Pre-compute per-bus energy profiles
        self._bus_trip_energy: Dict[str, List[Tuple[float, float, str]]] = {}
        self._bus_total_energy: Dict[str, float] = {}
        self._precompute_bus_energy()

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_bus_energy(self):
        """Pre-compute energy consumption per trip for every bus.

        Stores a list of (trip_start_time_s, trip_energy_wh, first_stop_id)
        sorted by start time.
        """
        for bus_id, trip_ids in self.bus_trips_dict.items():
            trip_profiles: List[Tuple[float, float, str]] = []

            # Determine per-line capacity and energy rate for this bus
            try:
                line_id = bus_id.split('_')[0].replace('line', '')
            except Exception:
                line_id = "unknown"
            bus_cap = self.line_battery_capacities_wh.get(
                line_id, self.default_battery_capacity_wh)
            bus_epm = energy_per_meter_for_capacity(bus_cap / 1000.0)

            sorted_trips = sorted(
                trip_ids,
                key=lambda t: (
                    self.sim.gtfs.stop_sequence(t)[0]["arrival"]
                    if self.sim.gtfs.stop_sequence(t)
                    else float('inf')
                )
            )

            total = 0.0
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
                                curr_stop["lon"], curr_stop["lat"],
                            )
                            dist = abs(dist)
                        except Exception:
                            dist = 0.0
                        trip_energy += dist * bus_epm

                    start_time = seq[0]["arrival"]
                    first_stop = seq[0].get("stop_id", "unknown")
                    trip_profiles.append((start_time, trip_energy, first_stop))
                    total += trip_energy
                except Exception:
                    continue

            self._bus_trip_energy[bus_id] = trip_profiles
            self._bus_total_energy[bus_id] = total

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide_charging(
        self,
        bus_id: str,
        line_id: str,
        stop_id: str,
        soc_wh: float,
        capacity_wh: float,
        current_time_s: float,
        layover_duration_s: float,
        all_bus_soc: Dict[str, float],
        line_charge_counts: Dict[str, int],
    ) -> ChargingDecision:
        """Decide whether and how to charge *bus_id* at *stop_id*.

        Returns a ``ChargingDecision``.
        """
        if self.num_maps <= 0:
            return ChargingDecision(False, None, soc_wh, 0.0,
                                   "No MAPs available")

        # 1. Remaining energy need for this bus
        remaining_energy = self._remaining_energy(bus_id, current_time_s)
        soc_ratio = soc_wh / capacity_wh if capacity_wh > 0 else 1.0

        # 2. Dynamic threshold: decide whether this bus needs charging NOW
        dyn_start, dyn_target = self._dynamic_thresholds(
            bus_id, soc_wh, capacity_wh, remaining_energy, current_time_s
        )

        if soc_ratio >= dyn_start:
            return ChargingDecision(False, None, soc_wh, 0.0,
                                   f"SOC {soc_ratio:.1%} >= dynamic start "
                                   f"threshold {dyn_start:.1%}")

        # 3. Compute priority
        priority = self._compute_priority(
            bus_id, line_id, soc_wh, capacity_wh,
            remaining_energy, current_time_s, all_bus_soc, line_charge_counts,
        )

        # 4. Select best MAP
        map_id = self._select_best_map(
            bus_id, line_id, stop_id, soc_wh, capacity_wh,
            layover_duration_s, line_charge_counts,
        )

        if map_id is None:
            return ChargingDecision(False, None, soc_wh, 0.0,
                                   "No available MAP with sufficient energy")

        # 5. Compute target SOC
        target_soc_wh = self._compute_target_soc(
            bus_id, soc_wh, capacity_wh, remaining_energy,
            layover_duration_s, map_id,
        )

        return ChargingDecision(
            should_charge=True,
            map_id=map_id,
            target_soc_wh=target_soc_wh,
            priority=priority,
            reason=(f"Dynamic threshold: start<={dyn_start:.1%}, "
                    f"target={target_soc_wh/capacity_wh:.1%}, "
                    f"priority={priority:.3f}"),
        )

    # ------------------------------------------------------------------
    # Dynamic thresholds
    # ------------------------------------------------------------------

    def _dynamic_thresholds(
        self,
        bus_id: str,
        soc_wh: float,
        capacity_wh: float,
        remaining_energy_wh: float,
        current_time_s: float,
    ) -> Tuple[float, float]:
        """Compute dynamic start/target SOC ratios for a bus.

        Instead of fixed 70%/80%, the thresholds adapt to the bus's future
        energy requirements and current state-of-charge.

        Returns (start_ratio, target_ratio) where values are in [0, 1].
        """

        if capacity_wh <= 0:
            return (DEFAULT_CHARGE_START_PCT, DEFAULT_CHARGE_TARGET_PCT)

        # Trip look-ahead: how many remaining trips?
        remaining_trips = self._remaining_trips(bus_id, current_time_s)
        trip_count = len(remaining_trips)

        if trip_count == 0:
            # No more trips - no need to charge
            return (1.0, 1.0)

        # Average energy per remaining trip
        avg_trip_energy = (remaining_energy_wh / trip_count) if trip_count > 0 else 0.0

        # Compute the worst-case SOC after the next trip(s) without charging
        # Look ahead at the next 2 trips for safety margin
        lookahead_energy = 0.0
        for _, energy, _ in remaining_trips[:2]:
            lookahead_energy += energy

        # The bus needs enough SOC to survive the look-ahead trips plus
        # a safety margin equal to 20% (BUS_MIN_SOC)
        min_safe_soc = BUS_MIN_SOC * capacity_wh + lookahead_energy

        # Dynamic start threshold: charge when SOC is at risk of dropping
        # below min_safe_soc within the next few trips.
        # This is higher when future energy needs are large.
        start_ratio = min(0.95, max(BUS_MIN_SOC + 0.05,
                                    min_safe_soc / capacity_wh))

        # Dynamic target: charge enough to survive all remaining trips,
        # but cap at a reasonable level to conserve MAP energy.
        # Balance between full charge and minimal needed charge.
        energy_deficit_ratio = remaining_energy_wh / capacity_wh
        # The target should ensure the bus can survive at least 3 more trips
        # even if no further charging opportunity exists
        trips_buffer = min(trip_count, 3)
        buffer_energy = avg_trip_energy * trips_buffer
        target_soc_needed = (BUS_MIN_SOC * capacity_wh + buffer_energy) / capacity_wh

        target_ratio = min(0.95, max(start_ratio + 0.05, target_soc_needed))

        return (start_ratio, target_ratio)

    # ------------------------------------------------------------------
    # Priority computation
    # ------------------------------------------------------------------

    def _compute_priority(
        self,
        bus_id: str,
        line_id: str,
        soc_wh: float,
        capacity_wh: float,
        remaining_energy_wh: float,
        current_time_s: float,
        all_bus_soc: Dict[str, float],
        line_charge_counts: Dict[str, int],
    ) -> float:
        """Multi-criteria priority scoring.

        Higher score = more urgent need for charging.

        Factors:
          - SOC urgency (lower SOC = higher priority)
          - Remaining energy vs. current SOC (energy shortfall)
          - Line fairness (underserved lines boosted)
          - System-wide contention (buses competing for MAPs)
        """

        soc_ratio = soc_wh / capacity_wh if capacity_wh > 0 else 1.0

        # Factor 1: SOC urgency (exponential increase near BUS_MIN_SOC)
        # Maps SOC from 1.0->BUS_MIN_SOC to urgency 0->1
        urgency = max(0.0, 1.0 - (soc_ratio - BUS_MIN_SOC) / (1.0 - BUS_MIN_SOC))
        # Exponential boost for critically low SOC
        if soc_ratio < BUS_MIN_SOC + 0.1:
            urgency = min(1.0, urgency * 2.0)

        # Factor 2: Energy shortfall — can the bus finish its day?
        if capacity_wh > 0 and remaining_energy_wh > 0:
            shortfall = max(0.0, remaining_energy_wh - soc_wh) / capacity_wh
        else:
            shortfall = 0.0

        # Factor 3: Line fairness (underserved lines get priority)
        line_count = line_charge_counts.get(line_id, 0)
        fairness = 1.0 / (1.0 + line_count)

        # Factor 4: System contention — how many other buses are also low?
        low_soc_buses = sum(
            1 for bid, soc in all_bus_soc.items()
            if bid != bus_id and soc / max(1.0, self._get_bus_capacity(bid)) < 0.5
        )
        # If many buses are low, prioritize the worst-off
        contention_boost = 1.0 + 0.1 * max(0, low_soc_buses - self.num_maps)

        # Weighted combination
        score = (W_URGENCY * urgency
                 + W_SHORTFALL * shortfall
                 + W_FAIRNESS * fairness
                 + W_CONTENTION * (contention_boost - 1.0))

        return score

    # ------------------------------------------------------------------
    # MAP selection
    # ------------------------------------------------------------------

    def _select_best_map(
        self,
        bus_id: str,
        line_id: str,
        stop_id: str,
        soc_wh: float,
        capacity_wh: float,
        layover_duration_s: float,
        line_charge_counts: Dict[str, int],
    ) -> Optional[int]:
        """Select the optimal MAP using multi-criteria scoring.

        Considers:
          - MAP availability (not self-charging, has energy above 10%)
          - Distance / travel time from MAP to bus stop
          - MAP remaining energy (enough to deliver meaningful charge)
          - Line fairness (underserved lines get priority)
          - Travel-time feasibility (MAP can reach in time for the layover)
          - MAP energy conservation (avoid draining a MAP that other
            buses will need soon)
        """
        if self.num_maps <= 0:
            return None

        best_map = None
        best_score = -1.0

        for map_id in range(self.num_maps):
            if not self.map_sched.is_map_available(map_id):
                continue

            avail = self.map_sched.available_energy_wh(map_id)
            if avail <= 0:
                continue

            # Distance & travel time
            map_loc = self.map_sched.get_map_location(map_id) or "depot"
            dist_m, travel_time_s = self.map_sched.calculate_travel_time(
                map_loc, stop_id
            )

            # Can the MAP reach in time?
            if travel_time_s > layover_duration_s * 0.9:
                continue  # Won't arrive with meaningful time to charge

            remaining_charge_time = max(0, layover_duration_s - travel_time_s)
            deliverable = min(avail, remaining_charge_time * self.charging_rate_wh_s)
            if deliverable <= 0:
                continue

            # --- Scoring components ---

            # Distance score (1.0 when co-located, decays with distance)
            distance_score = 1.0 / (1.0 + dist_m / 1000.0)

            # Energy score — prefer MAPs with enough energy for a good charge
            cap_map = self.map_sched.map_states[map_id].battery_capacity_wh
            energy_ratio = avail / cap_map if cap_map > 0 else 0.0
            energy_score = energy_ratio

            # Deliverable score — how much can actually be transferred?
            need = max(0, capacity_wh * 0.8 - soc_wh)  # rough need
            deliver_score = min(1.0, deliverable / max(1.0, need))

            # Fairness
            line_count = line_charge_counts.get(line_id, 0)
            fairness_score = 1.0 / (1.0 + line_count)

            # Bus urgency
            soc_ratio = soc_wh / capacity_wh if capacity_wh > 0 else 1.0
            urgency = max(0.01, 1.0 - soc_ratio)

            # Energy conservation: penalize draining a MAP below 30%
            # so it remains useful for future buses
            conservation_penalty = 1.0
            future_ratio = (avail - deliverable) / cap_map if cap_map > 0 else 0.0
            if future_ratio < 0.2:
                conservation_penalty = 0.5 + future_ratio * 2.5  # 0.5-1.0

            score = (W_MAP_DISTANCE * distance_score
                     + W_MAP_DELIVERABLE * deliver_score
                     + W_MAP_ENERGY * energy_score
                     + W_MAP_FAIRNESS * fairness_score
                     + W_MAP_URGENCY * urgency
                     + W_MAP_CONSERVATION * conservation_penalty)

            if score > best_score:
                best_score = score
                best_map = map_id

        return best_map

    # ------------------------------------------------------------------
    # Target SOC computation
    # ------------------------------------------------------------------

    def _compute_target_soc(
        self,
        bus_id: str,
        soc_wh: float,
        capacity_wh: float,
        remaining_energy_wh: float,
        layover_duration_s: float,
        map_id: int,
    ) -> float:
        """Compute the target SOC for this charging session.

        Instead of always charging to a fixed 80%, the target adapts to:
          - How much energy the bus needs for its remaining trips
          - How much the MAP can deliver within the layover
          - How much energy the MAP should conserve for other buses
        """

        # Maximum energy the MAP can deliver (limited by its SOC and rate)
        avail_map = self.map_sched.available_energy_wh(map_id)
        max_deliverable = min(avail_map, layover_duration_s * self.charging_rate_wh_s)

        # Energy the bus needs to finish its day with >= 20% SOC
        safe_margin = BUS_MIN_SOC * capacity_wh
        total_needed = remaining_energy_wh + safe_margin
        energy_gap = max(0, total_needed - soc_wh)

        # Target 1: Charge to cover the gap fully
        target_full = soc_wh + energy_gap

        # Target 2: Charge to 90% capacity (generous but not wasteful)
        target_generous = 0.90 * capacity_wh

        # Target 3: Only charge what the MAP can deliver
        target_feasible = soc_wh + max_deliverable

        # Pick the most reasonable target: enough for safety, but not more
        # than MAP can deliver, and not more than the generous cap.
        target = min(target_full, target_generous, target_feasible)

        # Ensure target is at least above current SOC + minimum useful charge
        min_useful_charge = 0.02 * capacity_wh  # 2% minimum
        target = max(soc_wh + min_useful_charge, target)

        # Hard cap at bus capacity
        target = min(target, capacity_wh)

        return target

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remaining_energy(self, bus_id: str, current_time_s: float) -> float:
        """Total energy still needed by *bus_id* from *current_time_s* onward."""
        trips = self._bus_trip_energy.get(bus_id, [])
        return sum(e for t, e, _ in trips if t > current_time_s)

    def _remaining_trips(
        self, bus_id: str, current_time_s: float
    ) -> List[Tuple[float, float, str]]:
        """Return list of (start_time, energy, first_stop_id) for remaining trips."""
        trips = self._bus_trip_energy.get(bus_id, [])
        return [(t, e, s) for t, e, s in trips if t > current_time_s]

    def _get_bus_capacity(self, bus_id: str) -> float:
        """Return battery capacity for a bus (Wh)."""
        try:
            line_id = bus_id.split('_')[0].replace('line', '')
        except Exception:
            line_id = "unknown"
        return self.line_battery_capacities_wh.get(
            line_id, self.default_battery_capacity_wh)
