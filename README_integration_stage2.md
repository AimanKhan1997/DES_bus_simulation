# Integration Stage 2 — DES Terminal & Dynamic MAP Charging

`integration_stage2.py` is the primary simulation module implementing **Stage 2** of the electric-bus Discrete Event Simulation (DES). It extends the base `DES_model.py` with terminal charging, preemptive strategies, MAP movement scheduling, and **dynamic en-route charging** across segments, stops, and layovers.

## Overview

```
Bus lifecycle (per day)
───────────────────────────────────────────────────────────
 Layover ──▶ Trip 1 ──▶ Layover ──▶ Trip 2 ──▶ ... ──▶ End

 Each phase supports MAP charging:
   • Layover:  MAP travels to bus, charges during idle time
   • Segment:  MAP follows bus, charges while moving
   • Stop:     MAP charges bus during dwell at any stop
```

The module simulates a full day (86 400 s) of bus operations using [SimPy](https://simpy.readthedocs.io/) and tracks energy consumption, state-of-charge (SOC), MAP battery levels, charging events, preemptions, and layovers.

---

## Key Classes

### `Stage2DESTerminalChargingPreemptive`

The main simulation class. Orchestrates bus processes, MAP scheduling, and charging decisions.

| Attribute | Type | Description |
|-----------|------|-------------|
| `sim` | `GTFSBusSim` | Reference to the core simulation (GTFS, OSM, geodesic). |
| `bus_trips_dict` | `Dict[str, List[str]]` | Mapping of `bus_id` → list of GTFS `trip_id`s. |
| `bus_lines` | `Dict[str, str]` | Mapping of `bus_id` → `line_id`. |
| `trip_change_stops` | `set` | Designated stops where trip changes occur. |
| `num_maps` | `int` | Number of MAPs available. |
| `battery_capacity_wh` | `float` | Default battery capacity (Wh). |
| `line_battery_capacities_wh` | `Dict[str, float]` | Per-line battery capacities (Wh). |
| `map_battery_capacity_wh` | `float` | MAP battery capacity (Wh). Default 150 000. |
| `use_advanced_heuristics` | `bool` | Use rolling-horizon scheduler. |

#### Methods

| Method | Description |
|--------|-------------|
| `run_simulation(duration_s)` | Run the full DES for `duration_s` seconds (default 86 400). Returns a statistics dict. |
| `_simulate_bus(bus_id, trip_ids)` | SimPy generator process for a single bus (layovers, trips, segments). |
| `_select_best_map(bus_id, ...)` | Greedy MAP selection (distance, energy, fairness, urgency). |
| `_check_and_request_dynamic_charging(...)` | Request a MAP dynamically during any phase (segment, stop, layover). |
| `_apply_enroute_charging(bus_id, duration_s, ...)` | Apply charging from an attached MAP for a given time window. |
| `_detach_map_from_bus(bus_id)` | Release an attached MAP once target SOC is reached. |
| `plot_soc(save_path)` | Plot bus SOC trajectories over time. |
| `plot_map_energy_delivery(save_path)` | Plot energy delivered by each MAP. |
| `plot_cumulative_energy_delivery(save_path)` | Plot cumulative energy delivery over time. |
| `plot_map_movement(save_path)` | Plot cumulative MAP distance traveled. |
| `plot_map_soc(save_path)` | Plot MAP battery SOC over time. |
| `plot_map_self_charge_heatmap(save_path)` | Heatmap of MAP self-charge events on the OSM network. |

---

### `PreemptiveStopChargingManager`

Manages exclusive charging slots with MAP tracking, queuing, and preemption.

| Feature | Description |
|---------|-------------|
| **Global concurrency** | At most `num_maps` buses charge simultaneously across the network. |
| **Preemption** | When a critically low bus arrives, it can interrupt a higher-SOC bus's session. |
| **Queuing** | High-priority and normal queues per stop. Requests are served by SOC urgency. |
| **MAP energy tracking** | Energy is eagerly deducted from the MAP so concurrent checks see correct SOC. |
| **Fairness** | Per-line charging counts prevent any single line from monopolising MAPs. |

### `MAPMovementScheduler`

Handles MAP location tracking, movement, SOC management, and self-charging.

| Feature | Description |
|---------|-------------|
| **Location tracking** | Each MAP has a `current_location` (stop ID or `"depot"`). |
| **Travel time** | Geodesic distance ÷ `map_speed_ms` (default 37.78 m/s ≈ 136 km/h). |
| **SOC management** | Energy deducted on charge delivery; floor at `MAP_MIN_SOC` (10%). |
| **Self-charging** | When MAP energy reaches the floor, a SimPy process recharges it at 233 Wh/s. |

### `MAPUsageTracker`

Collects detailed records for post-simulation analysis.

| Record Type | Fields |
|-------------|--------|
| `MAPChargingRecord` | map_id, bus_id, start/end time, energy, location, SOC before/after |
| `MAPMovementRecord` | map_id, start/end time, from/to location, distance, associated bus |
| `MAPSelfChargeRecord` | map_id, start/end time, location, SOC before/after |

### `PreemptionStrategyAnalyzer`

Analyses bus energy consumption patterns before the simulation to recommend an optimal preemption threshold.

---

## Dynamic En-Route Charging

### Concept

MAPs can now charge buses **wherever they are** — not only during layovers. When a bus's SOC drops below the charging threshold during a segment (between stops), at a stop, or during a layover, the bus can request a MAP. The MAP travels to the bus and stays **attached**, continuously charging across multiple segments and stops until the target SOC is reached or the MAP's energy is depleted.

### How It Works

```
Bus SOC drops below threshold during a segment
            │
            ▼
_check_and_request_dynamic_charging()
   │  Evaluates bus SOC against thresholds
   │  Selects best MAP (greedy or advanced)
   │  MAP travels to bus location
   │  MAP is attached: EnRouteChargingState stored
   │
   ▼
_apply_enroute_charging() — called each segment/stop
   │  Charges bus at MAP_CHARGING_RATE_WH_S (97.22 Wh/s)
   │  Deducts energy from MAP battery
   │  Records charging event with location_type
   │
   ▼
Target SOC reached  ──OR──  MAP depleted
   │                          │
   ▼                          ▼
_detach_map_from_bus()    MAP triggers self-charge
```

### Charging Locations

| Location | When | Duration |
|----------|------|----------|
| **Segment** | After bus moves between two stops | Travel time of the segment |
| **Stop** | When bus arrives at any stop with dwell time | Dwell time at the stop |
| **Layover** | During inter-trip idle periods | Full layover duration |

### Data Structure

```python
@dataclass
class EnRouteChargingState:
    map_id: int              # Which MAP is attached
    target_soc_wh: float     # Desired SOC level
    start_time_s: float      # When attachment began
    total_energy_charged_wh: float  # Cumulative energy delivered
    location_type: str       # "segment", "stop", or "layover"
```

---

## Parameters & Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `BUS_BATTERY_MAX_WH` | 470 000 | Maximum bus battery capacity (Wh) |
| `MAP_CHARGING_RATE_WH_S` | 97.22 | MAP charging rate (Wh/s = 350 kW) |
| `BUS_CHARGE_THRESHOLD_SOC` | 0.70 | Start charging below 70% SOC (greedy) |
| `BUS_CHARGE_CUTOFF_SOC` | 0.80 | Stop charging above 80% SOC (greedy) |
| `BUS_MIN_SOC` | 0.20 | Minimum allowed SOC (20%) |
| `MAX_CONCURRENT_CHARGERS` | 2 | Max simultaneous charging sessions |
| `CHARGER_SPEED_MS` | 37.78 | MAP travel speed (m/s ≈ 136 km/h) |
| `MAP_BATTERY_CAPACITY_WH` | 150 000 | MAP battery capacity (Wh) |
| `MAP_MIN_SOC` | 0.10 | MAP minimum SOC floor (10%) |
| `MAP_SELF_CHARGE_RATE_WH_S` | 233.0 | MAP self-charge rate (Wh/s) |

---

## Bus Lifecycle in `_simulate_bus`

```python
for each trip in bus_trips:
    # 1. LAYOVER (inter-trip idle)
    if idle_time > 0:
        # If MAP already attached → continue charging
        _apply_enroute_charging(bus_id, duration, capacity, "layover")

        # Else: evaluate need & optionally attach a new MAP
        # After layover charging, if target not met → MAP stays attached

    # 2. SEGMENTS (stop-to-stop movement)
    for each segment in trip:
        yield env.timeout(segment_travel_time)
        deduct energy consumption

        # Dynamic en-route charging during segment
        if MAP attached:
            _apply_enroute_charging(bus_id, dt, capacity, "segment")
        elif SOC < threshold:
            _check_and_request_dynamic_charging(...)
```

---

## Simulation Output

### Statistics Dictionary

The `run_simulation()` method returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `battery_capacity_wh` | float | Default battery capacity |
| `line_battery_capacities_wh` | dict | Per-line battery capacities |
| `num_maps` | int | Number of MAPs |
| `charging_enabled` | bool | Whether charging is active |
| `charging_strategy` | str | `"greedy"` or `"advanced"` |
| `preemption_threshold` | float | Preemption SOC threshold |
| `num_buses` | int | Total buses simulated |
| `num_layovers` | int | Total layover records |
| `num_preemptions` | int | Total preemption events |
| `min_soc_overall_ratio` | float | Lowest SOC ratio observed |
| `feasible` | bool | Whether min SOC ≥ 20% |
| `total_energy_charged_wh` | float | Total energy delivered by MAPs |
| `dynamic_charging` | dict | Dynamic charging breakdown (segment/stop/layover events and energy) |

### Generated Plots

| File | Description |
|------|-------------|
| `bus_soc_terminal_charging_optimized.png` | Bus SOC trajectories, color-coded by line |
| `map_energy_delivery.png` | Energy delivered by each MAP per event |
| `cumulative_energy_delivery.png` | Cumulative energy supplied over time |
| `map_movement_distance.png` | Cumulative MAP distance traveled |
| `map_soc_over_time.png` | MAP battery SOC trajectories |
| `map_self_charge_heatmap.png` | Geographic heatmap of self-charge events |

---

## Usage

### As Part of the Optimization Loop

```python
from integration_stage2 import run_terminal_charging_simulation

results, stage2 = run_terminal_charging_simulation(
    sim=sim,
    bus_trips_dict=bus_trips_dict,
    bus_lines=bus_lines,
    trip_change_stops=trip_change_stops,
    battery_capacity_wh=250000,
    num_maps=2,
    line_battery_capacities_wh={"1": 270000, "2": 130000},
    use_advanced_heuristics=True,
    skip_plots=False,
)
```

### Direct Instantiation

```python
from integration_stage2 import Stage2DESTerminalChargingPreemptive

stage2 = Stage2DESTerminalChargingPreemptive(
    sim=sim,
    bus_trips_dict=bus_trips_dict,
    bus_lines=bus_lines,
    trip_change_stops=trip_change_stops,
    initial_battery_capacity_wh=250000,
    num_maps=2,
    use_advanced_heuristics=False,
)

results = stage2.run_simulation(duration_s=86400)
stage2.plot_soc()
```
