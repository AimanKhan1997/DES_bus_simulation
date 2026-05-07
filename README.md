# DES Bus Simulation

A Discrete Event Simulation (DES) framework coupled with Mixed-Integer Linear Programming (MILP) optimization for planning electric bus charging infrastructure. The system simulates daily bus operations on Stockholm transit lines, models energy consumption and Mobile Access Point (MAP) scheduling, and optimizes battery capacities and infrastructure costs.

## Overview

Electric bus fleet planning requires balancing battery sizing, charging infrastructure investment, and operational feasibility. This project addresses that problem by:

1. **Simulating** a full day of bus operations using a SimPy-based discrete event simulation that tracks energy consumption, state-of-charge (SOC), and charging events.
2. **Optimizing** battery capacities and MAP (Mobile Access Point) deployment using a Gurobi-based MILP solver that minimizes total system cost.
3. **Iterating** between simulation and optimization in a feedback loop to converge on a feasible, cost-optimal solution.

## Architecture

```
┌───────────────────────────────────────────────┐
│           run_optimization.py (Entry Point)   │
│            MILP ↔ Simulation Feedback Loop     │
└──────────┬──────────────────────┬─────────────┘
           │                      │
    ┌──────▼──────┐      ┌───────▼──────────┐
    │  Stage 2 DES │      │  Gurobi MILP     │
    │  Simulation  │◄────►│  Optimizer       │
    └──────┬──────┘      └──────────────────┘
           │
    ┌──────▼──────────────────────────────┐
    │  Stage2DESTerminalChargingPreemptive │
    │  (integration_stage2.py)            │
    │  • Bus movement & energy tracking   │
    │  • Terminal charging simulation     │
    │  • Preemptive charging strategies   │
    │  • MAP movement scheduling          │
    └──────┬──────────────────────────────┘
           │
    ┌──────▼─────────────────┐
    │  GTFSBusSim (DES_model) │
    │  • SimPy bus processes  │
    │  • Energy consumption   │
    │  • Charging management  │
    └──────┬─────────────────┘
           │
     ┌─────┼──────────────┐
     │     │              │
 ┌───▼──┐ ┌▼────────┐ ┌──▼──────────┐
 │ GTFS │ │OSM Graph│ │Trip Assign  │
 │Loader│ │(Routing)│ │(Scheduling) │
 └──────┘ └─────────┘ └─────────────┘
```

## Project Structure

| File | Description |
|------|-------------|
| `run_optimization.py` | Main entry point. Orchestrates the MILP ↔ Simulation feedback loop. |
| `integration_stage2.py` | Stage 2 DES simulation with terminal charging, preemptive strategies, and MAP scheduling. |
| `post_simulation_milp.py` | Gurobi-based MILP optimizer for battery capacity and infrastructure cost minimization. |
| `DES_model.py` | Core simulation framework (`GTFSBusSim`). Handles bus processes, energy consumption, and GTFS integration. |
| `advanced_heuristics.py` | Rolling-horizon MAP scheduler with dynamic charge thresholds (alternative to fixed 70%/80%). |
| `gtfs_loader.py` | GTFS data loader with service calendar and date filtering. |
| `Trip_assign.py` | Trip-to-bus assignment algorithm based on line, turnover time, and schedule. |
| `osm_graph.py` | OpenStreetMap graph wrapper using OSMnx/NetworkX for routing and stop snapping. |

## Prerequisites

- **Python 3.10+**
- **Gurobi** optimizer with a valid license (required for MILP optimization)

### Python Dependencies

| Package | Purpose |
|---------|---------|
| `simpy` | Discrete event simulation engine |
| `pandas` | GTFS data parsing and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Visualization (SOC plots, heatmaps) |
| `networkx` | Graph algorithms for routing |
| `osmnx` | OpenStreetMap street network loading |
| `geopandas` | Geospatial data handling |
| `shapely` | Geometric operations |
| `pyproj` | Geographic coordinate transformations |
| `gurobipy` | Gurobi MILP solver Python interface |

Install dependencies with:

```bash
pip install simpy pandas numpy matplotlib networkx osmnx geopandas shapely pyproj gurobipy
```

## Input Data

The simulation requires two data sources placed in the repository root:

### GTFS Data (`gtfs_data/`)

Standard [GTFS](https://gtfs.org/) transit feed files:

- `stops.txt` — Transit stop locations (lat/lon)
- `stop_times.txt` — Trip stop sequences with arrival/departure times
- `trips.txt` — Trip metadata (route, service ID, direction)
- `routes.txt` — Route definitions (route short name)
- `shapes.txt` — Route geometry (lat/lon point sequences)
- `calendar.txt` — Service calendar (date ranges, day-of-week flags)
- `calendar_dates.txt` — Service exceptions and overrides

### OpenStreetMap Network (`map.xml`)

An OSM XML export covering the transit area, used for shortest-path routing between stops.

## Usage

### Full Optimization (MILP ↔ Simulation Loop)

Runs iterative optimization to find cost-optimal battery capacities and MAP deployment:

```bash
python run_optimization.py
```

The loop runs up to 50 iterations:
1. Simulates bus operations with current parameters
2. Extracts energy and SOC data from the simulation
3. Solves the MILP to find cheaper feasible configurations
4. Updates parameters and re-simulates
5. Tracks the best feasible solution across all iterations

### Simulation Only

Runs a single simulation without optimization (useful for validating specific parameters):

```bash
python run_optimization.py --sim-only
```

## Configuration

Key parameters are defined as constants in the source files:

### Transit Lines (`run_optimization.py`)

```python
LINES = ["1", "2", "3", "4", "6"]           # Stockholm bus lines
DATE_STR = "20231108"                         # Simulation date (YYYYMMDD)
LINE_BATTERY_CAPACITIES_KWH = {               # Initial battery capacities (kWh)
    "1": 270, "2": 130, "3": 280, "4": 200, "6": 50
}
```

### Charging Thresholds (`integration_stage2.py`)

```python
BUS_CHARGE_THRESHOLD_SOC = 0.70   # Start charging below 70% SOC
BUS_CHARGE_CUTOFF_SOC = 0.80      # Stop charging above 80% SOC
BUS_MIN_SOC = 0.20                # Minimum allowed SOC (20%)
MAP_CHARGING_RATE_WH_S = 97.22   # MAP charging rate (Wh/s)
MAX_CONCURRENT_CHARGERS = 2       # Max simultaneous charging sessions
```

### MILP Cost Parameters (`post_simulation_milp.py`)

```python
BATTERY_COST_PER_KWH = 115        # $/kWh for bus/MAP batteries
MAP_HARDWARE_COST = 40_000        # $ per MAP unit
OVERNIGHT_CHARGING_HOURS = 4.0    # Hours available for overnight depot charging
```

### Energy Model (`DES_model.py`)

Energy consumption per meter is a linear function of battery capacity:

```
rate = 2.7 − (470 − capacity_kwh) × 0.0005  Wh/m
```

## Output

### Visualizations

The simulation generates the following plots (saved as PNG at 300 DPI):

| Plot | Description |
|------|-------------|
| `bus_soc_terminal_charging_optimized.png` | Bus SOC trajectories over the simulation day, color-coded by line |
| `map_energy_delivery.png` | Energy delivered by each MAP to buses |
| `cumulative_energy_delivery.png` | Cumulative energy supplied by each MAP over time |
| `map_movement_distance.png` | Distance traveled by each MAP |
| `map_soc_over_time.png` | MAP battery SOC trajectories |
| `map_self_charge_heatmap.png` | Geographic heatmap of MAP self-charging events on the OSM network |

### Console Output

- Feasibility report (whether minimum SOC constraints are satisfied)
- Per-line charging statistics and preemption events
- MAP usage summary (assignments, movement, energy delivery)
- Cost breakdown (battery, hardware, overnight charging infrastructure)
- MILP solver status and recommended parameters per iteration

## Key Concepts

- **MAP (Mobile Access Point)**: A mobile charging unit that travels between stops to charge buses en route. MAPs have their own battery and can self-charge at designated locations.
- **Terminal Charging**: Charging that occurs only at terminal (end-of-line) stops during bus layovers.
- **Dynamic En-Route Charging**: MAPs can charge buses dynamically at any location — during segments (between stops), at stops, and during layovers. When a bus requests a charge, a MAP travels to it and stays attached, charging continuously until the desired SOC level is reached. See [`README_integration_stage2.md`](README_integration_stage2.md) for details.
- **Preemptive Charging**: Interrupting a bus's trip schedule to perform emergency charging when SOC drops critically low.
- **Rolling-Horizon Scheduling**: An advanced heuristic that dynamically adjusts charging thresholds based on predicted future energy needs rather than using fixed SOC thresholds. See [`README_advanced_heuristics.md`](README_advanced_heuristics.md) for details.
- **Overnight Charging**: Depot charging that occurs during off-service hours, with infrastructure costs based on tiered charger pricing ($/kW).

## Module Documentation

- **[`README_integration_stage2.md`](README_integration_stage2.md)** — Detailed documentation for the Stage 2 DES simulation, including dynamic en-route MAP charging at segments, stops, and layovers.
- **[`README_advanced_heuristics.md`](README_advanced_heuristics.md)** — Documentation for the rolling-horizon MAP scheduling algorithm with dynamic thresholds, priority scoring, and MAP selection.
