# DES Bus Simulation — Experiments for
Joint Optimization of Bus Batteries and Mobile Autonomous Charging Pods for Urban Bus Electrification

This repository contains the experimental code used in the paper
"Joint Optimization of Bus Batteries and Mobile Autonomous Charging Pods for Urban Bus Electrification".
It implements a two-level feasibility study that couples a discrete-event simulation (DES) of bus operations with heuristics and optimization routines to jointly reason about bus battery sizing and Mobile Access Point (MAP) deployment and usage.

Short summary
- Main heuristic / experiment driver: `run_optimization.py` (constraint-driven search and refinement loop; also supports simulation-only mode).
- Discrete-event simulation (Stage 2): `integration_stage2.py` (SimPy-based terminal charging, preemptive charging logic, MAP simulation and plotting utilities).
- Rolling-horizon, advanced MAP scheduling heuristic: `advanced_heuristics.py` (a dynamic scheduler that replaces the greedy MAP heuristic).

This README summarizes how to run the experiments, reproduce the results, and what to expect from the outputs.

--

## Stack
- Language: Python (3.10+ recommended)
- Key libraries: simpy, pandas, numpy, networkx/osmnx, geopandas/shapely, gurobipy (optional for full MILP flows)

Notes: The core simulation is pure-Python and uses SimPy. Gurobi is only required if you want to run/extend MILP-based optimization routines — the provided constraint-driven search in `run_optimization.py` can run without calling a MILP solver.

## What is here (top-level files)
- `run_optimization.py` — Main experiment driver. Orchestrates simulation runs, constraint-driven feasibility search, and the greedy refinement phases. Also supports `--sim-only` mode to validate parameters.
- `integration_stage2.py` — Stage 2 discrete-event simulation (terminal charging, preemptive charging, MAP movement/energy tracking, plotting helpers).
- `advanced_heuristics.py` — Rolling-horizon MAP scheduler (dynamic thresholds, multi-criteria MAP selection). Can be integrated into the stage-2 simulation as an alternative MAP scheduling policy.
- `DES_model.py` — Core DES building blocks (GTFS-based bus processes, energy model, helpers used by stage-2).
- `Trip_assign.py` — Trip assignment utilities used to create bus schedules from GTFS.
- `gtfs_loader.py` — GTFS parsing and feed helpers.
- `osm_graph.py` — OSMnx / NetworkX wrapper for routing and stop snapping.

## Quick start — dependencies
Install required packages (example):

```bash
pip install simpy pandas numpy matplotlib networkx osmnx geopandas shapely pyproj
```

If you want to use Gurobi-based optimization (optional): install Gurobi and its Python interface `gurobipy` and ensure a valid license is available.

## Input data (required to run experiments)
Place the following in the repository root (or change the paths in `run_optimization.py`):

- GTFS feed folder (example path used in code: `gtfs_data/`) containing standard GTFS files: `stops.txt`, `stop_times.txt`, `trips.txt`, `routes.txt`, `shapes.txt`, `calendar.txt`, `calendar_dates.txt`.
- OpenStreetMap network export covering the study area (example filename in code: `map.xml`).

The example scripts are set up to run experiments with Stockholm lines (see `run_optimization.py` constants); change `LINES`, `DATE_STR`, or GTFS path to run with other feeds/dates.

## Running the experiments
From a fresh clone with required data placed as above, the most common entry points are:

- Run the MILP ↔ Simulation constraint-driven loop (default):

```bash
python run_optimization.py
```

This executes the constraint-driven feasibility search (no external MILP required). It will:
1. Load GTFS + map data and assign trips to buses.
2. Run Stage-2 simulation with an initial configuration.
3. If infeasible, diagnose causes and apply feedback constraints (increase per-line battery minima, require more MAPs, etc.).
4. Iterate until a feasible configuration is found or max iterations are reached.
5. Run a refinement phase to greedily minimize battery capacities, MAP count, and MAP battery energy while preserving feasibility.

- Run a single simulation (validation mode — no feedback loop):

```bash
python run_optimization.py --sim-only
```

This mode runs a stage-2 simulation for the configured values in `run_optimization.py` and produces plots and a cost breakdown. Use this to validate specific battery / MAP configurations.

## How the pieces fit together
- `run_optimization.py` is the high-level experiment orchestrator. It calls into the DES (`integration_stage2.run_terminal_charging_simulation`) to evaluate candidate configurations and obtains simulation statistics.
- `integration_stage2.py` contains the SimPy simulation of buses and MAPs. It tracks bus SOC trajectories, MAP movement & energy delivery, preemptive charging events, and plotting utilities.
- `advanced_heuristics.py` implements a rolling-horizon MAP scheduler (dynamic start/target SOC thresholds, MAP selection by multi-criteria scoring). It can be used in place of the default greedy MAP scheduler by constructing an `AdvancedMAPScheduler` and passing it into the simulation (see `integration_stage2.py` where the scheduler is used/instantiated).
- `DES_model.py`, `Trip_assign.py`, `gtfs_loader.py`, and `osm_graph.py` provide the GTFS/OSM integration and lower-level simulation helpers used across the experiments.

## Important configuration knobs (where to look in code)
- In `run_optimization.py`:
  - `LINES` and `DATE_STR` — transit lines and date used for experiments.
  - `LINE_BATTERY_CAPACITIES_KWH` — per-line starting battery capacities used by the search.
  - `initial_capacity_wh` and `initial_num_maps` — default fallback bus capacity and starting MAP count.

- In `integration_stage2.py`:
  - Charging thresholds and constants (e.g. `BUS_CHARGE_THRESHOLD_SOC`, `BUS_CHARGE_CUTOFF_SOC`, `BUS_MIN_SOC`, `MAP_CHARGING_RATE_WH_S`, `MAX_CONCURRENT_CHARGERS`).
  - Function/entrypoint: `run_terminal_charging_simulation(...)` which accepts parameters for number of MAPs, line-specific capacities, map battery size, and other simulation flags.

- In `advanced_heuristics.py`:
  - `AdvancedMAPScheduler` — construct this scheduler with the simulation and pass it to the stage-2 simulation if you want the rolling-horizon heuristic.

## Reproducibility notes
- Randomized initial configurations: `run_optimization.py` contains utilities to generate randomized starting points (`generate_random_initial_values`) for stress-testing the search. To run multiple seeds, call the script in a loop or add a small wrapper that sets the RNG seed and logs outputs.
- Determinism: the simulation may rely on randomized tie-breaking. For bit-for-bit reproducibility, seed Python's `random` (and NumPy if used) and keep dataset and OSM sources identical.
- Gurobi: full MILP-based re-optimization is optional. The repository includes a constraint-driven fallback that does not require a solver.

## Outputs
By default the simulation/experiment pipeline produces the following artifacts (PNG plots and console summaries):
- bus_soc_terminal_charging_optimized.png — SOC trajectories per bus/line
- map_energy_delivery.png — per-MAP energy delivered over time
- cumulative_energy_delivery.png — cumulative energy supplied by MAPs
- map_movement_distance.png — distance travelled by each MAP
- map_soc_over_time.png — MAP battery SOC over time
- map_self_charge_heatmap.png — (geospatial) heatmap of MAP self-charging events

Console outputs include a feasibility report, per-line charging statistics, MAP usage summaries, and a cost breakdown (bus batteries, MAP batteries, MAP hardware, overnight charger cost, penalties for SOC constraint violations).

## How to use the rolling-horizon heuristic
1. Create an instance of `AdvancedMAPScheduler` from `advanced_heuristics.py` passing the simulation (`sim`), trip assignments, battery capacities, number of MAPs, and a MAP movement scheduler implementation.
2. Modify or call `integration_stage2.run_terminal_charging_simulation(...)` to accept and use the `AdvancedMAPScheduler` instance (the stage-2 simulation already contains hooks to plumb in alternative schedulers).

If you need help wiring the scheduler into a run, open an issue or point to the exact call-site in `integration_stage2.py` and I can provide a code snippet showing how to instantiate and inject the scheduler into the simulation.

## Citation
If you use this code in research, please cite the paper:

Joint Optimization of Bus Batteries and Mobile Autonomous Charging Pods for Urban Bus Electrification

(Use the citation format required by your venue; include authors, year, and DOI when available.)

## Contact / reproducibility help
Open an issue describing the dataset (GTFS + OSM extract) you're using and the command you ran. If you want, include the date string and the `LINES` you supplied and paste console output; I can help debug common causes of infeasibility or guide you through producing publication figures.

--

Prepared as the experiments README for the paper "Joint Optimization of Bus Batteries and Mobile Autonomous Charging Pods for Urban Bus Electrification".
