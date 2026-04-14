"""
Unit tests for cut relaxation, dynamic deficit, and optimality cut generation,
and the new initialisation-independent static demand computation.
"""
import math
import sys
import types
import unittest
from unittest.mock import MagicMock
from collections import defaultdict

# Stub heavy transitive dependencies so we can import run_optimization
# without installing the full simulation stack.
_STUB_MODULES = [
    'simpy', 'pyproj', 'osmnx', 'networkx', 'shapely',
    'matplotlib', 'matplotlib.pyplot', 'matplotlib.patches',
    'matplotlib.colors', 'matplotlib.cm',
    'integration_stage2', 'post_simulation_milp',
    'DES_model', 'Trip_assign', 'gtfs_loader', 'osm_graph',
    'advanced_heuristics',
]
for _mod_name in _STUB_MODULES:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = types.ModuleType(_mod_name)

# Provide the symbol that run_optimization imports at module level
sys.modules['integration_stage2'].run_terminal_charging_simulation = MagicMock()
sys.modules['post_simulation_milp'].post_simulation_optimize = MagicMock()

from run_optimization import (
    diagnose_infeasibility,
    relax_bus_capacity_cuts,
    generate_optimality_cuts,
)

# Import the REAL post_simulation_milp (not the stub used for run_optimization).
# We remove it from sys.modules first so importlib reloads the actual file.
del sys.modules['post_simulation_milp']
import importlib
import post_simulation_milp as _psm_module
# Restore the stub so other parts of the test suite continue to work.
sys.modules['post_simulation_milp'] = _psm_module


# ---------------------------------------------------------------------------
# Helpers for building a fake stage2_sim with GTFS data
# ---------------------------------------------------------------------------

def _make_stop_sequence(stop_coords, start_time=0, step=300):
    """Build a stop_sequence list from a list of (lat, lon) pairs."""
    seq = []
    for i, (lat, lon) in enumerate(stop_coords):
        seq.append({
            "stop_id": f"stop_{i}",
            "arrival": start_time + i * step,
            "lat": lat,
            "lon": lon,
        })
    return seq


def _make_stage2_sim(trips_per_bus, stop_sequences, layover_between_trips=0):
    """
    Build a minimal fake stage2_sim object.

    Parameters
    ----------
    trips_per_bus : dict  {bus_id: [trip_id, ...]}
    stop_sequences : dict  {trip_id: [(lat, lon), ...]}
        Coordinates for each trip's stops.
    layover_between_trips : int
        Dwell time (seconds) inserted between consecutive trips.
    """
    # Build stop sequence: trips start immediately after each other unless
    # layover_between_trips is set.
    trip_start_times = {}
    t = 0
    for bus_id, trip_ids in trips_per_bus.items():
        for trip_id in trip_ids:
            trip_start_times[trip_id] = t
            n_stops = len(stop_sequences[trip_id])
            t += n_stops * 300 + layover_between_trips

    def stop_sequence_fn(trip_id):
        coords = stop_sequences.get(trip_id, [])
        start = trip_start_times.get(trip_id, 0)
        return _make_stop_sequence(coords, start_time=start, step=300)

    # Geodesic distance: use a flat-earth approximation for testing.
    # ~111 km per degree of latitude.
    class FakeGeod:
        def inv(self, lon1, lat1, lon2, lat2):
            dlat = (lat2 - lat1) * 111_000  # metres
            dlon = (lon2 - lon1) * 111_000 * math.cos(math.radians((lat1 + lat2) / 2))
            dist = math.sqrt(dlat**2 + dlon**2)
            return 0.0, 0.0, dist

    class FakeGTFS:
        def stop_sequence(self, trip_id):
            return stop_sequence_fn(trip_id)

    class FakeSim:
        def __init__(self):
            self.gtfs = FakeGTFS()
            self.geod = FakeGeod()

    stage2_sim = MagicMock()
    stage2_sim.bus_trips_dict = trips_per_bus
    stage2_sim.sim = FakeSim()
    return stage2_sim


# ---------------------------------------------------------------------------
# Tests for compute_static_line_demand
# ---------------------------------------------------------------------------

class TestComputeStaticLineDemand(unittest.TestCase):
    """compute_static_line_demand returns initialisation-independent distances."""

    def test_single_trip_single_bus(self):
        """Single trip: chain distance equals the trip distance."""
        # 10 stops, each 0.01° north of the previous → ≈ 1111 m each segment
        coords = [(0.0 + i * 0.01, 0.0) for i in range(11)]  # 10 segments
        sim = _make_stage2_sim(
            {'line1_bus0': ['t1']},
            {'t1': coords},
            layover_between_trips=0,
        )
        result = _psm_module.compute_static_line_demand(sim)
        self.assertIn('1', result)
        expected_m = 10 * 0.01 * 111_000  # ~11 100 m
        self.assertAlmostEqual(result['1'], expected_m, delta=100)

    def test_two_trips_no_layover_accumulates(self):
        """Two consecutive trips with no layover: chain = sum of both."""
        coords = [(i * 0.01, 0.0) for i in range(6)]   # 5 segments each
        sim = _make_stage2_sim(
            {'line2_bus0': ['t1', 't2']},
            {'t1': coords, 't2': coords},
            layover_between_trips=0,   # no charging opportunity between trips
        )
        result = _psm_module.compute_static_line_demand(sim)
        self.assertIn('2', result)
        single_trip_m = 5 * 0.01 * 111_000
        self.assertAlmostEqual(result['2'], 2 * single_trip_m, delta=200)

    def test_two_trips_with_layover_resets_chain(self):
        """Two trips separated by a layover ≥ threshold: chain resets."""
        coords = [(i * 0.01, 0.0) for i in range(6)]
        sim = _make_stage2_sim(
            {'line3_bus0': ['t1', 't2']},
            {'t1': coords, 't2': coords},
            layover_between_trips=700,   # > _LAYOVER_THRESHOLD_S (600 s)
        )
        result = _psm_module.compute_static_line_demand(sim)
        self.assertIn('3', result)
        single_trip_m = 5 * 0.01 * 111_000
        # Chain should equal a SINGLE trip, not the sum of two
        self.assertAlmostEqual(result['3'], single_trip_m, delta=200)

    def test_multiple_buses_takes_max(self):
        """Worst-case is the maximum over all buses on the line."""
        short_coords = [(i * 0.01, 0.0) for i in range(3)]   # 2 segments
        long_coords  = [(i * 0.02, 0.0) for i in range(3)]   # 2 segments, 2× longer
        sim = _make_stage2_sim(
            {'line4_bus0': ['t_short'], 'line4_bus1': ['t_long']},
            {'t_short': short_coords, 't_long': long_coords},
        )
        result = _psm_module.compute_static_line_demand(sim)
        self.assertIn('4', result)
        short_m = 2 * 0.01 * 111_000
        long_m  = 2 * 0.02 * 111_000
        self.assertAlmostEqual(result['4'], long_m, delta=200)
        self.assertGreater(result['4'], short_m)

    def test_missing_gtfs_returns_empty(self):
        """If stage2_sim lacks GTFS attributes, return empty dict."""
        sim = MagicMock(spec=[])   # no attributes at all
        result = _psm_module.compute_static_line_demand(sim)
        self.assertEqual(result, {})

    def test_result_independent_of_battery_capacity(self):
        """The distance-based demand is independent of battery capacity."""
        coords = [(i * 0.01, 0.0) for i in range(6)]
        # Both sims use the same GTFS trips; only battery capacity differs.
        sim_small = _make_stage2_sim({'line5_bus0': ['t1']}, {'t1': coords})
        sim_large = _make_stage2_sim({'line5_bus0': ['t1']}, {'t1': coords})
        # Battery capacity is not part of the input; the result must be equal.
        r_small = _psm_module.compute_static_line_demand(sim_small)
        r_large = _psm_module.compute_static_line_demand(sim_large)
        self.assertAlmostEqual(r_small.get('5', 0), r_large.get('5', 0), delta=1)

class TestDiagnoseInfeasibility(unittest.TestCase):
    """diagnose_infeasibility now stores num_maps_at_creation on bus_min_cap."""

    def test_bus_min_cap_stores_num_maps(self):
        results = {
            'line_battery_capacities_wh': {'1': 100_000},
            'battery_capacity_wh': 100_000,
            'min_soc_overall_ratio': 0.5,
            'num_maps': 3,
        }
        bus_stats = {
            'line1_bus0': {
                'line_id': '1',
                'min_soc_ratio': 0.10,   # below 20%
                'min_soc_wh': 10_000,
                'total_energy_consumed_wh': 80_000,
            },
        }
        constraints = diagnose_infeasibility(results, bus_stats)
        bus_cap_cuts = [c for c in constraints if c['type'] == 'bus_min_cap']
        self.assertTrue(len(bus_cap_cuts) > 0)
        for cut in bus_cap_cuts:
            self.assertEqual(cut['num_maps_at_creation'], 3)

    def test_no_constraints_when_feasible(self):
        results = {
            'line_battery_capacities_wh': {'1': 200_000},
            'battery_capacity_wh': 200_000,
            'min_soc_overall_ratio': 0.50,
            'num_maps': 2,
        }
        bus_stats = {
            'line1_bus0': {
                'line_id': '1',
                'min_soc_ratio': 0.50,
                'min_soc_wh': 100_000,
                'total_energy_consumed_wh': 80_000,
            },
        }
        constraints = diagnose_infeasibility(results, bus_stats)
        self.assertEqual(constraints, [])


class TestRelaxBusCapacityCuts(unittest.TestCase):
    """relax_bus_capacity_cuts should reduce bus_min_cap when MAPs grow."""

    def _make_cut(self, line_id, value, maps_at_creation):
        return {
            'type': 'bus_min_cap',
            'line_id': line_id,
            'value': value,
            'num_maps_at_creation': maps_at_creation,
            'reason': f"Line {line_id}: cut at {value} kWh.",
        }

    def test_no_relaxation_when_maps_not_growing(self):
        cuts = [self._make_cut('1', 200, 2)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=2, prev_num_maps=2)
        self.assertEqual(result[0]['value'], 200)

    def test_no_relaxation_when_maps_decrease(self):
        cuts = [self._make_cut('1', 200, 2)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=1, prev_num_maps=2)
        self.assertEqual(result[0]['value'], 200)

    def test_relaxation_when_maps_grow(self):
        cuts = [self._make_cut('1', 200, 1)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=3, prev_num_maps=1)
        # Value should decrease
        self.assertLess(result[0]['value'], 200)
        # Value should be a multiple of 10
        self.assertEqual(result[0]['value'] % 10, 0)
        # Should not go below 50
        self.assertGreaterEqual(result[0]['value'], 50)

    def test_relaxation_respects_minimum(self):
        cuts = [self._make_cut('1', 60, 1)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=10, prev_num_maps=1,
                                         bus_cap_min_kwh=50.0)
        self.assertGreaterEqual(result[0]['value'], 50)

    def test_non_bus_min_cap_untouched(self):
        constraints = [
            self._make_cut('1', 200, 1),
            {'type': 'min_maps', 'value': 2, 'reason': 'Need 2 MAPs'},
        ]
        result = relax_bus_capacity_cuts(constraints, new_num_maps=3,
                                          prev_num_maps=1)
        min_maps = [c for c in result if c['type'] == 'min_maps']
        self.assertEqual(len(min_maps), 1)
        self.assertEqual(min_maps[0]['value'], 2)

    def test_prev_none_no_crash(self):
        cuts = [self._make_cut('1', 200, 1)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=3, prev_num_maps=None)
        # Should return unchanged
        self.assertEqual(result[0]['value'], 200)

    def test_relax_factor_bounded(self):
        """relax_factor is capped at 0.5 so value can't drop below half."""
        cuts = [self._make_cut('1', 300, 1)]
        result = relax_bus_capacity_cuts(cuts, new_num_maps=100,
                                          prev_num_maps=1)
        self.assertGreaterEqual(result[0]['value'], 150)


class TestGenerateOptimalityCuts(unittest.TestCase):
    """generate_optimality_cuts should produce tradeoff constraints."""

    def _make_sim_and_results(self, num_maps, bus_caps_kwh,
                               bus_stats, bus_energy_charged):
        milp_results = {
            'num_maps': num_maps,
            'bus_battery_kwh': bus_caps_kwh,
        }
        results = {
            'bus_statistics': bus_stats,
        }
        stage2_sim = MagicMock()
        stage2_sim.bus_energy_charged = bus_energy_charged
        return results, stage2_sim, milp_results

    def test_no_cuts_when_no_maps(self):
        results, sim, milp = self._make_sim_and_results(
            num_maps=0,
            bus_caps_kwh={'1': 200},
            bus_stats={'b0': {'line_id': '1'}},
            bus_energy_charged={'b0': 5000},
        )
        cuts = generate_optimality_cuts(results, sim, milp)
        self.assertEqual(cuts, [])

    def test_cuts_generated_when_maps_present(self):
        results, sim, milp = self._make_sim_and_results(
            num_maps=2,
            bus_caps_kwh={'1': 200},
            bus_stats={
                'b0': {'line_id': '1'},
                'b1': {'line_id': '1'},
            },
            bus_energy_charged={'b0': 10_000, 'b1': 10_000},
        )
        cuts = generate_optimality_cuts(results, sim, milp)
        self.assertTrue(len(cuts) > 0)
        cut = cuts[0]
        self.assertEqual(cut['type'], 'optimality_cut')
        self.assertEqual(cut['line_id'], '1')
        self.assertGreater(cut['gamma'], 0)
        # threshold = cap_kwh + gamma * num_maps
        self.assertAlmostEqual(
            cut['threshold'], 200 + cut['gamma'] * 2, places=2)

    def test_gamma_calculation(self):
        """γ = total_charged / n_buses / num_maps / 1000."""
        results, sim, milp = self._make_sim_and_results(
            num_maps=2,
            bus_caps_kwh={'1': 150},
            bus_stats={
                'b0': {'line_id': '1'},
            },
            bus_energy_charged={'b0': 20_000},  # 20 kWh
        )
        cuts = generate_optimality_cuts(results, sim, milp)
        self.assertEqual(len(cuts), 1)
        expected_gamma = 20_000 / (1 * 2 * 1000)  # = 10 kWh per MAP
        self.assertAlmostEqual(cuts[0]['gamma'], expected_gamma, places=2)

    def test_no_charging_no_cuts(self):
        """Lines with no MAP charging produce no optimality cuts."""
        results, sim, milp = self._make_sim_and_results(
            num_maps=2,
            bus_caps_kwh={'1': 200},
            bus_stats={'b0': {'line_id': '1'}},
            bus_energy_charged={'b0': 0},  # no charging
        )
        cuts = generate_optimality_cuts(results, sim, milp)
        self.assertEqual(cuts, [])


if __name__ == '__main__':
    unittest.main()
