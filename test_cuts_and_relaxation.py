"""
Unit tests for cut relaxation, dynamic deficit, and optimality cut generation.
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
        expected_gamma = 20_000 / 1 / 2 / 1000  # = 10 kWh per MAP
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
