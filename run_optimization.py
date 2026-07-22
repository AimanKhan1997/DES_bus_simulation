def export_runs_to_json(all_runs, filename='optimization_runs.json'):
    """
    Export all_runs to JSON for plotting.
    Keeps only necessary fields for plotting: log data with costs, configs, feasibility.
    Filters out non-serializable objects like stage2_sim.
    """
    export_data = []
    for run in all_runs:
        # Build iteration log with only JSON-serializable fields
        filtered_iteration_log = []
        for entry in run.get('log', []):
            filtered_entry = {
                'iteration': entry.get('iteration'),
                'method': entry.get('method'),
                'sim_feasible': entry.get('sim_feasible'),
                'bus_battery_kwh': entry.get('bus_battery_kwh'),
                'map_battery_kwh': entry.get('map_battery_kwh'),
                'num_maps': entry.get('num_maps'),
                'total_cost': entry.get('total_cost'),
                'constraints_applied': entry.get('constraints_applied', []),
            }
            filtered_iteration_log.append(filtered_entry)
        
        # Extract solution (only serializable parts)
        solution = run.get('solution')
        solution_export = None
        if solution:
            solution_export = {
                'bus_battery_kwh': solution.get('bus_battery_kwh'),
                'map_battery_kwh': solution.get('map_battery_kwh'),
                'num_maps': solution.get('num_maps'),
            }
        
        # Extract sim_results (only serializable parts)
        sim_results = run.get('sim_results')
        sim_results_export = None
        if sim_results:
            sim_results_export = {
                'feasible': sim_results.get('feasible'),
                'min_soc_overall_ratio': sim_results.get('min_soc_overall_ratio'),
            }
        
        run_export = {
            'run': run['run'],
            'initial_line_battery_kwh': run['initial_line_battery_kwh'],
            'initial_num_maps': run['initial_num_maps'],
            'initial_map_battery_kwh': run['initial_map_battery_kwh'],
            'total_cost': run.get('total_cost'),
            'solution': solution_export,
            'sim_results': sim_results_export,
            'iteration_log': filtered_iteration_log,
        }
        export_data.append(run_export)
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2, default=str)
    
    print(f"\n[EXPORT] Optimization runs exported to {filename}")
    return filename
