"""
Visualization module for optimization convergence and trajectory analysis.

Creates three plots for each random restart:
1. Trajectory scatter — X = total bus battery (kWh), Y = total MAP battery (kWh), each iteration annotated with its cost.
2. Trajectory with arrows — same axes, iterations connected i → i+1 via arrows, showing the search path.
3. Convergence — X = iteration, Y = total cost, one line per restart.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path


def load_optimization_runs(filename='optimization_runs.json'):
    """Load optimization runs from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)


def plot_trajectory_scatter(run_data, run_idx, ax):
    """
    Plot trajectory scatter: X = total bus battery (kWh), Y = total MAP battery (kWh).
    Each point is annotated with its total cost.
    """
    iteration_log = run_data['iteration_log']
    
    bus_batteries = []
    map_batteries = []
    costs = []
    iterations = []
    
    for entry in iteration_log:
        bus_caps = entry.get('bus_battery_kwh', {})
        if bus_caps:
            total_bus_battery = sum(bus_caps.values())
        else:
            total_bus_battery = 0
        
        map_battery = entry.get('map_battery_kwh', 0)
        cost = entry.get('total_cost', float('inf'))
        
        bus_batteries.append(total_bus_battery)
        map_batteries.append(map_battery)
        costs.append(cost)
        iterations.append(entry['iteration'])
    
    # Plot points with color based on cost
    scatter = ax.scatter(bus_batteries, map_batteries, c=costs, s=100, cmap='RdYlGn_r', 
                         edgecolors='black', linewidth=1.5, alpha=0.7, zorder=3)
    
    # Annotate each point with its cost
    for i, (bus_b, map_b, cost, iteration) in enumerate(zip(bus_batteries, map_batteries, costs, iterations)):
        if cost == float('inf'):
            label = f"Iter {iteration}\nInfeas"
        else:
            label = f"Iter {iteration}\n${cost/1e6:.2f}M"
        ax.annotate(label, (bus_b, map_b), fontsize=8, ha='center', va='center', zorder=4)
    
    ax.set_xlabel('Total Bus Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total MAP Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_title(f'Optimization Trajectory - Run {run_idx}\n(Points show cost progression)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    return scatter


def plot_trajectory_with_arrows(run_data, run_idx, ax):
    """
    Plot trajectory with arrows showing the search path.
    Iterations are connected i → i+1 via arrows.
    """
    iteration_log = run_data['iteration_log']
    
    bus_batteries = []
    map_batteries = []
    costs = []
    iterations = []
    feasibilities = []
    
    for entry in iteration_log:
        bus_caps = entry.get('bus_battery_kwh', {})
        if bus_caps:
            total_bus_battery = sum(bus_caps.values())
        else:
            total_bus_battery = 0
        
        map_battery = entry.get('map_battery_kwh', 0)
        cost = entry.get('total_cost', float('inf'))
        feasible = entry.get('sim_feasible', False)
        
        bus_batteries.append(total_bus_battery)
        map_batteries.append(map_battery)
        costs.append(cost)
        iterations.append(entry['iteration'])
        feasibilities.append(feasible)
    
    # Color points by feasibility
    colors = ['red' if not feas else 'green' for feas in feasibilities]
    
    scatter = ax.scatter(bus_batteries, map_batteries, c=colors, s=100, 
                         edgecolors='black', linewidth=1.5, alpha=0.7, zorder=3)
    
    # Draw arrows connecting consecutive iterations
    for i in range(len(bus_batteries) - 1):
        arrow = FancyArrowPatch(
            (bus_batteries[i], map_batteries[i]),
            (bus_batteries[i+1], map_batteries[i+1]),
            arrowstyle='->', mutation_scale=20, linewidth=2, color='blue', 
            alpha=0.6, zorder=2
        )
        ax.add_patch(arrow)
    
    # Annotate iterations
    for i, (bus_b, map_b, iteration) in enumerate(zip(bus_batteries, map_batteries, iterations)):
        ax.annotate(f"{iteration}", (bus_b, map_b), fontsize=9, ha='right', va='bottom', 
                    fontweight='bold', zorder=4)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', label='Infeasible')
    green_patch = mpatches.Patch(color='green', label='Feasible')
    ax.legend(handles=[red_patch, green_patch], loc='best', fontsize=10)
    
    ax.set_xlabel('Total Bus Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total MAP Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_title(f'Search Path with Arrows - Run {run_idx}\n(Red=Infeasible, Green=Feasible)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)


def plot_convergence(all_runs, ax):
    """
    Plot convergence: X = iteration, Y = total cost.
    One line per restart, with feasibility markers.
    """
    for run_data in all_runs:
        run_idx = run_data['run']
        iteration_log = run_data['iteration_log']
        
        iterations = []
        costs = []
        feasibilities = []
        
        for entry in iteration_log:
            iteration = entry['iteration']
            cost = entry.get('total_cost', float('inf'))
            feasible = entry.get('sim_feasible', False)
            
            iterations.append(iteration)
            costs.append(cost)
            feasibilities.append(feasible)
        
        # Plot line for this run
        ax.plot(iterations, costs, marker='o', label=f'Run {run_idx}', linewidth=2, markersize=6, alpha=0.7)
        
        # Mark feasible iterations with a star
        for i, (iteration, cost, feasible) in enumerate(zip(iterations, costs, feasibilities)):
            if feasible and cost != float('inf'):
                ax.plot(iteration, cost, marker='*', markersize=15, color='green', 
                        markeredgecolor='darkgreen', markeredgewidth=1, zorder=4)
    
    ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total Cost ($)', fontsize=11, fontweight='bold')
    ax.set_title('Cost Convergence Across All Runs\n(★ = Feasible solution)', 
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))


def generate_all_plots(json_file='optimization_runs.json', output_dir='plots'):
    """Generate and save all convergence plots."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Load data
    all_runs = load_optimization_runs(json_file)
    
    print(f"Loaded {len(all_runs)} runs from {json_file}")
    
    # --- Plot 1: Individual run trajectories (scatter) ---
    for run_data in all_runs:
        run_idx = run_data['run']
        fig, ax = plt.subplots(figsize=(10, 7))
        
        scatter = plot_trajectory_scatter(run_data, run_idx, ax)
        cbar = plt.colorbar(scatter, ax=ax, label='Total Cost ($)')
        cbar.formatter.set_powerlimits((0, 0))
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'trajectory_scatter_run_{run_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")
        plt.close()
    
    # --- Plot 2: Individual run trajectories (with arrows) ---
    for run_data in all_runs:
        run_idx = run_data['run']
        fig, ax = plt.subplots(figsize=(10, 7))
        
        plot_trajectory_with_arrows(run_data, run_idx, ax)
        
        plt.tight_layout()
        output_path = Path(output_dir) / f'trajectory_arrows_run_{run_idx}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved {output_path}")
        plt.close()
    
    # --- Plot 3: Convergence across all runs ---
    fig, ax = plt.subplots(figsize=(12, 7))
    
    plot_convergence(all_runs, ax)
    
    plt.tight_layout()
    output_path = Path(output_dir) / 'convergence_all_runs.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_plots()
