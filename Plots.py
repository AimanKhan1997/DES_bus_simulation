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
            label = f"Iter {iteration}\n${cost / 1e6:.2f}M"
        ax.annotate(label, (bus_b, map_b), fontsize=8, ha='center', va='center', zorder=4)

    ax.set_xlabel('Total Bus Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total MAP Battery Capacity (kWh)', fontsize=11, fontweight='bold')
    ax.set_title(f'Optimization Trajectory - Run {run_idx}\n(Points show cost progression)', fontsize=12,
                 fontweight='bold')
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
            (bus_batteries[i + 1], map_batteries[i + 1]),
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
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x / 1e6:.1f}M'))


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
# import pandas as pd
# import matplotlib.pyplot as plt
#
# fleet = {
#     'L1': 22,
#     'L2': 16,
#     'L3': 16,
#     'L4': 26,
#     'L6': 11
# }
#
# # Solutions
# data = [
#     [4949902.22,300,12,[90,60,80,110,50]],
#     [5157407.27,460,12,[80,60,80,110,50]],
#     [4885874.07,400,9,[80,60,90,120,60]],
#     [4989716.71,390,11,[80,60,80,120,50]],
#     [5171132.19,200,16,[90,60,80,140,60]],
#     [5347580.86,200,12,[120,60,90,130,310]],
#     [6511800.37,300,19,[80,60,70,440,60]],
#     [6022997.01,200,9,[120,60,100,420,240]],
#     [5503916.20,340,13,[80,60,130,100,330]],
#     [5233404.12,280,16,[80,70,80,110,80]],
#     [5763971.49,320,9,[80,60,130,400,50]],
#     [5656626.15,420,12,[300,60,80,100,50]],
#     [5369926.77,290,9,[80,60,150,200,220]],
#     [6217815.05,450,18,[80,160,290,90,40]],
#     [5927001.23,350,11,[80,420,80,110,270]],
#     [5187843.03,290,11,[90,60,90,130,230]],
#     [5994298.98,210,9,[90,60,400,310,90]],
#     [6189118.19,400,19,[80,50,130,250,50]],
#     [5302105.17,320,8,[100,110,260,150,50]],
#     [6661090.34,290,20,[70,130,350,190,230]],
#     [5212153.80,440,12,[80,80,100,110,50]],
#     [5269353.63,440,12,[80,140,80,110,50]],
#     [5396867.55,370,15,[80,50,80,100,190]],
#     [5026352.02,380,12,[80,60,80,100,60]],
#     [5156493.73,230,16,[90,60,80,120,60]],
#     [5144261.94,220,11,[90,60,90,200,90]],
#     [6089504.08,210,21,[230,60,160,120,130]],
#     [6138543.56,350,21,[80,60,190,90,250]],
#     [5106377.82,220,12,[80,90,130,150,50]],
#     [5177378.04,440,10,[90,90,90,100,170]]
# ]
#
# df = pd.DataFrame(
#     data,
#     columns=[
#         'TotalCost',
#         'MAPBattery',
#         'MAPCount',
#         'BusBatteries'
#     ]
# )
#
# # ---------------------------------------------------
# # Calculate total MAP battery
# # ---------------------------------------------------
#
# df['TotalMAPBattery'] = (
#     df['MAPBattery']
#     * df['MAPCount']
# )
#
# # ---------------------------------------------------
# # Calculate total bus battery
# # ---------------------------------------------------
#
# def calculate_bus_battery(row):
#
#     b = row['BusBatteries']
#
#     total = (
#         fleet['L1'] * b[0] +
#         fleet['L2'] * b[1] +
#         fleet['L3'] * b[2] +
#         fleet['L4'] * b[3] +
#         fleet['L6'] * b[4]
#     )
#
#     return total
#
# df['TotalBusBattery'] = df.apply(
#     calculate_bus_battery,
#     axis=1
# )
#
# # ---------------------------------------------------
# # Bubble plot
# # ---------------------------------------------------
# min_size = 130     # Smallest bubble
# max_size = 270    # Largest bubble
#
# # Bubble size based on Total MAP Battery
# bubble_sizes_map = (
#     min_size +
#     (df['TotalMAPBattery'] - df['TotalMAPBattery'].min()) *
#     (max_size - min_size) /
#     (df['TotalMAPBattery'].max() - df['TotalMAPBattery'].min())
# )
#
# # Bubble size based on Total Cost
# bubble_sizes_cost = (
#     min_size +
#     (df['TotalCost'] - df['TotalCost'].min()) *
#     (max_size - min_size) /
#     (df['TotalCost'].max() - df['TotalCost'].min())
# )
#
# plt.figure(figsize=(11,7))
#
# scatter =(
#     plt.scatter(
#     df['TotalBusBattery'],
#     df['TotalCost'],
#     s=bubble_sizes_map,#df['TotalMAPBattery']/5,      # scale bubble size
#     c=df['MAPCount'],
#     cmap='viridis',
#     alpha=0.95,
#     edgecolors='black'
# ))
#
# plt.figure(figsize=(11,7))
#
# scatter2 = (
#     plt.scatter(
#     df['TotalBusBattery'],
#     df['TotalMAPBattery'],
#     s=bubble_sizes_cost,#df['TotalCost']/3000,
#     c=df['TotalCost'],
#     cmap='plasma',
#     alpha=0.95,
#     edgecolors='black'
# ))
# plt.xlabel("Total Bus Battery Capacity (kWh)")
# plt.ylabel("Total MAP Battery Capacity (kWh)")
# plt.title("Pareto-Style Visualization of Bus and MAP Battery Capacities")
# plt.colorbar(scatter2, label="Total Cost (M$)")
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
#
# plt.savefig("pareto_bus_map.svg", bbox_inches='tight')
# plt.show()
#
# ##Random Runs
#
# # -----------------------------
# # Read CSV
# # -----------------------------
# df = pd.read_csv("random_runs.csv")
#
# # -----------------------------
# # Calculate total MAP battery
# # -----------------------------
# df['TotalMAPBattery'] = df['MAPs'] * df['MAP_kWh']
#
# # -----------------------------
# # Calculate total bus battery
# # -----------------------------
# df['TotalBusBattery'] = (
#         df['L1'] * fleet['L1'] +
#         df['L2'] * fleet['L2'] +
#         df['L3'] * fleet['L3'] +
#         df['L4'] * fleet['L4'] +
#         df['L6'] * fleet['L6']
# )
#
# # -----------------------------
# # Colors
# # -----------------------------
# colors = {
#     'YES': 'green',
#     'NO': 'red'
# }
#
# # -----------------------------
# # Plot
# # -----------------------------
# plt.figure(figsize=(10, 7))
# markers = {
#     'YES': 'o',
#     'NO': 'x'
# }
#
# for feasibility in ['YES', 'NO']:
#
#     subset = df[df['feasible'] == feasibility]
#
#     plt.scatter(
#         subset['TotalBusBattery'],
#         subset['TotalMAPBattery'],
#         c=colors[feasibility],
#         marker=markers[feasibility],
#         s=120,
#         label=feasibility
#     )
#
# plt.xlabel("Total Bus Battery Capacity (kWh)")
# plt.ylabel("Total MAP Battery Capacity (kWh)")
# plt.title("Feasible vs Infeasible Solutions")
#
# plt.grid(True, alpha=0.3)
# plt.legend(title="Feasibility")
# plt.savefig("random.svg", bbox_inches='tight')
# #plt.show()
# plt.tight_layout()
# plt.show()
# # import pandas as pd
# # import matplotlib.pyplot as plt
# #
# # fleet = {
# #     'L1': 22,
# #     'L2': 16,
# #     'L3': 16,
# #     'L4': 26,
# #     'L6': 11
# # }
# #
# # # Solutions
# # data = [
# #     [4949902.22,300,12,[90,60,80,110,50]],
# #     [5157407.27,460,12,[80,60,80,110,50]],
# #     [4885874.07,400,9,[80,60,90,120,60]],
# #     [4989716.71,390,11,[80,60,80,120,50]],
# #     [5171132.19,200,16,[90,60,80,140,60]],
# #     [5347580.86,200,12,[120,60,90,130,310]],
# #     [6511800.37,300,19,[80,60,70,440,60]],
# #     [6022997.01,200,9,[120,60,100,420,240]],
# #     [5503916.20,340,13,[80,60,130,100,330]],
# #     [5233404.12,280,16,[80,70,80,110,80]],
# #     [5763971.49,320,9,[80,60,130,400,50]],
# #     [5656626.15,420,12,[300,60,80,100,50]],
# #     [5369926.77,290,9,[80,60,150,200,220]],
# #     [6217815.05,450,18,[80,160,290,90,40]],
# #     [5927001.23,350,11,[80,420,80,110,270]],
# #     [5187843.03,290,11,[90,60,90,130,230]],
# #     [5994298.98,210,9,[90,60,400,310,90]],
# #     [6189118.19,400,19,[80,50,130,250,50]],
# #     [5302105.17,320,8,[100,110,260,150,50]],
# #     [6661090.34,290,20,[70,130,350,190,230]],
# #     [5212153.80,440,12,[80,80,100,110,50]],
# #     [5269353.63,440,12,[80,140,80,110,50]],
# #     [5396867.55,370,15,[80,50,80,100,190]],
# #     [5026352.02,380,12,[80,60,80,100,60]],
# #     [5156493.73,230,16,[90,60,80,120,60]],
# #     [5144261.94,220,11,[90,60,90,200,90]],
# #     [6089504.08,210,21,[230,60,160,120,130]],
# #     [6138543.56,350,21,[80,60,190,90,250]],
# #     [5106377.82,220,12,[80,90,130,150,50]],
# #     [5177378.04,440,10,[90,90,90,100,170]]
# # ]
# #
# # # data = [
# # #     [1949902.22,300,12,[90,60,80,110,50]],
# # #     # [1787787.37,290,9,[90,60,90,120,70]],
# # #     [2157407.27,460,12,[80,60,80,110,50]],
# # #     [1885874.07,400,9,[80,60,90,120,60]],
# # #     [1989716.71,390,11,[80,60,80,120,50]],
# # #     [2171132.19,200,16,[90,60,80,140,60]],
# # #     [2347580.86,200,12,[120,60,90,130,310]],
# # #     [3511800.37,300,19,[80,60,70,440,60]],
# # #     [3022997.01,200,9,[120,60,100,420,240]],
# # #     [2503916.20,340,13,[80,60,130,100,330]],
# # #     [2233404.12,280,16,[80,70,80,110,80]],
# # #     [2763971.49,320,9,[80,60,130,400,50]],
# # #     [2656626.15,420,12,[300,60,80,100,50]],
# # #     [2369926.77,290,9,[80,60,150,200,220]],
# # #     [3217815.05,450,18,[80,160,290,90,40]],
# # #     [2927001.23,350,11,[80,420,80,110,270]],
# # #     [2187843.03,290,11,[90,60,90,130,230]],
# # #     [2994298.98,210,9,[90,60,400,310,90]],
# # #     [3189118.19,400,19,[80,50,130,250,50]],
# # #     [2302105.17,320,8,[100,110,260,150,50]],
# # #     [3661090.34,290,20,[70,130,350,190,230]],
# # #     [2212153.80,440,12,[80,80,100,110,50]],
# # #     [2269353.63,440,12,[80,140,80,110,50]],
# # #     [2396867.55,370,15,[80,50,80,100,190]],
# # #     [2026352.02,380,12,[80,60,80,100,60]],
# # #     [2156493.73,230,16,[90,60,80,120,60]],
# # #     [2144261.94,220,11,[90,60,90,200,90]],
# # #     [3089504.08,210,21,[230,60,160,120,130]],
# # #     [3138543.56,350,21,[80,60,190,90,250]],
# # #     [2106377.82,220,12,[80,90,130,150,50]],
# # #     [2177378.04,440,10,[90,90,90,100,170]]
# # # ]
# #
# # df = pd.DataFrame(
# #     data,
# #     columns=[
# #         'TotalCost',
# #         'MAPBattery',
# #         'MAPCount',
# #         'BusBatteries'
# #     ]
# # )
# #
# # # ---------------------------------------------------
# # # Calculate total MAP battery
# # # ---------------------------------------------------
# #
# # df['TotalMAPBattery'] = (
# #     df['MAPBattery']
# #     * df['MAPCount']
# # )
# #
# # # ---------------------------------------------------
# # # Calculate total bus battery
# # # ---------------------------------------------------
# #
# # def calculate_bus_battery(row):
# #
# #     b = row['BusBatteries']
# #
# #     total = (
# #         fleet['L1'] * b[0] +
# #         fleet['L2'] * b[1] +
# #         fleet['L3'] * b[2] +
# #         fleet['L4'] * b[3] +
# #         fleet['L6'] * b[4]
# #     )
# #
# #     return total
# #
# # df['TotalBusBattery'] = df.apply(
# #     calculate_bus_battery,
# #     axis=1
# # )
# #
# # # ---------------------------------------------------
# # # Bubble plot
# # # ---------------------------------------------------
# #
# # plt.figure(figsize=(11,7))
# # best = df['TotalCost'].idxmin()
# #
# # scatter =(
# #     plt.scatter(
# #     df['TotalBusBattery'],
# #     df['TotalCost'],
# #     s=df['TotalMAPBattery']/5,      # scale bubble size
# #     c=df['MAPCount'],
# #     cmap='viridis',
# #     alpha=0.75,
# #     edgecolors='black'
# # ))
# #
# # # Label each solution
# # for i, row in df.iterrows():
# #
# #     plt.annotate(
# #         f"S{i+1}",
# #         (row['TotalBusBattery'], row['TotalCost']),
# #         xytext=(5,5),
# #         textcoords='offset points'
# #     )
# #
# # cbar = plt.colorbar(scatter)
# # cbar.set_label("MAP Count")
# #
# # plt.xlabel("Total Bus Battery Capacity (kWh)")
# # plt.ylabel("Total Cost ($)")
# # plt.title(
# #     "Pareto-Style Visualization of Cost, Bus Battery and MAP Battery"
# # )
# #
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.show()
# #
# # plt.scatter(
# #     df['TotalBusBattery'],
# #     df['TotalMAPBattery'],
# #     s=df['TotalCost']/3000,
# #     c=df['TotalCost'],
# #     cmap='plasma'
# # )
# # plt.xlabel("Total Bus Battery Capacity (kWh)")
# # plt.ylabel("Total MAP Battery Capacity (kWh)")
# # plt.title("Pareto-Style Visualization of Bus and MAP Battery Capacities")
# # plt.colorbar(label="Total Cost ($)")
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# # plt.show()
# #
# # plt.figure(figsize=(11,7))
# #
# # scatter2 = (
# #     plt.scatter(
# #     df['TotalBusBattery'],
# #     df['TotalMAPBattery'],
# #     s=df['TotalCost']/3000,
# #     c=df['TotalCost'],
# #     cmap='plasma',
# #     alpha=0.75,
# #     edgecolors='black'
# # ))
# # plt.xlabel("Total Bus Battery Capacity (kWh)")
# # plt.ylabel("Total MAP Battery Capacity (kWh)")
# # plt.title("Pareto-Style Visualization of Bus and MAP Battery Capacities")
# # plt.colorbar(scatter2, label="Total Cost (M$)")
# # plt.grid(True, alpha=0.3)
# # plt.tight_layout()
# #
# # plt.savefig("pareto_bus_map.svg", bbox_inches='tight')
# # plt.show()
# #
# # ##Random Runs
# #
# # # -----------------------------
# # # Read CSV
# # # -----------------------------
# # df = pd.read_csv("random_runs.csv")
# #
# # # -----------------------------
# # # Calculate total MAP battery
# # # -----------------------------
# # df['TotalMAPBattery'] = df['MAPs'] * df['MAP_kWh']
# #
# # # -----------------------------
# # # Calculate total bus battery
# # # -----------------------------
# # df['TotalBusBattery'] = (
# #         df['L1'] * fleet['L1'] +
# #         df['L2'] * fleet['L2'] +
# #         df['L3'] * fleet['L3'] +
# #         df['L4'] * fleet['L4'] +
# #         df['L6'] * fleet['L6']
# # )
# #
# # # -----------------------------
# # # Colors
# # # -----------------------------
# # colors = {
# #     'YES': 'green',
# #     'NO': 'red'
# # }
# #
# # # -----------------------------
# # # Plot
# # # -----------------------------
# # plt.figure(figsize=(10, 7))
# # markers = {
# #     'YES': 'o',
# #     'NO': 'x'
# # }
# #
# # for feasibility in ['YES', 'NO']:
# #
# #     subset = df[df['feasible'] == feasibility]
# #
# #     plt.scatter(
# #         subset['TotalBusBattery'],
# #         subset['TotalMAPBattery'],
# #         c=colors[feasibility],
# #         marker=markers[feasibility],
# #         s=120,
# #         label=feasibility
# #     )
# #
# # plt.xlabel("Total Bus Battery Capacity (kWh)")
# # plt.ylabel("Total MAP Battery Capacity (kWh)")
# # plt.title("Feasible vs Infeasible Solutions")
# #
# # plt.grid(True, alpha=0.3)
# # plt.legend(title="Feasibility")
# # plt.savefig("random.svg", bbox_inches='tight')
# # #plt.show()
# # plt.tight_layout()
# # plt.show()
