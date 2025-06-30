import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import os

plt.style.use('bmh')

SAVE_DIR = "figures/"
os.makedirs(SAVE_DIR, exist_ok=True)

def calculate_bootstrap_ci(data: np.ndarray, confidence_level: float = 0.95):
    """Calculate bootstrap confidence interval for the mean."""
    result = bootstrap((data,), np.mean, confidence_level=confidence_level, method='basic')
    return np.mean(data), result.confidence_interval.low, result.confidence_interval.high

def plot_baseline(results_df: pd.DataFrame, cluster_sizes: list, save_path: str):
    """
    Plot baseline comparison in the exact style of the original plot.py
    """
    colors = plt.cm.viridis(np.linspace(0, .4, 2, endpoint=True))

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Convert times to milliseconds
    results_df = results_df.copy()
    results_df['recapc_time'] = results_df['recapc_time'] * 1000  # Convert to ms
    results_df['sarsop_time'] = results_df['sarsop_time'] * 1000  # Convert to ms

    # Prepare data containers
    recapc_means, recapc_lows, recapc_highs = [], [], []
    sarsop_means, sarsop_lows, sarsop_highs = [], [], []

    # Calculate means and confidence intervals for each size
    for size in cluster_sizes:
        size_data = results_df[results_df['size'] == size]

        # RECAPC data
        recapc_data = size_data['recapc_time'].values
        mean, low, high = calculate_bootstrap_ci(recapc_data)
        recapc_means.append(mean)
        recapc_lows.append(low)
        recapc_highs.append(high)

        # SARSOP data
        sarsop_data = size_data['sarsop_time'].values
        mean, low, high = calculate_bootstrap_ci(sarsop_data)
        sarsop_means.append(mean)
        sarsop_lows.append(low)
        sarsop_highs.append(high)

    # Plot RECAPC (equivalent to "B&B (ours)" in the original)
    ax.plot(cluster_sizes, recapc_means, marker='s', label='B&B (ours)', c=colors[0])
    ax.fill_between(cluster_sizes, recapc_lows, recapc_highs, alpha=0.25, color=colors[0])

    # Plot SARSOP
    ax.plot(cluster_sizes, sarsop_means, marker='s', label='SARSOP', c=colors[1], ls='--')
    ax.fill_between(cluster_sizes, sarsop_lows, sarsop_highs, alpha=0.25, color=colors[1])

    # Formatting exactly like plot.py - no axis labels, no grid
    plt.legend(fontsize=20)
    ax.set_ylim(bottom=0)

    print(f'{save_path}')
    plt.savefig(save_path, bbox_inches='tight')

def print_summary_statistics(results_df: pd.DataFrame):
    """Print summary statistics for each matrix size."""
    print("\nSummary Statistics (times in milliseconds):")
    print("Size | RECAPC Time (mean ± std) | SARSOP Time (mean ± std) | Speed Improvement")
    print("-" * 85)

    # Convert to milliseconds for display
    results_df = results_df.copy()
    results_df['recapc_time'] = results_df['recapc_time'] * 1000
    results_df['sarsop_time'] = results_df['sarsop_time'] * 1000

    cluster_sizes = sorted(results_df['size'].unique())
    for size in cluster_sizes:
        size_data = results_df[results_df['size'] == size]

        recapc_mean = size_data['recapc_time'].mean()
        recapc_std = size_data['recapc_time'].std()
        sarsop_mean = size_data['sarsop_time'].mean()
        sarsop_std = size_data['sarsop_time'].std()

        speedup = sarsop_mean / recapc_mean if recapc_mean > 0 else float('inf')

        print(f"{size:4d} | {recapc_mean:8.2f} ± {recapc_std:6.2f}     | {sarsop_mean:8.2f} ± {sarsop_std:6.2f}     | {speedup:6.2f}x")

def main():
    # Load the results from run_experiments.py
    results_file = 'comparison_results.csv'

    if not os.path.exists(results_file):
        print(f"Error: {results_file} not found. Please run run_experiments.py first.")
        return

    # Read the results
    results_df = pd.read_csv(results_file)

    print(f"Loaded results with {len(results_df)} data points")
    cluster_sizes = sorted(results_df['size'].unique())
    print(f"Matrix sizes tested: {cluster_sizes}")

    # Create the plot
    save_path = os.path.join(SAVE_DIR, 'baseline_time.png')
    plot_baseline(results_df, cluster_sizes, save_path)

    # Print summary statistics
    print_summary_statistics(results_df)

if __name__ == "__main__":
    main()