import time
import os
import sys
import numpy as np
import pandas as pd
import pathlib
from tqdm import tqdm
from scipy.stats import bootstrap
from typing import List, Tuple

np.random.seed(42)  # For reproducibility

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from Modules.baselines import sarsop
from Modules.simulations import run_RECAPC
from movielens_simulations.aggregate_clusters import aggregate_clusters
from create_clusters_main import main as create_clusters

class TeeOutput:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def calculate_bootstrap_ci(data: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float, float]:
    result = bootstrap((data,), np.mean, confidence_level=confidence_level)
    return np.mean(data), result.confidence_interval.low, result.confidence_interval.high

def create_summary_table(results_df: pd.DataFrame, cluster_sizes: List[int]) -> pd.DataFrame:
    summary_data = []
    for size in cluster_sizes:
        size_data = results_df[results_df['size'] == size]
        row_data = {'Size': size}
        for metric in ['time_diff', 'c_stat']:
            mean, low, high = calculate_bootstrap_ci(size_data[metric].values)
            row_data[f'{metric}_mean'] = mean
            row_data[f'{metric}_ci_low'] = low
            row_data[f'{metric}_ci_high'] = high
        summary_data.append(row_data)
    return pd.DataFrame(summary_data)

def print_summary_table(summary_df: pd.DataFrame):
    print("\nSummary of time comparison by matrix size:")
    print("Size | Time Difference (mean [95% CI]) | C-statistic (mean [95% CI])")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        print(f"{int(row['Size']):4d} | "
              f"{row['time_diff_mean']:+.4f} [{row['time_diff_ci_low']:.4f}, {row['time_diff_ci_high']:.4f}] | "
              f"{row['c_stat_mean']:.4f} [{row['c_stat_ci_low']:.4f}, {row['c_stat_ci_high']:.4f}]")

def create_q_vector(clusters_dir):
    user_clusters = pd.read_csv(os.path.join(clusters_dir, 'user_clusters.csv'))
    cluster_counts = user_clusters['Cluster'].value_counts()
    q = cluster_counts / len(user_clusters)

    max_cluster = q.index.max()
    q_vector = np.zeros(max_cluster + 1)
    for cluster, proportion in q.items():
        q_vector[cluster] = proportion

    return q_vector / q_vector.sum()

def add_noise_to_matrix(P, noise_std):
    noise = np.random.normal(0, noise_std, P.shape)
    noised_P = P + noise
    return np.clip(noised_P, 0.01, 0.99)

def run_comparison_for_size(cluster_size, sim_num=1, noised_mat_num=500, noise_std=0.005, verbose=True):
    clusters_dir = create_clusters(n_clusters=cluster_size)
    P, count_matrix = aggregate_clusters(cluster_size=cluster_size)
    q = create_q_vector(clusters_dir)

    if verbose:
        print(f"Matrix size: {P.shape}")
        print(f"Prior distribution size: {q.shape}")

    results = []
    total_iterations = noised_mat_num * sim_num

    with tqdm(total=total_iterations, desc=f"Size {cluster_size}x{cluster_size}") as pbar:
        for noise_iter in range(noised_mat_num):
            noised_P = add_noise_to_matrix(P, noise_std)
            for sim_iter in range(sim_num):
                result = {'size': cluster_size, 'noise_iteration': noise_iter,
                         'simulation_iteration': sim_iter}

                sarsop_time, sarsop_backups, _ = sarsop(noised_P, q, verbose=False)
                _, _, _, _, recapc_time, c_stat = run_RECAPC(noised_P, q, verbose=False)

                result.update({
                    'sarsop_time': float(sarsop_time),
                    'recapc_time': float(recapc_time),
                    'time_diff': float(recapc_time - sarsop_time),
                    'c_stat': float(c_stat)
                })

                results.append(result)
                pbar.update(1)

    return results

def main():
    parent_folder = pathlib.Path(__file__).parent.absolute()
    output_file = os.path.join(parent_folder, 'comparison_output.txt')
    if os.path.exists(output_file):
        os.remove(output_file)
    sys.stdout = TeeOutput(output_file)

    cluster_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    all_results = []

    for size in cluster_sizes:
        print(f"\nRunning comparison for size {size}x{size}")
        all_results.extend(run_comparison_for_size(size))

    results_df = pd.DataFrame(all_results)
    summary_df = create_summary_table(results_df, cluster_sizes)

    print_summary_table(summary_df)

    # Save results to CSV
    results_csv_path = os.path.join(parent_folder, 'comparison_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nResults saved to CSV: {results_csv_path}")

    sys.stdout = sys.__stdout__
    print(f"Experiment completed. Results saved to {output_file}")
    print(f"CSV data saved to {results_csv_path}")

if __name__ == "__main__":
    main()