import pandas as pd
from scipy.stats import wilcoxon
import os

def run_wilcoxon_tests(csv_file='comparison_results_005.csv'):
    """
    Run Wilcoxon signed-rank tests on the comparison results.
    Compares RECAPC vs SARSOP times for each matrix size and overall.
    """

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run run_experiments.py first.")
        return

    # Read the results
    df = pd.read_csv(csv_file)

    print("Wilcoxon Signed-Rank Test Results")
    print("=" * 50)
    print("Comparing RECAPC vs SARSOP execution times")
    print("Null hypothesis: No difference in median times")
    print("Alternative: RECAPC and SARSOP have different median times")
    print()

    # Get unique matrix sizes
    sizes = sorted(df['size'].unique())

    print("Results by Matrix Size:")
    print("-" * 30)

    # Test for each matrix size
    for size in sizes:
        mask = df['size'] == size
        size_data = df[mask]

        recapc_times = size_data['recapc_time']
        sarsop_times = size_data['sarsop_time']

        # Run Wilcoxon signed-rank test
        try:
            statistic, p_value = wilcoxon(recapc_times, sarsop_times)
            n_pairs = len(recapc_times)

            # Calculate some descriptive stats
            recapc_median = recapc_times.median()
            sarsop_median = sarsop_times.median()

            print(f"Size {size}x{size}:")
            print(f"  Sample size: {n_pairs} pairs")
            print(f"  RECAPC median: {recapc_median:.6f}s")
            print(f"  SARSOP median: {sarsop_median:.6f}s")
            print(f"  Wilcoxon statistic: {statistic}")
            print(f"  p-value: {p_value:.2e}")

            # Interpret significance
            if p_value < 0.001:
                sig_level = "***"
            elif p_value < 0.01:
                sig_level = "**"
            elif p_value < 0.05:
                sig_level = "*"
            else:
                sig_level = "ns"

            print(f"  Significance: {sig_level}")
            print()

        except Exception as e:
            print(f"Size {size}x{size}: Error - {e}")
            print()

    # Overall test across all sizes
    print("Overall Comparison (All Sizes Combined):")
    print("-" * 40)

    try:
        statistic, p_value = wilcoxon(df['recapc_time'], df['sarsop_time'])
        n_pairs = len(df)

        recapc_median = df['recapc_time'].median()
        sarsop_median = df['sarsop_time'].median()

        print(f"Total sample size: {n_pairs} pairs")
        print(f"RECAPC median: {recapc_median:.6f}s")
        print(f"SARSOP median: {sarsop_median:.6f}s")
        print(f"Wilcoxon statistic: {statistic}")
        print(f"p-value: {p_value:.2e}")

        # Interpret significance
        if p_value < 0.001:
            sig_level = "*** (highly significant)"
        elif p_value < 0.01:
            sig_level = "** (very significant)"
        elif p_value < 0.05:
            sig_level = "* (significant)"
        else:
            sig_level = "ns (not significant)"

        print(f"Significance: {sig_level}")

        # Effect size interpretation
        if recapc_median < sarsop_median:
            faster_algo = "RECAPC"
            speedup = sarsop_median / recapc_median
        else:
            faster_algo = "SARSOP"
            speedup = recapc_median / sarsop_median

        print(f"\nInterpretation: {faster_algo} is faster by a factor of {speedup:.2f}")

    except Exception as e:
        print(f"Overall test error: {e}")

    print("\n" + "=" * 50)
    print("Significance levels: *** p<0.001, ** p<0.01, * p<0.05, ns p>=0.05")

if __name__ == "__main__":
    run_wilcoxon_tests()