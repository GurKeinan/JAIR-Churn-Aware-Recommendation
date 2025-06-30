import time
import os
from tqdm import tqdm
import numpy as np
import json
import sys
import pathlib

np.random.seed(0)  # for reproducibility

# handle relative imports
if os.path.basename(os.getcwd()) != "Modules":
    from Modules.branch_and_bound import RECAPC
    from Modules.utils import drop_controlled_actions, SparseRepeatedArray
    from Modules.plot import plot_branches_and_c, plot_baseline, plot_c, plot_beliefs, save_df
else:
    from branch_and_bound import RECAPC
    from utils import drop_controlled_actions, SparseRepeatedArray
    from plot import plot_branches_and_c, plot_baseline, plot_c, plot_beliefs, save_df


USE_BNB_CPP = True
if USE_BNB_CPP:
    import subprocess
    root_path = pathlib.Path(__file__).parent.parent
    BNB_PATH = str(root_path / "bnb_cpp")  # path to the compiled C++ binary
    # mingw_path = "path/to/mingw64/bin"  # path to mingw64/bin folder
    # MINGW_ENV = os.environ.copy()
    # MINGW_ENV["PATH"] = mingw_path + ";" + MINGW_ENV["PATH"]
    MINGW_ENV = os.environ.copy()  # No need to modify the PATH for macOS


def get_random_P(n_actions, n_types, threshold=1e-2):
    """
    Samples sigmoid activations from normal distribution and transforms into probabilities
    :return: P
    """
    # log_P = np.random.normal(mean, std, (n_actions, n_types))
    # P = 1 / (1 + np.exp(-log_P))

    dim = n_actions
    latent_action_vectors = np.random.normal(0, 1, (n_actions, dim))
    latent_type_vectors = np.random.normal(0, 1, (n_types, dim))
    norm_a = np.linalg.norm(latent_action_vectors, ord=2, axis=1).reshape(-1, 1)
    norm_t = np.linalg.norm(latent_type_vectors, ord=2, axis=1).reshape(1, -1)
    norm = norm_a @ norm_t
    P = (latent_action_vectors @ latent_type_vectors.T) / norm
    P = (P + 1) / 2

    P = drop_controlled_actions(P)
    P = np.clip(P, threshold, 1-threshold)
    return P


def get_random_q(n_types, std=.5, threshold=1e-2):
    """
    Samples softmax activations from normal distribution and transforms into probabilities
    :return: q
    """
    log_q = np.random.normal(0, std, (n_types,))
    q = np.exp(log_q)
    q /= q.sum()
    q = np.clip(q, threshold/n_types, 1-threshold)
    q /= q.sum()
    return q


def get_random_RECAPC(n_actions, n_types, q_std=.5, verbose=True):
    max_generations = 100
    for _ in range(max_generations):
        P, q = get_random_P(n_actions, n_types), get_random_q(n_types, q_std)
        # filter trivial problems
        bandit = RECAPC(P, q, verbose=False)
        if bandit.V_upper(q) > bandit.V_lower(q) + 1e-8:
            break
    if verbose:
        print(f'P: {P.round(3)}')
        print(f'q: {q.round(3)}')
        print()
    return P, q


def get_many_random_RECAPC(n_actions, n_types, n_runs, q_std=.5):
    Ps, qs = [], []
    for _ in range(n_runs):
        P, q = get_random_RECAPC(n_actions, n_types, q_std=q_std, verbose=False)
        Ps.append(P), qs.append(q)
    return Ps, qs


def run_bandit(P, q, min_horizon):
    bandit = RECAPC(P, q, verbose=False)
    c_statistic = bandit.c_statistic()
    if not USE_BNB_CPP:
        start_time = time.time()
        best_value, best_prefix, belief_log, explored_branches = bandit.branch_and_bound(min_horizon)
        elapsed_time = time.time() - start_time
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT

        # method 1: pass all to command line -- fails if size of P is too big
        # P_str = ' '.join(map(str, P.flatten()))
        # q_str = ' '.join(map(str, q))

        # method 2: provide as input -- works for any size
        data = {"P": P.tolist(), "q": q.tolist()}
        data_str = json.dumps(data).encode('utf-8')

        for _ in range(5):  # in case of rare fragmentation errors
            try:
                proc = subprocess.run(
                    # [BNB_PATH, str(P.shape[0]), str(P.shape[1]), P_str, q_str],  # use for method 1
                    [BNB_PATH], input=data_str,  # use for method 2
                    env=MINGW_ENV,
                    stdout=stdout,
                    stderr=stderr,
                    check=True
                )
                break
            except subprocess.CalledProcessError:
                pass
        stdout_text = proc.stdout.decode().split("\n")
        best_value = eval(stdout_text[0].split(": ")[1])
        best_prefix = eval(stdout_text[1].split(": ")[1])
        best_prefix = SparseRepeatedArray(best_prefix)
        explored_branches = eval(stdout_text[2].split(": ")[1])
        elapsed_time = eval(stdout_text[3].split(": ")[1])
        belief_log = None
    return best_value, best_prefix, belief_log, explored_branches, elapsed_time, c_statistic


def run_RECAPC(P, q, min_horizon=0, verbose=True):
    best_value, best_prefix, belief_log, explored_branches, elapsed_time, c_statistic = run_bandit(P, q, min_horizon)
    if verbose:
        print(f'value: {round(best_value, 5)}, '
              f'prefix: [{best_prefix}], '
              f'prefix length: {len(best_prefix)}, '
              f"c statistic={c_statistic}, "
              f"elapsed time: {round(elapsed_time, 6)} seconds\n")
    return best_value, best_prefix, belief_log, explored_branches, elapsed_time, c_statistic


def time_many_RECAPC(Ps, qs, n_runs, min_horizon=0):
    time_bnb, time_sarsop, branches, values, prefixes, c_statistics, beliefs, backups = [], [], [], [], [], [], [], []
    for P, q in tqdm(zip(Ps, qs)):
        value, prefix, belief, branch, t_bnb, c_statistic = run_RECAPC(P, q, min_horizon, verbose=False)
        time_bnb.append(int(t_bnb*1000))

        t_sarsop, n_backups, _ = sarsop(P, q)
        time_sarsop.append(int(t_sarsop*1000))

        branches.append(branch), values.append(value), prefixes.append(prefix), c_statistics.append(c_statistic)
        backups.append(n_backups), beliefs.append(belief)
    print(f'{n_runs} runs: '
          f'total time bnb: {round(sum(time_bnb), 2)}, '
          f'total time sarsop: {round(sum(time_sarsop), 2)}, '
          f'avg branches: {int(sum(branches) / n_runs)}, '
          f'avg sarsop backups: {int(sum(backups) / n_runs)}')
    return branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop


def time_many_random_RECAPC(actions_types, n_runs, min_horizon=0):
    branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop = {}, {}, {}, {}, {}, {}, {}, {}
    for n_actions, n_types in actions_types:
        print(f'{n_actions} actions, {n_types} types')
        Ps, qs = get_many_random_RECAPC(n_actions, n_types, n_runs)
        key = (n_actions, n_types)
        branches[key], values[key], prefixes[key], c_statistics[key], beliefs[key], backups[key], \
            time_bnb[key], time_sarsop[key] = time_many_RECAPC(Ps, qs, n_runs, min_horizon)
        print(f'Frequency of non-trivial solutions: {np.mean([p.num_values>1 for p in prefixes[key]])}')
        print()
    return branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop


def time_and_plot_branches(n_runs, seed=None):
    np.random.seed(seed)

    # long prob matrix
    horizon = 150
    fn = 'types'
    x_axis = [5, 10, 20, 40, 60, 80, 100]
    keys = [(10, x) for x in x_axis]
    branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop = time_many_random_RECAPC(
        keys, n_runs, horizon)
    save_df(branches, values, prefixes, c_statistics, backups, time_bnb, time_sarsop, f'{fn}.csv')
    plot_branches_and_c(keys, branches, c_statistics, f'{fn}.png', x_axis)
    plot_baseline(keys, branches, backups, f'{fn}_baseline.png', x_axis)
    plot_baseline(keys, time_bnb, time_sarsop, f'{fn}_baseline_time.png', x_axis)
    # plot_c(keys, branches, c_statistics, f'{fn}_c.png')

    if not USE_BNB_CPP:
        x_beliefs = [5, 10, 40, 100]
        keys = [(10, x) for x in x_beliefs]
        plot_beliefs(keys, beliefs, f'uncertainty.png', horizon)

    # square prob matrix
    fn = 'types_and_actions'
    x_axis = [5, 10, 15, 20, 30, 40]
    keys = [(x, x) for x in x_axis]
    branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop = time_many_random_RECAPC(
        keys, n_runs)
    save_df(branches, values, prefixes, c_statistics, backups, time_bnb, time_sarsop, f'{fn}.csv')
    plot_branches_and_c(keys, branches, c_statistics, f'{fn}.png', x_axis)
    plot_baseline(keys, branches, backups, f'{fn}_baseline.png', x_axis)
    plot_baseline(keys, time_bnb, time_sarsop, f'{fn}_baseline_time.png', x_axis)

    # high prob matrix
    fn = 'actions'
    x_axis = [5, 10, 20, 40, 60, 80, 100]
    keys = [(x, 10) for x in x_axis]
    branches, values, prefixes, c_statistics, beliefs, backups, time_bnb, time_sarsop = time_many_random_RECAPC(
        keys, n_runs)
    save_df(branches, values, prefixes, c_statistics, backups, time_bnb, time_sarsop, f'{fn}.csv')
    plot_branches_and_c(keys, branches, c_statistics, f'{fn}.png', x_axis)
    plot_baseline(keys, branches, backups, f'{fn}_baseline.png', x_axis)
    plot_baseline(keys, time_bnb, time_sarsop, f'{fn}_baseline_time.png', x_axis)


if __name__ == '__main__':
    from baselines import sarsop

    # one random simulation with detailed info
    # P, q = get_random_RECAPC(5, 5, q_std=.5)
    # run_RECAPC(P, q)

    # many random simulations with aggregated info
    # n_actions, n_types = 10, 10
    # n_runs = 1000
    # np.random.seed(42)
    # time_many_random_RECAPC([(n_actions, n_types)], n_runs)

    # all experiments
    time_and_plot_branches(n_runs=500, seed=42)
