import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
plt.style.use('bmh')


SAVE_DIR = "figures/"
os.makedirs(SAVE_DIR, exist_ok=True)


def save_df(branches, values, prefixes, c_statistics, backups, time_bnb, time_sarsop, fn):
    df = pd.DataFrame()
    for key in branches:
        df_ = pd.DataFrame()
        n_actions, n_types = key
        df_['branches'] = branches[key]
        df_['prefix_length'] = [len(p) for p in prefixes[key]]
        df_['prefix_n_actions'] = [p.num_values for p in prefixes[key]]
        df_['values'] = values[key]
        df_['c_statistics'] = c_statistics[key]
        df_['backups'] = backups[key]
        df_['time_bnb'] = time_bnb[key]
        df_['time_sarsop'] = time_sarsop[key]
        df_['n_actions'] = n_actions
        df_['n_types'] = n_types
        df = pd.concat([df, df_])
    print(f'{SAVE_DIR}{fn}')
    df.to_csv(f'{SAVE_DIR}{fn}', index=False)
    return df


def plot_branches_and_c(keys, branches, c_statistics, fn, x_axis=None, x_label=None, xlog=False):
        colors = plt.cm.viridis(np.linspace(0, .4, 2, endpoint=True))

        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.tick_params(axis='y', labelcolor=colors[0])
        if x_axis is None:
            x_axis = range(len(keys))
            ax1.set_xticks(keys, fontsize=20)
        if x_label is not None:
            ax1.set_xlabel(x_label, fontsize=20)
        if xlog:
            ax1.set_xscale('log')
        ax1.set_ylabel('branches', fontsize=20)

        b_means, b_lows, b_highs, c_means, c_lows, c_highs = [], [], [], [], [], []
        for k in keys:
            b_log = np.array(branches[k])
            b_means.append(np.median(b_log))
            interval = bootstrap(np.array([b_log]), np.median, method='basic').confidence_interval
            b_lows.append(interval.low), b_highs.append(interval.high)

            c_log = np.array(c_statistics[k])
            c_means.append(np.median(c_log))
            interval = bootstrap(np.array([c_log]), np.median, method='basic').confidence_interval
            c_lows.append(interval.low), c_highs.append(interval.high)

        ax1.plot(x_axis, b_means, marker='s', label='B&B (ours)', c=colors[0])
        ax1.fill_between(x_axis, b_lows, b_highs, alpha=0.25, color=colors[0])
        ax1.set_ylim(bottom=0)

        ax2 = ax1.twinx()
        ax2.set_yscale('log')
        ax2.plot(x_axis, c_means, c=colors[1], ls='--')
        ax2.fill_between(x_axis, c_lows, c_highs, alpha=0.25, color=colors[1])
        ax2.tick_params(axis='y', which='major', labelsize=20, labelcolor=colors[1])
        ax2.set_ylabel(r'$c(\mathbf{q}, \mathbf{P})$', fontsize=20)

        print(f'{SAVE_DIR}{fn}')
        plt.savefig(f'{SAVE_DIR}{fn}', bbox_inches='tight')


def plot_c(keys, branches, c_statistics, fn):
        colors = plt.cm.plasma(np.linspace(0, .75, len(keys), endpoint=True))

        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)

        from sklearn.svm import SVR
        for i, k in enumerate(keys):
            label = f'm={k[1]}'
            b_log = np.array(branches[k])
            c_log = np.log(np.array(c_statistics[k]))

            svr = SVR(kernel='rbf')
            svr.fit(c_log.reshape(-1, 1), b_log)
            x = np.linspace(np.quantile(c_log, 0.01), np.quantile(c_log, 0.99), 100, endpoint=True)
            y = svr.predict(x.reshape(-1, 1))
            ax1.plot(x, y, c=colors[i], label=label)

        ax1.tick_params(axis='both', which='major', labelsize=20)
        ax1.set_xlabel(r'$\log(c)$', fontsize=20)
        ax1.legend(fontsize=16)
        ax1.set_ylim(bottom=0)

        print(f'{SAVE_DIR}{fn}')
        plt.savefig(f'{SAVE_DIR}{fn}', bbox_inches='tight')


def plot_baseline(keys, branches, backups, fn, x_axis=None, x_label=None, xlog=False):
        colors = plt.cm.viridis(np.linspace(0, .4, 2, endpoint=True))

        fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)
        ax1.tick_params(axis='both', which='major', labelsize=20)
        if x_axis is None:
            x_axis = range(len(keys))
            ax1.set_xticks(keys, fontsize=20)
        if x_label is not None:
            ax1.set_xlabel(x_label, fontsize=20)
        if xlog:
            ax1.set_xscale('log')

        b_means, b_lows, b_highs, c_means, c_lows, c_highs = [], [], [], [], [], []
        for k in keys:
            b_log = np.array(branches[k])
            b_means.append(np.mean(b_log))
            interval = bootstrap(np.array([b_log]), np.mean, method='basic', confidence_level=.95).confidence_interval
            b_lows.append(interval.low), b_highs.append(interval.high)

            c_log = np.array(backups[k])
            c_means.append(np.mean(c_log))
            interval = bootstrap(np.array([c_log]), np.mean, method='basic', confidence_level=.95).confidence_interval
            c_lows.append(interval.low), c_highs.append(interval.high)

        ax1.plot(x_axis, b_means, marker='s', label='B&B (ours)', c=colors[0])
        ax1.fill_between(x_axis, b_lows, b_highs, alpha=0.25, color=colors[0])

        ax1.plot(x_axis, c_means, marker='s', label='SARSOP', c=colors[1], ls='--')
        ax1.fill_between(x_axis, c_lows, c_highs, alpha=0.25, color=colors[1])

        plt.legend(fontsize=20)
        ax1.set_ylim(bottom=0)

        print(f'{SAVE_DIR}{fn}')
        plt.savefig(f'{SAVE_DIR}{fn}', bbox_inches='tight')


def plot_beliefs(keys, beliefs, fn, horizon):
    plt.figure(figsize=(7, 5), dpi=300)
    colors = plt.cm.plasma(np.linspace(0, .75, len(keys), endpoint=True))
    for c, k in zip(colors, keys):
        n_actions, n_types = k
        label = f'{n_types} types'
        x = np.arange(horizon+1)
        y = []
        print("Correlation of average uncertainty with closest exponential functions:")
        for i, belief_seq in enumerate(beliefs[k]):
            if len(belief_seq) > 2:
                belief_seq = np.array(belief_seq)
                vertex = np.diff(belief_seq, axis=0)[-1].argmax()
                y_temp = 1-belief_seq[:horizon+1, vertex]
                if 4 < i < 9:  # cherry-picking more interesting examples to show
                    plt.plot(x, y_temp, color=c, alpha=0.15)
                y.append(y_temp)
        y = np.stack(y, axis=1).mean(axis=1)
        plt.plot(x, y, label=label, color=c)

        def exp_f(x, scale):
            return y[0] * np.exp(-scale * x)
        from scipy.optimize import curve_fit
        scale = curve_fit(exp_f, np.arange(y.shape[0]), y, p0=(0.1,))[0]
        y_exp = np.apply_along_axis(exp_f, 0, x, scale=scale)
        r2 = np.corrcoef(y, y_exp)
        print(f"{n_actions} actions, {n_types} types: R^2 = {r2[0][1]}")

    plt.xlim(0, horizon)
    plt.ylim(0, 1)
    plt.xticks(np.arange(0, horizon+1, 50, dtype=int), fontsize=20)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1], fontsize=20)
    plt.xlabel("number of rounds", fontsize=20)
    plt.legend(fontsize=16)

    print(f'{SAVE_DIR}{fn}')
    plt.savefig(f'{SAVE_DIR}{fn}', bbox_inches='tight')


if __name__ == '__main__':
    all_files = os.listdir(SAVE_DIR)
    csv_files = list(filter(lambda f: f.endswith('.csv'), all_files))

    for fn_csv in csv_files:
        fn = fn_csv.split('.csv')[0]
        df = pd.read_csv(f'{SAVE_DIR}{fn_csv}', index_col=False)
        df["n_actions"] = df["n_actions"].astype(int)
        df["n_types"] = df["n_types"].astype(int)

        keys = df[["n_actions", "n_types"]].drop_duplicates()
        bad_keys = [(60, 60), (10, 15), (15, 10)]
        keys = [(keys.loc[idx, "n_actions"], keys.loc[idx, "n_types"]) for idx in keys.index]
        keys = [k for k in keys if k not in bad_keys]
        if fn == 'types':
            x_label = '10 actions, x types'
            x_axis = [k[1] for k in keys]
        elif fn == 'types_and_actions':
            x_label = 'x actions, x types'
            x_axis = [k[0] for k in keys]
        elif fn == 'actions':
            x_label = 'x actions, 10 types'
            x_axis = [k[0] for k in keys]

        branches, c_statistics, backups, time_bnb, time_sarsop = {}, {}, {}, {}, {}
        for key in keys:
            mask = (df["n_actions"] == key[0]) & (df["n_types"] == key[1])
            branches[key] = df.loc[mask, 'branches']
            c_statistics[key] = df.loc[mask, 'c_statistics']
            backups[key] = df.loc[mask, 'backups']
            time_bnb[key] = df.loc[mask, 'time_bnb']
            time_sarsop[key] = df.loc[mask, 'time_sarsop']

        # plot_branches_and_c(keys, branches, c_statistics, f'{fn}.png', x_axis, x_label=x_label)
        # plot_baseline(keys, branches, backups, f'{fn}_baseline_time.png', x_axis, x_label=x_label)
        plot_baseline(keys, time_bnb, time_sarsop, f'{fn}_baseline_time.png', x_axis)
