import pandas as pd
from scipy.stats import wilcoxon


if __name__ == "__main__":
    df_a = pd.read_csv('figures/actions.csv', index_col=False)
    df_t = pd.read_csv('figures/types.csv', index_col=False)
    df_ta = pd.read_csv('figures/types_and_actions.csv', index_col=False)

    keys_a = df_a[["n_actions", "n_types"]].drop_duplicates()
    keys_t = df_t[["n_actions", "n_types"]].drop_duplicates()
    keys_ta = df_ta[["n_actions", "n_types"]].drop_duplicates()

    keys_a = [(keys_a.loc[idx, "n_actions"], keys_a.loc[idx, "n_types"]) for idx in keys_a.index]
    keys_t = [(keys_t.loc[idx, "n_actions"], keys_t.loc[idx, "n_types"]) for idx in keys_t.index]
    keys_ta = [(keys_ta.loc[idx, "n_actions"], keys_ta.loc[idx, "n_types"]) for idx in keys_ta.index]

    print("Actions")
    for key in keys_a:
        mask = (df_a["n_actions"] == key[0]) & (df_a["n_types"] == key[1])
        print(key, wilcoxon(df_a.loc[mask, "time_bnb"], df_a.loc[mask, "time_sarsop"]).pvalue)
    print('all', wilcoxon(df_a["time_bnb"], df_a["time_sarsop"]).pvalue)

    print("Types")
    for key in keys_t:
        mask = (df_t["n_actions"] == key[0]) & (df_t["n_types"] == key[1])
        print(key, wilcoxon(df_t.loc[mask, "time_bnb"], df_t.loc[mask, "time_sarsop"]).pvalue)
    print('all', wilcoxon(df_t["time_bnb"], df_t["time_sarsop"]).pvalue)

    print("Types and Actions")
    for key in keys_ta:
        mask = (df_ta["n_actions"] == key[0]) & (df_ta["n_types"] == key[1])
        print(key, wilcoxon(df_ta.loc[mask, "time_bnb"], df_ta.loc[mask, "time_sarsop"]).pvalue)
    print('all', wilcoxon(df_ta["time_bnb"], df_ta["time_sarsop"]).pvalue)
