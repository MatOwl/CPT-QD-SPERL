"""Compare social-welfare sums under different state-filter definitions.

Paper Definition 5: SW(pi) = sum_x V^pi(x). The question is which x are included.
Barberis featurizer enumerates all (t, z in [-t*bet, t*bet] step bet), but reachable
states require z and t to share parity (each step is +/-bet). Terminal step t=T
has no decision and is rarely included in welfare comparisons.
"""
import ast
import sys
import numpy as np
import pandas as pd

# Barberis: bet=10, T=5 decisions, n_t=6 states
BET = 10
T = 5

def reachable_no_term(state):
    t, z = state
    if t >= T:
        return False
    if abs(z) > t * BET:
        return False
    return ((z // BET) + t) % 2 == 0

def reachable_with_term(state):
    t, z = state
    if abs(z) > t * BET:
        return False
    return ((z // BET) + t) % 2 == 0

def all_states(state):
    return True

def summarize(name, paths, col):
    rows = []
    for cfg, path in paths:
        sums_all, sums_rt, sums_r = [], [], []
        for seed in range(3):
            df = pd.read_csv(f"{path}/barberis_sperl_p0.72_cpt_a0.88_r0.65_l2.25/seed{seed}/per_state_values.csv")
            df["s"] = df["state"].apply(ast.literal_eval)
            sums_all.append(df[col].sum())
            sums_rt.append(df.loc[df["s"].apply(reachable_with_term), col].sum())
            sums_r.append(df.loc[df["s"].apply(reachable_no_term), col].sum())
        rows.append((cfg, np.mean(sums_all), np.mean(sums_rt), np.mean(sums_r)))
    print(f"\n=== {name} (col={col}) ===")
    print(f"{'config':<24} {'all_36':>10} {'reach+term_21':>14} {'reach_no_term_15':>18}")
    for cfg, sa, srt, sr in rows:
        print(f"{cfg:<24} {sa:>10.2f} {srt:>14.2f} {sr:>18.2f}")

paths = [
    ("baseline_off", "runs/results_alg34_off"),
    ("sticky_only", "runs/results_alg34_stickyonly"),
    ("filter_only", "runs/results_alg34_filteronly"),
    ("sticky+filter", "runs/results_alg34"),
]

summarize("SPERL policy V_tilde", paths, "v_tilde")
summarize("SPE oracle V_hat", [paths[0]], "v_hat")

# Show state counts under each filter
df0 = pd.read_csv("runs/results_alg34_off/barberis_sperl_p0.72_cpt_a0.88_r0.65_l2.25/seed0/per_state_values.csv")
df0["s"] = df0["state"].apply(ast.literal_eval)
print(f"\nstate counts: all={len(df0)}, reach+term={df0['s'].apply(reachable_with_term).sum()}, reach_no_term={df0['s'].apply(reachable_no_term).sum()}")
print("\npaper CPT88/p=0.72: SPE Welfare = -24.15 +/- 1.21")
