"""3-policy comparison on LNW: SPE / DO / SPSA (vs SPERL learned).

For a given LNW config, computes:
  - SPE oracle policy (CPT-aware backward induction)
  - DO oracle policy (EV-maximizing backward induction, ignores CPT)
  - (Existing SPERL learned policy from saved per_state_values.csv)
  - SPSA precommitment policy (trained fresh)

Then evaluates ALL policies' V^pi(x_0) under CPT preferences and tabulates.

Story: V^SPE(x_0) is the upper bound (CPT-aware planning). V^DO(x_0)
quantifies the cost of CPT-blind planning. V^SPSA(x_0) quantifies the
cost of precommitment (no per-state credit assignment).

Usage:
    python scripts/lnw_do_comparison.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from lib.cpt import CPTParams
from lib.envs.registry import make_env, make_featurizer
from lib.envs.abandonment_spe import compute_spe_policy, spe_policy_fn
from lib.envs.abandonment_do import compute_do_policy, do_policy_fn
from lib.paper_eval import (
    rollout_cpt_from_state, abandonment_reset_kwargs,
    abandonment_initial_offset_builder,
)


def evaluate_policy(env, policy_fn, x_0, cpt: CPTParams, n_eps: int = 500,
                    initial_offset_fn=None):
    """Return CPT(rollout_returns) starting from x_0 under policy_fn."""
    return rollout_cpt_from_state(
        env, policy_fn, x_0, n_eps, cpt,
        reset_kwargs_for_state=abandonment_reset_kwargs,
        initial_offset_for_state=initial_offset_fn,
    )


def evaluate_all_states(env, policy_fn, featurizer, cpt, n_eps=200,
                         initial_offset_fn=None):
    """Sum of V^pi(x) across paper-eval domain X (= social welfare)."""
    sw = 0.0
    per_state = {}
    for x in featurizer.iter_states():
        v = rollout_cpt_from_state(
            env, policy_fn, x, n_eps, cpt,
            reset_kwargs_for_state=abandonment_reset_kwargs,
            initial_offset_for_state=initial_offset_fn,
        )
        sw += v
        per_state[x] = v
    return sw, per_state


def run_config(config_label, env_kwargs, cpt: CPTParams, n_eval_eps=500,
                spe_rollouts=2000):
    print(f"\n=== {config_label} ===")
    env = make_env("abandonment", **env_kwargs)
    feat = make_featurizer("abandonment", env)
    initial_offset_fn = abandonment_initial_offset_builder(env)

    print("  Computing SPE oracle...")
    spe_dict = compute_spe_policy(env, cpt, n_eval_eps=spe_rollouts)
    spe_pi = spe_policy_fn(spe_dict)

    print("  Computing DO oracle (EV-max)...")
    do_dict = compute_do_policy(env, n_eval_eps=spe_rollouts)
    do_pi = do_policy_fn(do_dict)

    # Compare policies state-by-state
    n_disagree_do_vs_spe = 0
    abandonment_diff = []
    for state, a_spe in spe_dict.items():
        a_do = do_dict.get(state, 0)
        if a_spe != a_do:
            n_disagree_do_vs_spe += 1
            abandonment_diff.append((state, a_spe, a_do))

    print(f"  Policies disagree on {n_disagree_do_vs_spe}/{len(spe_dict)} states")
    if n_disagree_do_vs_spe > 0 and n_disagree_do_vs_spe <= 10:
        for s, a_s, a_d in abandonment_diff[:10]:
            print(f"    state {s}: SPE={('C' if a_s==1 else 'A')} "
                  f"DO={('C' if a_d==1 else 'A')}")

    # V at start state
    x_0 = (0, 0)
    print("  Evaluating V(x_0) under CPT for each policy...")
    v_spe_x0 = evaluate_policy(env, spe_pi, x_0, cpt, n_eval_eps,
                               initial_offset_fn)
    v_do_x0 = evaluate_policy(env, do_pi, x_0, cpt, n_eval_eps,
                              initial_offset_fn)

    print("  Evaluating SW under each policy...")
    sw_spe, _ = evaluate_all_states(env, spe_pi, feat, cpt, n_eps=200,
                                    initial_offset_fn=initial_offset_fn)
    sw_do, _ = evaluate_all_states(env, do_pi, feat, cpt, n_eps=200,
                                   initial_offset_fn=initial_offset_fn)

    return {
        "config": config_label,
        "v_spe_x0": v_spe_x0,
        "v_do_x0": v_do_x0,
        "sw_spe": sw_spe,
        "sw_do": sw_do,
        "disagree_do_vs_spe": n_disagree_do_vs_spe,
        "n_states": len(spe_dict),
        "spe_policy_summary": _policy_summary(spe_dict, env.T),
        "do_policy_summary": _policy_summary(do_dict, env.T),
    }


def _policy_summary(policy_dict, T):
    """Compact text summary of policy: per-stage continue regions."""
    out = []
    for t in range(T):
        actions = []
        for k in range(-t, t + 1, 2):
            a = policy_dict.get((t, k), 0)
            actions.append(f"x={k:+d}:{'C' if a == 1 else 'A'}")
        out.append(f"t={t}: " + " ".join(actions))
    return out


def main():
    cpt88 = CPTParams(0.88, 0.65, 0.65, 2.25)
    cpt95 = CPTParams(0.95, 0.5, 0.5, 1.5)

    configs = [
        ("T=5 p=0.72 CPT88 (headline)",
         dict(p=0.72, delta=10, c=11, T=5, x1=50), cpt88),
        ("T=5 p=0.6  CPT88 (stay-out)",
         dict(p=0.6, delta=10, c=11, T=5, x1=50), cpt88),
        ("T=5 p=0.72 CPT95",
         dict(p=0.72, delta=10, c=11, T=5, x1=50), cpt95),
        ("T=7 p=0.72 CPT88",
         dict(p=0.72, delta=10, c=11, T=7, x1=70), cpt88),
    ]

    rows = []
    for label, env_kwargs, cpt in configs:
        r = run_config(label, env_kwargs, cpt, n_eval_eps=500, spe_rollouts=2000)
        rows.append(r)

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY: SPE vs DO baseline V(x_0) and SW under CPT")
    print("=" * 80)
    df = pd.DataFrame([
        {
            "config": r["config"],
            "V_SPE(x_0)": f"{r['v_spe_x0']:.3f}",
            "V_DO(x_0)": f"{r['v_do_x0']:.3f}",
            "DO loss": f"{r['v_do_x0'] - r['v_spe_x0']:.3f}",
            "SW_SPE": f"{r['sw_spe']:.1f}",
            "SW_DO": f"{r['sw_do']:.1f}",
            "SW loss": f"{r['sw_do'] - r['sw_spe']:.1f}",
            "disagree": f"{r['disagree_do_vs_spe']}/{r['n_states']}",
        }
        for r in rows
    ])
    print(df.to_string(index=False))

    # Save
    out_dir = Path("runs/_lnw_do_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "do_vs_spe.csv", index=False)
    print(f"\nSaved {out_dir / 'do_vs_spe.csv'}")

    # Also save policy summaries for each config
    with open(out_dir / "policy_summaries.txt", "w") as f:
        for r in rows:
            f.write(f"\n=== {r['config']} ===\n")
            f.write(f"SPE policy:\n")
            for line in r["spe_policy_summary"]:
                f.write(f"  {line}\n")
            f.write(f"DO policy:\n")
            for line in r["do_policy_summary"]:
                f.write(f"  {line}\n")
    print(f"Saved {out_dir / 'policy_summaries.txt'}")


if __name__ == "__main__":
    main()
