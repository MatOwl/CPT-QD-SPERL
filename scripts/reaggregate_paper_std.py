"""Re-aggregate paper §4.1 PE / VE σ on existing sweep data, without retraining.

Why: the original `lib/io.py:_aggregate_paper_formula` took σ on (a_hat, v_hat)
which under rebuilt-once-CRN gives ≈0 — wildly off paper Tables 3/4. Switching
to (a_tilde, v_tilde) is correct per paper §4.1; this script rebuilds the
`policy_error_paper` / `value_error_paper` entries in each cell's
`aggregate.json` from the per-seed `per_state_values.csv` without the cost of a
re-run.

Usage:
    PYTHONPATH=. python scripts/reaggregate_paper_std.py runs/results_paper_tables_3_4_v3
    PYTHONPATH=. python scripts/reaggregate_paper_std.py runs/results_paper_tables_1_2_v3
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np


def _load_per_state(seed_dir: str):
    """Read per_state_values.csv → list of dicts."""
    path = os.path.join(seed_dir, "per_state_values.csv")
    if not os.path.exists(path):
        return None
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "state": row["state"],
                "a_tilde": float(row["a_tilde"]),
                "a_hat": float(row["a_hat"]),
                "v_tilde": float(row["v_tilde"]),
                "v_hat": float(row["v_hat"]),
            })
    return rows


def _paper_aggregate(per_state_by_seed: list[list[dict]]) -> dict:
    """Paper §4.1:
        PE μ = Σ_x | μ_seeds[π̃(x)] - μ_seeds[π̂(x)] |
        VE μ = Σ_x | μ_seeds[Ṽ(x)] - μ_seeds[V̂(x)] |
        PE σ = (1/|X|) Σ_x σ_seeds[π̃(x)]
        VE σ = (1/|X|) Σ_x σ_seeds[Ṽ(x)]
    """
    # Collect per-state arrays across seeds, joining by state-key string.
    state_keys = [r["state"] for r in per_state_by_seed[0]]
    state_data = {s: {"a_tilde": [], "a_hat": [], "v_tilde": [], "v_hat": []}
                  for s in state_keys}
    for rows in per_state_by_seed:
        for r in rows:
            s = r["state"]
            if s not in state_data:
                state_data[s] = {"a_tilde": [], "a_hat": [],
                                 "v_tilde": [], "v_hat": []}
            state_data[s]["a_tilde"].append(r["a_tilde"])
            state_data[s]["a_hat"].append(r["a_hat"])
            state_data[s]["v_tilde"].append(r["v_tilde"])
            state_data[s]["v_hat"].append(r["v_hat"])

    pe_mean_terms = []
    ve_mean_terms = []
    pe_std_terms = []
    ve_std_terms = []
    for s, d in state_data.items():
        if not d["a_tilde"]:
            continue
        a_tilde = np.asarray(d["a_tilde"], dtype=np.float64)
        a_hat = np.asarray(d["a_hat"], dtype=np.float64)
        v_tilde = np.asarray(d["v_tilde"], dtype=np.float64)
        v_hat = np.asarray(d["v_hat"], dtype=np.float64)
        pe_mean_terms.append(abs(a_tilde.mean() - a_hat.mean()))
        ve_mean_terms.append(abs(v_tilde.mean() - v_hat.mean()))
        pe_std_terms.append(a_tilde.std())
        ve_std_terms.append(v_tilde.std())

    n_x = len(pe_mean_terms)
    return {
        "policy_error_paper": {
            "mean": float(sum(pe_mean_terms)),
            "std": float(sum(pe_std_terms) / n_x),
            "n_states": n_x,
        },
        "value_error_paper": {
            "mean": float(sum(ve_mean_terms)),
            "std": float(sum(ve_std_terms) / n_x),
            "n_states": n_x,
        },
    }


def reaggregate_cell(cell_dir: str) -> bool:
    """Update the paper-formula entries inside ``cell_dir/aggregate.json``.

    Returns True if updated, False if missing input data.
    """
    seed_dirs = sorted(
        d for d in os.listdir(cell_dir)
        if d.startswith("seed") and os.path.isdir(os.path.join(cell_dir, d))
    )
    per_state_by_seed = []
    for sd in seed_dirs:
        rows = _load_per_state(os.path.join(cell_dir, sd))
        if rows is None:
            continue
        per_state_by_seed.append(rows)
    if not per_state_by_seed:
        return False

    paper = _paper_aggregate(per_state_by_seed)

    agg_path = os.path.join(cell_dir, "aggregate.json")
    if os.path.exists(agg_path):
        with open(agg_path) as f:
            agg = json.load(f)
    else:
        agg = {}
    agg["policy_error_paper"] = paper["policy_error_paper"]
    agg["value_error_paper"] = paper["value_error_paper"]
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    return True


def walk_cells(root: str):
    """Yield each cell dir (one level above seed{N}) under ``root``."""
    for dirpath, dirnames, _ in os.walk(root):
        if any(d.startswith("seed") for d in dirnames):
            yield dirpath


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"not a directory: {root}", file=sys.stderr)
        sys.exit(1)
    n_done = 0
    n_skip = 0
    for cell in walk_cells(root):
        ok = reaggregate_cell(cell)
        if ok:
            n_done += 1
        else:
            n_skip += 1
    print(f"reaggregated: {n_done} cells, skipped (no per_state_values.csv): {n_skip}")


if __name__ == "__main__":
    main()
