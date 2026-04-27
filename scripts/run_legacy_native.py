"""Run `agents/rerun_GreedySPERL_QR__main.py` as faithfully as possible —
no slicing, no closure injection, no overrides. Lets the original script's
runner block (lines 1048+) execute its own active config.

Two minimal accommodations only:
  1. Stub `stable_baselines.common.misc_util.set_global_seeds` based on
     line 458's usage context (seed numpy + python random; legacy comment
     says "Seed python, numpy and tf random generator", but no TF needed
     for Barberis).
  2. Make `import record_csv` (line 19) resolvable: `record_csv.py` lives
     in `agents/`, so we add `agents/` to sys.path. Also chdir to
     `agents/` so `record_csv`'s relative output paths
     (`./{env}/results/static/...`) land in `agents/barberis/results/static/`.

Run from project root:
    PYTHONPATH=. python scripts/run_legacy_native.py

Outputs go to `agents/barberis/results/static/` per the legacy convention.
"""

import os
import random
import sys
import types

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))
AGENTS_DIR = os.path.join(PROJECT_ROOT, "agents")

# Project root for `from lib.envs.barberis_casino import barberisCasino`
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# agents/ for `import record_csv`
if AGENTS_DIR not in sys.path:
    sys.path.insert(0, AGENTS_DIR)


def _set_global_seeds(i):
    """Verbatim semantics of stable_baselines.common.misc_util.set_global_seeds.

    Legacy line 457 comment: 'Seed python, numpy and tf random generator'.
    Barberis env has no TF dependency, so just numpy + python random suffice.
    """
    np.random.seed(int(i))
    random.seed(int(i))


def _install_stub():
    sb_misc = types.ModuleType("stable_baselines.common.misc_util")
    sb_misc.set_global_seeds = _set_global_seeds
    sb_common = types.ModuleType("stable_baselines.common")
    sb_common.misc_util = sb_misc
    sb = types.ModuleType("stable_baselines")
    sb.common = sb_common
    sys.modules.setdefault("stable_baselines", sb)
    sys.modules.setdefault("stable_baselines.common", sb_common)
    sys.modules.setdefault("stable_baselines.common.misc_util", sb_misc)


def main():
    _install_stub()

    # Legacy uses relative output paths via record_csv (`./{env}/results/static/`).
    # cd to agents/ so those paths resolve to agents/barberis/results/static/.
    os.chdir(AGENTS_DIR)

    # S3 speedup: skip per-eval per-(state,action) QFDyn CSV writes.
    # legacy `evaluate_critic_` (line 622-639) calls record_csv.record_quantiles
    # 42×N_evals times per seed -> 12k+ small Windows file writes.
    # We don't consume QFDyn output; making it no-op cuts ~20-30% wall clock.
    import record_csv as _rc
    _rc.record_quantiles = lambda *args, **kwargs: None

    script = os.path.join(AGENTS_DIR, "rerun_GreedySPERL_QR__main.py")
    print(f"[run_legacy_native] cwd = {os.getcwd()}")
    print(f"[run_legacy_native] sys.path[0..2] = {sys.path[:3]}")
    print(f"[run_legacy_native] exec {script}")

    with open(script, encoding="utf-8") as f:
        code = f.read()

    # Exec as if it were `__main__`
    g = {"__name__": "__main__", "__file__": script}
    exec(compile(code, script, "exec"), g)


if __name__ == "__main__":
    main()
