"""Make `agents/rerun_GreedySPERL_QR__main.py` importable without
record_csv / stable_baselines installed.

Inject lightweight stubs into sys.modules BEFORE importing the legacy
script. record_csv calls become no-ops; set_global_seeds becomes
`np.random.seed`.

Usage:
    import scripts.legacy_compat_stubs  # noqa: F401  (must come first)
    # ...then use the legacy classes via direct copy/extraction
"""

import sys
import types

import numpy as np


def _make_record_csv_stub():
    mod = types.ModuleType("record_csv")

    def _noop(*args, **kwargs):
        return None

    mod.record_params = _noop
    mod.record_results = _noop
    mod.record_quantiles = _noop
    mod.record_csv = _noop
    return mod


def _make_stable_baselines_stub():
    sb = types.ModuleType("stable_baselines")
    sb_common = types.ModuleType("stable_baselines.common")
    sb_misc = types.ModuleType("stable_baselines.common.misc_util")

    def set_global_seeds(seed):
        np.random.seed(int(seed))
        try:
            import random
            random.seed(int(seed))
        except Exception:
            pass

    sb_misc.set_global_seeds = set_global_seeds
    sb_common.misc_util = sb_misc
    sb.common = sb_common
    return sb, sb_common, sb_misc


sys.modules.setdefault("record_csv", _make_record_csv_stub())
_sb, _sb_common, _sb_misc = _make_stable_baselines_stub()
sys.modules.setdefault("stable_baselines", _sb)
sys.modules.setdefault("stable_baselines.common", _sb_common)
sys.modules.setdefault("stable_baselines.common.misc_util", _sb_misc)
