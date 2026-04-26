"""Environment + featurizer registry for the generic experiment runner."""

from __future__ import annotations

from .barberis_casino import barberisCasino
from .optimal_execution import OptimalExecution
from .abandonment_project import AbandonmentProject
from .featurizers import (
    BarberisFeaturizer,
    OptExFeaturizer,
    AbandonmentFeaturizer,
)


def make_env(name: str, **kwargs):
    name = name.lower()
    if name in ("barberis", "barberis_casino"):
        return barberisCasino(**kwargs)
    if name in ("optex", "optimal_execution"):
        return OptimalExecution(**kwargs)
    if name in ("abandonment", "abandonment_project", "lnw"):
        return AbandonmentProject(**kwargs)
    raise ValueError(f"Unknown env: {name}")


def make_featurizer(name: str, env):
    name = name.lower()
    if name in ("barberis", "barberis_casino"):
        return BarberisFeaturizer(env)
    if name in ("optex", "optimal_execution"):
        return OptExFeaturizer(env)
    if name in ("abandonment", "abandonment_project", "lnw"):
        return AbandonmentFeaturizer(env)
    raise ValueError(f"Unknown featurizer: {name}")


REGISTERED_ENVS = ("barberis", "optex", "abandonment")
