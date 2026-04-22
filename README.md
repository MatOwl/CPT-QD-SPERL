# CPT-QD-SPERL

Quantile-based distributional RL under Cumulative Prospect Theory (CPT), with
a time-consistent (subgame-perfect equilibrium) learning algorithm and
precommitment baselines. Reference paper: *"Quantile-based Distributional
Reinforcement Learning under Cumulative Prospect Theory and its Dynamic
Optimality Characterization"* (`paperRef/MSci_MANUSCRIPT.pdf`).

## Layout

```
lib/
  cpt.py                  # compute_CPT, CPTParams
  paper_eval.py           # paper §4.1–4.2 metrics (policy/value err, welfare)
  io.py                   # result persistence (config, stats, per-state CSV)
  envs/
    barberis_casino.py    # paper's main env (Barberis 2012 casino)
    barberis_spe.py       # backward-induction SPE oracle for Barberis
    optimal_execution.py  # OptEx env migrated from CumSPERL_ref
    optex_spe.py          # OptExTree + SPE oracle loader (.npy)
    featurizers.py        # per-env feature index + state enumerator
    registry.py           # make_env / make_featurizer dispatcher
    ...                   # other CPT MDPs (shi_PM, gridworlds, blackjack)

agents/
  sperl_qr_generic.py     # Greedy-SPERL w/ QR critic (main algo, env-agnostic)
  spsa_generic.py         # SPSA precommitment baseline (env-agnostic)
  run_experiments.py      # unified CLI: learn + evaluate (+ optional SPE cmp)
  run_paper_eval.py       # paper-style evaluation w/ seed sweep + aggregation

  rerun_GreedySPERL_QR__main.py   # legacy Barberis reproduction (paper tables)
  rerun_SPSA_QRMC__main.py        # legacy SPSA baseline on Barberis
  record_csv.py                   # CSV logging + plotting used by legacy scripts

scripts/
  benchmark.py            # train + paper-eval wall-time + optional cProfile
```

## Environments

| Name       | Obs space                         | Action space       | Notes                                    |
|------------|-----------------------------------|--------------------|------------------------------------------|
| `barberis` | `(t, wealth) ∈ Discrete × Discrete` | `Discrete(2)`      | Paper's main env; T=5, bet=10            |
| `optex`    | 3-tuple of ints (price/remain/X bins) | `Discrete(K)`    | Migrated from CumSPERL_ref; horizon 5    |

For OptEx, SPE oracles come from pre-computed `.npy` files in
`CumSPERL_ref/SPE_OptEx_*.npy`; matching `--sigma` / `--horizon` / `--num-w`
is required.

## Algorithms

- **Greedy-SPERL** (paper main): quantile-regression critic + per-state greedy
  improvement under CPT → time-consistent (SPE) policy.
- **SPSA** (baseline): Spall-style simultaneous perturbation on a tabular
  softmax actor, optimising CPT at s₀ only → precommitment policy.

Both algorithms share `CPTParams` (α, ρ₁, ρ₂, λ) and a tabular featurizer
that returns a flat index + hashable key.

### Env-sensitive tricks in the QR critic

The quantile learner in `QRCritic` is **not pure quantile regression**; it
bakes in a few heuristics that are tuned for Barberis-scale rewards and
should be revisited when porting to a new env. See the `QRCritic` docstring
for the full list. In particular:

- first-visit mean initialisation couples the critic to the reward scale;
- MC targets collapse all quantiles to the scalar return-to-go (point
  estimate, not distributional);
- the legacy CumSPERL lbub-filter is available as an opt-in via
  `--critic-clip LO HI` but is **off by default**;
- default `--critic-lr 0.1` / `--support-size 50` assume Barberis-scale
  rewards (O(10)); OptEx rewards are ~100× smaller — retune before
  blaming the algorithm.

## Quick start

```bash
# Activate the virtualenv at C:\Users\User\rl (Python 3.10)
PYTHON=C:/Users/User/rl/Scripts/python.exe

# Train Greedy-SPERL on Barberis, compare with analytical SPE oracle
$PYTHON agents/run_experiments.py --env barberis --algo sperl \
    --p-win 0.6 --train-eps 2000 --batch 10 --support-size 20

# Train Greedy-SPERL on OptEx (σ=0.015), benchmark vs preloaded SPE oracle
$PYTHON agents/run_experiments.py --env optex --algo sperl \
    --sigma 0.015 --num-w 4 --horizon 5 \
    --train-eps 500 --batch 20 --support-size 20 \
    --spe-file ../CumSPERL_ref/SPE_OptEx_5_0.015_4.npy

# Train SPSA baseline (precommitment) on OptEx
$PYTHON agents/run_experiments.py --env optex --algo spsa \
    --sigma 0.015 --seed 2 --train-eps 1000 --batch 50 \
    --spe-file ../CumSPERL_ref/SPE_OptEx_5_0.015_4.npy
```

## Paper-style evaluation

Reproduces the four metrics from MSci_MANUSCRIPT §4.1–4.2:

- **Policy Error**    `Σ_x |π̃(x) − π̂(x)|`
- **Value Error**     `Σ_x |V^π̃(x) − V^π̂(x)|`
- **Optimality**      `V^π̃(x₀)`
- **Social Welfare**  `Σ_x V^π̃(x)`

Aggregated as `(mean, stdev)` across seeds.

```bash
# Barberis: SPE oracle solved in-process via backward induction
$PYTHON agents/run_paper_eval.py --env barberis --p-win 0.72 --seeds 3 \
    --train-eps 2000 --eval-per-state 80 --spe-rollouts 200

# OptEx: SPE oracle loaded from .npy
$PYTHON agents/run_paper_eval.py --env optex --sigma 0.015 --seeds 3 \
    --train-eps 500 --eval-per-state 30 \
    --spe-file ../CumSPERL_ref/SPE_OptEx_5_0.015_4.npy
```

## Notes on reproducing the paper

The legacy scripts (`rerun_*__main.py`) are the faithful reproduction path for
the paper's Barberis tables/figures; they record detailed per-iteration
diagnostics via `record_csv`. The generic `sperl_qr_generic` / `spsa_generic`
path is the maintainable baseline for extending to new envs (OptEx, PM, …)
and runs `run_paper_eval.py` for the four core metrics.

SPSA on Barberis is known to be sensitive to the random `θ` initialisation
(some seeds collapse to `argmax(θ[s₀]) = exit` and never escape); the legacy
runner sweeps and filters initialisations, which has not been ported to the
generic path.

## Persistence

`agents/run_paper_eval.py` writes a structured result tree by default:

```
results/{env}_{algo}_{env_id}_{cpt_id}/
    config.json                  # full CLI args
    seed{i}/
        stats.npz                # learning curves
        qtile_theta.npy          # critic params (SPERL)
        greedy_action.npy        # learned policy
        per_state_values.csv     # x, a_tilde, a_hat, v_tilde, v_hat
        metrics.json             # 4 scalar metrics
    aggregate.{json,csv}         # mean/std across seeds
```

Pass `--no-save` to disable. The generic training script
(`agents/run_experiments.py`) does NOT persist by default — use
`run_paper_eval.py` when you want reproducible artifacts.

## Performance notes

`scripts/benchmark.py` measures training + evaluation wall-time.

A single tabular critic update previously allocated a full
`[n_states, n_actions, I]` zero matrix on every call, which dominated cost
on OptEx-size state spaces (48 521 states). The current `QRCritic.update`
applies the quantile-regression gradient directly on the visited slice with
no extra allocation — measured **~107× speedup** on OptEx (73.9 s → 0.69 s
for 100 training episodes). Correctness was cross-checked against earlier
runs (final CPT within 3 % of pre-optimisation numbers).

Remaining known costs:
- `compute_paper_metrics` runs `n_states × n_eps_per_state × 2` rollouts
  sequentially; the main loop is Python-level and `env.step` is cheap, so
  wall-time scales roughly linearly with `eval_per_state`. For OptEx's 493
  reachable states this is noticeable but acceptable.
- For very large sweeps, parallelising seeds (multiprocessing) is the
  largest available win and has not been wired in.

## Dependencies

`gym`, `numpy`, `scipy`, `pandas`, `matplotlib`, `pypdf` (paper parsing only),
`stable-baselines` (legacy scripts only — for `set_global_seeds`).
