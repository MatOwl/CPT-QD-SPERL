# 2026-04-26 — Welfare 求和域修复 + Paper Table 1/2 复现 (Phase 3)

## 任务

1. 解决 HANDOFF 长期遗留问题: Welfare 量级和 paper 差 6× (paper -24.15 vs 我们 -154.86 for CPT88/p=0.72)
2. 跨 CPT 配置复现 paper Table 1 (Optimality) + Table 2 (Social Welfare)
3. 在 paper 称为 "hardest convergence" 的 CPT88/p=0.66 上做 Alg 3/4 ablation

## Welfare 求和域修复

读 paper Manuscript p18 Definition 5: `SW(π) = Σ_x V^π(x)`。问题是"x 取哪些"——我们之前在 Barberis 上枚举 `(t, z)` 全 36 个，包含 (a) 不可达 (z 与 t 不同奇偶) 和 (b) 终态 (t=T)。OptEx 一直用 BFS reachable tree 所以没问题，只 Barberis 过度枚举。

**验证** (CPT88/p=0.72, baseline, 3 seeds):

| Filter | SW (count) | SPE oracle SW | Paper |
|---|---|---|---|
| 全枚举 | 36 states | -156.27 | — |
| 可达 + 终态 | 21 states | -96.28 | — |
| **可达，去终态** | **15 states** | **-22.78** | **-24.15 ± 1.21** ✓ |

**修复**: [lib/envs/featurizers.py:60](../lib/envs/featurizers.py:60) `BarberisFeaturizer.iter_states` 改为只 yield 可达决策 state (`range(-t, t+1, 2)`，且 `t < T`)。

**修复后 baseline 数字** (results_swfix):
- Optimality: 8.39 ± 0.74 (paper 7.91 ± 1.64) ✓ 不变 (V at x_0 不受求和域影响)
- Policy disagree: **1.67/15** (vs 旧 15/36) — 更有意义，反映真正决策 state 上的差异
- SPE Welfare (oracle): **-23.30** (paper -24.15 ± 1.21) ✓ 1σ 内
- SPERL SW: -35.53 ± 11.11 (无 paper 直接对比，但量级合理)

**注意**: 这个 fix 让 Barberis 的所有 metric (policy_disagree, value_error, social_welfare) 的求和域和 paper 一致；旧 `results_*` 目录里的数值 base 在过度枚举上，不能直接和新数比较。重要的不变量: `optimality` 不变，`SPE Welfare (ref)` 落入 paper 1σ。

## Paper Table 1/2 跨配置复现 (CPT88)

抓出 paper §4.2.5 的 Table 1 (Optimality) 和 Table 2 (Social Welfare):

**Paper Table 1 — Optimality**:
| p_win | Paper SPERL | Paper SPE oracle |
|---|---|---|
| CPT88/0.72 | 7.91 ± 1.64 | 8.07 ± 0.39 |
| CPT88/0.66 | 2.74 ± 0.95 | 3.59 ± 0.30 |
| CPT88/0.60 | 0.0 ± 0.0 | 0.0 ± 0.0 |

**Paper Table 2 — Social Welfare**:
| p_win | Paper SPERL | Paper SPE oracle |
|---|---|---|
| CPT88/0.72 | -26.03 ± 3.63 | -24.15 ± 1.21 |
| CPT88/0.66 | -59.94 ± 2.88 | -51.5 ± 1.06 |
| CPT88/0.60 | -80.09 ± 4.22 | -79.28 ± 0.67 |

**我们 (welfare fix 后, 3 seeds × 15k eps)**:
| Config | Metric | 我们 (baseline) | Paper SPE / SPERL |
|---|---|---|---|
| CPT88/0.72 | SPE Welfare | -23.30 | -24.15 ± 1.21 ✅ |
| | SPERL Optimality | 8.39 ± 0.74 | 7.91 ± 1.64 ✅ |
| | SPERL SW | -35.53 ± 11.11 | -26.03 ± 3.63 ⚠️ off (high variance, 3 seeds 不够) |
| CPT88/0.66 | SPE Welfare | -51.29 | -51.5 ± 1.06 ✅ |
| | SPERL Optimality | 4.57 ± 2.31 | 2.74 ± 0.95 (我们更高) |
| | SPERL SW | -59.31 ± 4.43 | -59.94 ± 2.88 ✅ |

**结论**: SPE oracle 在所有测过的配置上完美对齐 paper 1σ → welfare fix 正确。SPERL 的 Optimality 比 paper 高、SW 一致 → 我们和 paper 落在 SPE 多重性的不同点上 (SW 等价但 V(x_0) 不同)。

## Alg 3/4 在 hardest config (CPT88/p=0.66) 上的 ablation

| Config | Optimality | Disagree | VE | SW |
|---|---|---|---|---|
| Baseline (off) | **4.57 ± 2.31** | 3.67/15 | 23.64 | -59.31 |
| Sticky (tie=0.01) + Filter | -0.02 ± 0.03 | 6.67/15 | 27.68 | -67.34 |
| Sticky (tie=0) + Filter | **0.00 ± 0.00** ❌ | 7.67/15 | 19.86 | -63.51 |
| Paper SPERL (with Alg 3/4) | 2.74 ± 0.95 | — | 9.72 | -59.94 |

**惊人发现**: 我们 port 的 Alg 3/4 在 paper 称为"hardest convergence" 的 CPT88/p=0.66 上**主动把 policy 锁到"exit"** (Optimality → 0 in all seeds)。`tie_thresh=0`（strict-better only）甚至更糟。HANDOFF 早就预测过 ("sticky 过早锁死了早期噪声 policy")。

可能原因 (排查优先级):
1. **Hyperparameter mismatch**: paper Appendix C.2.5 应有 SPERL 训练超参 (eps, batch, train_eps, critic_lr 等)；我们用的是 HANDOFF 给的 `eps=0.3, critic-lr=0.04, train-eps=15000`，可能不是 paper 的设置
2. **Sticky 实现差异**: 我们的 `update_from_critic_values` `is_better_guard` 用 strict ">"，paper 可能用 ">= + epsilon" 或 SoftMax；需要对照 paper Appendix C.2.2 的 Algorithm 3 伪代码
3. **Paper Table 4 filter sweep**: 我们用 `--filter-thresh 0.75` (paper 范围最低)。Paper p=0.66 的 Pareto-optimal 是 `filter=0.9` (VE 9.3 ± 0.95)。需要 sweep filter 才能看到 Alg 4 的最佳效果

## 产物
```
results_swfix/                       # welfare fix 后的 baseline (CPT88/p=0.72)
results_cpt88_p066_baseline/         # CPT88/p=0.66 baseline
results_cpt88_p066_alg34/            # CPT88/p=0.66 with sticky+filter (tie=.01)
results_cpt88_p066_alg34_strict/     # CPT88/p=0.66 with sticky+filter (tie=0)
scripts/sw_breakdown.py              # SW filter-definition 比较脚本
```
