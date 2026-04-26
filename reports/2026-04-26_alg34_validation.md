# 2026-04-26 — Algorithm 3/4 验证 (Phase 1 + Phase 2)

## 任务

验证 [2026-04-24 port](2026-04-24_refactor_bugs_and_alg34_port.md) 的 Algorithm 3 (sticky tie-break) 和 Algorithm 4 (quantile filter) 在 Barberis (regression) 和 OptEx (主战场) 上的实际效果。

## 关键概念修正

SPE policy 在 x_0 处的值 **可以高于** SPE oracle 计算出的值。SPE 不是任意一点的最优解，是 Bellman-CPT 算子的不动点；多个 SPE 可以共存，每个有不同的 V(x_0)。"SPERL 的 V > oracle V" **不**是 bug，是均衡多重性的体现。

## Phase 1 — Barberis Alg 3/4 回归测试

**配置**: CPT88 (α=0.88, ρ₁=ρ₂=0.65, λ=2.25), p=0.72, 3 seeds × 15k eps，[lib/envs/featurizers.py](../lib/envs/featurizers.py) 修复**前**的 36-state 求和域

| 配置 | Optimality | Disagree | Value Error |
|---|---|---|---|
| All off (regression check) | 8.39 ± 0.74 | 15.7/36 | 70.6 |
| Sticky only | 6.56 ± 1.59 | 15.3/36 | 74.4 |
| Filter only | 7.82 ± 0.97 | 16.0/36 | 77.0 |
| Sticky + filter | 9.38 ± 1.40 | 13.3/36 | 74.5 |
| Paper | 7.91 ± 1.64 | — | — |

**结论**:
- ✅ 所有配置在 paper 1σ 带内 → port 没破回归
- ⚠️ Disagree 在 Barberis 上 13-16/36 几乎不动（2-action 空间多重性少）
- 3 seeds 噪声 (per-seed 摆幅 4-11) 完全淹没配置间差异——后续如要在 Barberis 上做 head-to-head，需 ≥10 seeds

## Phase 2 — OptEx σ=0.015 Alg 3/4 ablation

**配置**: σ=0.015, num_w=4, horizon=5, 3 seeds × 3k eps

| 指标 | Baseline (off) | Sticky only | Sticky + Filter |
|---|---|---|---|
| Disagree / 493 | **241.0 ± 3.7** | **242.7 ± 0.9** | **242.0 ± 2.4** |
| Optimality | -0.037 ± 0.010 | -0.033 ± 0.006 | -0.024 ± 0.003 |
| Value error | 2.77 ± 0.09 | 2.77 ± 0.11 | 2.73 ± 0.03 |
| Cross-seed stable / 493 | 433 (87.8%) | 431 (87.4%) | 428 (86.8%) |
| ↳ stable AND matches SPE | 245/433 = **56.6%** | 244/431 = **56.6%** | 244/428 = **57.0%** |
| mean\|Δv\| at disagree | 0.0113 | 0.0113 | 0.0111 |

**当时的解读 (HANDOFF "启动包" 第 2 项的 B 档假设)**:
- Disagreement 在 3 个配置间完全不动 (241/242/242)，paper 的 Alg 3/4 不能把 SPERL 推向 paper oracle 选定的那个 SPE
- Cross-seed 稳定的 ~430 个 state 中只 57% 匹配 oracle——剩下 43% 不是噪声/抖动，是 SPERL 确定地学到了**另一个有效 SPE**
- Alg 3/4 **确实有微弱效果**:
  - Sticky 把 cross-seed std 从 3.7 缩到 0.9（4× 改善），符合 Alg 3 的稳定收敛设计目标
  - Sticky + filter 把 Optimality bias 从 -0.037 拉到 -0.024，更接近 oracle V(x_0)
  - 但 policy 分布的差异看似无法消除

> ⚠️ 这个解读后被 [Phase 4 visit-frequency 分析](2026-04-26_optex_visit_frequency.md) 修正：49% disagreement 实际是 **off-equilibrium-path artifact**，on-path SPERL 几乎完美匹配 SPE。

## 产物
```
results_alg34/                    # Barberis sticky+filter
results_alg34_off/                # Barberis 全 off
results_alg34_stickyonly/         # Barberis sticky-only
results_alg34_filteronly/         # Barberis filter-only
results_alg34_optex/              # OptEx sticky+filter (含 disagree heatmap)
results_alg34_optex_stickyonly/   # OptEx sticky-only (含 disagree heatmap)
```
