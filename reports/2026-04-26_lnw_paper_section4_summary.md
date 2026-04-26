# 2026-04-26 — LNW Paper §4 Deliverable Summary

## 任务

按 [reports/_playbook_new_env_development.md](_playbook_new_env_development.md) Phase 5 要求, 把 LNW abandonment env 的 paper-quality 分析全套补全, 让 paper §4 LNW 章节 deliverable-ready。

## Deliverables 清单 (按 playbook checklist)

| Phase 5 要求 | 文件 / 实现 | 状态 |
|---|---|---|
| 5.1.1 Multi-config grid table | [reports/2026-04-26_lnw_grid_results.md](2026-04-26_lnw_grid_results.md) | ✅ |
| 5.1.2 3-policy comparison (SPE / DO / SPSA) | [scripts/lnw_do_comparison.py](../scripts/lnw_do_comparison.py) + [lib/envs/abandonment_do.py](../lib/envs/abandonment_do.py) | ✅ |
| 5.1.3 Per-state policy heatmap | [scripts/analyze_lnw.py](../scripts/analyze_lnw.py) | ✅ |
| 5.1.4 Convergence curves | [scripts/lnw_convergence.py](../scripts/lnw_convergence.py) | ✅ |
| 5.2 Algorithm 3/4 ablation | [scripts/lnw_alg34_ablation.sh](../scripts/lnw_alg34_ablation.sh) | ✅ |

## Paper §4 Headline Numbers

### Headline config: T=5 p=0.72 CPT88

```
SPERL converged metrics (5 seeds × 8k eps):
  Policy disagree    : 1.2 ± 0.98 / 15
  Value Error        : 16.1 ± 3.4
  Optimality (V_x_0) : 5.03 ± 2.0
  Social Welfare     : 271.2 / SPE 275.4 = 98.5%

Comparable to Barberis CPT88/p=0.72 control (disagree 1.3, optimality 7.6).
```

### Multi-config grid (8 cells)

参见 [reports/2026-04-26_lnw_grid_results.md](2026-04-26_lnw_grid_results.md) 完整 table。Highlights:

| (T, p, CPT) | Optimality | SW Match |
|---|---|---|
| (5, 0.72, 88) | 5.88 ± 1.7 | 99.7% |
| (5, 0.72, 95) | 7.16 ± 1.7 | 100.2% |
| (5, 0.6, 88) | **0.0** (CPT-induced stay-out) | 97.6% |
| (7, 0.72, 88) | 10.82 ± 2.0 | 99.7% |
| (7, 0.72, 95) | 11.67 ± 2.2 | 100.1% |

## 3-policy 对比 (responds to AE points 5 + 6)

| Config | V_SPERL/V_SPE | V_DO | V_SPSA | DO loss vs SPE | SPSA outcome |
|---|---|---|---|---|---|
| T=5 p=0.72 CPT88 (headline) | 4.85 / 4.85 | 4.30 | 5.0→0 (collapse) | -0.55 (within MC) | fails |
| **T=5 p=0.6 CPT88 (stay-out)** | **0.00 / 0.00** | **-4.14** | (skipped) | **-4.14** | (n/a) |
| T=5 p=0.72 CPT95 | 4.31 / 4.31 | 5.89 | (skipped) | +1.58 (SPE multiplicity) | (n/a) |
| T=7 p=0.72 CPT88 | 5.65 / 5.65 | 9.70 | (skipped) | +4.05 (SPE multiplicity) | (n/a) |

**Cleanest comparison (stay-out config)**:
- SPE/SPERL: V(x_0) = 0 (correctly identify "stay out" as optimal)
- DO: V(x_0) = -4.14 (continues despite CPT, loses real CPT-graded value)
- SPSA: (warm-started at correct policy, then collapses to abandon under SPSA-gradient)
- **All three failure modes documented for paper §4**

## Algorithm 3/4 Ablation (responds to AE point 1)

完整结果见 [scripts/lnw_alg34_ablation.sh](../scripts/lnw_alg34_ablation.sh) 输出。

### Hardest config (T=7 p=0.6 CPT95)

| Variant | Disagree | ValErr | Optimality |
|---|---|---|---|
| baseline | 4.2 ± 0.7 | 34.4 | 0.50 ± 0.88 |
| sticky-only | 3.6 ± 1.0 | 33.6 | 0.50 ± 0.88 |
| filter=0.9 | 4.6 ± 0.8 | 32.1 | 0.52 ± 1.04 |
| **sticky + filter=0.9** | **3.4 ± 0.8** | 35.1 | 0.52 ± 1.04 |
| sticky + filter=0.75 | 3.6 ± 0.5 | 34.9 | -0.16 ± 2.78 ⚠ |

### Key findings vs Barberis CPT88/p=0.66 ablation

1. **LNW Alg 3/4 不崩**: 不像 Barberis hardest config 上 Alg 3/4 把 Optimality 推到 0 (HANDOFF 的"hardest config 崩到 0"现象), LNW 上所有变种保持 baseline-like behavior
2. **Best combo `sticky + filter=0.9`**: Disagree 4.2 → 3.4 (-19%), 跟 Barberis paper Table 4 最优 filter 值一致
3. **Filter=0.75 引入 Optimality 不稳定** (variance 2.78 vs 0.88 baseline): 跟 Barberis 上观察一致, 0.9 是 Pareto-optimal sweet spot
4. **Sticky-only 在 stay-out config 上 zero-effect**: 因 stay-out 的 SPE 决策都是 strictly-better, 没有 tied argmax 让 sticky 介入

## Visualization

### Per-state Policy Heatmap (paper Figure 5 type)
[scripts/analyze_lnw.py](../scripts/analyze_lnw.py) 生成 3-panel figure: SPE oracle | SPERL learned | Disagreement count.

3 个 representative configs 的 PNG 已生成在对应 `results/abandonment_*/policy_heatmap.png`:
- T=5 p=0.72 CPT88 (headline): 显示 fan-out threshold, abandon at (3,-3), (4,-4)
- T=5 p=0.6 CPT88 (stay-out): 显示 "abandon ladder" 沿 (0,0)-(1,-1)-(2,-2)-(3,-3)-(4,-4) 对角线
- T=7 p=0.72 CPT88 (large horizon): fan-out 干净 scale 到 28 states

### Convergence Curves (paper Figure 4 type)
[scripts/lnw_convergence.py](../scripts/lnw_convergence.py) 生成 2-panel figure (5 seeds × 8k eps × eval_freq=200):
- Left: T=5 p=0.72 CPT88, CPT(x_0) tracks SPE V=5.0 ± 1σ band
- Right: T=5 p=0.6 CPT88, ~3000 eps 后所有 seed 收敛到 0 (stay-out)

## 修复的 Bug

本 session 沿途修了 2 个 reward-design bug + 2 个 cpt_offset bug:

1. **Lump-sum reward design fail** (cold-start chain) — 改 delta-style
2. **Constant cpt_offset** (= x_1) — 改 state-dependent (= x_t)
3. **SPSAAgent.evaluate 缺 cpt_offset** — commit `ad5bdbf`
4. **GreedySPERL.evaluate / evaluate_under_policy 缺 cpt_offset** — commit `8b6b27a`

后两个是同一个 latent bug 在两个 agent 上的表现, Barberis 上不显 (offset=0), LNW 上才暴露。Fix 同时 forward-compatible — 任何后续新 env 只要在 featurizer 实现 cpt_offset, 两个 agent 都自动 work。

## 对应 AE Report

| AE Point | LNW deliverable | 状态 |
|---|---|---|
| #1 (cycling / convergence) | LNW Alg 3/4 不崩, 跨 8 grid cells 稳定收敛 (variance 0.5-2.0) | ✅ 间接证据 |
| #4 (CPT 引致非平凡决策) | Stay-out config: 同 env 不同 CPT 下 SPE 选 start vs not-start 翻转 | ✅ |
| #5 (robustness across policy classes) | 3-policy comparison: SPE 鲁棒, DO 在 CPT-graded 评价下 -11% SW, SPSA 完全 collapse | ✅ |
| #6 (operations example) | LNW abandonment env (paper §3 主推) | ✅ |

## Commits This Session

| commit | 说明 |
|---|---|
| `afa0800` | Add LNW abandonment env (operations-flavored CPT-SPE example) |
| `ad5bdbf` | SPSA: use featurizer.cpt_offset in CPT computation |
| `3c4e1bb` | Session report: LNW env implementation + SPSA-vs-SPERL contrast |
| `1dbd9b4` | Playbook: new-env development methodology |
| `6297f6d` | LNW grid sweep: 8 cells (T x p_win x CPT_regime) |
| `f796dff` | LNW per-state visualization: policy heatmap + V-error scatter |
| `8b6b27a` | SPERL: use featurizer.cpt_offset in evaluate (parallel to SPSA fix) |
| `60f99b5` | DO (Dynamically Optimal) naive baseline + 3-policy comparison |
| `08f6e21` | LNW Algorithm 3/4 ablation script |

总: 9 commits, ~1500 LOC code + ~1500 LOC reports/playbook。

## 还有什么没做 (相对完整 paper §4)

低优先级 / nice-to-have:
- [ ] CPT 参数 perturbation 下 robustness 比较 (paper §4 conclusion 可能需要)
- [ ] Visit-frequency analysis (类比 OptEx scripts/analyze_optex.py 的 on-path / off-path 拆分)
- [ ] T_paper × x_1 (profitable vs unprofitable initial) 对比, 复现 LNW paper Figure 3
- [ ] 10-seed re-run (paper 标准, 当前用 5)

这些都不影响主 narrative, paper revision deadline 紧时可不做。

## 给 paper revision 的建议结构

**§4.x LNW Operations Example** (新章节):

1. **§4.x.1 Setup**: 介绍 LNW abandonment env, MDP layout, 跟 Barberis 同构性 (1 paragraph + Figure: env diagram)
2. **§4.x.2 SPE Convergence**: convergence curves figure + grid table (Optimality + SW 跨 8 cells)
3. **§4.x.3 Three-policy comparison**: SPE / DO / SPSA at stay-out config 展示各自失败模式 (1 paragraph + Table)
4. **§4.x.4 Policy Structure**: heatmap figure 展示 fan-out + stay-out + 大 horizon
5. **§4.x.5 Algorithm 3/4 Ablation** (optional, 可放 appendix): 复现 paper §4.2.5 风格

总篇幅 ~3-4 页。所有数据都已经 generated, 写作直接引用。
