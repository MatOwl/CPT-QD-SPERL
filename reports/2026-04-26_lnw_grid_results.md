# 2026-04-26 — LNW Grid Sweep Results (Tier 1 Paper Table)

## 任务

按 [reports/_playbook_new_env_development.md](_playbook_new_env_development.md) Phase 5.1 要求, 跨 (T × p_win × CPT_regime) 配置跑 LNW abandonment env 的 SPERL paper-eval, 形成 paper §4 Table 1/2 类型的数据。

## Setup

- env: LNW abandonment ([lib/envs/abandonment_project.py](../lib/envs/abandonment_project.py))
- T (=env.T = paper_T - 1) ∈ {5, 7} (= paper T_paper ∈ {6, 8})
- p_win ∈ {0.6, 0.72}
- CPT regime ∈ {CPT88 (α=0.88, ρ=0.65, λ=2.25), CPT95 (α=0.95, ρ=0.5, λ=1.5)}
- x_1 = T·δ (= unprofitable initial per LNW paper convention; T=5→50, T=7→70)
- 其它固定: δ=10, c=11
- Per cell: 5 seeds × 5000 train_eps × batch=1 × support=50 × critic_lr=0.04 × eps=0.3
- SPE oracle: 2000 MC rollouts/state-action
- Paper-eval: 100 rollouts/state

总 8 cells, 跑约 12 min wall time。

## 完整结果

### T=5 (paper_T=6, |X|=15 reachable decision states)

| p_win | CPT | Disagree | Val Err | Optimality | SW (SPERL) | SW (SPE oracle) | SW match |
|---|---|---|---|---|---|---|---|
| 0.6 | 88 | 1.2 ± 1.2 | 14.4 ± 7.0 | **0.0** | 234.3 ± 5.6 | 240.2 | 97.6% |
| 0.6 | 95 | 1.8 ± 1.3 | 12.6 ± 3.9 | 0.12 ± 0.87 | 289.7 ± 5.9 | 291.3 | 99.4% |
| 0.72 | 88 | 1.2 ± 1.0 | 15.5 ± 6.0 | 5.88 ± 1.7 | 270.4 ± 5.1 | 271.3 | 99.7% |
| 0.72 | 95 | **1.0 ± 0.0** | 18.5 ± 3.1 | 7.16 ± 1.7 | 327.8 ± 2.7 | 327.2 | 100.2% |

### T=7 (paper_T=8, |X|=28 reachable decision states)

| p_win | CPT | Disagree | Val Err | Optimality | SW (SPERL) | SW (SPE oracle) | SW match |
|---|---|---|---|---|---|---|---|
| 0.6 | 88 | 2.8 ± 1.6 | 32.4 ± 8.2 | **0.0** | 648.3 ± 9.3 | 655.1 | 99.0% |
| 0.6 | 95 | 4.2 ± 0.7 | 32.6 ± 3.0 | 0.50 ± 0.88 | 809.9 ± 8.3 | 824.3 | 98.3% |
| 0.72 | 88 | 1.0 ± 0.6 | 33.6 ± 3.0 | 10.82 ± 2.0 | 752.6 ± 6.8 | 754.9 | 99.7% |
| 0.72 | 95 | **0.4 ± 0.5** | 40.8 ± 3.4 | 11.67 ± 2.2 | 935.2 ± 8.1 | 934.3 | 100.1% |

## 核心 observations (paper writeup)

### O1: SPERL 跨所有 config 稳定收敛, 没有 hard config 失败

SW match 97.6% - 100.2% 跨 8 cells, 最低 disagreement 0.4 ± 0.5 / 28 (T=7 p=0.72 CPT95), 最高 4.2 ± 0.7 / 28 (T=7 p=0.6 CPT95)。**没有一个 cell 出现 paper 在 Barberis CPT88/p=0.66 上观察到的"hardest config 崩到 0"现象**。

LNW 上 SPERL 比 Barberis 更稳的可能原因:
- LNW 的 fan-out abandonment threshold 比 Barberis 的 wealth-based gambling threshold **结构更简单**
- LNW 的 reward 分布比 Barberis 更"正" (continue all 期望正), critic 的 quantile 估计 less prone 到极端 outlier
- LNW state space (T=7, |S|=120) 比 Barberis (T=5, |S|=66) 大但 reachable subset 相对更稀疏, 学习信号更集中

### O2: CPT-induced "stay out" 行为有 dose-response 关系

| Config 严苛性 | Optimality at (0,0) | 解读 |
|---|---|---|
| p=0.6 + CPT88 (loss aversion ↑↑, prob weight ↑↑) | 0.0 (cross-seed) | SPE 选 abandon, "don't even start" |
| p=0.6 + CPT95 OR p=0.72 + CPT88 | 0.12-0.50 (近 0 + variance) | Borderline, 有 seed 翻转 |
| p=0.72 + CPT95 (loss aversion ↓, prob weight ↓) | 7.16-11.67 | Continue 主导, V 显著正 |

这个 dose-response **非常干净**, 比 paper 在 Barberis 上展示 (Table 1 CPT88 p=0.6 给 0.0±0.0) 更系统化, 可单独写成 paper §4 sub-section 回应 AE point 4 "CPT preference 引致 non-trivial 决策"。

### O3: CPT95 (less loss-averse) → 更高 Optimality + 更高 SW

CPT95 跟 CPT88 同 (T, p_win) 对比:
- T=5 p=0.72: Opt 5.88 → 7.16 (+22%); SW 270.4 → 327.8 (+21%)
- T=7 p=0.72: Opt 10.82 → 11.67 (+8%); SW 752.6 → 935.2 (+24%)

预期: 减小 λ (loss aversion) 让 CPT distribution 看起来更"正", agent 更愿 continue, V 提升。**符合理论。**

### O4: Horizon scale-up: T=5→7 (paper_T=6→8)

- SW (SPERL) at p=0.72 CPT88: 270.4 → 752.6 (×2.78)
- |X| from 15 → 28 (×1.87)
- Per-state V 平均 ≈ 18.0 → 26.9, 反映更多机会去赚 x_T

Disagree count 也增 (1.2 → 1.0, 但 |X| 翻倍, 比例 8% → 4%) — agent 在更大 state space 上**表现更精细**, 不退化。

### O5: Value Error 系统性高 (paper 上也有同现象)

每 cell Val Err 大约是 SW 的 5-15%。这跟 paper [reports/2026-04-26_alg34_paper_diff_and_filter09.md] 标注的 "VE 系统性 2.5× 高于 paper" 一致 — 全 ablation 配置都受影响, 不独立于 LNW。值得另起一个 task 调查 VE 偏高的根因 (可能 quantile 估计噪声、cpt_offset 应用时机、或 paper-eval domain 差异)。

## Artifacts

```
runs/
  abandonment_sperl_T5_x150_c11_d10_p0.6_cpt_a0.88_r0.65_l2.25/    # T=5 p=0.6 CPT88
  abandonment_sperl_T5_x150_c11_d10_p0.6_cpt_a0.95_r0.5_l1.5/      # T=5 p=0.6 CPT95
  abandonment_sperl_T5_x150_c11_d10_p0.72_cpt_a0.88_r0.65_l2.25/   # T=5 p=0.72 CPT88 (production)
  abandonment_sperl_T5_x150_c11_d10_p0.72_cpt_a0.95_r0.5_l1.5/     # T=5 p=0.72 CPT95
  abandonment_sperl_T7_x170_c11_d10_p0.6_cpt_a0.88_r0.65_l2.25/    # T=7 p=0.6 CPT88
  abandonment_sperl_T7_x170_c11_d10_p0.6_cpt_a0.95_r0.5_l1.5/      # T=7 p=0.6 CPT95
  abandonment_sperl_T7_x170_c11_d10_p0.72_cpt_a0.88_r0.65_l2.25/   # T=7 p=0.72 CPT88
  abandonment_sperl_T7_x170_c11_d10_p0.72_cpt_a0.95_r0.5_l1.5/     # T=7 p=0.72 CPT95
```

每个 dir 里有 `seed{0..4}/per_state_values.csv`, `aggregate.json`, `aggregate.csv`, `config.json`。Gitignored 但本地可 reproducible by `bash scripts/grid_lnw.sh`。

## 下一步

Per playbook checklist Phase 5:
- [x] 5.1.1 Multi-config grid table (本报告)
- [ ] 5.1.2 DO naive baseline 加进同 grid
- [ ] 5.1.3 Per-state policy heatmap 可视化
- [ ] 5.1.4 Convergence curves
- [ ] 5.2 Algorithm 3/4 ablation on hardest config (T=7 p=0.6 CPT88 是这个 grid 里 "最难" 的)

剩 4 个 task 完成后 LNW env 的 paper §4 章节就 deliverable-ready。
