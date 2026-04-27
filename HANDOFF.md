# 交接文档 — CPT-QD-SPERL paper 复现

**最后更新**: 2026-04-27 (BLN consumption env MVP 实施完成；2026-04-26 之前: LNW abandonment env 实施完成，paper §4 deliverable 全套就位；Phase A+B+C refactor verification 完成在前)

任务: 验证 refactor 后的 generic SPERL 是否和 paper (`paperRef/MSci_MANUSCRIPT.pdf`) 对得上。

## 当前状态

| 项 | 状态 |
|---|---|
| Refactor 引入的 3 个语义 bug | ✅ 已修 (cpt_offset hook + featurizer/SPE 修补) |
| Refactor pure-fn 一致性 (Phase A unit isolation) | ✅ compute_CPT 100% ≡ legacy；filter ⚠️ paper-pseudocode-faithful（legacy 偏离 paper 文字）；env ✓ |
| Refactor critic update 一致性 (Phase B unit isolation) | ✅ first-visit ≡；QR grad ≡ legacy 用向量化版后 bit-exact；当前 generic 比 legacy fp 更精确 |
| Refactor e2e 一致性 (Phase C 10-seed legacy aggregate) | ⚠️ Phase C 用的 "L-仿" driver 在 ~19% near-tie state 上 fp-flip ≠ L-原 native；-0.89 数字需用 L-原 native 重测 |
| Paper Tables 数字源于 legacy？| ❌ legacy 自己都跑不出 paper 2.74，paper 用的不是这份代码 |
| Welfare 求和域 vs paper Definition 5 | ✅ 已对齐 (Barberis featurizer fix) |
| Algorithm 3/4 port 到 generic path | ✅ 完成，默认 off, 向后兼容 |
| Barberis Optimality + SW (CPT88/p=0.72, 0.66) | ✅ 对齐 paper 1σ |
| OptEx on-path SPERL vs SPE | ✅ 几乎完美匹配 (0 stable_mismatch on-path) |
| README CLI 示例 + Alg 3/4 flags 文档 | ✅ 已 sync |
| Alg 3/4 在 p=0.66 上 partial 对齐 (10 seeds 后) | ⚠️ 最佳 sticky+filter=0.9+accept=∞: 1.46±1.71 vs paper 2.74±0.95 (1.3σ 差距) |
| OptEx 5 个 contested decision points | ⚠️ Alg 3/4 未能解决，需更长 training 或别的 tie-break |
| ≥10 seeds head-to-head ablation 完成 | ✅ done; 3-seed 数据全部翻案 (见 alg34_paper_diff report) |
| VE 系统性 2.5× 高于 paper (我们 ~24 vs paper ~9.3) | ❌ 全 ablation 配置都受影响，独立于 Alg 3/4 |
| Acceptance gate=0.5 cost ~1.65 Optimality | ⚠️ 跨 3-seed/10-seed 都存在，paper default 反而压低我们 |
| Legacy `rerun_GreedySPERL_QR__main.py` 独立校验 | ❌ 未做 |
| CPT95 全 p_win 完整复现 | ❌ 未做 |
| OptEx 自我 backward induction SPE oracle | ❌ 未做 |
| **LNW abandonment env 实施 (回应 AE point 6)** | ✅ env class + SPE oracle + featurizer + CLI 全到位 |
| **LNW grid sweep (T × p_win × CPT_regime, 8 cells)** | ✅ SW match 97.6%-100.2%, Optimality 0-11.67 |
| **LNW 3-policy 对比 (SPE / DO / SPSA)** | ✅ stay-out config 上 V_DO=-4.14 vs V_SPE=0, SPSA collapse |
| **LNW Algorithm 3/4 ablation** | ✅ best combo sticky+filter=0.9, LNW 不崩 (vs Barberis hardest 崩) |
| **LNW per-state heatmap + convergence figure** | ✅ paper Figure 风格 PNG 生成 |
| **新-env 开发 playbook 沉淀** | ✅ [reports/_playbook_new_env_development.md](reports/_playbook_new_env_development.md) (含 5 个 bug pattern, BLN 加了 #5) |
| 修补 SPSA + SPERL 的 cpt_offset latent bug (LNW 才暴露) | ✅ commit ad5bdbf + 8b6b27a |
| **BLN consumption env MVP (van Bilsen-Laeven-Nijman 2020 MS, 第 3 个 env)** | ✅ env + SPE + featurizer + CLI, ternary action + fixed π=0.5, endogenous-reference 机制 |
| **BLN MVP smoke test (5k eps × 3 seeds, 19.6s)** | ✅ Disagree 6/20 = 30% (BFS reachable set), SW gap 1.5/state ≈ Barberis 0.8/state |
| **BLN BFS reachability fix (iter_states bug #5)** | ✅ 240 grid → 20 reachable; 一次 fix 把 SPERL Disagree 从 80% 降到 30% |
| BLN Phase 5 deliverables (grid / DO / heatmap / convergence / Alg34) | ❌ 未做 (MVP scope 不含, 视后续展开) |
| BLN salary / portfolio decision extension | ❌ MVP 不含 (paper §4 时声明 future work) |

## 关键概念

- **SPE = Bellman-CPT 算子的不动点，不是任意点最优**。多个 SPE 可共存，每个 V(x_0) 不同。SPERL 的 V(x_0) > oracle 不是 bug，是均衡多重性。
- **OptEx BFS-tree disagreement 是误导性指标**。SPE 实际玩 2000 局只触及 ~53 个 (t,state)；其余 ~440 个 state SPERL 学不准但本来也不会被访问。用 [scripts/analyze_optex.py](scripts/analyze_optex.py) 自动出 visit-weighted on-path 数。
- **Barberis 上 SPERL 系统性 under-gamble**：disagree state 全是 "oracle 说 gamble，SPERL 选 exit"。10 seeds 看 7/15 状态 ≥50% seeds 错；paper Table 3 filter=0.9 上是 2.89/15。用 [scripts/state_disagree_breakdown.py](scripts/state_disagree_breakdown.py) 出 per-state 聚合。

## 环境

```
Python: C:/Users/Jingxiang Tang/FNN/Scripts/python.exe (3.10.7)
Packages: numpy==1.26.3 (gym 0.26 硬要求 <2), scipy, pandas, matplotlib, gym==0.26.2, pypdf
```

## 关键代码位置

| 模块 | 文件 |
|---|---|
| CPT-offset hook | [lib/envs/featurizers.py](lib/envs/featurizers.py) `Featurizer.cpt_offset` |
| Barberis SPE backward induction | [lib/envs/barberis_spe.py](lib/envs/barberis_spe.py) |
| OptEx SPE oracle loader | [lib/envs/optex_spe.py](lib/envs/optex_spe.py) `SPEOracle` |
| **LNW abandonment env** | [lib/envs/abandonment_project.py](lib/envs/abandonment_project.py) |
| **LNW SPE backward induction** | [lib/envs/abandonment_spe.py](lib/envs/abandonment_spe.py) |
| **LNW DO (EV-max) baseline** | [lib/envs/abandonment_do.py](lib/envs/abandonment_do.py) |
| **BLN consumption env** | [lib/envs/bln_consumption.py](lib/envs/bln_consumption.py) |
| **BLN SPE backward induction** | [lib/envs/bln_spe.py](lib/envs/bln_spe.py) |
| **BLN featurizer + BFS reachable iter_states** | [lib/envs/featurizers.py](lib/envs/featurizers.py) `BLNFeaturizer` |
| Algorithm 3 sticky tie-break | [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `GreedyPolicy.update_from_critic_values` |
| Algorithm 4 quantile filter | [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `filter_quantiles` + `QRCritic._cpt_from_quantiles` |
| Paper 4 metrics | [lib/paper_eval.py](lib/paper_eval.py) `compute_paper_metrics` |
| Paper-style CLI | [agents/run_paper_eval.py](agents/run_paper_eval.py) |
| **LNW grid sweep** | [scripts/grid_lnw.sh](scripts/grid_lnw.sh) |
| **LNW per-state visualization** | [scripts/analyze_lnw.py](scripts/analyze_lnw.py) |
| **LNW convergence curves** | [scripts/lnw_convergence.py](scripts/lnw_convergence.py) |
| **LNW DO comparison** | [scripts/lnw_do_comparison.py](scripts/lnw_do_comparison.py) |
| **LNW Alg 3/4 ablation** | [scripts/lnw_alg34_ablation.sh](scripts/lnw_alg34_ablation.sh) |

## 跑实验踩坑

- stderr 噪声: `2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"`
- 时长: Barberis 3 seeds × 15k eps ≈ 30-60s; OptEx 3 seeds × 3k eps ≈ 8-10 min
- 3 seeds 的 per-seed 摆幅在 Barberis 上很大 (4-11)，要 ≥10 seeds 才能 head-to-head 比较 Alg 3/4 ablation
- `run_paper_eval.py` 生成的 `per_state_values.csv` 是逐 state 黄金数据
- 旧 `results_alg34*` (welfare fix 前) 的 SW 数字基于过度枚举 36 states，不能直接和新结果比

## 下一步 TODO (按价值)

1. **联系 paper 作者 / 翻 Lesmana et al 2022 / Lesmana & Pun 2025** — Tables 数字源于另一份 SPERL 代码（这份 legacy 跑出 -0.89 Optimality，paper 2.74）
2. **Filter-thresh sweep 补完整** — 已测 0.75 / 0.9，剩 1.0 / 0.95 / 0.85 / 0.8 复现 Table 4
3. **CPT95 整套配置** — Tables 1/2 复现 (用 generic, 已知比 legacy 好)
4. **写 paper §4.x LNW 章节** — 数据全有 ([reports/2026-04-26_lnw_paper_section4_summary.md](reports/2026-04-26_lnw_paper_section4_summary.md) 已 outline 章节结构), 直接 draft
5. **LNW 10-seed re-run** — 当前 5 seeds 已稳定, paper 标准是 10
6. **OptEx 自我 backward induction SPE oracle**
7. **BLN Phase 5 deliverables** — 视 paper §4 narrative 需要展开 (grid / DO baseline / heatmap / convergence / Alg34 ablation; salary + portfolio extension)
8. **加 Henderson liquidation env** — 第 4 个 env, 1D state + binary action, 用 playbook 走 (估 ~LNW 工作量)
9. ~~Phase B/C / filter 决策~~：✅ done (refactor verdict: faithful + improved)
10. ~~LNW abandonment env 实施~~：✅ done (2026-04-26 session, 9 commits afa0800..1c58019)
11. ~~BLN consumption env MVP~~：✅ done (本 session, 1 commit 0dbcb72)

详见各 session report。

## Session 报告

| 日期 | 主题 | 文件 |
|---|---|---|
| 2026-04-24 | Refactor bug 修复 + Algorithm 3/4 port | [reports/2026-04-24_refactor_bugs_and_alg34_port.md](reports/2026-04-24_refactor_bugs_and_alg34_port.md) |
| 2026-04-26 | Algorithm 3/4 在 Barberis 和 OptEx 上的 ablation | [reports/2026-04-26_alg34_validation.md](reports/2026-04-26_alg34_validation.md) |
| 2026-04-26 | Welfare 求和域修复 + paper Tables 1/2 复现 + p=0.66 ablation | [reports/2026-04-26_welfare_and_paper_tables.md](reports/2026-04-26_welfare_and_paper_tables.md) |
| 2026-04-26 | OptEx 49% disagreement 是 off-path artifact (visit-frequency 分析) | [reports/2026-04-26_optex_visit_frequency.md](reports/2026-04-26_optex_visit_frequency.md) |
| 2026-04-26 | Paper Alg 3/4 vs 实现 diff + filter=0.9 解决 p=0.66 崩塌 | [reports/2026-04-26_alg34_paper_diff_and_filter09.md](reports/2026-04-26_alg34_paper_diff_and_filter09.md) |
| 2026-04-26 | Refactor verification Phase A+B: 全 unit isolated 组件忠于 legacy，无 regression | [reports/2026-04-26_refactor_verification.md](reports/2026-04-26_refactor_verification.md) |
| 2026-04-26 | MSci 候选论文文献综述 (27 篇, 13 个 OM/MS 子领域) | [reports/2026-04-26_msci_literature_survey.md](reports/2026-04-26_msci_literature_survey.md) |
| 2026-04-26 | LNW env 实施 + 2 reward bug 修 + SPSA-vs-SPERL 对照 | [reports/2026-04-26_lnw_env_implementation.md](reports/2026-04-26_lnw_env_implementation.md) |
| 2026-04-26 | 新-env 开发方法论 playbook (5 phases + bug patterns + checklist) | [reports/_playbook_new_env_development.md](reports/_playbook_new_env_development.md) |
| 2026-04-26 | LNW grid sweep 8 cells (T × p_win × CPT_regime) | [reports/2026-04-26_lnw_grid_results.md](reports/2026-04-26_lnw_grid_results.md) |
| 2026-04-26 | LNW paper §4 deliverable summary (整合所有 artifacts) | [reports/2026-04-26_lnw_paper_section4_summary.md](reports/2026-04-26_lnw_paper_section4_summary.md) |
| 2026-04-27 | BLN consumption env MVP (3rd env, endogenous reference + ternary action; iter_states bug #5) | [reports/2026-04-27_bln_env_mvp.md](reports/2026-04-27_bln_env_mvp.md) |
