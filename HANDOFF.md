# 交接文档 — CPT-QD-SPERL paper 复现

**最后更新**: 2026-04-26

任务: 验证 refactor 后的 generic SPERL 是否和 paper (`paperRef/MSci_MANUSCRIPT.pdf`) 对得上。

## 当前状态

| 项 | 状态 |
|---|---|
| Refactor 引入的 3 个语义 bug | ✅ 已修 (cpt_offset hook + featurizer/SPE 修补) |
| Welfare 求和域 vs paper Definition 5 | ✅ 已对齐 (Barberis featurizer fix) |
| Algorithm 3/4 port 到 generic path | ✅ 完成，默认 off, 向后兼容 |
| Barberis Optimality + SW (CPT88/p=0.72, 0.66) | ✅ 对齐 paper 1σ |
| OptEx on-path SPERL vs SPE | ✅ 几乎完美匹配 (0 stable_mismatch on-path) |
| README CLI 示例 + Alg 3/4 flags 文档 | ✅ 已 sync |
| Alg 3/4 在 hardest config (CPT88/p=0.66) 上崩到 0 | ⚠️ 待诊断 (超参或实现细节差异) |
| OptEx 5 个 contested decision points | ⚠️ Alg 3/4 未能解决，需更长 training 或别的 tie-break |
| Legacy `rerun_GreedySPERL_QR__main.py` 独立校验 | ❌ 未做 |
| CPT95 全 p_win 完整复现 | ❌ 未做 |
| OptEx 自我 backward induction SPE oracle | ❌ 未做 |

## 关键概念

- **SPE = Bellman-CPT 算子的不动点，不是任意点最优**。多个 SPE 可共存，每个 V(x_0) 不同。SPERL 的 V(x_0) > oracle 不是 bug，是均衡多重性。
- **OptEx BFS-tree disagreement 是误导性指标**。SPE 实际玩 2000 局只触及 ~53 个 (t,state)；其余 ~440 个 state SPERL 学不准但本来也不会被访问。用 [scripts/analyze_optex.py](scripts/analyze_optex.py) 自动出 visit-weighted on-path 数。

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
| Algorithm 3 sticky tie-break | [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `GreedyPolicy.update_from_critic_values` |
| Algorithm 4 quantile filter | [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `filter_quantiles` + `QRCritic._cpt_from_quantiles` |
| Paper 4 metrics | [lib/paper_eval.py](lib/paper_eval.py) `compute_paper_metrics` |
| Paper-style CLI | [agents/run_paper_eval.py](agents/run_paper_eval.py) |

## 跑实验踩坑

- stderr 噪声: `2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"`
- 时长: Barberis 3 seeds × 15k eps ≈ 30-60s; OptEx 3 seeds × 3k eps ≈ 8-10 min
- 3 seeds 的 per-seed 摆幅在 Barberis 上很大 (4-11)，要 ≥10 seeds 才能 head-to-head 比较 Alg 3/4 ablation
- `run_paper_eval.py` 生成的 `per_state_values.csv` 是逐 state 黄金数据
- 旧 `results_alg34*` (welfare fix 前) 的 SW 数字基于过度枚举 36 states，不能直接和新结果比

## 下一步 TODO (按价值)

1. **Paper Algorithm 3/4 伪代码 vs 我们实现的 diff** — 读 paper Appendix C.2.2，对照 `GreedyPolicy.update_from_critic_values` 和 `filter_quantiles`。零成本，可能直接解释 p=0.66 上 Alg 3/4 崩
2. **Paper Appendix C.2.5 超参对照** — 用 paper 的 setting 重跑 CPT88/p=0.66，看是否 reproduce paper 的 2.74 Optimality
3. **Filter-thresh sweep on p=0.66** — 复现 paper Table 4: filter ∈ {1, .95, .9, .85, .8, .75}，paper Pareto-optimal 是 filter=0.9 (VE 9.3 ± 0.95)
4. **Legacy `rerun_GreedySPERL_QR__main.py` 当独立 ground truth**
5. **CPT95 整套配置** — 跑 p ∈ [0.48..0.72] 完成 Tables 1/2 复现
6. **OptEx 自我 backward induction SPE oracle** — 看是否和 `CumSPERL_ref/SPE_OptEx_5_0.015_4.npy` 一致

详见各 session report。

## Session 报告

| 日期 | 主题 | 文件 |
|---|---|---|
| 2026-04-24 | Refactor bug 修复 + Algorithm 3/4 port | [reports/2026-04-24_refactor_bugs_and_alg34_port.md](reports/2026-04-24_refactor_bugs_and_alg34_port.md) |
| 2026-04-26 | Algorithm 3/4 在 Barberis 和 OptEx 上的 ablation | [reports/2026-04-26_alg34_validation.md](reports/2026-04-26_alg34_validation.md) |
| 2026-04-26 | Welfare 求和域修复 + paper Tables 1/2 复现 + p=0.66 ablation | [reports/2026-04-26_welfare_and_paper_tables.md](reports/2026-04-26_welfare_and_paper_tables.md) |
| 2026-04-26 | OptEx 49% disagreement 是 off-path artifact (visit-frequency 分析) | [reports/2026-04-26_optex_visit_frequency.md](reports/2026-04-26_optex_visit_frequency.md) |
