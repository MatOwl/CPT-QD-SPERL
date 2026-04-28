# Paper Tables 1/2 v3 vs Paper — 6-bug-fix 后重跑对比

**日期**: 2026-04-28
**数据**: `runs/results_paper_tables_1_2_v3/` (10 cells × 10 seeds, paper §C.2.5 hyperparams + 2026-04-28 6-bug fix)
**Paper PDF 来源**: `paperRef/MSci_MANUSCRIPT.pdf` p.25 Tables 1 & 2 SPERL 列

---

## 修复了什么

vs v2 (2026-04-27 sweep)：
- **F1** 删 `_cpt_from_quantiles` 的 `np.sort` pre-sort（让 paper §C.2.3 crossing-quantile detection 真正生效）
- **F2** CLI default 全 paper-align（特别是 `--filter-gate-mode absolute`、`--spe-rollouts 2000`）
- **F3** SPE oracle iteration 改为 parity-correct only
- **F4** `compute_paper_metrics` 加 per-state common random numbers (CRN)，v_tilde / v_hat 用同 seed
- **F5/F6**: doc-only 修正（不影响数据）

## Hyperparams (paper §C.2.5)

| Hyperparam | Paper | v3 sweep |
|---|---|---|
| M (NumofTrainEpisodes) | 15000 | 15000 ✓ |
| T̄ = T+1 | 6 (T=5) | 6 ✓ |
| ξ (ExploreRate) | 0.3 | 0.3 ✓ |
| K (SupportSize) | 50 | 50 ✓ |
| ϱ (LearningRate) | 0.04 | 0.04 ✓ |
| filterTresh | per-cell §C.4 *-mark | per-cell ✓ |
| treshRatio | 0.5 / 0 | 0.5 / 0 ✓ |
| **gate semantics** | **absolute (paper text)** | **absolute ✓ (was "relative" in v2)** |
| **SPE rollouts (Alg 5 M)** | **2000** | **2000 ✓ (was 300 in v2)** |
| n_batch | (paper 未明示) | 5 (与 legacy verification cell 一致) |
| sticky tie-break | Algorithm 3 | --sticky-policy ✓ |
| firstVisit | yes | yes ✓ |
| seeds | 10 | 10 |

## Table 1: Optimality (= V^π̃(x_0))

| Cell | filter | Paper (μ ± σ) | Refactor v3 (μ ± σ) | \|R−P\|/σ_pool | 1σ? |
|---|---|---|---|---|---|
| CPT88 / p=0.72 | 0.95 | **7.91 ± 1.64** | 9.47 ± 1.55 | 0.97σ | ✓ |
| **CPT88 / p=0.66** | **0.90** | **2.74 ± 0.95** | **0.87 ± 2.06** | **1.17σ** | **✗** |
| CPT88 / p=0.60 | 0.85 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |
| CPT88 / p=0.54 | 0.80 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |
| CPT88 / p=0.48 | 1.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |
| CPT95 / p=0.72 | 0.95 | −0.03 ± 0.33 | −0.22 ± 0.65 | 0.36σ | ✓ |
| CPT95 / p=0.66 | 1.00 | 0.00 ± 0.00 | 0.07 ± 0.20 | 0.47σ | ✓ |
| CPT95 / p=0.60 | 0.80 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |
| CPT95 / p=0.54 | 0.85 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |
| CPT95 / p=0.48 | 0.90 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00σ | ✓ |

**1σ within: 9/10**（仅 CPT88/0.66 fail at 1.17σ — paper §4.2.5 自己也称这是 "hardest convergence" cell）

## Table 2: Social Welfare (= Σ_x V^π̃(x))

| Cell | filter | Paper (μ ± σ) | Refactor v3 (μ ± σ) | \|R−P\|/σ_pool | 1σ? |
|---|---|---|---|---|---|
| CPT88 / p=0.72 | 0.95 | −26.03 ± 3.63 | −30.09 ± 6.90 | 0.74σ | ✓ |
| **CPT88 / p=0.66** | **0.90** | **−59.94 ± 2.88** | **−66.45 ± 5.88** | **1.41σ** | **✗** |
| CPT88 / p=0.60 | 0.85 | −80.09 ± 4.22 | −81.63 ± 5.77 | 0.30σ | ✓ |
| CPT88 / p=0.54 | 0.80 | −91.49 ± 2.93 | −92.79 ± 3.04 | 0.44σ | ✓ |
| CPT88 / p=0.48 | 1.00 | −104.41 ± 2.13 | −104.93 ± 3.73 | 0.17σ | ✓ |
| CPT95 / p=0.72 | 0.95 | −7.31 ± 3.32 | −9.83 ± 7.75 | 0.42σ | ✓ |
| CPT95 / p=0.66 | 1.00 | −17.56 ± 1.20 | −17.11 ± 4.58 | 0.14σ | ✓ |
| CPT95 / p=0.60 | 0.80 | −23.67 ± 1.04 | −25.15 ± 3.75 | 0.54σ | ✓ |
| CPT95 / p=0.54 | 0.85 | −30.36 ± 0.99 | −31.81 ± 3.04 | 0.64σ | ✓ |
| CPT95 / p=0.48 | 0.90 | −35.61 ± 2.37 | −39.31 ± 7.70 | 0.65σ | ✓ |

**1σ within: 9/10** (vs v2 only 6/10 — **CPT95 低 p_win SW 系统偏负的问题已基本解决**)

## v2 → v3 改善

| 指标 | v2 (relative gate, spe=300, 无 CRN, filter pre-sort) | v3 (paper-aligned + 6 bug 修) |
|---|---|---|
| Optimality 1σ | 9/10 | **9/10** (CPT88/0.66 1.66σ → 1.17σ) |
| Social Welfare 1σ | 6/10 | **9/10** (CPT95 低 p_win 三个 cell 全部回到 1σ 内) |
| CPT88/0.66 Opt | −0.19 ± 2.31 (1.66σ) | 0.87 ± 2.06 (1.17σ) — 偏差减半 |
| CPT88/0.66 SW | −67.84 ± 6.15 (1.65σ) | −66.45 ± 5.88 (1.41σ) |

## 验证 cell (CPT88/p=0.66/filter=0.9) 三路径对照

| 路径 | Optimality | Social Welfare |
|---|---|---|
| Paper | 2.74 ± 0.95 | −59.94 ± 2.88 |
| Legacy (10 seeds, 2026-04-27 native rerun) | 0.45 ± 2.63 | −67.46 ± 7.77 |
| Refactor v3 (10 seeds) | 0.87 ± 2.06 | −66.45 ± 5.88 |

| Pair | Opt σ_pool | SW σ_pool |
|---|---|---|
| Legacy vs Paper | 1.16σ | 1.28σ |
| Refactor vs Paper | 1.17σ | 1.41σ |
| **Refactor vs Legacy** | **0.18σ** | **0.15σ** |

→ Refactor 与 Legacy 完全一致；两者都偏离 paper 1.16-1.41σ。证据指向：**这份代码（refactor + legacy）和 paper Tables 1/2 报告的数字不是出自同一份实现**——这一论断从 v1 就有，v3 paper-align 修复后仍然成立（但现在 9/10 cell 通过了，剩 CPT88/0.66 一个）。

## 解读

1. **三路径已闭合到 statistical noise 量级**: refactor ↔ legacy 0.15-0.18σ，两者 vs paper 都 1.17-1.41σ。剩下的 paper gap 来自 implementation 之外，不是 refactor 的 regression。
2. **CPT95 低 p_win SW 偏负的问题解决**: v2 时 CPT95/0.60/0.54/0.48 SW 偏 1-2σ，v3 全部回到 0.5-0.7σ 内。最大功臣是 **F4 (CRN)** + **F2 (filter-gate=absolute)**——VE/SW 的独立 RNG 噪声不再叠加。
3. **CPT88/0.66 仍是单点 outlier**: paper §4.2.5 自己称之为 "hardest convergence"。1.17σ Opt + 1.41σ SW 的 gap，在 paper 自己的 1σ 标尺下也只是边缘。
4. **F1 (filter pre-sort 修复)** 在这些 cells 里影响相对小: 大部分 cell 的 quantile 经过 15000 步训练后 crossing 概率低；filter 的 主要价值是 discrete-distribution sharpening，已通过 `gaps <= d_star` 路径正常生效。

## 数据来源

- v3 sweep: `runs/results_paper_tables_1_2_v3/barberis_sperl_p*_cpt_*/aggregate.json`
- 重跑命令: `bash scripts/sweep_paper_tables_1_2.sh`
- 总 wall time: ~27 min on default Windows venv (16 sec / seed × 10 cells × 10 seeds)
- Paper PDF: `paperRef/MSci_MANUSCRIPT.pdf` Table 1 (p.25) Optimality 列, Table 2 (p.25) Social Welfare 列
- v2 (修复前) 数据: `runs/results_paper_tables_1_2_v2/`（保留作 ablation 对照）
- Legacy verification 数据: `agents/barberis/results/static/SPERL_27042026{215842,...,223834}.csv`
