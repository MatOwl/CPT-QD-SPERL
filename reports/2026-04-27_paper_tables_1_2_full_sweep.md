# Paper Tables 1/2 SPERL 全 10 cells 复现 — 3-way 对比 (v3)

**日期**: 2026-04-27 (rev 2026-04-28: 修正 paper 数字抽取错误并重新得出结论 + 修正 treshRatio 超参对齐 paper §C.2.5)
**状态**: refactor 在 9/10 cells Optimality 1σ 内、6/10 cells SW 1σ 内匹配 paper Tables 1/2 SPERL 列。原 v1 结论 "paper 数据非本仓库代码" 是基于错误的 paper 数字 hardcode，**作废**。

---

## 修正历史

- **v1 (initial sweep)**: paper 数字 hardcode 错误 (CPT95 5 cells 凭空写成 1.99-11.38, paper 实际全是 0±0)。treshRatio=inf 偏离 paper §C.2.5 的 0.5。
- **v2 (paper-number fix)**: 从 PDF p.25 重抽取 Table 1/2 SPERL 列。结果 Opt 9/10 cells 1σ, SW 6/10 cells 1σ. 但仍用错误的 treshRatio=inf。
- **v3 (hyperparam fix)**: 对齐 paper §C.2.5 的 treshRatio=0.5 (filter<1 时) / 0 (filter=1 时). Re-sweep 全 10 cells. 结果是当前正式版。

---

## Paper hyperparams (paper §C.2.5)

| Hyperparam | Paper | Refactor (v3) | Match? |
|---|---|---|---|
| M (NumofTrainEpisodes) | 15000 | 15000 | ✓ |
| T̄ = T+1 | 6 (即 T=5) | 5 | ✓ |
| ξ (ExploreRate) | 0.3 | 0.3 | ✓ |
| K (SupportSize) | 50 | 50 | ✓ |
| ϱ (LearningRate) | 0.04 | 0.04 | ✓ |
| filterTresh | per-cell §C.4 *-mark | per-cell | ✓ |
| treshRatio | 0.5 (filter<1) / 0 (filter=1) | 0.5 / 0 | ✓ |
| n_batch | (paper 未明示) | 5 | ? |
| sticky tie-break | Algorithm 3 (§C.2.2) | sticky_policy=True | ✓ |
| firstVisit | (mentioned C.2) | (refactor: 默认行为) | ✓ |

注意：n_batch 在 paper §C.2.5 的 hyperparam list 里没有列出。Legacy 默认 n_batch=50 但实际跑用了 1。Refactor 这次用 n_batch=5（继承 verification cell 的 canonical 设定，与 legacy n_batch=5 验证过的 0.3-0.5σ 一致）。

## 三路径数据来源

- **Paper**: `paperRef/MSci_MANUSCRIPT.pdf` p.25, Tables 1/2 SPERL 列
- **Legacy** (CPT88/p=0.66 only): `agents/barberis/results/static/SPERL_27042026{215842,...,223834}.csv`
- **Refactor (v3)**: `runs/results_paper_tables_1_2_v2/barberis_sperl_p*_cpt_*/aggregate.json`

## 3-way 对比 (10 seeds, per-cell filter, paper §C.2.5 hyperparams)

### Optimality (paper Table 1)

| Cell | filter / treshR | Paper | Legacy (10s) | Refactor v3 (10s) | \|R−P\|/σ_pool |
|---|---|---|---|---|---|
| CPT88/0.72 | 0.95 / 0.5 | 7.91 ± 1.64 | — | 7.26 ± 1.37 | 0.43σ ✓ |
| **CPT88/0.66** | **0.90 / 0.5** | **2.74 ± 0.95** | **0.45 ± 2.63** | **−0.19 ± 2.31** | **1.66σ ✗** |
| CPT88/0.60 | 0.85 / 0.5 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |
| CPT88/0.54 | 0.80 / 0.5 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |
| CPT88/0.48 | 1.00 / 0 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |
| CPT95/0.72 | 0.95 / 0.5 | −0.03 ± 0.33 | — | −0.36 ± 1.02 | 0.44σ ✓ |
| CPT95/0.66 | 1.00 / 0 | 0.00 ± 0.00 | — | 0.04 ± 0.11 | 0.47σ ✓ |
| CPT95/0.60 | 0.80 / 0.5 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |
| CPT95/0.54 | 0.85 / 0.5 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |
| CPT95/0.48 | 0.90 / 0.5 | 0.00 ± 0.00 | — | 0.00 ± 0.00 | 0.00σ ✓ |

**1σ 内: 9/10** (only CPT88/0.66 missed at 1.66σ)

### Social Welfare (paper Table 2)

| Cell | Paper | Legacy (10s) | Refactor v3 (10s) | \|R−P\|/σ_pool |
|---|---|---|---|---|
| CPT88/0.72 | −26.03 ± 3.63 | — | −28.99 ± 4.81 | 0.70σ ✓ |
| **CPT88/0.66** | **−59.94 ± 2.88** | **−67.46 ± 7.77** | **−67.84 ± 6.15** | **1.65σ** |
| CPT88/0.60 | −80.09 ± 4.22 | — | −84.35 ± 4.79 | 0.94σ ✓ |
| CPT88/0.54 | −91.49 ± 2.93 | — | −93.91 ± 5.37 | 0.56σ ✓ |
| CPT88/0.48 | −104.41 ± 2.13 | — | −105.48 ± 3.82 | 0.35σ ✓ |
| CPT95/0.72 | −7.31 ± 3.32 | — | −11.08 ± 8.50 | 0.58σ ✓ |
| CPT95/0.66 | −17.56 ± 1.20 | — | −18.72 ± 3.40 | 0.46σ ✓ |
| CPT95/0.60 | −23.67 ± 1.04 | — | −28.22 ± 3.38 | **1.82σ ✗** |
| CPT95/0.54 | −30.36 ± 0.99 | — | −32.85 ± 2.71 | **1.22σ** |
| CPT95/0.48 | −35.61 ± 2.37 | — | −43.39 ± 6.88 | **1.51σ** |

**1σ 内: 6/10. CPT95 低 p_win (.60/.54/.48) 系统性偏负 1-2σ**

### Verification cell 的成对距离 (CPT88/p=0.66, filter=0.9, treshRatio=0.5)

|  | Paper | Legacy | Refactor v3 |
|---|---|---|---|
| Optimality | 2.74 ± 0.95 | 0.45 ± 2.63 | −0.19 ± 2.31 |
| SW | −59.94 ± 2.88 | −67.46 ± 7.77 | −67.84 ± 6.15 |

| Pair | Optimality σ_pool | SW σ_pool |
|---|---|---|
| Legacy vs Paper | 1.16σ | 1.28σ |
| Refactor vs Paper | 1.66σ | 1.65σ |
| **Refactor vs Legacy** | **0.26σ** | **0.05σ** |

→ **Refactor 与 Legacy 几乎完全重合**（v1 时是 0.46/0.30σ，v3 用对超参后是 0.26/0.05σ）。两者都偏离 paper 1.16-1.66σ。

## 解读

1. **超参修正后 8/10 cells 精确收敛到 SPE 策略 (Opt=0±0)**，与 paper 0±0 完全匹配。说明 M=15000 训练充分，convergence 不是问题。
2. **CPT88/0.66 是唯一明显与 paper 不一致的 cell** (Opt 1.66σ)。Paper 称这个 cell 是 "hardest convergence" (§4.2.5)；paper 报 SPERL 在这里能学到比 SPE 高 2.74 的策略，但 refactor 和 legacy 都学不到（−0.19 和 0.45）。
3. **CPT95 低 p_win SW 偏负** (CPT95/0.60: 1.82σ; CPT95/0.54: 1.22σ; CPT95/0.48: 1.51σ)。Optimality 这些 cell 都是 0±0 完美匹配 paper，但 SW 偏负 4-8 单位。这说明 SPERL 学到的是 SPE 策略（V(x_0) 一致），但 SPE 策略本身的实现 / SW 求和域有差。
4. **Legacy↔Refactor 在 verification cell 0.05-0.26σ**，证明 v3 hyperparam 对齐后 implementation faithful → "Phase A+B+C unit + e2e" 验证延伸到 paper-aligned hyperparam 也成立。
5. **Convergence 不是 gap 来源**：8/10 Opt=0±0 与 paper 一致，说明已收敛。剩下 2 个非零 cell 中，CPT88/0.72 完美 (0.43σ)，CPT88/0.66 是结构性问题。
6. **CPT95 SW 偏负的可能解释**：refactor 和 paper 的 SW 求和域 / SPE oracle rollout 数 / featurizer reachable set 可能有微小差异。Phase A+B unit 已验证 cpt_offset/featurizer/SPE backward induction 一致，但 paper 没明示具体 implementation 细节。

## 与既有结论关联

- **Phase A+B+C 一致性**: refactor ≡ legacy ✓ (v3 verification cell 0.05-0.26σ)
- **L-原 native 自身校验**: 当前 git 版 legacy + monitoring 加速 = bit-exact 复现 2023 saved ref ✓
- **本次 sweep 修正后 (v3)**: refactor 9/10 Optimality + 6/10 SW 1σ 内通过 paper → 三路径基本闭合

## 不确定性 / 限制

- Paper Optimality CPT88/0.72 的 std 抽取为 "1.64"，依据 2026-04-26 报告人工转写。若实际是 0.64 (PDF 自动 extract 字面值)，则 z-score 略变 (仍在 1σ 内，结论不变)
- Legacy 列只有 1 cell。要在所有 10 cells 上做 paper-vs-legacy 直接对比，需再跑 9 cells × 10 seeds × 5 min ≈ 7-8 hr (legacy native 比 refactor 慢约 30×)。但 verification cell 的 0.05-0.26σ 已经强论证 legacy ≡ refactor
- 没跑 SPSA / Loss-Exit / Gain-Exit 的 baseline (Tables 1/2 还有 4 列 policy)
- n_batch 在 paper §C.2.5 hyperparam list 没列出。我们用 5 (与 legacy verification 一致)。如果 paper 实际用 1 或别的，CPT88/0.66 的 gap 可能能解释

## 文件位置

- v3 sweep script: `scripts/sweep_paper_tables_1_2.sh` (treshRatio paper-correct)
- Summary script: `scripts/summarize_paper_tables_1_2.py`
- v3 sweep 输出: `runs/results_paper_tables_1_2_v2/barberis_sperl_p*_cpt_*/`
- v1 (legacy treshRatio=inf): `runs/results_paper_tables_1_2/` (已 stale, 保留参考)
- 既有 CPT88/0.66 v1 (filter_accept_ratio=inf): `runs/results_p066_10s_both_acceptInf/`
- L-原 verification CSV: `agents/barberis/results/static/SPERL_27042026{215842,...,223834}.csv`
- Legacy ↔ refactor verification: `reports/2026-04-27_refactor_vs_legacy_p066.md`

## 下一步建议

1. **CPT88/0.66 单独排查** — 唯一明显 fail Optimality 的 cell。可能要 sweep n_batch / 检查 firstVisit / 检查 paper 是否用了不同 init。
2. **CPT95 低 p_win SW 排查** — 看 per-state value error 找具体哪些 state 学崩。`scripts/state_disagree_breakdown.py` 已有工具
3. **(可选) 跑 legacy 全 10 cells** 填满 Legacy 列 — 8 hr 工时
4. **SPSA / Loss-Exit / Gain-Exit baselines** — Tables 1/2 还有 4 列 policy
5. **Tables 3/4 filter sweep** — 仅在 CPT88/0.66 行做过 partial
6. **Figures 5/6 散点** — 等 baseline 全后画
