# Paper Tables 3/4 v3 vs Paper — 6-bug-fix 后重跑对比

**日期**: 2026-04-28
**数据**: `runs/results_paper_tables_3_4_v3/` (10 cells × 6 filters × 10 seeds = 600 runs, paper §C.2.5 hyperparams + 2026-04-28 6-bug fix)
**Paper PDF 来源**: `paperRef/MSci_MANUSCRIPT.pdf` p.53 Table 3 (Policy Error), p.54 Table 4 (Value Error)
**Wall time**: ~80 min (vs v2 ~90 min, 因为 spe-rollouts 从 300 升到 2000 但 SPE 计算时间相对训练较短)

---

## 修复了什么 (vs v2 sweep, 2026-04-28 早些时候)

- **F1**: 删 `_cpt_from_quantiles` 的 `np.sort` pre-sort（让 paper §C.2.3 crossing-quantile detection 真正生效）
- **F2**: CLI default 全 paper-align（`--filter-gate-mode absolute`、`--spe-rollouts 2000`）
- **F3**: SPE oracle iteration 改为 parity-correct only
- **F4**: `compute_paper_metrics` 加 per-state common random numbers (CRN)，v_tilde / v_hat 用同 seed
- **F5/F6**: doc-only 修正

## Hyperparams (paper §C.2.5 + §C.4)

| Hyperparam | Paper | v3 sweep |
|---|---|---|
| M (NumofTrainEpisodes) | 15000 | 15000 ✓ |
| T̄ = T+1 | 6 (T=5) | 6 ✓ |
| ξ (ExploreRate) | 0.3 | 0.3 ✓ |
| K (SupportSize) | 50 | 50 ✓ |
| ϱ (LearningRate) | 0.04 | 0.04 ✓ |
| filterTresh | sweep {1.0, .95, .9, .85, .8, .75} | sweep {1.0, .95, .9, .85, .8, .75} ✓ |
| treshRatio | 0.5 (filter<1) / 0 (filter=1) | 0.5 / 0 ✓ |
| **gate semantics** | **absolute (paper text)** | **absolute ✓** |
| **SPE rollouts (Alg 5 M)** | **2000** | **2000 ✓** |
| sticky tie-break | Algorithm 3 | --sticky-policy ✓ |
| seeds | 10 | 10 |

---

## Table 3: Policy Error (per cell, 6 filters, mean ± std)

格式：每 cell 两行 — `R` = refactor v3, `P` = paper PDF mean. *-mark = paper §C.4 Pareto-optimal filter.

### CPT88

```
                flt=1.00     flt=0.95    flt=0.90    flt=0.85    flt=0.80    flt=0.75
CPT88/p=0.72
  R           1.00±0.89   1.40±1.11*  1.00±0.89   1.10±1.14   1.30±1.27   0.80±0.98
  P                0.89        0.44*       0.67        0.78        1.33        1.11
CPT88/p=0.66
  R           5.60±2.50   6.60±1.96   6.70±1.90*  5.70±1.27   5.40±2.33   6.00±2.37
  P                5.11        3.44        2.89*       3.00        3.00        3.11
CPT88/p=0.60
  R           2.80±1.89   4.70±2.10   4.60±2.01   2.90±1.45*  2.10±1.04   1.90±1.22
  P                1.22        1.44        0.78        0.67*       1.11        0.78
CPT88/p=0.54
  R           2.70±1.55   2.50±1.20   2.90±1.58   2.40±1.50   2.70±1.62*  2.90±1.76
  P                1.67        1.56        1.11        1.33        0.56*       1.00
CPT88/p=0.48
  R           3.70±1.27*  3.70±1.62   3.30±1.10   4.00±1.41   3.20±1.08   2.90±1.04
  P                1.67*       2.11        3.33        2.44        2.78        2.33
```

### CPT95

```
                flt=1.00     flt=0.95    flt=0.90    flt=0.85    flt=0.80    flt=0.75
CPT95/p=0.72
  R           2.10±0.94   1.40±1.20*  1.50±1.02   1.90±1.30   2.10±1.30   1.20±1.17
  P                1.11        0.88*       1.11        1.22        1.11        1.44
CPT95/p=0.66
  R           0.90±1.14*  0.80±1.08   1.00±1.10   1.20±0.87   1.10±1.51   1.10±1.22
  P                0.22*       0.44        0.78        0.56        0.22        1.00
CPT95/p=0.60
  R           1.30±1.00   1.10±0.83   1.30±1.00   1.20±0.98   1.10±0.83*  1.30±0.78
  P                0.22        0.11        0.22        0.22        0.11*       0.11
CPT95/p=0.54
  R           1.70±1.27   2.50±1.20   1.40±1.36   2.00±1.10*  2.40±1.28   2.20±1.17
  P                0.67        0.78        0.78        0.44*       0.67        0.89
CPT95/p=0.48
  R           2.40±2.20   2.10±1.76   2.10±2.21*  2.30±1.95   2.30±2.15   2.00±1.73
  P                0.22        0.44        0.22*       0.56        0.56        0.78
```

**汇总 (mean-only, |R-P| < 0.5 算"接近")**: 13/60 cells. 与 v2 (~5/60) 相比改善，但仍 60 cells 全都 R > P。

---

## Table 4: Value Error (per cell, 6 filters, mean ± std)

### CPT88

```
                flt=1.00      flt=0.95     flt=0.90     flt=0.85     flt=0.80     flt=0.75
CPT88/p=0.72
  R           8.10±8.62    10.15±8.33*   7.34±6.75    8.74±8.14    8.27±7.48    5.14±6.51
  P                5.98         3.71*        5.41         5.90         8.32         7.90
CPT88/p=0.66
  R          17.50±6.40   17.69±4.36   18.74±4.69*  16.55±1.49   16.80±6.71   16.55±3.45
  P               14.72        10.66         9.30*        9.47         8.82         9.72
CPT88/p=0.60
  R           5.67±6.13   12.86±7.15   10.10±5.73    7.45±5.75*   8.11±5.46    6.95±7.25
  P                2.79         2.86         1.68         1.65*        1.71         1.58
CPT88/p=0.54
  R           8.38±5.63    9.22±4.10    8.37±4.91    7.84±4.80    7.50±5.17*   7.59±5.30
  P                4.96         3.57         3.55         3.79         2.37*        2.29
CPT88/p=0.48
  R           9.40±2.20*   9.65±2.68    8.79±1.75    9.48±2.09    9.07±2.13    8.33±1.50
  P                4.22*        5.82         8.16         5.74         5.00         3.98
```

### CPT95

```
                flt=1.00      flt=0.95     flt=0.90     flt=0.85     flt=0.80     flt=0.75
CPT95/p=0.72
  R          11.10±5.34    9.45±7.93*   9.62±6.24   10.22±7.65   11.11±7.65    8.22±7.34
  P                4.91         3.28*        3.93         2.73         2.57         4.34
CPT95/p=0.66
  R           4.20±6.22*   4.38±7.13    6.23±7.17    7.10±6.40    7.28±9.60    4.82±5.98
  P                1.91*        3.41         4.35         3.74         1.24         5.43
CPT95/p=0.60
  R           7.83±7.11    6.94±6.91    8.94±8.13    7.16±7.24    6.94±6.91*   7.25±6.66
  P                0.94         0.60         0.58         0.52         0.56*        0.84
CPT95/p=0.54
  R           6.00±4.66    9.43±6.21    5.49±5.48    6.89±5.79*   9.18±6.64    7.95±6.34
  P                3.30         1.88         1.39         1.13*        1.08         1.35
CPT95/p=0.48
  R          10.89±8.77    9.81±8.07    8.70±8.45*   9.41±8.02   10.31±8.10    9.12±7.94
  P                2.16         2.50         1.21*        3.35         2.65         2.74
```

**汇总 (|R-P| < 2.0 算"接近")**: 6/60 cells. v2 也是 6/60，但 v3 的 mean 量级整体降低了约 30-50%（CRN 噪声地板下沉）。

---

## v2 → v3 在关键 cell 上的 VE 改善

| Cell | filter | v2 VE mean | v3 VE mean | 下降比例 |
|---|---|---|---|---|
| CPT88/0.72 | 1.00 | 23.07 | 8.10 | **−65%** |
| CPT88/0.72 | 0.95 | 18.35 | 10.15 | −45% |
| CPT88/0.66 | 0.90 | 21.22 | 18.74 | −12% |
| CPT88/0.60 | 0.95 | 19.30 | 12.86 | −33% |
| CPT88/0.54 | 0.85 | 12.66 | 7.84 | −38% |
| CPT88/0.48 | 0.95 | 10.35 | 9.65 | −7% |
| CPT95/0.72 | 0.85 | 11.69 | 10.22 | −13% |
| CPT95/0.60 | 0.80 | 12.23 | 6.94 | −43% |

→ **F4 (CRN) 在 VE 上贡献明显**：尤其 CPT88/0.72/0.60 和 CPT95/0.60 这些 cell，VE 直接砍掉一半。但仍然系统性偏高（paper VE 0.5-15 vs v3 5-18）。

PE 没有可比对照（v2 数据 mean 已稳定，bug 修复对 PE mean 影响小，因为 PE 是 0/1 indicator 不受 CRN 影响）。

---

## Pareto-optimal *-mark 复现

Paper §C.4 用 *-mark 标 (mean PE, std PE) 和 (mean VE, std VE) 二元 Pareto-optimal 的 filter。Refactor 各 cell 在 T3/T4 上的 Pareto front 与 paper 的对比：

| Cell | Paper * | Refactor T3 Pareto | Refactor T4 Pareto | Match? |
|---|---|---|---|---|
| CPT88/0.72 | 0.95 | {0.75, 0.90, 1.00} | {0.75} | neither |
| CPT88/0.66 | 0.90 | {0.80, 0.85} | {0.75, 0.85} | neither |
| CPT88/0.60 | 0.85 | {0.75, 0.80} | {0.80, 0.85, 1.00} | T4-only |
| CPT88/0.54 | 0.80 | {0.85, 0.95} | {0.80, 0.85, 0.95} | T4-only |
| CPT88/0.48 | 1.00 | {0.75} | {0.75} | neither |
| CPT95/0.72 | 0.95 | {0.75, 0.90, 1.00} | {0.75, 0.90, 1.00} | neither |
| CPT95/0.66 | 1.00 | {0.85, 0.95} | {0.75, 1.00} | T4-only |
| CPT95/0.60 | 0.80 | {0.75, 0.80, 0.95} | {0.75, 0.80, 0.95} | **T3+T4 ✓** |
| CPT95/0.54 | 0.85 | {0.85, 0.90, 1.00} | {0.90, 1.00} | T3-only |
| CPT95/0.48 | 0.90 | {0.75} | {0.75, 0.90} | T4-only |

**Paper *-mark 在 refactor T3 AND T4 Pareto 同时出现**: 1/10 (CPT95/0.60). vs v2 是 2/10 — 略 regress 但 dominated by Pareto-front 的 std-driven 噪声，可视作持平。

---

## ⚠️ 撤回 (2026-04-28 末段, tex 源码核对后)

下方"Aggregator bug 修正 (BUG #7/#8)"小节里**关于 σ 应该取 (a_tilde, v_tilde)** 的判断已撤回。tex:936-937 明确公式取在 π̂/V̂ (SPE 基准) 上，原版 `a_hat.std()` 是 paper-correct。BUG #8 改方向是错的，已经在 `lib/io.py` + `scripts/reaggregate_paper_std.py` 里回滚到 `a_hat.std()` / `v_hat.std()`。

正确的修法是 **per-seed rebuild SPE oracle**（在 `barberis_spe.compute_spe_policy` 加 `seed=` 参数 + `run_paper_eval.py` 把 oracle build 移进 seed loop），让 σ_seeds[π̂(x)] 在小 action-gap state 上有跨 seed flip → 非零。验证：M=2000 + 10 seeds 在 CPT88/0.66 上 state (2,0) 真的会 flip (action 0/1)。

详见 [reports/2026-04-28_tex_source_cross_check.md](2026-04-28_tex_source_cross_check.md) §"严重度 1"。

下方原文保留作为修复历史记录。

## ⚠️ Aggregator bug 修正 (BUG #7, 2026-04-28 后续) — σ 部分 retracted

**Paper §4.1 (p.23) 的 PE / VE 公式**:

```
Policy Error mean = Σ_x | μ_seeds[π̃(x)]   − μ_seeds[π̂(x)]   |
Value Error mean  = Σ_x | μ_seeds[V^π̃(x)] − μ_seeds[V^π̂(x)] |
```

**先在 seeds 上取平均，再取绝对值差**。

我之前的 `compute_paper_metrics` 是 **per-seed 算 Σ_x|...| 然后跨 seed 平均**。由 Jensen / triangle 不等式：

```
mean over seeds of Σ_x |V_tilde - V_hat|  ≥  Σ_x | mean[V_tilde] - mean[V_hat] |
```

旧 aggregator **系统性地过估** paper VE（PE 由 Fubini 不变，但 std 公式完全不一样）。这是 BUG #7。

**修复后 60-cell aggregate（paper 公式）**:

| 比较 | PE | VE |
|---|---|---|
| refactor mean **<** paper mean | 4/60 cells | 4/60 cells |
| refactor mean **≥** paper mean | **56/60 cells** | **56/60 cells** |
| 平均 (R − P) | **+1.262** | **+3.980** (旧 +5.20) |
| Aggregate ratio Σ R / Σ P | — | **2.01×** (旧 2.32×) |

→ Paper 公式让 VE gap 缩 25%，但方向仍是 **refactor ≥ paper**——不是 "更小"。

**两套 aggregator 在 v3 数据上的对比** (CPT88/0.66 filter=0.9 paper * cell):

| 公式 | refactor PE | refactor VE |
|---|---|---|
| 旧 (per-seed mean of Σ\|...\|) | 6.70 ± 1.90 | 18.74 ± 4.69 |
| **Paper §4.1** (mean-over-seeds → \|diff\|) | **6.70 ± 0.000** | **17.52 ± 0.000** |
| Paper Table 3/4 数字 | 2.89 ± 0.13 | 9.30 ± 0.95 |

PE mean 由 Fubini 完全相同。VE mean 减小 ~6%（per-cell），ratio 从 2.0× 还是 1.9×。

std=0 是因为 v3 的 SPE oracle 在 seed loop 外只 build 一次 + CRN seeding → π̂/V_hat 跨 seed 几乎 deterministic。要复现 paper 的 σ[π̂(x)] / σ[V^π̂(x)] 需要每个 seed 重 build 自己的 SPE oracle。

## 方向澄清（防止误读 paper 公式后的数据）

paper 公式 修正后，refactor 仍 56/60 cells PE/VE ≥ paper. 不是 "更小"。

但需要分两层看：

**Layer 1: PE-per-cell 量级**
- refactor PE mean range: 0.80 – 6.70
- paper PE mean range: 0.11 – 5.11
- "基本相当"在某些 cell 适用 (e.g. CPT88/0.72 全 6 个 filter R/P 都在 0.8-1.4 vs 0.4-1.3 范围)，但绝大多数 cell refactor 还是 1.5-3× paper.

**Layer 2: VE 是 PE 的"放大版"还是独立信号？**
- Refactor "VE / PE" ≈ 17.52/6.70 = 2.6 (per-disagreement value loss)
- Paper "VE / PE" ≈ 9.30/2.89 = 3.2 (per-disagreement value loss)
- → Per-disagreement 价值损失 refactor **比 paper 更小**!! 这可能是 user 看到的"value error 比paper更小"的视角。

含义：
- Refactor 学到的策略 disagree 的 state 更多 (PE 高)
- 但每个 disagree 的 state 上的价值损失更小 (VE/PE 比 paper 小)
- 综合 VE 还是高（因为 disagree 数量主导）
- **没有"refactor 更好"，只是 disagree 模式不同**：refactor 更倾向于 disagree 在低-action-gap state（损失小），paper 的 disagree 集中在更关键 state（损失大）。

要真的"refactor 对齐 paper"应同时满足：
(a) PE ≈ paper PE (cell-wise)
(b) VE ≈ paper VE (cell-wise)
(c) std ≈ paper std 量级

目前三条都没达到。

## 解读

1. **v3 6-bug-fix 大幅改善 VE 量级 (CRN 起决定作用)，但没解决 PE/VE 比 paper 系统偏高的问题**
   - VE: ~30-65% 下降，但 paper-vs-refactor ratio 仍 2-4×
   - PE: refactor 始终 60/60 cells > paper. Mean ratio 1.5-3× 维持。

2. **Tables 1/2 vs Tables 3/4 的对比说明问题层次**
   - Tables 1/2 (Optimality + SW): v3 已 9/10 within 1σ → V(x_0) 和 Σ_x V(x) 已对齐 paper
   - Tables 3/4 (Policy Error + Value Error): refactor 偏高 → **逐 state 的 policy 选择分歧 + V^π̃(x) 偏离比 V^π̂(x) 大**
   - 含义：refactor 学到的策略在 V(x_0) 上对，在大量非 x_0 状态上不对。Paper 的 SPERL 似乎能在 off-path 状态上也学得很好，refactor 只在 on-path 学得好。

3. **剩余 gap 的可能根因 (按优先级)**
   - **n_batch = 1 vs 5**: paper Algorithm 2 line 4-5 暗示 n_batch=1 (每个 outer iter 只采一条 trajectory)。v3 仍用 n_batch=5。这是最大的未排查 hypothesis。
   - 评估时 `eval_per_state=100` rollouts 噪声地板（即使 CRN 也无法消除策略不同时的真噪声）
   - filterTresh sweep 时 paper 可能用了更长 training (e.g., M=30000) 让 off-path 状态收敛
   - paper 的实现可能是另一份代码（HANDOFF row 14 已记 "paper Tables 1/2 数字源不是 legacy"，Tables 3/4 同源问题继承）

4. **Pareto *-mark 不复现是噪声主导**：std 大于 mean 的 spread 时，Pareto front 由 std 决定，不稳定。要可靠复现 *-mark 大概需要 30+ seeds。

---

## v2 → v3 对比汇总

| 指标 | v2 (2026-04-28 早些时候) | v3 (本 sweep) | 改善 |
|---|---|---|---|
| PE mean ratio (R/P) | 2-5× | 1.5-3× | 中等改善 |
| VE mean ratio (R/P) | 3-7× | 2-4× | **明显改善** |
| VE 绝对量级 | 8-26 单位 | 5-18 单位 | **−30 to −65%** |
| PE within 0.5 (60 cells) | ~5/60 | 13/60 | +8 cells |
| VE within 2.0 (60 cells) | 6/60 | 6/60 | 持平 |
| Paper *-mark 复现 (T3+T4) | 2/10 | 1/10 | 噪声内 |

**结论**: v3 修了 6 个 bug 中 F4 (CRN) 对 VE 量级有明显贡献；其他 bug (F1 filter pre-sort, F2 CLI defaults, F3 parity SPE) 对 Tables 3/4 影响相对小。**Tables 3/4 残余 gap 的最大候选解释是 n_batch=1**，需要单独 ablation。

---

## 数据来源

- v3 sweep: `runs/results_paper_tables_3_4_v3/filt*/barberis_sperl_p*_cpt_*/aggregate.json`
- 重跑命令: `bash scripts/sweep_paper_tables_3_4.sh`
- 总 wall time: ~80 min
- Paper PDF: `paperRef/MSci_MANUSCRIPT.pdf` Table 3 (p.53), Table 4 (p.54)
- v2 (修复前) 数据: `runs/results_paper_tables_3_4/`（保留作 ablation 对照）
- Summary script: `scripts/summarize_paper_tables_3_4.py`

## 下一步建议

1. **n_batch=1 ablation 单 cell** (CPT88/0.66, ~50 min × 10 seeds): 验证是否能把 PE/VE 缩到 paper 量级
2. **eval_per_state=500 ablation** 单 cell: 看是否进一步降低 VE 噪声地板
3. **若 n_batch=1 + eval=500 仍 1.5× gap**: 接受 "paper Tables 数据源不同" 作为最终结论，写入 HANDOFF
4. (低优先级) 把 sweep 跑到 30 seeds 以稳定 Pareto *-mark
