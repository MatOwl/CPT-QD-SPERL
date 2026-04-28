# 用 LaTeX 源码 (`paperRef/Version 1.1/DRL=SPE.tex`) 校对前期 4 篇笔记

**日期**: 2026-04-28
**目的**: 用作者新同步的 tex 源码（比 PDF 更准确、更易引用）核对 2026-04-28 同日 4 篇报告 + HANDOFF 中所有"按 paper 文字"做出的判断。
**结论**: tex 大部分确认前期判断；但有 **1 处方向性错误（PE/VE std 公式改反了）**、**1 处数据笔误**、**3 处 section 引用号错配**、**1 处 SW 求和域含糊**。下文按严重度由高到低排列。

被核对的报告：
- `reports/2026-04-28_implementation_vs_paper_verification.md` (M1-M9 + Addendum F1-F6)
- `reports/2026-04-28_paper_tables_1_2_v3_comparison.md`
- `reports/2026-04-28_paper_tables_3_4_v3_comparison.md` (含 BUG #7 / BUG #8)
- `reports/2026-04-28_tables_3_4_sweep_findings.md`

tex 源码引用：以 `tex:LINE` 标注。

---

## 严重度 1 ──「BUG #8」改错方向（关键）

### 论文原文（tex:936-937）

```
Policy Error  mean  = Σ_x | μ[π̃(x)] − μ[π̂(x)] |
              stdev = (1/|X|) Σ_x σ[π̂(x)]
Value Error   mean  = Σ_x | μ[V^π̃(x)] − μ[V^π̂(x)] |
              stdev = (1/|X|) Σ_x σ[V^π̂(x)]
```

**关键**: σ 取在 **π̂ / V^π̂（SPE/hat 基准）** 上，不是 **π̃ / V^π̃（SPERL/tilde）** 上。

### 当前实现做了什么

`scripts/reaggregate_paper_std.py:78-79`:
```python
pe_std_terms.append(a_tilde.std())     # ← 改成了 tilde
ve_std_terms.append(v_tilde.std())     # ← 改成了 tilde
```

而原版 `lib/io.py:_aggregate_paper_formula` 用的是 `a_hat.std()` / `v_hat.std()`，是 **paper-correct** 的。

我们 04-28 的「BUG #8 fix」（HANDOFF + 3-4 v3 报告）把它从 hat 改成了 tilde，**改反了**。

### 改反的根本原因（追溯）

报告 3-4 v3 §"⚠️ Aggregator bug 修正" 自己就承认：
> std=0 是因为 v3 的 SPE oracle 在 seed loop 外只 build 一次 + CRN seeding → π̂/V_hat 跨 seed 几乎 deterministic。要复现 paper 的 σ[π̂(x)] / σ[V^π̂(x)] 需要每个 seed 重 build 自己的 SPE oracle。

正确的修法是 **per-seed re-build SPE oracle（drop CRN at seed boundary）**，而不是改 std 公式的取数对象。

paper Tables 3/4 std 数字（0.05-0.31）说明：paper 的 SPE oracle 跨 seed **有真实噪声**，因为没有 CRN seeding。

### 修复方向

两个方案：

1. **回滚 std 公式到 hat**：
   - `scripts/reaggregate_paper_std.py:78-79` 改回 `a_hat.std()` / `v_hat.std()`
   - `lib/io.py:_aggregate_paper_formula` 同步
   - 同时 `lib/paper_eval.py` 让每个 seed 跑一次自己的 SPE oracle（即去掉 oracle 的 outer-once cache + CRN）
   - 再重跑 Tables 3/4 sweep

2. **保留 CRN，承认 paper-公式不可重现 std 量级**：
   - 留 `a_tilde.std()` 当作 *近似* 但 mark 报告里"非 paper 公式"
   - 写 HANDOFF 说明无法逐字复现 paper σ 列

我推荐方案 1。

### 影响

- v3 的 PE σ μ ≈ 0.20、VE σ μ ≈ 0.71 是用 **tilde-std** 算出来的，不能直接和 paper 的 **hat-std** 比 σ 量级。**所谓"σ 已经回到 paper 区间"的结论需要重审**。
- PE/VE **mean** 不受影响（绝对值差，前后顺序不变 → Fubini）。所以 mean 系统偏高 1.5-3× 的结论仍然成立。

---

## 严重度 2 ──「Table 1 CPT88/p=0.72 paper σ 笔误」

`reports/2026-04-28_paper_tables_1_2_v3_comparison.md:40`:
> `| CPT88 / p=0.72 | 0.95 | **7.91 ± 1.64** | 9.47 ± 1.55 | 0.97σ | ✓ |`

tex:1066:
```
$7.91_{.64}{}^\dag$    →  paper 实际是 7.91 ± 0.64（不是 1.64）
```

报告里 "0.97σ" 这一栏的计算其实是用 σ=0.64 算的（√(0.64²+1.55²) = 1.68，差 1.56 → 0.93σ ≈ 0.97σ），所以 **数字栏对、表格里抄错了一位**。改成 `7.91 ± 0.64` 即可，结论不变。

其他 9 个 cell 的 paper σ 抽样核对一致（CPT88/0.66 2.74_.95、CPT88 SW -59.94_2.88、CPT95/0.72 SW -7.31_3.32 等都对）。

---

## 严重度 3 ──「Section 引用号错配」

我在 4 篇报告里用了 §C.2.1 / §C.2.2 / §C.2.3 / §C.2.4 / §C.2.5 来索引 paper 的「filtering」「consistent tie-break」「crossing quantiles」「first visit init」「hyperparams」。

实际 tex 的章节结构：

| 我标的 | tex 真实位置 | 实际章节标题 |
|---|---|---|
| §C.2.1 Algorithm 2 | §6 (body) | "A CPT QDRL Algorithm" |
| §C.2.2 Consistent tie-break | §6.2 (body, tex:831) | "Selecting Greedy Actions" |
| §C.2.3 Crossing quantiles | §6.3.1 (body, tex:866) | "Filtering" → Crossing 子项 |
| §C.2.4 firstVisit | §6.3.2 (body, tex:910) | "Initialization" |
| §C.2.5 hyperparams | Appendix `cpt5-cpt-qr-implement` (tex:1828) | "SPERL (CPT QR Q-learning)" |
| §C.4 sweep tables 3/4 | §7.1 (body, tex:931) | "Equilibrium Learning and Approximation" |

也就是说：

- 1-3-4 实际位置都在 paper 主体的算法章节（§6），不在附录 C
- 5 在附录但不是 §C.2.5（具体编号取决于 PDF 渲染，但 tex `\bmsubsubsection{SPERL (CPT QR Q-learning)}` 是 Appendix C 的一个子项）
- §C.4 的 Tables 3/4 实际在 paper 主体 §7.1，不在附录

**修复**：报告里把所有 `§C.2.X` / `§C.4` 改成更稳的"算法章节内 N-th 子节"或者直接用 tex label（如 `sec: chp5-algo-improvements`、`sec: cptsperl-experiments-filtering`）。

实质内容（公式、伪代码、超参数）都对，所以这是 **doc-level** 错误，不影响代码或结论。

---

## 严重度 4 ──「SW 求和域 X 含糊」

tex 的定义（line 504-505）：
> Given any joint strategy profile π, **SW(π) = Σ_x U_x(π)**.

其中 `U_x(π) = V^π(x)`（line 501），x 取自所有 player ≡ 状态空间 X。

模糊点：finite-horizon `X = {(t, s): t ∈ [0,...,T], s ∈ S_t}` — **包含 terminal t=T 状态**。但是：
- terminal 状态没有 nontrivial action（A_x = {bar{a}}），game-theoretic 意义上不算 player
- 算 V^π((T,s)) = u(s) 是 deterministic，跨 policy 加 constant offset

paper 实际数字（Table 2 CPT88/0.66 SPERL = -59.94）若包含 6 个 terminal（CPT88 下贡献 ≈ -73），总 SW 应该 ≈ -132。所以 **paper 实际把 terminal 排除了**，与 refactor `featurizer.iter_states()` 排除 terminal 的做法一致。

**结论**: 实现行为正确，但 M8 描述 "SW 求和域 = 15 decision states" 应该补一句"paper 形式上写 'all players X' 但隐含排除 terminal（因 terminal 无 strategy）" 以避免误读。

---

## 严重度 5 ──「CPT 单参数 δ 与 ρ_gain ≠ ρ_loss」

tex:359:
```
w(p; δ) = p^δ / (p^δ + (1−p)^δ)^(1/δ),  δ ∈ (0, 1).   ← 单 δ
```

而 refactor `lib/cpt.py:prob_weight(F, pos, rho1, rho2)` 允许 ρ_gain ≠ ρ_loss。

实验 (tex:921):
- CPT88: (α, δ, λ) = (.88, .65, 2.25)
- CPT95: (α, δ, λ, B) = (.95, .5, 1.5)

**两个实验都用单 δ**，所以 refactor 的扩展不影响 paper 复现。M1 已经标 "等价"，正确。

---

## 已确认完全无误的部分

以下报告判断在 tex 中得到完全确认：

| 报告原文 | tex 引用 | 状态 |
|---|---|---|
| Algorithm 2 backward sweep `for t = T̄−1 to 0` | tex:791（橙色高亮） | ✓ |
| Algorithm 2 quantile lookup index 是 i+1（backward） | tex:796, 798, 800 | ✓ |
| Algorithm 2 line 13 a* with consistent tie-break | tex:799（橙色） | ✓ |
| Algorithm 3 strict-improvement only update | tex:846-850 | ✓ |
| Algorithm 4 line 5 `d* = sampleQuantiles(d_{2:K}, filterTresh)` | tex:887 | ✓ |
| Algorithm 4 line 6 `bool_j ← 1[0 ≤ d_j ≤ d*]; bool_1 ← bool_2` | tex:888 | ✓ |
| Algorithm 4 line 11-15 snap 到更近邻居 | tex:893-897 | ✓ |
| **Algorithm 4 line 20 gate 是绝对 \|Q_filt − Q\| > treshRatio** | tex:902 | ✓✓ 关键确认 |
| Algorithm 5 (True SPE) M = 2000 | tex:1846 | ✓✓ 关键确认 |
| §C.2.5 hyperparams M=15000, T̄=6, ξ=.3, K=50, ϱ=.04, treshRatio=.5 | tex:1828-1838 | ✓ |
| filterTresh ∈ [.75, 1.] | tex:1836 | ✓ |
| firstVisit init: φ^0_{1:K}(x,a) ← (1/K) Σ_j T̂φ_j | tex:910 | ✓ |
| Optimality mean = μ[V^π(x_0)], stdev = σ[V^π(x_0)] | tex:1052 | ✓ |
| SW mean = μ[Σ_x V^π(x)], stdev = σ[Σ_x V^π(x)] | tex:1053 | ✓ |
| CPT formula eq 13-15 | tex:346-360 | ✓ |
| CPT88 (.88, .65, 2.25) / CPT95 (.95, .5, 1.5) | tex:921 | ✓ |
| Tables 1-2 paper 数字（除 CPT88/0.72 σ 笔误外） | tex:1058-1104 | ✓ |
| Tables 3-4 paper 数字 | tex:940-984 | ✓ |
| n_batch / batch size 在 paper 里**没出现** | tex 全文搜过 | ✓（confirms "paper 未明示"） |
| Algorithm 2 hyperparam tuple 是 (M, T̄, ξ, K, ϱ)（不含 κ Huber） | tex:784 | ✓（pure ρ_τ，非 Huber） |

---

## paper §4.2.5 "hardest convergence" 定位修正

`reports/2026-04-28_paper_tables_1_2_v3_comparison.md:51` 写 "paper §4.2.5 自己也称这是 'hardest convergence' cell"。

tex 实际只在 §7.1 (tex:1006) 写 "the worst SPE learning result from Figure ref{fig: SPE-match-Pol-8866}"（即 CPT88/p_win=.66）和 (tex:1144) "the furthest distance is realized in Figure ref{fig: frontier-88}(b)"。

**没有 §4.2.5 编号，也没有"hardest convergence"原文**。但 paper 确实用 "worst" 描述了同一个 cell。

修法：把"§4.2.5 自己也称这是 'hardest convergence'"改成"paper §7.1 在 fig: SPE-match-Pol-8866 + fig: frontier-88(b) 也单独把这个 cell 标为 worst SPE learning result"。

实质判断（CPT88/0.66 是 paper 自己报告的 worst cell）正确。

---

## 关于 §C.4 这个标号

`reports/...3_4_sweep_findings.md` 用 §C.4 来指 Tables 3/4 sweep 的来源。tex 里没有 §C.4 编号 — Tables 3/4 实际在 body 的 §7.1（tex `sec: cptsperl-experiments-filtering`，tex:931），不是附录。filterTresh per-cell 取值 `*-mark` 在 tex:940-984 表格中。

实质内容对，标号错。

---

## 修法清单（按优先级）

### P0（数据正确性）

1. **回滚 PE/VE std 公式到 hat**
   - 文件：`scripts/reaggregate_paper_std.py:78-79`、`lib/io.py:_aggregate_paper_formula`
   - 同时改 `lib/paper_eval.py` 让 SPE oracle per-seed rebuild（drop outer-once cache + CRN-seeding for V_hat）
   - 重跑 `scripts/reaggregate_paper_std.py runs/results_paper_tables_3_4_v3`

2. **HANDOFF + 3-4 v3 报告里把"BUG #8 fix"重写**
   - 把"σ_seeds[π̂] ≈ 0 → 改用 σ_seeds[π̃]"这个 reasoning 标为 retraction
   - 替换为"per-seed SPE oracle rebuild + 保留 hat-std"

### P1（笔误）

3. **`reports/2026-04-28_paper_tables_1_2_v3_comparison.md:40`**: 把 "7.91 ± 1.64" 改成 "7.91 ± 0.64"

### P2（doc 引用号修正，非阻塞）

4. 4 篇报告里的 §C.2.X / §C.4 标号改成 tex label（`sec: chp5-algo`, `sec: chp5-algo-greedy-actions`, `sec: chp5-algo-improvements`, `sec:filtering`, `sec: cptsperl-experiments-filtering`, `app: cpt5-cpt-qr-implement` 等）。

5. M8 SW 求和域注释：补一句 "tex 形式上写 Σ_x with x ∈ X 包含 terminal，但 terminal 无 strategy 不算 player；paper 实际数字（如 CPT88/0.66 SPERL SW=-59.94）只能在排除 terminal 时成立"。

6. CPT88/0.66 "hardest convergence" wording → "paper §7.1 worst SPE learning result"。

---

## 数据/代码的下一步建议

**先做 P0**（回滚 std 公式 + per-seed SPE oracle），然后用现有 `runs/results_paper_tables_3_4_v3` 数据再跑 `reaggregate_paper_std.py`，看 σ 是否能直接对到 paper 的 0.05-0.31 区间。

如果 σ 落在 paper 区间 → 数据复现度（除 PE/VE mean 仍偏高之外）会更高，且 paper 公式得到逐字执行。

如果 σ 仍 5-50× 偏高（像之前 v2 那样）→ 说明 refactor 学到的 SPE 估计跨 seed 噪声本来就比 paper 大（M=2000 仍不够）；这是结构性发现，应该写进 HANDOFF。
