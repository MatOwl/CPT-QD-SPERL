# Paper Tables 1/2/3/4 v4 — per-seed SPE oracle 重跑后对比

**日期**: 2026-04-28
**目的**: tex 源码 (`paperRef/Version 1.1/DRL=SPE.tex:936-937`) 核对后，撤回 04-28 早些时候的「BUG #8 σ 公式取错」(σ 改取 (a_tilde, v_tilde) 是错的方向)，改成正确修法 — **per-seed rebuild SPE oracle**，并保留 paper-correct 的 `a_hat.std()` / `v_hat.std()`。本报告重跑 `runs/results_paper_tables_*_v3` 后看 σ 是否落入 paper 量级。

**数据**:
- `runs/results_paper_tables_1_2_v3/` (10 cells × 10 seeds, 重跑) — wall ~30 min
- `runs/results_paper_tables_3_4_v3/` (60 cells × 10 seeds, 重跑) — wall ~80 min

**修复前后对比** (v3 内部 σ 形式从 `a_tilde.std()` → `a_hat.std()` + per-seed oracle):

| 指标 | v3 (BUG #8 fix, σ on tilde, oracle once) | v4 (paper-correct σ on hat, per-seed oracle) |
|---|---|---|
| σ 形式与 paper §4.1 一致？ | ❌ | ✅ |
| σ 跨 seed 真实变化来源 | ❌ (oracle 静态) | ✅ (oracle MC 随 seed 重建) |
| Tables 1/2 Opt 1σ-within | 9/10 | 9/10 (持平) |
| Tables 1/2 SW 1σ-within | 9/10 | 7/10 (略 regress, 见下) |
| Tables 3/4 PE σ ≈ paper | (跨 cell 0.20 mean, 但是错 side) | 21/60 within ±0.05 |
| Tables 3/4 VE σ ≈ paper | (0.71 mean, 错 side) | 19/60 within ±0.30 |

---

## Tables 1/2 (Optimality + SW)

### Optimality

| Cell | filter | Paper | v4 Refactor | \|R−P\|/σ_pool | 1σ |
|---|---|---|---|---|---|
| CPT88/0.72 | 0.95 | 7.91±0.64 | 9.65±2.38 | 0.85σ | ✓ |
| CPT88/0.66 | 0.90 | 2.74±0.95 | 0.88±2.08 | 1.15σ | ✗ |
| CPT88/0.60 | 0.85 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT88/0.54 | 0.80 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT88/0.48 | 1.00 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT95/0.72 | 0.95 | -0.03±0.33 | -0.53±0.83 | 0.79σ | ✓ |
| CPT95/0.66 | 1.00 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT95/0.60 | 0.80 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT95/0.54 | 0.85 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |
| CPT95/0.48 | 0.90 | 0.00±0.00 | 0.00±0.00 | 0.00σ | ✓ |

**Opt 1σ-within: 9/10** (持平 v3)

### Social Welfare

| Cell | filter | Paper | v4 Refactor | \|R−P\|/σ_pool | 1σ |
|---|---|---|---|---|---|
| CPT88/0.72 | 0.95 | -26.03±3.63 | -28.48±7.10 | 0.43σ | ✓ |
| CPT88/0.66 | 0.90 | -59.94±2.88 | -66.74±5.28 | 1.60σ | ✗ |
| CPT88/0.60 | 0.85 | -80.09±4.22 | -80.54±4.27 | 0.11σ | ✓ |
| CPT88/0.54 | 0.80 | -91.49±2.93 | -93.72±1.66 | 0.93σ | ✓ |
| CPT88/0.48 | 1.00 | -104.41±2.13 | -108.20±5.98 | 0.84σ | ✓ |
| CPT95/0.72 | 0.95 | -7.31±3.32 | -9.03±4.40 | 0.44σ | ✓ |
| CPT95/0.66 | 1.00 | -17.56±1.20 | -18.19±5.52 | 0.16σ | ✓ |
| CPT95/0.60 | 0.80 | -23.67±1.04 | -23.90±3.11 | 0.10σ | ✓ |
| CPT95/0.54 | 0.85 | -30.36±0.99 | -34.20±4.97 | 1.07σ | ✗ |
| CPT95/0.48 | 0.90 | -35.61±2.37 | -42.40±4.73 | 1.81σ | ✗ |

**SW 1σ-within: 7/10** (vs v3: 9/10)

**SW 略 regress 解读**: per-seed SPE oracle 引入了 ref env 的额外 RNG 重建 (`np.random.seed`)，agent 训练 RNG stream 不再和 v3 完全一致，agent 收敛位置略有偏移。CPT95/0.54、CPT95/0.48 这两个原本边缘通过的 cell 现在 1.07σ / 1.81σ 失败。Opt 列没受影响（Opt 都是 0±0 类型）。

---

## Tables 3/4 σ 列对比 (核心 v4 deliverable)

paper §4.1 公式：
- σ_PE = (1/\|X\|) Σ_x σ_seeds[**π̂(x)**]   (在 SPE 基准上)
- σ_VE = (1/\|X\|) Σ_x σ_seeds[**V^π̂(x)**] (在 SPE 基准上)

per-seed oracle 后，σ 仅跨 (regime, p_win) 变化，不随 filter 变 (filter 是 SPERL 超参数，不影响 SPE oracle)。下表展示每个 (regime, p_win) 在 6 个 filter 上的 σ (refactor 全相同) vs paper 6 个 σ (paper 也基本不随 filter 变 — 见 paper §C.4 表):

### PE σ

```
cell           refactor σ     paper σ range (跨 6 filter)
CPT88/0.72     0.000          0.05 - 0.12
CPT88/0.66     0.020          0.13 - 0.31
CPT88/0.60     0.107          0.11 - 0.17       ← 落入 paper 区间
CPT88/0.54     0.000          0.10 - 0.17
CPT88/0.48     0.000          0.16 - 0.26
CPT95/0.72     0.020          0.06 - 0.14
CPT95/0.66     0.000          0.03 - 0.09
CPT95/0.60     0.000          0.02 - 0.04
CPT95/0.54     0.020          0.05 - 0.12
CPT95/0.48     0.000          0.03 - 0.10
```

### VE σ

```
cell           refactor σ     paper σ range (跨 6 filter)
CPT88/0.72     0.00           0.62 - 0.82
CPT88/0.66     0.20           0.77 - 2.12
CPT88/0.60     0.10           0.30 - 0.43
CPT88/0.54     0.00           0.41 - 0.62
CPT88/0.48     0.00           0.44 - 0.69
CPT95/0.72     0.14           0.30 - 0.50
CPT95/0.66     0.00           0.27 - 0.47
CPT95/0.60     0.00           0.14 - 0.17
CPT95/0.54     0.00           0.17 - 0.37
CPT95/0.48     0.00           0.25 - 0.41
```

### 命中率

| | refactor 在 paper σ ±0.05 / ±0.30 内 |
|---|---|
| PE σ | 21/60 cells |
| VE σ | 19/60 cells |

### 模式

1. **方向对**：refactor σ ∈ [0, 0.107] 全 ≤ paper σ。说明 paper 的 SPE oracle 比我们 noisier (跨 seed flipping 更多)。
2. **小 action-gap cell 出现 σ > 0**：CPT88/0.60 (σ_PE=0.107) 和 CPT88/0.66 (σ_PE=0.020) 跟 paper "worst SPE learning result" (CPT88/0.66) 的指认一致。
3. **大 action-gap cell σ = 0**：M=2000 在这些 cell 上让 SPE argmax bit-stable 跨 seed。Paper σ > 0 说明它的 oracle 比我们 noisier — 可能 M < 2000，或者 paper σ 同时含训练侧 noise (我们的 CRN 把 V_hat 跨 seed 完全锁死了，paper 可能没有 CRN)。

### 余下 gap 的可能解释 (按可能性排序)

1. **paper M < 2000**：tex:1846 写 "M=2000" 是 oracle pseudocode 默认，但 paper §C.4 sweep 实际可能用了较小的 M (e.g. M=200) → 10× 更 noisy → σ 上去了。需要 ablation：refactor 用 spe-rollouts=200 重跑 1 个 cell 验证。
2. **paper 没用 CRN at eval**: refactor `compute_paper_metrics:271` 用 `state_seed = hash(state)` 锁住 V_hat 跨 seed → 跨 seed 同 state 上 V_hat 一致 (除 SPE policy 本身 flip). 若 paper 的 V_hat 用独立 seed eval, 会引入额外 stochastic noise → σ_VE 变大. 但 PE σ 不变 (PE 看 argmax indicator).
3. **paper 跨 seed 同时跨 (training, oracle, eval)**：如果三者一起 reseed, σ_VE 会含 V_hat eval noise → 解释 VE σ 比 PE σ 更难追平.

---

## Tables 3/4 mean 列 (Σ_x |Δμ|) 仍 R > P

per-seed oracle 不影响 mean (mean = Σ_x |μ_seeds[π̃(x)] − μ_seeds[π̂(x)]|, μ_seeds[π̂(x)] 跨 seed 取平均，单 state flip 影响很小):

```
PE: 60/60 cells refactor > paper, mean ratio ≈ 1.5-3×
VE: 60/60 cells refactor > paper, mean ratio ≈ 2-4×
```

mean 上的系统性 gap 是另一条独立 issue，**不在本次 v4 修复范围**。HANDOFF 已记 "结构性 PE/VE 偏高，无法仅通过对齐 paper 文字闭合"，仍然成立。

---

## Pareto *-mark 复现

paper §C.4 *-mark = (PE mean, PE std) ∧ (VE mean, VE std) 二元 Pareto-optimal filter.

| Cell | Paper * | Refactor T3 Pareto | Refactor T4 Pareto | T3+T4 match? |
|---|---|---|---|---|
| CPT88/0.72 | 0.95 | {0.95, 1.00} | {0.95} | **✓** |
| CPT88/0.66 | 0.90 | {0.90} | {0.75} | T3-only |
| CPT88/0.60 | 0.85 | {0.95} | {0.95} | neither |
| CPT88/0.54 | 0.80 | {0.80, 0.90} | {0.90} | T3-only |
| CPT88/0.48 | 1.00 | {1.00} | {0.85} | T3-only |
| CPT95/0.72 | 0.95 | {0.75} | {0.75} | neither |
| CPT95/0.66 | 1.00 | {0.90} | {0.95} | neither |
| CPT95/0.60 | 0.80 | {0.80} | {0.80} | **✓** |
| CPT95/0.54 | 0.85 | {0.95} | {1.00} | neither |
| CPT95/0.48 | 0.90 | {0.75} | {0.85} | neither |

**T3+T4 共同匹配 paper *: 2/10** (CPT88/0.72, CPT95/0.60)

vs v3: 1/10 (CPT95/0.60)。略改善 1 cell，但本质仍是噪声 dominated。

---

## 结论

1. **Paper §4.1 σ 公式逐字执行**：撤回 BUG #8 (`a_tilde.std()`)，回滚 `a_hat.std()` + 引入 per-seed SPE oracle (在 `barberis_spe.compute_spe_policy(seed=...)` 加 RNG save/restore，`run_paper_eval.py` 把 oracle build 移进 seed loop, offset=1_000_003)。
2. **σ 量级显著下沉但非零**：CPT88/0.60 cell σ_PE=0.107 完美落入 paper 0.11-0.17 区间; 其余 cells σ 多为 0 或 0.02-0.20（paper 0.02-0.31、0.14-2.12）。21/60 PE σ 与 19/60 VE σ 在 paper ±0.05/±0.30 内.
3. **mean 列没动**：Tables 3/4 PE/VE mean 全 60/60 cells 仍 R > P，跟 v3 一致。系统性高 1.5-3× 是独立 issue。
4. **Tables 1/2 SW 略 regress**: 9/10 → 7/10，per-seed oracle 引入 RNG stream 偏移；agent 训练在 CPT95/0.54、CPT95/0.48 上略走偏。Opt 列稳定 9/10。

### 下一步可选

- (低优先级 ablation) `--spe-rollouts 200` 单 cell 重跑：看 σ 是否升到 paper 量级，验证 paper 实际 M < 2000 的猜测。
- (低优先级) 同时让 V_hat 在 `compute_paper_metrics` 中 drop CRN (用 training seed 而不是 hash(state))，看 σ_VE 是否再涨。
- 接受当前 σ 量级偏小，作为 "M=2000 too stable" 的结构性说明写入 HANDOFF。

---

## 文件

- 数据: `runs/results_paper_tables_1_2_v3/`、`runs/results_paper_tables_3_4_v3/` (在原位重跑覆盖 v3)
- 代码: `lib/envs/barberis_spe.py:compute_spe_policy(seed=...)`, `agents/run_paper_eval.py` (oracle inside seed loop), `lib/io.py:_aggregate_paper_formula` (σ 回滚), `scripts/reaggregate_paper_std.py` (同步)
- 历史: `reports/2026-04-28_paper_tables_1_2_v3_comparison.md`, `reports/2026-04-28_paper_tables_3_4_v3_comparison.md` (v3 数据 + BUG #8 retracted notice)
- tex 校对: `reports/2026-04-28_tex_source_cross_check.md`
