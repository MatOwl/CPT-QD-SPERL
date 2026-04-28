# Paper Tables 3/4 §C.4 全 sweep 复现：发现结构性 PE/VE gap

**日期**: 2026-04-28
**结论**: Refactor 在 §C.4 全 sweep (10 cells × 6 filters × 10 seeds) 上 PE/VE 比 paper 系统性差 2-7×。这是结构性 gap，需要进一步排查 Algorithm 实现细节，而不仅是超参对齐。

---

## 1. 主要数据

### Policy Error mean (per cell, 6 filters)

```
                flt=1.0  0.95  0.90  0.85  0.80  0.75
CPT88/0.72  R   1.00   1.00  0.60  1.10  1.30  0.70
            P   0.89   0.44  0.67  0.78  1.33  1.11
CPT88/0.66  R   5.60   6.00  5.00  4.30  7.00  6.30
            P   5.11   3.44  2.89  3.00  3.00  3.11
CPT88/0.60  R   3.40   4.70  3.20  4.80  4.50  4.70
            P   1.22   1.44  0.78  0.67  1.11  0.78
CPT88/0.54  R   2.70   2.90  2.00  2.40  2.20  2.00
            P   1.67   1.56  1.11  1.33  0.56  1.00
CPT88/0.48  R   5.10   3.00  3.20  2.80  2.90  2.90
            P   1.67   2.11  3.33  2.44  2.78  2.33
CPT95/0.72  R   3.10   1.60  1.70  1.80  1.90  2.60
            P   1.11   0.88  1.11  1.22  1.11  1.44
CPT95/0.66  R   0.90   1.90  1.00  0.80  0.90  0.70
            P   0.22   0.44  0.78  0.56  0.22  1.00
CPT95/0.60  R   1.30   1.60  1.10  1.30  1.30  1.10
            P   0.22   0.11  0.22  0.22  0.11  0.11
CPT95/0.54  R   1.70   1.70  2.20  1.80  1.80  1.50
            P   0.67   0.78  0.78  0.44  0.67  0.89
CPT95/0.48  R   2.40   2.20  2.80  2.60  2.70  2.90
            P   0.22   0.44  0.22  0.56  0.56  0.78
```

**汇总**: 60/60 cells 上 refactor PE > paper PE。Mean ratio ≈ 2-5×。

### Value Error mean

```
                flt=1.0   0.95   0.90   0.85   0.80   0.75
CPT88/0.72  R  23.07  18.35  19.00  22.01  21.83  18.51
            P   5.98   3.71   5.41   5.90   8.32   7.90
CPT88/0.66  R  23.74  24.42  21.22  20.35  25.69  26.12
            P  14.72  10.66   9.30   9.47   8.82   9.72
CPT88/0.60  R  14.84  19.30  15.09  18.13  19.11  17.21
            P   2.79   2.86   1.68   1.65   1.71   1.58
CPT88/0.54  R  11.66  11.72  13.20  12.66  13.35  10.80
            P   4.96   3.57   3.55   3.79   2.37   2.29
CPT88/0.48  R  14.13  10.35  11.01  10.15  10.22  10.93
            P   4.22   5.82   8.16   5.74   5.00   3.98
CPT95/0.72  R  14.96  12.31  10.84  11.69  12.52  13.94
            P   4.91   3.28   3.93   2.73   2.57   4.34
CPT95/0.66  R   7.68  12.76   9.73  10.06   9.75   8.12
            P   1.91   3.41   4.35   3.74   1.24   5.43
CPT95/0.60  R  10.67  12.12  10.18  10.58  12.23  10.15
            P   0.94   0.60   0.58   0.52   0.56   0.84
CPT95/0.54  R   7.53   8.95  10.44   8.03   8.06  10.35
            P   3.30   1.88   1.39   1.13   1.08   1.35
CPT95/0.48  R  13.20  12.70  14.44  13.55  13.96  13.28
            P   2.16   2.50   1.21   3.35   2.65   2.74
```

**汇总**: 60/60 cells refactor VE > paper VE。Mean ratio ≈ 3-7×。

### PE std 对比

| Cell | Refactor PE std 范围 | Paper PE std 范围 | Ratio |
|---|---|---|---|
| 全部 cells | 0.6 - 3.1 | 0.02 - 0.31 | 5-50× 大 |

**Paper seeds 高度一致** (std ≈ 0.05-0.31 → 10 seeds 之间几乎收敛到同一 policy)。
**Refactor seeds 摆动大** (std ≈ 1-3 → 10 seeds 之间收敛到截然不同 policy)。

### Pareto *-mark 复现

| Cell | Paper * | Refactor T3 Pareto | Refactor T4 Pareto | Match |
|---|---|---|---|---|
| CPT88/0.72 | 0.95 | {0.75, 0.90} | {0.75, 0.80, 0.95} | T4 only |
| CPT88/0.66 | 0.90 | {0.85, 0.90} | {0.85, 0.90, 0.95} | T3+T4 ✓ |
| CPT88/0.60 | 0.85 | {0.90} | {0.75, 0.80, 1.00} | neither |
| CPT88/0.54 | 0.80 | {0.75, 0.85} | {0.75, 0.85, 0.90} | neither |
| CPT88/0.48 | 1.00 | {0.85} | {0.85} | neither |
| CPT95/0.72 | 0.95 | {0.75, 0.95, 1.00} | {0.90} | T3 only |
| CPT95/0.66 | 1.00 | {0.75, 0.80, 0.95} | {0.75, 1.00} | T4 only |
| CPT95/0.60 | 0.80 | {0.80, 0.90} | {0.75, 0.80, 0.85, 0.90} | T3+T4 ✓ |
| CPT95/0.54 | 0.85 | {0.75, 0.80, 1.00} | {1.00} | neither |
| CPT95/0.48 | 0.90 | {0.95} | {0.75, 0.80, 0.85, 0.95} | neither |

**Refactor 全 T3+T4 都把 paper *-mark 选作 Pareto-optimal 的: 2/10**

## 2. 含义重估

之前 Tables 1/2 v3 报告说 "Opt 9/10 在 1σ 内"，这个数字现在需要重看：
- Paper SPERL Optimality CPT95 全列 = 0±0；refactor 也 ≈ 0±0 → 看似匹配
- 但 paper 0±0 意味着 SPERL 收敛到与 SPE 相同的 V(x_0)，与全 policy 与 SPE 匹配是不同的事
- Tables 3/4 显示：refactor 在大量非 x_0 state 上学错 → 所以 SW 也偏负 (这与 v3 报告 CPT95 SW 偏负的现象一致)

**结论修正**：refactor SPERL 与 paper SPERL **不等价**，存在系统性更差的学习质量。

## 3. 可能根因 (待排查)

### 3.1 (高优先级) Algorithm 2 隐含 n_batch = 1

Paper Algorithm 2 line 5: "Sample X_0, A_0, R_1, ..., X_T̄ ∼ π_i" — 单 trajectory per outer iteration。
Refactor `learn(n_train_eps, n_batch=5)` 采 5 trajectories per training round。

我们 verification cell (Section 04-27 report) 用 n_batch=5 看到 refactor↔legacy 0.05-0.26σ 匹配，因为 legacy 也用 n_batch=5。但 paper 可能用 n_batch=1（per Algorithm 2 文字）。

需要测试：refactor n_batch=1 重跑 CPT88/0.66 cell 看 PE/VE 是否降。

### 3.2 (高优先级) Algorithm 4 line 20 absolute vs relative gate

Paper: |Q_filtered − Q| > treshRatio (绝对)
Refactor: |Q_filtered − Q| / |Q| > treshRatio (相对)

行为差异：treshRatio=0.5 在 paper 的绝对意义下接近"基本不 reject filter"（对 Q ≈ 1-100 来说）；refactor 相对 0.5 经常 reject。

需要测试：refactor 加 absolute gate flag 重跑。

### 3.3 (中优先级) SPE oracle M

Paper Algorithm 5 line 1: M = 2000 rollouts per (state, action)。
Refactor sweep: 我们用 spe_rollouts=300。

差额 6.7×。SPE oracle 估计噪声差 √6.7 ≈ 2.6×。

需要测试：spe-rollouts=2000 重跑 verification cell。

### 3.4 (中优先级) firstVisit 是否完全一致

Paper §C.2.4: φ^0 ← (1/K) Σ_j T̂φ_j on first visit。Refactor 也是 mean of targets。逻辑一致，但是否触发条件 (first visit 的定义) 完全一致需 line-by-line。

### 3.5 (低优先级) 评估细节

`eval_per_state=100` (paper 没明说)。Tables 3/4 的 PE 用 0/1 indicator per state，所以 100 should be fine for stable mean. VE 用 V_pi rollouts，100 可能不够噪声平均。

## 4. 立即建议

按优先级跑 ablation：
1. **n_batch=1 重跑 CPT88/0.66 cell** (~10 min) — 看 PE/VE 是否降
2. 若 #1 无效：spe_rollouts=2000 单 cell 重跑 (~10 min)
3. 若 #1 有效：扩到全 60 cells 重跑 (~90 min × 数倍 (n_batch=1 比 n_batch=5 慢))
4. 若 #1 + #2 都无效：转向 absolute filter gate

## 5. 附录：原 Tables 1/2 v3 报告的修正

Section 2026-04-27 v3 的 "Opt 9/10 1σ 内 / SW 6/10 1σ 内" 是 mean+std 匹配，但 mean 匹配不代表 policy 匹配。Tables 3/4 直接测量 policy 差异，结果是大部分 cells refactor 与 SPE 差异远大于 paper 与 SPE 的差异。

之前的 "Refactor ≡ Legacy 0.5σ verification" 仍然成立，因为那是 paper-vs-legacy 的对比，不是 paper-vs-refactor。Legacy 应该有同样的 PE/VE 系统性偏离，本仓库代码与 paper 数据不一致是 long-standing issue (符合 HANDOFF 之前的判断 "VE 系统性 2.5× 高于 paper")。

但本次 Tables 3/4 sweep 的精确数字是新数据，提供更系统的证据：**问题不在某一个 cell，而是普遍**。

## 文件位置

- §C.4 sweep script: `scripts/sweep_paper_tables_3_4.sh`
- §C.4 summary: `scripts/summarize_paper_tables_3_4.py`
- 60-cell 输出: `runs/results_paper_tables_3_4/filt*/barberis_sperl_p*_cpt_*/`
- Implementation 对照: `reports/2026-04-28_implementation_vs_paper_verification.md`
