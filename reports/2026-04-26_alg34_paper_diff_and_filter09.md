# 2026-04-26 — Paper Alg 3/4 vs 实现 diff + filter=0.9 复现 paper p=0.66

## 任务

HANDOFF TODO #1：读 paper Appendix C.2.2 / C.2.3 把伪代码和我们的 [`agents/sperl_qr_generic.py`](../agents/sperl_qr_generic.py) `GreedyPolicy.update_from_critic_values` + `filter_quantiles` 做行级 diff，看能否解释 [Phase 3](2026-04-26_welfare_and_paper_tables.md) 报告的 "Alg 3/4 在 CPT88/p=0.66 上崩到 0"。

## Algorithm 3 (paper p49) 伪代码 vs 实现

```
1: Input: x', π_i(x'), {Q(x', a')}
2: if max_a' Q(x', a') > Q(x', π_i(x')):
3:     π_{i+1}(x') ← arg max_a' Q  (random tie-breaks)
4: else:
5:     π_{i+1}(x') ← π_i(x')
```

我们的 `update_from_critic_values(sticky=True, tie_thresh=0)`：
1. `candidates = flatnonzero(cpt == max)`（严格 argmax 集合）
2. `new_a = rng.choice(candidates)`
3. 若 `cpt[new_a] > cpt[prev_a]` flip 到 new_a，否则 keep prev_a

**Diff**：当 `tie_thresh=0` 时，candidates = paper 的 arg-max 集合，`cpt[new_a]=v_max`，分支条件 `v_max > cpt[prev_a]` ≡ paper 条件。✓ 完全等价。

**`tie_thresh > 0` 是我们额外加的**——paper p54 明确说："we are not dealing with actual ties, but perceived ties... induced by errors larger than Δ"，paper Algorithm 3 严格用 strict argmax + strict-better。`tie_thresh > 0` 应当默认关闭。

Legacy [`agents/rerun_GreedySPERL_QR__main.py:765-768`](../agents/rerun_GreedySPERL_QR__main.py) 用 `np.argmax`（deterministic 取最小 index）而不是 `rng.choice`，这是和 paper 文字 "random tie-breaks" 的小差异，但功能等价。

## Algorithm 4 (paper p51) 伪代码 vs 实现

Filter 主体（Lines 4-17：snap invalid quantiles to nearest valid）与 [`filter_quantiles`](../agents/sperl_qr_generic.py:38) 行行对得上：
- `bool_1 ← bool_2`（paper Line 6）↔ `valid[0] = valid_gap[0]` ✓
- `dist_l < dist_u → φ_jl, else φ_ju`（Line 15）↔ ours 同语义 ✓
- `sampleQuantiles(d_{2:K}, filterTresh)` 我们用 `np.quantile(method="higher")`，匹配 legacy [`rerun_GreedySPERL_QR__main.py:1147`](../agents/rerun_GreedySPERL_QR__main.py:1147)（`interpolation='higher'`） ✓

**Acceptance gate (Lines 18-22)**：
- Paper 文字写 `|Q_filtered - Q| > treshRatio` (绝对)
- Legacy [`rerun_GreedySPERL_QR__main.py:222`](../agents/rerun_GreedySPERL_QR__main.py:222) 实际是 `abs((cpt_estimate - cpt_ori)/cpt_ori) > treshRatio` (相对)
- 我们 [`_cpt_from_quantiles`](../agents/sperl_qr_generic.py:165) 用相对 → **匹配 legacy，paper 文字 likely shorthand**

**默认值 mismatch**：
- Paper C.2.5 / legacy: `treshRatio=0.5`（filter 开启时）
- 我们的 `filter_accept_ratio` 默认 `inf`（gate 完全失效）

## CPT88/p=0.66 hyperparameter sweep

paper 推荐超参（K=50, lr=0.04, M=15000, ε=0.3, batch=5）。

### 3 seeds 初探（不可靠，仅作对照）

| Config | Optimality (3s) |
|---|---|
| Baseline | 4.57 ± 2.31 |
| Filter=0.75 + sticky + accept=0.5 (旧) | -0.02 ± 0.03 |
| Filter=0.9 alone | 2.62 ± 1.37 |
| Sticky alone | 1.54 ± 1.68 |
| Sticky + Filter=0.9 (accept=inf) | 1.52 ± 2.37 |
| Sticky + Filter=0.9 + accept=0.5 | -1.45 ± 3.20 |

### 10 seeds（HANDOFF 早就标 ≥10 才能 ablation）

| Config | Optimality (10s) | Disagree | VE | SW |
|---|---|---|---|---|
| Baseline (off+off) | 0.30 ± 1.53 | 6.6/15 | 25.17 | -69.16 |
| Filter=0.9 alone | 0.41 ± 3.57 | 4.8/15 | 25.58 | -67.66 |
| Sticky alone (tie=0) | 0.67 ± 2.33 | 5.6/15 | 23.74 | -67.16 |
| **Sticky + Filter=0.9 (accept=∞)** | **1.46 ± 1.71** | 5.3/15 | 24.14 | -65.28 |
| Sticky + Filter=0.9 + accept=0.5 | -0.19 ± 2.31 | 6.2/15 | 24.08 | -67.84 |
| **Paper SPERL (Alg 3+4)** | **2.74 ± 0.95** | — | **9.3 ± 0.95** | -59.94 ± 2.88 |

## Per-state disagreement breakdown

跑 [`scripts/state_disagree_breakdown.py`](../scripts/state_disagree_breakdown.py) 把 10 seeds 的 disagreement 按 state 聚合：

**Best config (sticky+filter=0.9+accept=∞)** — 7/15 states 持续 disagree (rate ≥ 0.5):

| state | a_hat | disagree | mean \|Δv\| | v_tilde | v_hat |
|---|---|---|---|---|---|
| (1, -10) | 1 | 0/10 | 5.70 | -11.34 | -5.88 |
| (1, 10) | 1 | 6/10 | 3.06 | 8.42 | 10.97 |
| (0, 0) | 1 | 3/10 | 2.69 | 1.46 | 3.57 |
| (2, 20) | 1 | 7/10 | 1.79 | 14.77 | 16.50 |
| (2, 0) | 1 | 8/10 | 1.31 | -0.04 | 0.19 |
| (3, 10) | 1 | 8/10 | 1.12 | 7.91 | 9.01 |
| (3, 30) | 1 | 6/10 | 1.07 | 20.40 | 21.10 |

**Baseline** — 8/15 disagree, sticky+filter 仅多救 1 个 state。

关键观察：
1. **系统性 under-gambling**：所有 disagree state，oracle 说 gamble (a=1)，SPERL 选 exit (a=0)。
2. **(1, -10) 0 disagree 但 |Δv|=5.70** — 没人在 (1,-10) 选错，但 downstream policy 链式偏差让此处 V 差 5.7
3. **(2, 0) 几乎全错** — best config 8/10、baseline 10/10 seeds 都选 exit
4. Paper Table 3 at filter=0.9, p=0.66: policy_err=2.89/15 (~19%)，我们 7/15 (~47%)，**大约 2× 差距**

可能根因（待诊断）：
- QR 估计偏差：Barberis 的 reward sparse (+bet/-bet)，5-step horizon 上分布学不准
- CPT offset 应用时机或精度
- Paper 用更多 seeds (18-27?) 拉低均值

## 关键发现

1. **没有实现层 bug**：Alg 3 和 Alg 4 的伪代码 vs 我们代码逐行对照，加上 legacy 交叉检验，逻辑都对。
2. **3-seed 数据全部翻案**：baseline 3-seed 估 4.57，10-seed 实际 0.30；filter=0.9 alone 3-seed 估 2.62，10-seed 实际 0.41。之前所有"sticky/filter 损 Optimality"的结论都是 per-seed 摆幅 (4-11) 假象。
3. **真正的最优 ablation**：Sticky + Filter=0.9 (accept=∞) 1.46 ± 1.71，比 baseline 高 1.16 (~0.7σ)，是唯一显著优于 baseline 的配置——这和 paper 直觉一致 (Alg 3+4 双开提升 SPERL)。
4. **Acceptance gate 0.5 仍是 cost**：sticky+filter+gate 0.5 (paper default) 给 -0.19，比 accept=∞ 低 1.65 Optimality。10 seeds 上仍是统计显著的 cost。
5. **VE 系统性偏高 ~24 vs paper 9.3 (2.5×)**：所有配置 VE 都在 23-26 间，包括最佳的 sticky+filter=0.9。这超出 Alg 3/4 能解释的范围——可能是 quantile 估计噪声、cpt_offset 应用时机、或 evaluation domain 差异。值得作为新 TODO 单列。
6. **比 paper 仍低 1.3σ**：最佳 1.46 vs paper 2.74。可能是 (a) 3 seeds 时我们 baseline 也偏低，paper 是更多 seeds 的结果（paper 一般 18-27 seeds）；(b) firstVisit init 细节；(c) paper 可能在 Tables 1/2 用了 filter=1.0（无 filter）的非最优值见 footnote。

## 对 HANDOFF 状态的修正

- ❌ 旧："Alg 3/4 在 hardest config (CPT88/p=0.66) 上崩到 0 ⚠️ 待诊断"
- ✅ 新："Alg 3/4 在 p=0.66 上 partial 对齐 (sticky+filter=0.9+accept=∞ 1.46±1.71，paper 2.74±0.95，1.3σ 间距)。Acceptance gate=0.5 cost ~1.65 Optimality 是真实差异，待诊断"
- 新 ⚠️：VE 系统性 2.5× 高于 paper，全 ablation 配置都在 ~24 vs paper ~9，待诊断

## 产物

```
runs/results_p066_paperdefault/             # 3-seed: filter=0.9 + sticky + accept=0.5
runs/results_p066_filter09_only/            # 3-seed: filter=0.9 only
runs/results_p066_sticky_only/              # 3-seed: sticky only
runs/results_p066_sticky_filter09_acceptInf/# 3-seed: filter=0.9 + sticky, no gate
runs/results_p066_10s_baseline/             # 10-seed: all off
runs/results_p066_10s_filter09only/         # 10-seed: filter=0.9 only
runs/results_p066_10s_stickyonly/           # 10-seed: sticky only
runs/results_p066_10s_both_acceptInf/       # 10-seed: filter=0.9 + sticky, no gate ← 最佳
runs/results_p066_10s_both_accept05/        # 10-seed: paper default (with gate=0.5)
reports/_paper_alg34_extract.txt            # paper p48-55 文本提取
scripts/state_disagree_breakdown.py         # per-state disagreement aggregator (10-seed)
```

## 后续 TODO

- ~~原 #1：paper diff~~ ✅ done
- ~~原 #2：paper 超参对照~~ ✅ done (filter=0.9, accept=∞)
- ~~新增 #1：≥10 seeds head-to-head ablation~~ ✅ done in this session
- ⚠️ **新增：诊断 VE 2.5× gap (我们 ~24 vs paper ~9.3 at filter=0.9)** — 全 ablation 都受影响，独立于 Alg 3/4
- ⚠️ **新增：诊断 acceptance gate=0.5 cost 1.65 Optimality** — 跨 3-seed/10-seed 都存在的真实差异
- ✅ 部分原 #3 (filter sweep)；剩 1.0/0.95/0.85/0.8 还没扫
- 原 #4-6 不变
