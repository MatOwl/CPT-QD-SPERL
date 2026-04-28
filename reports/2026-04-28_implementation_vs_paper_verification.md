# Implementation vs Paper line-by-line verification

**日期**: 2026-04-28
**目的**: 系统性核对 refactor 实现的每一步公式 + 参数 vs paper 文字。覆盖 Algorithm 2/3/4/5 + §2.1.4 (CPT) + §C.2.1-5 + Definition 5.

---

## 总览

| Module | 检验结果 | 严重度 |
|---|---|---|
| Algorithm 2 主循环 | ✅ 拓扑等价 | — |
| Algorithm 2 line 13 a* selection | ⚠️ refactor 用 `policy.predict` (cached), paper 写 `argmax Q` (compute on fly) — sticky 语义下等价 | 低 |
| Algorithm 3 sticky tie-break | ✅ | — |
| Algorithm 4 line 4-15 (filter + snap) | ✅ (含 fp tolerance 1e-9/1e-6) | — |
| **Algorithm 4 line 20 acceptance gate** | **❌ paper text 绝对; legacy/refactor 相对** | **中** |
| Algorithm 5 (True SPE) line-by-line | ✅ topology | — |
| **Algorithm 5 M (rollouts)** | **❌ paper M=2000; refactor default 500; 我们 sweep 用 300** | **中** |
| §C.2.4 firstVisit init | ✅ | — |
| §C.2.5 hyperparams (M, K, ϱ, ξ, T̄, filterTresh, treshRatio) | ✅ (v3 后) | — |
| CPT 公式 (§2.1.4 eq 13-15) | ✅ (B=0 hardcoded) | — |
| Definition 5 (Social Welfare) | ✅ | — |
| Policy / Value Error 定义 | ✅ | — |

## 详细对照

### M1. CPT 公式 (paper §2.1.4 eq 13-15)

**Paper**:
- C[ϑ] = ∫_{-∞}^B u(z) dw(F(z)) + ∫_B^∞ u(z) d(-w(1-F(z)))   (eq 13)
- u(z; α, λ, B) = (z-B)^α (z≥B); -λ(B-z)^α (z<B)   (eq 14)
- w(p; δ) = p^δ / (p^δ + (1-p)^δ)^(1/δ)   (eq 15)

**Refactor** (`lib/cpt.py`):
```python
def utility(x, pos, alpha, lmbd):
    if pos: return x**alpha if x >= 0 else 0.0
    return -lmbd * (-x)**alpha if x <= 0 else 0.0

def prob_weight(F, pos, rho1, rho2):
    rho = rho1 if pos else rho2
    return F**rho / ((F**rho + (1-F)**rho)**(1/rho))

def compute_CPT(x_list, ...):
    for x_i (sorted ascending):
        if x_i >= 0:  # gain side
            dF = w(1 - F(x_{i-1})) - w(1 - F(x_i))
            cpt_pos += u(x_i, pos=True) * dF
        else:  # loss side
            dF = w(F(x_i)) - w(F(x_{i-1}))
            cpt_neg += u(x_i, pos=False) * dF
```

**Match**:
- ✓ Functional form of u(x): paper (z-B)^α with B=0 = refactor x^α
- ✓ Functional form of w(p): exact match
- ✓ Riemann-Stieltjes 离散化为 sample CPT 与 paper 等价
- ⚠️ refactor 允许 ρ_gain ≠ ρ_loss; paper eq 15 只有单一 δ。CPT88/CPT95 实验 δ 两端相等 → 等价
- ⚠️ B=0 hardcoded; paper 算法允许 B≠0 (Algorithm 2 line 11 用了 1[φ<B])。Casino 实验 B=0 → 等价

### M2. Algorithm 2 line-by-line (paper p.48)

| Paper | Operation | Refactor | Match |
|---|---|---|---|
| L3 init | φ⁰=0; π⁰ = argmax C[1/K Σδ_φ⁰] | qtile_theta = zeros; greedy_action = -1 (uniform) | ⚠️ subtle: paper 用 "argmax of zero quantile distributions" 实际上是任意 (全 0 → 全 0 CPT → tie)，refactor 用 uniform，等价 |
| L4 | for i=0..M-1 | for i_ep in 1..n_train_eps+1 | ✓ |
| L5 | sample under π_i ξ-greedy | _rollout via policy.predict eps-greedy | ✓ |
| L6 | for t = T̄-1 to 0 | player_set bwd | ✓ |
| L7 | x=X_t, a=A_t, r=R_{t+1}, x'=X_{t+1} | tr = (state, action, reward, next_state, done) | ✓ |
| L8-9 | if x'=x_void: T̂φ_j=r ∀j | if done: targets = [reward]*I | ✓ |
| L11 q_j | 1[φ<B](w(τ_j)-w(τ_{j-1})) + 1[φ≥B](w(1-τ_{j-1})-w(1-τ_j)) | inside compute_CPT loop with same formula | ✓ |
| L12 Q | Q(x',a') = Σ q_j u(φ_j) ∀a' | cpt_values_all_actions(state) | ✓ |
| **L13 a*** | **a* = argmax_a' Q(x',a') with consistent tie-break** | **opp_action = argmax(policy.predict(next_state, det=True))** | ⚠️ 见下注 |
| L14 TD target | T̂φ_j = r + γφ^{i+1}_j(x', a*) | targets = [r + γ*q for q in next_q[opp_action]] | ✓ |
| L17 SA | φ_k ← φ_k + ϱ ∇_φ Σ_j ρ_τ̂_k(ε_kj) | grad_slice = mids*M - cmp.sum; current += lr*grad_slice | ✓ (Phase B bit-exact w/ legacy) |

⚠️ **L13 注解**: refactor 读 `policy.greedy_action[next_state]`, 这是上一次 sticky update 在 next_state 处 commit 的 action。Paper L13 写 `argmax Q on the fly`，但同时 §C.2.2 说 "consistent tie-break: only update when strictly better"。两者在 sticky 语义下等价（因为上一次 sticky update 已经 commit 了当时 critic 的 argmax，且 sticky 不会因当前 critic 略优而 flip）。但严格按 paper 文字 L13 应该 reread Q ∀a' — refactor 没这么做，而是 trust 缓存 policy。**实际效果可能略有不同当 critic 在该 batch 内更新 next_state 多次**。

### M3. Algorithm 3 (paper p.49)

| Paper | Operation | Refactor | Match |
|---|---|---|---|
| L3 | if max_a' Q > Q(π_i(x')) | `if cpt_values[new_a] > cpt_values[prev_a]` | ✓ |
| L4 | π_{i+1} = argmax (random tie) | new_a = rng.choice(candidates) where candidates = argmax | ✓ |
| L6 | else π_{i+1} = π_i | else keep prev_a | ✓ |

✅ Match.

### M4. Algorithm 4 (paper p.51)

| Paper | Operation | Refactor | Match |
|---|---|---|---|
| L4 | d_j = φ_j - φ_{j-1} ∀j=2..K | gaps = np.diff(q) | ✓ |
| L5 | d* = sampleQuantiles(d_2:K, filterTresh) | np.quantile(gaps, p_filter, method="higher") | ✓ |
| L6 | bool_j = 1[0 ≤ d_j ≤ d*]; bool_1 = bool_2 | (gaps >= -1e-9) & (gaps <= d_star + 1e-6); valid[0]=valid_gap[0] | ⚠️ refactor 加 fp tolerance 1e-9/1e-6 (paper 没说)，行为差异微小 |
| L11-15 snap | dist_l < dist_u → φ_jl else φ_ju | `q[lb] if dist_lb < dist_ub else q[ub]` | ✓ |
| **L20 gate** | **\|Q_filt - Q\| > treshRatio (绝对)** | **\|(v_filt-v_ori)/v_ori\| > τ (相对)** | **❌ 文字-代码不一致** |
| L21 reset | reset φ_filtered ← φ | return v_ori (skip filter) | ✓ semantically |

❌ **L20 是 paper 文字与 legacy/refactor 长期不一致点**。Paper 写绝对差 \|Q_f - Q\|，代码用相对差 \|ΔQ\|/\|Q\|。treshRatio=0.5 含义：
- **绝对**: filter 把 CPT 改动 > 0.5 单位就 reject (对 Q 量级 1-100 来说基本不会触发，filter 几乎总被接受)
- **相对**: filter 把 CPT 改动 > 50% 就 reject (常常触发)

两者行为差距大。若按 paper 文字，应改为绝对 gate。需测试是否会改善 Tables 1/2/3/4 复现。

### M5. Algorithm 5 (True SPE oracle, paper p.52)

| Paper | Operation | Refactor `lib/envs/barberis_spe.py` | Match |
|---|---|---|---|
| L1 | M = 2000 | n_eval_eps default 500; 我们 sweep 用 300 | **❌ rollout 数量** |
| L3 | π̂(x) ← ∅ | policy = {} | ✓ |
| L4 | for t = T-1 to 0 | for t in range(T, -1, -1) | ⚠️ refactor 多迭代 t=T (无害但浪费) |
| L5 | for x_t ∈ X_t | for k in range(-t, t+1): z = k*bet | ✓ |
| L6 | for a_t ∈ {exit, gamble} | for a in range(env.action_space.n) | ✓ |
| L8-10 | rollouts under π̂, accumulate Z | n_eval_eps rollouts, follow(nxt) for τ>t | ✓ |
| L12 | Q(x_t, a_t) = C[Z] | cpt_params.compute(rewards) | ✓ |
| L14 | π̂(x_t) = argmax Q | argmax(q_vals) | ✓ |

❌ **rollout 数量 mismatch**: paper M=2000, refactor 我们 sweep 用 300 → SPE oracle 噪声 √(2000/300) = 2.6× 大。CPT95 低 p_win SW 偏负可能与此相关（V_SPE 本身估计有噪声）。

**建议**: 改 `--spe-rollouts 2000` 重跑 verification cell 看 SW 是否改善。

### M6. §C.2.4 firstVisit init

**Paper**: φ^0_{1:K}(x,a) ← 1/K Σ_j T̂φ_j (mean of TD targets) on first visit

**Refactor** `QRCritic.update`:
```python
if not self._visited[loc, a]:
    self.qtile_theta[loc, a, :] = targets.mean()
    self._visited[loc, a] = True
```

✅ Match: refactor sets ALL K quantiles = mean of targets on first visit.

⚠️ refactor 没有 first_visit 的开关 — 总是启用。Paper §C.2.5 说默认启用，所以 OK。但若想测试 no-firstVisit 对比，需要加开关。

### M7. §C.2.5 Hyperparameters

| Hyperparam | Paper | Refactor (v3) | Match |
|---|---|---|---|
| M | 15000 | 15000 | ✓ |
| T̄ = T+1 | 6 | 6 (T=5) | ✓ |
| ξ | 0.3 | 0.3 | ✓ |
| K | 50 | 50 | ✓ |
| ϱ | 0.04 | 0.04 | ✓ |
| filterTresh | per-cell §C.4 *-mark | per-cell ✓ | ✓ |
| treshRatio | 0.5 / 0 | 0.5 / 0 | ✓ |

⚠️ n_batch 在 §C.2.5 list 不在；refactor 用 5。Legacy 默认 50 但实际跑 1。Paper 没说，无法对照。

### M8. Definition 5 (Social Welfare, paper p.18)

**Paper**: SW(π) = Σ_x U_x(π) where U_x(π) = V^π(x)

**Refactor** `compute_paper_metrics`:
```python
v_tilde_total = sum over x in featurizer.iter_states() of rollout_cpt_from_state(...)
return {"social_welfare": v_tilde_total, ...}
```

✅ Match. 求和域是 `featurizer.iter_states()` (env-specific reachable set)。Barberis: **15 decision states** (T=5 个 decision-time slots × parity-correct wealth grid: 1+2+3+4+5)。Terminal time t=T=5 不算 decision，被 `iter_states` 排除（与 paper Algorithm 5 line 4 `for t = T-1 to 0` 的语义一致）。

**修正历史**: 本节原写 "21 states (6 time × variable wealth, BFS reachable from x_0)"，是错的。21 是 SPE oracle 跨 t=0..T 的全 parity-correct 状态数（含 terminal），但 SW 求和域只看 decision 状态 (t<T) → 15。

### M9. Optimality + Policy Error + Value Error 定义

**Paper §4.1-4.2 (推断)**:
- Optimality = V^π(x_0)
- Policy Error = Σ_x \|μ_seeds[π_tilde(x)] - μ_seeds[π_hat(x)]\| → per-seed = Σ_x 1{a_tilde≠a_hat}
- Value Error = Σ_x \|V_tilde(x) - V_hat(x)\|

**Refactor**: 全部 ✓ (per-seed 公式与 paper 一致；跨 seed aggregate μ ± σ 也匹配)

---

## P1 (§C.4 sweep) 进行中

启动时间: 22:46 (本地)。预计 60 cells × 90 sec ≈ 90 min wall clock。

完成后将比对：
- Refactor Tables 3 (Policy Error μ±σ) vs paper 60 数字
- Refactor Tables 4 (Value Error μ±σ) vs paper 60 数字
- *-mark Pareto-optimal identification 是否复现 paper

---

## 关键 mismatch 总结 + 影响评估

| # | Mismatch | 影响 cells | 修复优先级 |
|---|---|---|---|
| **M2-L13** | a* selection 用 cached policy vs paper 写 on-fly Q argmax | sticky 语义下基本等价 | 低 |
| **M4-L20** | filter gate 绝对 vs 相对 | filter≠1.0 的 cells 都受影响 (8/10 Tables 1/2 cells) | **高** |
| **M5-L1** | SPE oracle M=2000 vs 我们 300 | 全 10 cells 的 SPE oracle 都用更少 rollout，CPT95 低 p_win SW 可能受影响最大 | **高** |
| L6 fp tol | 1e-9 / 1e-6 in valid check | 边界 case，影响小 | 低 |

## 建议的下一步

### 立即 (sweep 跑完前)
完成 P1 (§C.4 sweep) 解析 + 对比

### sweep 完成后
1. **修 M5**: 把 SPE oracle 改成 spe_rollouts=2000 重跑 verification cell；看 CPT95 SW 是否改善
2. **修 M4**: 加一个 `filter_gate_mode` flag 支持 "absolute" / "relative"，按 paper 重跑 Tables 1/2 看 CPT88/0.66 是否回到 paper 2.74
3. P1 完成后视 Tables 3/4 复现度决定是否需要更多调整

### 工时
- P1 sweep + parse: 60 min (sweep 自动) + 30 min (parse + report)
- M5 fix + verify: 30 min (单 cell 重跑)
- M4 fix + sweep: 1 hr (改代码 + 重跑 Tables 1/2)
- 总: 约 3 hr 完成本轮系统化检验

---

## 文件位置

- Paper PDF 抽取脚本: 一次性 inline (pypdf)
- v3 sweep script: `scripts/sweep_paper_tables_1_2.sh`
- §C.4 sweep script: `scripts/sweep_paper_tables_3_4.sh`
- Implementation: `agents/sperl_qr_generic.py` + `lib/cpt.py` + `lib/envs/barberis_spe.py` + `lib/paper_eval.py`

---

## Addendum (later same day): 6 个本报告原版漏掉/弄错的问题（已修）

本报告 v1 提交后，进一步对照 paper §C.2.3 "Crossing quantiles" 与 legacy `rerun_GreedySPERL_QR__main.py` 又找到 4 个本报告没标记的实质 bug + 2 个 doc/note 错误。修正概览：

| # | Bug | 影响 | 修法 |
|---|---|---|---|
| **F1** | `QRCritic._cpt_from_quantiles` 在 filter 前 `np.sort` → paper §C.2.3 "crossing quantile detection" 死代码 | 中-高 (filter 不再修 crossing) | 删除 pre-sort，把 raw quantile 传给 `filter_quantiles`；同时把 valid_gap 的负-tol 从 `1e-9` 调至 `1e-6` 对齐 legacy |
| **F2** | `run_paper_eval.py` 7 个默认值都偏离 paper §C.2.5 (train-eps=2000/K=20/lr=.1/treshRatio=inf/sticky=False/spe-rollouts=300/gate=relative) | 高 (用户 footgun) | 全部改成 paper-aligned；`--sticky-policy` 改用 `BooleanOptionalAction`（默认 True，可 `--no-sticky-policy`）；`--filter-gate-mode` 默认 `absolute` (paper 文字) |
| **F3** | `lib/envs/barberis_spe.py` 用 `range(-t, t+1)` 含 parity-violating z 值 (legacy 用 `np.linspace(-bet*t, bet*t, t+1)` 只 parity-correct) | 中 (1.7× 慢且与 legacy 不严格一致) | 改为 `range(-t, t+1, 2)` |
| **F4** | `lib/paper_eval.py:rollout_cpt_from_state` v_tilde 和 v_hat 用独立 RNG → 即使 policy 一致 \|v_tilde - v_hat\| 也非零，膨胀 VE 总和 | 中 (可能解释 §C.4 sweep VE 系统偏高的一部分) | 加 `seed` 参数（save/restore np.random state），`compute_paper_metrics` per-state 用 `hash(x)` 作 CRN seed |
| **F5** (doc) | 本报告 M8 写 "Barberis: 21 states" — 实际 decision 状态是 15 | 低 (报告自检) | 已修正本节 + 加修正历史 |
| **F6** (doc) | `HANDOFF.md` row 14/21 数字未跟随 v3 hyperparam 修正（仍是 1.46±1.71，应是 −0.19±2.31） | 低 | 已修 (见 HANDOFF) |

### F1 修复细节
- 旧：`q_sorted = np.sort(q_arr); q_filt = filter_quantiles(q_sorted, ...)` — sorted 输入下 `gaps = np.diff(q_sorted) >= 0`，所以 `(gaps >= -1e-9)` 检查永远 True，filter 不能 detect crossing。
- 新：`v_ori = compute(q_arr, sort=True)`（CPT 积分对 input order 不敏感），`q_filt = filter_quantiles(q_arr, ...)` — filter 直接看到 raw（可能 crossing）quantile，按 paper Algorithm 4 line 6 的 `1{0 ≤ d_j ≤ d*}` 把 crossing 处的 quantile snap 到邻居。
- 验证：用 `q=[1, 5, 3, 4, 10]` (q[1]>q[2] crossing) 测试，新 filter 输出 `[1, 5, 4, 4, 10]`（snap q[2] 到右邻居 4），匹配 legacy 行为 ✓

### F4 修复细节
- 加 `seed: int | None = None` 到 `rollout_cpt_from_state`：进入函数前 `np.random.get_state()` 保存，`np.random.seed(seed)` 重设，`finally` 块里 `np.random.set_state` 恢复。
- `compute_paper_metrics` per-state seed = `abs(hash(tuple(int(v) for v in x))) & 0x7FFFFFFF`，v_tilde 和 v_hat 同 seed → 相同 stochastic stream。
- 验证：同 policy + 同 seed 给 bit-exact 相同 CPT (|v1-v2|=0)；不同 seed 给不同 CPT。✓

### 影响 sweep 数据
- `runs/results_paper_tables_1_2_v2/` 和 `runs/results_paper_tables_3_4/` 的 aggregate.json 是修复前生成的（filter pre-sort + relative gate + 独立 RNG VE 噪声 + spe-rollouts=300）。要拿到 paper-aligned 数据需要重跑两个 sweep 脚本。
- F1+F4 是新的 candidate explanations 解释 §C.4 sweep PE/VE 比 paper 系统性偏高 2-7×。需要重跑 sweep 验证 gap 是否收窄。

### 修复后建议优先级
1. 重跑 `scripts/sweep_paper_tables_3_4.sh`（已含新 spe-rollouts=2000）→ 看 PE/VE 是否回到 paper 量级
2. 重跑 `scripts/sweep_paper_tables_1_2.sh` → 看 CPT88/0.66 Optimality 是否回到 paper 2.74±0.95
3. 如果 #1 和 #2 还有 gap，回到 2026-04-28 v1 的 candidate（n_batch=1, M=15000 真做 15000 个独立 batch 而不是 3000 个 batch×5）
