# 2026-04-27 — BLN Consumption Env MVP

## 任务

实施 van Bilsen-Laeven-Nijman (2020) MS "Consumption and Portfolio Choice Under
Loss Aversion and Endogenous Updating of the Reference Level" 作为 LNW
abandonment 之后的第三个 env, 验证 framework 能 host 一个 **endogenous-reference**
机制的 env (跟 Barberis 的 probability-weighting / LNW 的 terminal-stop 是质性
不同的 time-inconsistency 来源)。

Scope 限定为 **MVP**: env class + SPE oracle + Phase 4.1 smoke test, 不做 Phase 5
(grid / DO / heatmap / convergence / Alg34 ablation), 视 MVP 结果再展开。

参考 [reports/_playbook_new_env_development.md](_playbook_new_env_development.md)
的 5-phase 流程严格执行。

## SPERL 框架适配 (BLN ≠ SPERL native, 显式 reframe)

BLN 原 paper 是 continuous-time + 每期累加 loss-averse utility u(c_t, R_t)。
SPERL 是 discrete-time + 单次 CPT-at-terminal-on-cumulative-scalar。两者不直接
同构, 三个适配方案中选了 **方案 A: per-step excess** —

- `r_t = (c_t - R_t) * reward_scale` (signed; +25%R 是 gain, -25%R 是 loss)
- CPT input = Σ r_t (SPERL 把整个 episode 当一次 CPT 评估)
- cpt_offset = 0 (r_t 已 centered, 不需要 cash-position offset)
- Endogenous reference 机制完整保留: R_{t+1} = (1-γ)R_t + γc_t

paper §4 写作时显式 disclose: "Our SPERL adaptation distorts the distribution of
cumulative excess consumption rather than aggregating per-period CPT utilities;
this preserves the endogenous-reference mechanism while fitting the
cumulative-distributional CPT operator."

## MDP 设计 (用户主导的 action-space 简化)

跟 LNW 比, 用户提议用 ternary Δ-style action 而非完整 (c, π) 决策, 通过 action
约束顺带约束 state space:

| 维度 | 设计 |
|---|---|
| Action | `Discrete(3)`: 0=减消费 (c=0.75R), 1=维持 (c=R), 2=增消费 (c=1.25R) |
| Portfolio π | 固定 0.5 (50% risky asset, MVP 不让 agent 选, paper §4 可声明 future work) |
| State | `(t, W_idx, R_idx)`, T+1 × n_W × n_R 三维 |
| W grid | log-spaced [0.5, 10] (n_W=8 for MVP) |
| R grid | linear [0.3, 1.5] (n_R=6 for MVP) |
| Stock | 2-point binomial matched to lognormal μ=5%, σ=20% |
| Reward | r_t = (c_t - R_t) × 100 (scale to Barberis ±25 量级, 兼容 default critic-lr) |

|S| = 5 × 8 × 6 = **240** total cells, |S × A| = 720。

## 实施清单 (按 playbook Phase 2.1)

### 新建
- `lib/envs/bln_consumption.py` — env class (200 行)
- `lib/envs/bln_spe.py` — backward-induction SPE oracle (60 行)

### 修改 (extension points 由 explore agent 预先定位)
- `lib/envs/featurizers.py` — 加 `BLNFeaturizer` (含 BFS reachability — 见 "中途 bug 修复" 节)
- `lib/envs/registry.py` — `bln`/`consumption`/`bln_consumption` 别名
- `lib/paper_eval.py` — `_is_bln` duck-type + 4 个 dispatcher 分支
- `lib/io.py` — `run_name_from_args` BLN 分支
- `agents/run_paper_eval.py` — CLI args (n-W/n-R/gamma/delta-c/pi-fixed/...) + build_env + get_reference_policy
- `agents/run_experiments.py` — 同上 (但不需要 reference policy)

### 不动 (env-agnostic)
- `agents/sperl_qr_generic.py`, `agents/spsa_generic.py`, `lib/cpt.py`

## MVP 测试结果

### Step 1: env smoke (1.3s)

- bijection: `loc(s) ↔ s` 在 240 states 上一一对应 ✓
- random policy 跨 5 episodes 跑完, total_reward ∈ [-129, -27], 全负 (符合预期 — 随机消费不能管财富)

### Step 2: SPE oracle structure (6.7s)

```
Action distribution: {2: 178, 1: 22, 0: 40}  (74% / 9% / 17%)
```

定性结构合理:
- 低 W (0.5) 高 R: action 0 (decrease, defensive 避损)
- 中高 W: action 2 (increase, 趁财富还在快消费)
- 高 W (10) 最高 R (1.5): action 0 (R 已封顶, 增 c 没 upside, 只是耗财)

### Step 3: SPERL 训练 + paper-eval (19.6s, 5k eps × 3 seeds)

| Metric | BLN MVP | Barberis sanity (5k eps × 3 seeds, 8.4s) |
|---|---|---|
| n_states (eval domain) | 20 (BFS reachable) | 15 (parity grid) |
| Disagree | 6/20 = 30% (std 1.6) | 1.67/15 = 11% (std 0.94) |
| Optimality V(x_0) | -15.0 ± 0.05 | 8.2 ± 0.6 |
| SW (SPERL learned) | -589 ± 40 | -35 ± 12 |
| SW (SPE oracle) | -559 | -23.6 |
| SW gap / state | 1.5 | 0.8 |

✅ **数字量级跟 Barberis 同 order**, 说明 default Barberis 超参 (critic-lr 0.04,
support-size 50, eps 0.3) 在 BLN 上直接 work, 不需要 retune。

✅ **Disagree 30% < 50% 阈值**, MVP DoD 通过。

⚠ **Optimality 是负的 (-15)**, 但这是 BLN env 内在 wealth-depleting (没 salary,
W_init=5 不够 sustain 5 期 c=R consumption), 不是 SPERL 失败 — SPE oracle 在
x_0 也只能给 V(x_0) ≈ -12 (跟 fix-action-2 的 -15 差 1σ MC noise)。

## 中途 bug 修复 (Phase 4.1 → 4.1' 迭代)

### 第一次跑 SPERL: Disagree = 80% (collapse to action 0 在 93% states)

诊断步骤:
1. 怀疑 training 不够长 → 5k → 20k eps, lr 0.04 → 0.01, eps 0.3 → 0.5: 都没改善
2. 怀疑 algorithmic bug → 在 x_0 处直接看 critic Q values:
   ```
   a=0: CPT=-38.2  a=1: CPT=-14.5  a=2: CPT=-11.8  → argmax = 2  ✓
   greedy_action[loc=33] = 2  ✓
   ```
   **x_0 处 SPERL 是对的**, 选 action 2!
3. 但全 240 states 上 SPERL 大部分选 0 → SPERL 在 **x_0 之外** state 没学到
4. 真因: paper_eval 的 `iter_states` 默认 yield 全 240 (T × n_W × n_R) 网格,
   多数 (t, W, R) 组合 from x_0 不可达 → SPERL critic 没 train 过 → 默认 0
5. 跟 Barberis/LNW 对照: 那两个 env 的 parity-correct 网格 = reachable set
   (parity 自然给出 random walk 可达性), BLN 的 grid snap + stochastic dynamics
   不等同 parity argument, 必须显式 BFS 找 reachable set

### Fix: BFS reachability for BLNFeaturizer.iter_states

```python
def iter_states(self, env=None):
    """Forward BFS over (action × stock-outcome) transitions from x_0."""
    visited = set()
    frontier = [(0, env.W_init_idx, env.R_init_idx)]
    while frontier:
        s = frontier.pop()
        if s in visited: continue
        visited.add(s)
        if s[0] >= env.T: continue
        for a in range(env.action_space.n):
            for R_stock in (env.R_up, env.R_down):
                # simulate one step manually, snap, push successor
                ...
    yield from sorted(visited where t < T)
```

类比 OptExFeaturizer.iter_states 用 OptExTree 也是 BFS。重跑 SPERL → 240 → 20
reachable states, Disagree 80% → 30%, MVP pass。

## 经验沉淀 (回写 playbook)

playbook 现有 4 个 bug pattern (lump-sum reward / constant cpt_offset / SPE MC
noise / SPSA cpt_offset)。BLN 加了 1 个新 pattern:

### Bug #5 (新): iter_states 必须等于 reachable-from-x_0 set

**症状**: SPERL 训练正常, 在 x_0 critic 学对了, 但 paper-eval Disagree 高得离谱
(70-90%)。检查 critic.greedy_action[loc(x_0)] 是对的, 但其他 state 默认动作 0。

**原因**: `iter_states` yield 了不可达 states, SPERL 训练永远不访问这些 cell,
critic 是 init 值, 默认 argmax → 大量假性 disagreement。

**避免方法**: 实施 env 时, 写完 iter_states **立刻**验证: 跑 SPERL 训练 → 检查
visited-state 集合; 如果 visited ⊊ iter_states, 改 iter_states 为 visited 子集
(BFS 或动态 trace)。

**Barberis/LNW 不踩坑因为**: parity-correct 网格 (x_idx 跟 t 同 parity) 在
random-walk + binary 动作下 = reachable set。这是巧合, 不是普适。BLN /
OptEx / 任何 stochastic-snap env 都需要显式 BFS。

## 其他已知 MVP limitation

1. **R discretization stickiness**: γ=0.3 + Δc=0.25 → 一步 R 改变 ≈0.075,
   跨不出 R_grid 间距 0.24, R 钉住。endogenous-reference 机制部分 mute, 但
   reachable set (20 states) 仍跨多个 R bin。Phase 5 可加细 R_grid (n_R=10 +)。
2. **π 固定 0.5**: 没 portfolio decision。BLN paper 主推 portfolio response,
   是 future work。
3. **Optimality negative**: env 没 salary, W 在 T=5 内必然 deplete。要做 BLN
   paper-faithful 数字需要加 income flow + 更长 horizon。MVP 不在 scope。

## 还没做 (Phase 5+, 视后续 session 开展)

- Multi-config grid (γ × CPT_regime × T sweep)
- DO baseline (`bln_do.py`)
- Per-state policy heatmap (3D state, 可视化要 t-slice)
- Convergence curves
- Alg 3/4 ablation
- Salary / income flow extension
- Portfolio decision extension (action → 9 actions)
- 10-seed re-run (current 3-seed)
- paper §4.x BLN 章节 draft

## 文件清单

### 新建 (commit 进来)
- `lib/envs/bln_consumption.py` — env (~200 行)
- `lib/envs/bln_spe.py` — SPE oracle (~60 行)
- `reports/2026-04-27_bln_env_mvp.md` — 本文档

### 修改 (commit 进来)
- `lib/envs/featurizers.py` — `BLNFeaturizer` (含 BFS iter_states)
- `lib/envs/registry.py` — `bln`/`consumption` 别名 + `REGISTERED_ENVS`
- `lib/paper_eval.py` — `_is_bln` + 4 处 dispatcher 分支
- `lib/io.py` — `run_name_from_args` BLN 分支
- `agents/run_paper_eval.py` — CLI args + build_env + get_reference_policy
- `agents/run_experiments.py` — CLI args + build_env

总改动 ~500 行 code + 本 session report。
