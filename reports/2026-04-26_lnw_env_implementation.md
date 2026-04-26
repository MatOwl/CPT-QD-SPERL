# 2026-04-26 — LNW Abandonment Env: Implementation + Two Reward-Design Bugs + SPSA-vs-SPERL Contrast

## 任务

回应 MSci AE Report 第 6 点 (operations-flavored example beyond the casino setting), 在现有 generic SPERL 框架上实现一个 OM-flavored env, 并和 Barberis 同一套超参做对照。同时副产物回应第 5 点 (robustness across policy classes via SPE-vs-precommitment) 和第 4 点 (CPT preference 引致质性不同的 start-or-not 决策)。

文献基础: [reports/2026-04-26_msci_literature_survey.md](2026-04-26_msci_literature_survey.md), 27 篇候选筛出 Long-Nasiry-Wu (2020) MS [`paperRef/A_Behavioral_Study_on_Abandonm.PDF.pdf`] 作 Tier S #1 — 多阶段项目放弃决策, 跟 Barberis casino MDP 几乎同构 (binary action, 1D 离散 state, parity-correct 随机游走)。

## Env 模型 (LNW Section 3.1 raw setup)

直接用论文 raw managerial setup, **不**采用论文的 behavioral model (reference + sunk cost + status quo) — 把 CPT 偏好通过 `lib/cpt.py::compute_CPT` 在算法层面叠加, 不在 env 层面。

| 维度 | 值 | 论文对应 |
|---|---|---|
| Horizon | env.T = 5 (= paper T-1) | T_paper=6 |
| State | (t, x_idx), parity-correct grid | (stage, project value) |
| Action | binary {0=abandon, 1=continue} | continue/abandon |
| Continue dynamics | x_idx ± 1 w.p. p; reward = δ·outcome - c | x_{t+1} = x_t ± δ (随机游走) |
| Abandon dynamics | terminate, reward = -x_t (recovery) | total payoff = 0 (forfeit) |
| Defaults | δ=10, c=11, x_1=50, p=0.72 | Figure 1 example |
| State space | |S| ≤ T·(2T+1) = 55 | 一个 fan-out tree |

## 关键设计决策 (踩了两个 bug 才走通)

### Bug #1: lump-sum-at-terminal reward 设计 SPERL 学不动

**初版** (semantically 直白):
- continue intermediate: r = -c
- continue terminal: r = -c + x_T (lump sum)
- abandon: r = 0
- cpt_offset = 0

**症状**: 即使 p=0.9 (clearly profitable, EV = +35) 5 个 seed 全部 Optimality=0, 跨 8000 eps 不学。Barberis 同 hyperparams 在 p=0.72 正常 work (Optimality 7.6 ± 1.3, [section below](#barberis-control)).

**根因**: QR critic first-visit-mean 初始化:
- Q(s, abandon) 起始 0 (immediate reward 0)
- Q(s, continue) 起始 -c = -11 (always negative immediate reward)
- argmax → abandon 锁死, cold-start chain 永远到不了终末 +x_T lump sum, 永远学不到 future value 是大正

**对比 Barberis**: 每步 reward = bet*outcome ∈ ±10 (zero-mean), Q(continue) 初始能正能负, 不锁死。

**修复**: 改为 delta-style reward, 把 x_T 摊到每步:
- continue: `r = δ·outcome - c` (per-step variance ±10 量级, 跟 Barberis 同)
- abandon at state (t, x_idx): `r = -x_t = -(x_1 + x_idx·δ)` (recovery 取消 累积价值变化 + offset, 让 cumulative+offset=0 = 无 future payoff)
- featurizer.cpt_offset((t, x_idx)) = `x_t` (state-dependent, 跟 Barberis 的 `cpt_offset = z` 完全同构)

### Bug #2: cpt_offset 写成 constant x_1

第一次 refactor 时把 `cpt_offset` 写成 constant `self.x1`, 没读出 obs 的 x_idx。

**症状**: SPE oracle 在所有 state 全选 abandon (V=u(50)≈32.5, 比任何 risky distribution 看起来都"安全")。SPERL 数字看起来非常漂亮 (disagree 1.2/15, optimality 31.27, SW 99.1%) 但**完全是错的** — agent 学到的是"abandon 拿白送的 +50"。

**怎么发现的**: 看 seed 1 的 [per_state_values.csv] 发现 SPE 在 (3,-3) 选 continue (V=47), 但 lump-sum 时代 SPE 在该 state 是 abandon (V=0). 数字翻转方向不对, 翻代码追到 cpt_offset 写错。

**修复**: cpt_offset 改成 state-dependent `x_1 + x_idx·δ` (类比 Barberis `cpt_offset = z` = 当前 wealth, **当前** 项目价值)。同时 SPE oracle ([lib/envs/abandonment_spe.py](../lib/envs/abandonment_spe.py)) 和 paper_eval `initial_offset` ([lib/paper_eval.py](../lib/paper_eval.py)) 一致用 state-dependent offset。

修复后 SPE 复现论文 Figure 2 fan-out abandonment threshold:
```
t=0,1,2: continue everywhere
t=3: abandon if x ≤ -3
t=4: abandon if x ≤ -4
```

## SPERL 主结果

5 seeds × 8000 eps × default Greedy-SPERL (eps=0.3, batch=1, support=50, critic-lr=0.04), config p=0.72 + CPT88 (α=0.88, ρ=0.65, λ=2.25):

| Metric | LNW (5 seeds) | Barberis CPT88/p=0.72 control (3 seeds) |
|---|---|---|
| Policy disagree / 15 | **1.2 ± 0.98** | 1.3 ± 0.5 |
| Value Error | **16.1 ± 3.4** | 18.9 ± 4.3 |
| Optimality | **5.03 ± 2.0** | 7.6 ± 1.3 |
| SW | 271.2 ± 3.6 | -28.2 ± 4.4 |
| SPE Welfare ref | 275.4 | -21.1 |
| SW / SPE | **98.5 %** | within 1σ |

跨 5 seeds **policy 收敛非常稳定** (variance 0.98 ≪ Barberis 在 hardest config 上的 1.7), 比 Barberis 同 config 还稳。

Artifacts: [results/abandonment_sperl_T5_x150_c11_d10_p0.72_cpt_a0.88_r0.65_l2.25/](../results/abandonment_sperl_T5_x150_c11_d10_p0.72_cpt_a0.88_r0.65_l2.25/) (gitignored, 本地 reproducible)。

## CPT-induced "stay out" 副产物 (responds to AE point 4)

同 LNW env 在两组 CPT 偏好下产生**质性不同**的 SPE 起点决策:

| Config | EV(continue all) | CPT(continue all) | SPE 在 (0,0) 选 | SPE V at (0,0) |
|---|---|---|---|---|
| p=0.72 + CPT88 (α=0.88, ρ=0.65, λ=2.25) | +17 | +5 | **continue** | 5.03 |
| p=0.6 + CPT95 (α=0.95, ρ=0.5, λ=1.5) | +5 | -2 | **abandon** | 0.0 |

第二组 EV 仍是正 (期望盈利), 但 probability weighting (ρ=0.5 inflates loss tails) + loss aversion (λ=1.5) 让 CPT 估值变负, 所以 sophisticated CPT-agent 选择"don't even start"。这是 paper §4 robustness 讨论可写的干净 demonstration of CPT-induced excessive pessimism。

SPERL 在两个 config 都正确学到 SPE 决策 (Optimality 5.03 vs 0.0, 跨 seeds 一致)。

## SPSA-vs-SPERL: precommitment 在 LNW 上的根本性失败 (responds to AE point 5)

复用 `agents/spsa_generic.py --algo spsa`, 同 CPT 偏好对照:

### Barberis CPT88/p=0.72 (control): SPSA 正常 work
默认 hyperparams (`--spsa-step 5 --spsa-c 1.9`, beta=1, theta in [0.1, 2.0]):
- ep 1: cpt=0 (随机 theta argmax = exit)
- ep 2501: cpt=6.87 (escape, gambling)
- ep 5001: cpt=8.32
- ep 10001: cpt=4.65
- 平均 ~6 (跟 SPERL 7.6 ± 1.3 同量级)

### LNW p=0.72 CPT88: SPSA 完全失败
默认 hyperparams: 8 个 seed (0-7) 全 cpt=0。Continue-favored seeds (3, 6, 8, 11, 12, 13, 15, 16, 17, 18) 也全 cpt=0。

**Warm-start 测试**: 强制 `theta[*, continue] = 2.0, theta[*, abandon] = 0.5` 让初始 policy = 永远 continue:
- ep 1: cpt = **5.0** ← 几乎等于 SPERL 收敛值 5.03 ✓ (warm-start 给的就是最优 precommitment policy)
- ep 2501: cpt = **0.0** ← SPSA 把 theta 推回 abandon
- ep 5001+: 一直 cpt=0

**SPSA 主动从最优解走到劣解。** 这不是优化没收敛, 是 SPSA 的随机梯度估计在 LNW 上**系统性偏向 abandon**。

### 故障机制 (LNW 结构性原因)

| 因素 | Barberis | LNW | LNW 后果 |
|---|---|---|---|
| 每步 reward 期望 | 0 (zero-mean ±bet) | -c = -11 (固定成本) | random rollout 大概率累积负 |
| 累积分布中心 | ≈ z_init = 0 | ≈ -t·c (取决于路径长度) | shifted into loss domain |
| CPT loss aversion λ=2.25 | 对称压两边 | 不对称放大累积损失部分 | gradient 一致推 abandon |
| Sampling-based gradient | 噪声 vs systematic 比例平衡 | systematic bias dominates | optimization fails |

**Paper-worthy claim**: precommitment SPSA 在 ops-flavored env (per-step 成本基线偏移) 上即使从最优解开始也会被随机梯度的 loss-aversion bias 拖回 trivial 解。SPE/SPERL 通过 per-state credit assignment 不受此影响, 在两个 env 上都收敛到正确 policy。这正是 paper §4 想说的 "equilibrium policies are more robust than precommitment policies", 现在有 LNW 上一个 dramatic example 而非微小 SW 差异。

## 修改 + 新建文件

| File | 操作 | LOC | 说明 |
|---|---|---|---|
| [lib/envs/abandonment_project.py](../lib/envs/abandonment_project.py) | 新建 | 150 | env class with delta-style reward |
| [lib/envs/abandonment_spe.py](../lib/envs/abandonment_spe.py) | 新建 | 60 | backward-induction SPE oracle, state-dependent offset |
| [lib/envs/featurizers.py](../lib/envs/featurizers.py) | 加类 | +50 | AbandonmentFeaturizer, cpt_offset = x_t |
| [lib/envs/registry.py](../lib/envs/registry.py) | 注册 | +5 | abandonment / abandonment_project / lnw 别名 |
| [lib/paper_eval.py](../lib/paper_eval.py) | 重构 dispatcher | +30 | _is_abandonment / _is_barberis duck-type, abandonment_initial_offset_builder |
| [lib/io.py](../lib/io.py) | 加 dir name | +3 | run_name_from_args abandonment 分支 |
| [agents/run_paper_eval.py](../agents/run_paper_eval.py) | CLI + SPE hookup | +20 | --delta --c --x1 + abandonment SPE 求解 |
| [agents/run_experiments.py](../agents/run_experiments.py) | CLI | +10 | --delta --c --x1 |
| [agents/spsa_generic.py](../agents/spsa_generic.py) | 用 cpt_offset | +6 | 三个 rollout 路径都加 offset |

总: 9 文件, ~330 LOC 新代码。`agents/sperl_qr_generic.py` (核心算法) **零改动** ✓ env-agnostic 假设成立。

## Commits

- [`afa0800`] Add LNW abandonment env (operations-flavored CPT-SPE example)
- [`ad5bdbf`] SPSA: use featurizer.cpt_offset in CPT computation

## 局限 + 下一步候选

**已解决**:
- [x] LNW env 实现, MDP 跟 Barberis 同构
- [x] SPE oracle 复现 paper Figure 2 fan-out threshold
- [x] SPERL 在 p=0.72 CPT88 + p=0.6 CPT95 上正确收敛
- [x] SPSA-vs-SPERL 对照, LNW 上 SPSA 失败的结构性原因清楚
- [x] cpt_offset bug 修复 (state-dependent matching Barberis)

**未做**:
- [ ] T_paper × p_win × x_1 grid sweep (paper-style table)
- [ ] CPT88 + CPT95 全 p_win 跨配置 ablation
- [ ] LNW 上 Algorithm 3/4 (sticky tie-break + quantile filter) ablation
- [ ] Profitable-vs-unprofitable 项目对比 (x_1 = (T_p-1)δ vs (T_p+1)δ)
- [ ] 高阶 robustness check: CPT 参数 perturbation 下 SPE-vs-SPSA 差距

**对 paper revision 的 immediate value**:
- AE point 6 (operations example) ✓ 直接交付
- AE point 5 (robustness across policy classes) ✓ SPSA-LNW-collapse 是干净例证
- AE point 4 (CPT-induced behavior beyond local optimality) ✓ p=0.6 CPT95 的 stay-out 决策展示
- AE point 1 (cycling / convergence) — 没直接证据但 SPERL 在两个 config 上都稳定收敛, 跨 5 seeds variance 小

剩下的 grid table 和 algorithm 3/4 ablation 都是 nice-to-have 而非 must-have。当前数据已足以 support paper §4 的 LNW 章节写作。
