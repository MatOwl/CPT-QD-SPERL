# Playbook: 在 CPT-QD-SPERL 框架上开发一个新 Environment

**用途**: 当未来要再加一个 env (e.g., LNW 之后的 BLN consumption / Henderson liquidation / 新的 OM env), 按这个 playbook 走能少踩很多坑。

**最后更新**: 2026-04-26 (基于 LNW abandonment 实施经验)

---

## Phase 0 — 选 env

### 选择 criteria (按重要性排)

1. **MDP 兼容性 (must)**:
   - 多期决策 (≥ 2 时间步, 否则 time-inconsistency 不存在)
   - State 离散或可离散化 (与现有 `Featurizer` 接口兼容)
   - Action 离散 (≤ 10, 现有 SPERL 是 tabular)
   - State space ≤ ~50k (OptEx 上限)
   - 有 closed-form / numerical / experimental benchmark 用于 cross-validate

2. **CPT 兼容性 (must)**:
   - Reward 结构能被 CPT 评估 (单一 cumulative scalar at terminal)
   - 偏好里含 reference-point / loss-aversion / probability-weighting **任一** (CPT 才能落地)
   - Time-inconsistency 来源清楚 (probability weighting / endogenous reference / hyperbolic 任一)

3. **Paper value (should)**:
   - 来自 published OM/MS/OR 期刊 (有 reviewer 认可)
   - 跟现有 env (Barberis casino, OptEx) 不同 framing (operations / managerial / ...)
   - 有 closed-form 阈值/政策可作 sanity check

4. **实施成本 (should be low)**:
   - 单 1D state 跟 multi-D state 工作量差 5-10× (Featurizer + state enumeration)
   - Continuous action 需要 discretizer
   - Continuous time 模型需要 time discretization (注意 dt 选取影响)

### 已评估候选 (2026-04-26)

参考 [reports/2026-04-26_msci_literature_survey.md](2026-04-26_msci_literature_survey.md)。Tier S 排名:
1. Long-Nasiry-Wu (2020) MS — multistage abandonment ⭐ 已实施
2. van Bilsen-Laeven-Nijman (2020) MS — life-cycle consumption (continuous, 2D state)
3. Henderson (2012) MS — liquidation (continuous time, 1D state)

---

## Phase 1 — MDP 设计 (最关键, 决定能否学得动)

### 1.1 决定 reward 结构: delta-style vs lump-sum

**关键经验**: **必须 delta-style, 绝对不要 lump-sum-at-terminal**。

延伸: 即使原始论文 model 描述像 lump-sum (e.g., LNW 的 "终末 payoff x_T"), implementation 上要把 lump 摊到每步。

**为什么 lump-sum fails**: QR critic 用 first-visit-mean 初始化:
- 如果 continue 的 immediate reward 全是 -c (固定成本), Q(s, continue) 起始 ~ -c < 0
- Q(s, abandon) 起始 ~ 0 (immediate reward 0)
- argmax 永远是 abandon, 永远到不了终末 lump sum, learn 不到 future value

**delta-style 怎么做**: 把 state 的 "current cash position" 概念分摊到每步:
- continue: r = `Δstate · scale - cost` (state 变化 - 成本)
- abandon: r = `-current_cash` (recovery, 取消累积变化 + offset)
- featurizer.cpt_offset(s) = `current_cash` (state-dependent)

**验证**: 跑一个 trivially profitable config (e.g., LNW 的 p=0.9), 看 Optimality 是否非零。如果还是 0, reward 结构有问题。

### 1.2 决定 cpt_offset

**关键经验**: **cpt_offset 必须 state-dependent**, 反映 "agent 在当前 state 'in possession of' 多少 cash"。

类比 Barberis: `cpt_offset(s=(t, z)) = z` (当前 wealth, state-dependent)。

**踩坑**: 第一次很容易写成 constant (e.g., `cpt_offset = x_1` for LNW)。这样 abandon-at-any-state 给 CPT input = constant, SPE oracle 会全选 abandon 拿 deterministic CPT, 数字看起来漂亮但**完全错误**。

**怎么发现 bug**: SPE oracle 给的 policy 跟原 paper 的阈值结构对不上 (e.g., LNW 应该是 fan-out abandonment, 但 SPE 全 abandon 或全 continue), 翻代码到 cpt_offset。

**自检**: 在 abandon 状态下, cumulative + offset 应该等于 LNW 的 abandon 语义 (e.g., -t·c, 不含项目价值)。在 continue all 状态下, cumulative + offset 应该等于 paper 描述的最终 payoff (e.g., x_T - T·c)。手算一两个 case 验证。

### 1.3 决定 abandon / exit 语义

**关键问题**: 终止 action 是 "cash out" (保留当前 state 的价值, 像 Barberis 的 exit) 还是 "forfeit" (放弃所有未来, 像 LNW 的 abandon)?

| 语义 | abandon_reward |
|---|---|
| Cash out (保留 current_cash) | 0 (cumulative 和 offset 自然给出 current_cash) |
| Forfeit (归零 + sunk cost) | `-current_cash` (recovery, 让 cumulative+offset 变成 -t·c) |

**踩坑**: 默认假设 "exit = 拿走" 不对所有 env 适用。LNW 是明显的 forfeit 语义。读原 paper 时**专门确认**这一点。

### 1.4 决定 horizon convention

不同 paper 用 T 表示不同东西:
- Barberis paper T=5 = 5 个决策 = env.T
- LNW paper T=6 = 6 个 stage = env.T+1 = 5 个决策

**决定**: env.T 一律 = 决策步数 (跟 Barberis 代码同), 文档里清楚写出"用户传 T = paper_T - 1"或类似换算。

### 1.5 State space 大小估算

公式: |S| = (T+1) × (state-coordinate 离散值数)。

LNW: |S| = (T+1)(2T+1), T=5 → 66。Barberis 同公式 → 66 (T=5)。

如果 |S| > 50k, tabular SPERL 跑不动, 需要 function approximation。

---

## Phase 2 — 实施 (机械步骤)

### 2.1 文件清单

新建 (复制 barberis_casino.py / barberis_spe.py 改):
- [ ] `lib/envs/<env_name>.py` — env class, gym.Env 接口
- [ ] `lib/envs/<env_name>_spe.py` — backward-induction SPE oracle

修改 (扩展 dispatcher):
- [ ] `lib/envs/featurizers.py` — 加 `<EnvName>Featurizer` (含 `loc`/`key`/`cpt_offset`/`iter_states`)
- [ ] `lib/envs/registry.py` — 注册 `<env_name>` 别名
- [ ] `lib/paper_eval.py` — 加 `<env_name>_reset_kwargs`, 扩 `_is_<envname>(env)` duck-type 检查, 扩 `reset_kwargs_builder`/`initial_offset_builder`/`_state_time`/`_to_obs`
- [ ] `lib/io.py` — `run_name_from_args` 加 env-specific 分支
- [ ] `agents/run_paper_eval.py` — `build_env` + `get_reference_policy` + env-specific CLI args
- [ ] `agents/run_experiments.py` — `build_env` + env-specific CLI args

**重要不动**:
- `agents/sperl_qr_generic.py` — env-agnostic, 不应该需要改
- `agents/spsa_generic.py` — env-agnostic, 不应该需要改 (但 2026-04-26 修过 cpt_offset bug, 后续应该 OK)
- `lib/cpt.py` — 算法本体, 不动

**总代码量预估**: 单 1D state + binary action env ≈ 200-300 行新代码。Multi-D state 按需翻倍。

### 2.2 Featurizer 接口的关键 method

```python
class XxxFeaturizer(Featurizer):
    def loc(self, obs) -> int:
        # obs → flat int index in [0, n_states). 用作 critic/policy 表索引.
    
    def key(self, obs):
        # obs → hashable. 用作 dict key (for replay buffer / logging).
    
    def cpt_offset(self, obs) -> float:
        # 当前 state 的 "cash position". 必须 state-dependent (见 Phase 1.2).
    
    def iter_states(self, env=None):
        # yield 所有 paper-eval domain X 的 states (= reachable, non-terminal).
        # 必须严格 parity-correct! 否则 SW 会被多算 (历史教训, 见 2026-04-26_welfare_and_paper_tables).
```

### 2.3 paper_eval dispatcher 扩展模式

`paper_eval.py` 用 duck-type 检查路由 reset/offset 函数。加新 env 时:

```python
def _is_xxx(env) -> bool:
    return hasattr(env, "<unique_attr>") and ...

def reset_kwargs_builder(env):
    if _is_xxx(env):
        return xxx_reset_kwargs
    if _is_barberis(env):
        return barberis_reset_kwargs
    return optex_reset_kwargs

def initial_offset_builder(env):
    if _is_xxx(env):
        return xxx_initial_offset_builder(env)  # 闭包捕获 env 参数
    ...
```

注意优先级: 加更具体的 _is_ 判断在前 (e.g., `_is_abandonment` 在 `_is_barberis` 前, 因 abandonment 也有 `T` 属性)。

---

## Phase 3 — SPE oracle 验证 (sanity check 不可跳)

### 3.1 写完 SPE oracle 后立刻验证 policy structure

不要直接跑 paper-eval, 先单独调 SPE oracle, 打印 policy on (t, state) grid:

```python
spe = compute_spe_policy(env, cpt, n_eval_eps=2000)
for t in range(env.T):
    actions = []
    for k in range(parity_correct_grid_at_t):
        a = spe.get((t, k), '?')
        actions.append(f'x={k:+d}:{"C" if a == 1 else "A"}')
    print(f't={t}: ' + ' '.join(actions))
```

**对比原 paper Figure**: 看 policy 是 monotone threshold? Fan-out structure? 这个对比是发现 reward / offset bug 的最快路径。

### 3.2 SPE rollouts 噪声敏感性

borderline state (CPT 接近 0) 在低 n_eval_eps (e.g., 500) 下会随机翻action。**默认 n=2000**, 在边界 config 时 n=5000。

### 3.3 Backward-induction 顺序

```python
for t in range(T - 1, -1, -1):  # T-1 down to 0
    for x_idx in parity_correct_range(t):
        # 每个 action MC rollout 估 Q
        # argmax 设 policy[(t, x_idx)]
```

注意 follow(s) 默认 action 应该是"安全 fallback" (e.g., abandon=0)。如果 default 选 continue 可能死循环 / 跑出 grid。

---

## Phase 4 — SPERL 训练 + 单 config 验证

### 4.1 第一次训练用 default Barberis 超参, 不要 tune

```bash
$PYTHON agents/run_paper_eval.py --env <new_env> --seeds 3 \
    --train-eps 8000 --batch 1 --support-size 50 \
    --critic-lr 0.04 --eps 0.3 --spe-rollouts 2000 \
    --eval-per-state 100
```

跑出来如果:
- **Optimality > 0 + Disagree < 5/15 + SW > 80% SPE**: env reward 设计正确, 可进 Phase 5
- **Optimality = 0 + Disagree > 7/15**: 说明 reward / offset 有 bug, 回 Phase 1 检查 (大概率 bug #1 lump-sum 或 bug #2 constant offset)
- **跨 seeds 高 variance (std/mean > 1)**: hard config 的正常现象 (HANDOFF: "3-seed Barberis unreliable, 需 ≥10 seeds")

### 4.2 Sanity check: 跟 Barberis 同 config 对照

跑 `agents/run_paper_eval.py --env barberis --seeds 3 ...` 同样超参做对照。如果 LNW 数字差 Barberis 一个量级以上, 通常是 reward 设计问题。

### 4.3 SPSA precommitment 训练

记得检查 `agents/spsa_generic.py` 的 `cpt_offset` 加法是否正确 (2026-04-26 加的, 应该 OK 但后续要重新加新 env 时确认)。

**SPSA 在某些 env 上结构性失败** (e.g., LNW 上 SPSA 必塌缩到 abandon 因为 per-step 成本 + CPT loss aversion 把梯度推向 trivial 解)。这不是 bug 而是 paper-worthy 现象, 应该记录到 session report 而不是修。

---

## Phase 5 — Paper-quality 分析 suite

参考 paper §4 对 Barberis 做的分析, 一个完整 env 章节需要的:

### 5.1 必做 (core deliverables)

1. **Multi-config grid table**: CPT_regime × p_win × env-param 至少 6-8 cells
   - paper Table 1 对应: Optimality 跨配置
   - paper Table 2 对应: SW 跨配置
   - **每 cell 至少 5 seeds, 理想 10 seeds**

2. **3 类 policy 对照**: SPERL (SPE) vs SPSA (precommitment) vs DO (dynamically optimal naive)
   - DO = "假装无 time-inconsistency, 用 expected-CPT 当 reward 跑标准 DP" 的 baseline
   - paper §4 标配, 缺了不完整

3. **Per-state policy 可视化**:
   - SPE oracle 的 policy heatmap on (t, state)
   - SPERL 学到的 policy 同 plot, 标差异
   - 如果 env 有自然阈值结构 (e.g., LNW fan-out), 这个 plot 一图胜千言

4. **Convergence curves**:
   - CPT(x_0) vs training episodes
   - 多 seed 平均 + 1σ shaded region
   - paper Figure 标配

### 5.2 强烈建议

5. **Algorithm 3/4 ablation** (paper §4.2.5):
   - sticky-policy on/off + filter-thresh sweep
   - 在 hardest config 跑 (paper 上 hardest 是 CPT88/p=0.66)

6. **Hyperparameter robustness** (response to AE point 5):
   - critic-lr / support-size / batch sweep
   - 看 SPERL 性能对 hyperparam 的敏感度

### 5.3 Nice-to-have

7. **CPT 参数 perturbation 下 SPE-vs-precommitment robustness**:
   - paper claim: "equilibrium policies are more robust"
   - 在 CPT 参数 ±20% perturb 下两类 policy 的 V(x_0) 退化

8. **Visit-frequency analysis** (类比 OptEx 上 `scripts/analyze_optex.py`):
   - 计算 SPE policy 下哪些 state 真被访问
   - Disagreement 集中在 on-path 还是 off-path

---

## 附录 A — 已知 bug 模式 (按发现顺序)

### Bug #1: Lump-sum-at-terminal reward 设计

**症状**: 即使 trivially profitable config (e.g., p=0.9) Optimality=0 跨所有 seeds。Barberis 同超参正常。

**原因**: QR critic first-visit-mean 初始化让 Q(continue) 起始负, agent 锁 abandon, cold-start chain 不闭合。

**修复**: 改 delta-style reward, 让每步有 ±量级 signal。

**怎么避免**: Phase 1.1 时就直接选 delta-style, 不要"先简单 lump-sum 再优化"。

### Bug #2: cpt_offset 写成 constant

**症状**: SPE oracle policy 跟原 paper Figure 阈值结构对不上 (e.g., LNW 应该 fan-out, 但 SPE 全 abandon)。SPERL 数字漂亮但 policy 错。

**原因**: cpt_offset 应反映 "current cash position", 必须 state-dependent。

**修复**: `cpt_offset((t, x_idx)) = current_cash_at_state` (类比 Barberis `cpt_offset = z`)。

**怎么避免**: Phase 1.2 时手算 abandon-at-(0,0) 的 CPT input 应该等于多少 (按原 paper 语义), 验证 cpt_offset 实现给出对的值。

### Bug #3: SPE oracle MC noise 在 borderline config 上翻车

**症状**: 同样 config 不同 random seed 下 SPE policy 在某些 state 翻 action, paper-eval 的 SPE Welfare 抖动很大。

**原因**: CPT 接近 0 时 argmax 受 MC sampling noise 影响。

**修复**: `--spe-rollouts 2000` 起步, 边界 config (e.g., LNW p=0.6 default CPT) 用 5000。

**怎么避免**: Phase 3.2 默认 n=2000 不要 < 1000。

### Bug #4: SPSA cpt_offset 没加

**症状**: SPSA 训练目标和 SPERL 不一致 (优化的是 CPT(payoff - x_1) 而非 CPT(payoff))。

**修复**: `agents/spsa_generic.py` 三个 rollout 路径都加 `+ featurizer.cpt_offset(initial_obs)`。已在 commit `ad5bdbf` 修。

**怎么避免**: 加新 env 时验证 SPSA 在该 env 上 cpt_rewards 的初始 evaluation 跟 SPE oracle 在 (0,0) 的 V_hat 接近 (warm-start at SPE 应该给 ≈ V_hat)。

---

## 附录 B — 超参移植启发

Barberis 上调好的超参直接拿来用通常 **work**, 因为 reward 量级类似 (delta-style 后 ±10 量级)。

|  超参 | Barberis CPT88/p=0.72 (paper) | LNW (verified) |
|---|---|---|
| `--train-eps` | 15000 | 8000 (5k 也够 quick check) |
| `--batch` | 1 | 1 |
| `--support-size` | 50 | 50 |
| `--critic-lr` | 0.04 | 0.04 |
| `--eps` | 0.3 | 0.3 |
| `--spe-rollouts` | 300-500 | 2000 (LNW 边界 case 需要更多) |
| `--eval-per-state` | 100 | 100 |

**注意**: 如果 reward 量级跟 Barberis 差异大 (e.g., OptEx 是 1e-3 量级), `--critic-lr` 要 retune。LNW delta-style 后跟 Barberis 同量级, 不需要 retune。

---

## 附录 C — Quick verification commands

```bash
# Phase 3.1: SPE oracle structure check (replace ENV_NAME / params as needed)
$PYTHON -c "
import sys; sys.path.insert(0, '.')
from lib.cpt import CPTParams
from lib.envs.registry import make_env
from lib.envs.<ENV_NAME>_spe import compute_spe_policy

env = make_env('<ENV_NAME>', ...)
cpt = CPTParams(0.88, 0.65, 0.65, 2.25)
spe = compute_spe_policy(env, cpt, n_eval_eps=2000)
# Print policy in your env's natural state grid
"

# Phase 4.1: SPERL smoke test (no save)
$PYTHON agents/run_paper_eval.py --env <ENV_NAME> --seeds 3 \
    --train-eps 5000 --batch 1 --support-size 50 \
    --critic-lr 0.04 --eps 0.3 --spe-rollouts 2000 \
    --eval-per-state 100 --no-save

# Phase 4.2: Barberis same-CPT-params control
$PYTHON agents/run_paper_eval.py --env barberis --seeds 3 \
    --p-win 0.72 --alpha 0.88 --rho1 0.65 --rho2 0.65 --lmbd 2.25 \
    --train-eps 8000 --batch 1 --support-size 50 \
    --critic-lr 0.04 --eps 0.3 --spe-rollouts 2000 \
    --eval-per-state 100 --no-save

# Phase 4.3: SPSA precommitment check
$PYTHON agents/run_experiments.py --env <ENV_NAME> --algo spsa \
    --train-eps 10000 --batch 50 --eval-eps 200 --eval-freq 2500 \
    --spsa-step 5 --spsa-c 1.9 --seed 0
```

---

## 附录 D — Checklist before declaring "env done"

- [ ] env class 实现 (Phase 2.1)
- [ ] SPE oracle 实现 + policy structure 跟原 paper 对应 ✓ (Phase 3.1)
- [ ] paper-eval domain X 严格 parity-correct (Phase 2.2)
- [ ] 单 config (paper-default) 跑出非平凡 Optimality + 低 Disagree (Phase 4.1)
- [ ] 跟 Barberis 同 config 对照, 数字量级一致 (Phase 4.2)
- [ ] SPSA 训练验证 (Phase 4.3)
- [ ] **Multi-config grid table** (Phase 5.1.1)
- [ ] **DO baseline** 实现 + run on grid (Phase 5.1.2)
- [ ] **Per-state policy 可视化** (Phase 5.1.3)
- [ ] **Convergence curves** (Phase 5.1.4)
- [ ] Session report 沉淀经验 (类比本 playbook + 2026-04-26_lnw_env_implementation.md)
- [ ] Project memory 更新 (类比 `.claude/projects/.../memory/project_msci_env_extension.md`)

---

## 引用历史

本 playbook 基于以下实施经验沉淀:
- LNW abandonment env (2026-04-26): [reports/2026-04-26_lnw_env_implementation.md](2026-04-26_lnw_env_implementation.md)
- Welfare 求和域修复 (2026-04-26): [reports/2026-04-26_welfare_and_paper_tables.md](2026-04-26_welfare_and_paper_tables.md)
- Refactor bugs + Algorithm 3/4 port (2026-04-24): [reports/2026-04-24_refactor_bugs_and_alg34_port.md](2026-04-24_refactor_bugs_and_alg34_port.md)
- 文献综述 (2026-04-26): [reports/2026-04-26_msci_literature_survey.md](2026-04-26_msci_literature_survey.md)
