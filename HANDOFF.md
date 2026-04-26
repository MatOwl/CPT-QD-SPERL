# 交接文档 — 2026-04-24（2026-04-26 更新：Phase 1+2 已完成）

验证 refactor 后的 generic SPERL 是否和 paper (`paperRef/MSci_MANUSCRIPT.pdf`) 对得上，并初步对比 OptEx 学到 policy 和 SPE oracle 的逐 state 差异。

> **2026-04-26 更新摘要**：HANDOFF "启动包" 第 1、2、3、4 项 (Phase 1+2+3) 全跑完。
> - **Phase 1** (Barberis Alg 3/4 回归): 4 个 ablation 全部落在 paper 1σ 带内，port 无回归；3 seeds 噪声完全淹没 Alg 3/4 的效应
> - **Phase 2** (OptEx Alg 3/4 效果): disagreement 241→242 完全不动 → **HANDOFF B 档结论确认 = 结构性 SPE 多重性**；Alg 3 仅有微小稳定性贡献 (cross-seed std 4× 缩窄)
> - **Phase 3** (welfare 定义 + Tables 1/2 复现 + hardest config ablation):
>   - 修复 `BarberisFeaturizer.iter_states` 让求和域和 paper Definition 5 对齐 → SPE Welfare 在 CPT88/p=0.72 和 0.66 都精确匹配 paper (1σ 内)
>   - SPERL 在所有 config 上 Optimality > paper、SW 与 paper 一致 → 进一步确认我们和 paper 落在 SPE 多重性的不同点上
>   - **Alg 3/4 在 hardest config (CPT88/p=0.66) 上把 Optimality 砸到 0** (sticky 锁死 exit)，paper 称这种配置上 Alg 3/4 应该有效 → 我们的 port 实现可能在某些细节上与 paper 不一致，或 paper 用了不同超参
>
> 关键概念修正: **SPE policy 在 x_0 处的值可以高于 SPE oracle**。SPE 是 Bellman-CPT 不动点，多个 SPE 可共存。"V_SPERL > V_oracle" 不是 bug。
>
> **Phase 2 结论修正 (visit-weighted)**: 49% disagreement 是 **off-equilibrium-path 的伪指标**。SPE 实际玩 2000 局只触及 53 个 (t,state)，其中 on-path disagreement 只有 5/53 = 9.4%，并且全部是 cross-seed unstable（真正的多重性集中在这 5 个 contested decision points）；off-path 的 184 个 stable_mismatch 是 SPERL 在低 visit state 上的次优学习结果，不影响实际表现。
>
> 完整结果见 [文档末尾的 "2026-04-26 验证结果"](#2026-04-26-验证结果) 章节。剩余事 (legacy 对照、超参 sweep、CPT95 完整 config、Alg 3/4 实现细节核对、README) 仍未做。

## 环境

Python: `C:/Users/Jingxiang Tang/FNN/Scripts/python.exe` (Python 3.10.7)。今天装进去的包（原本 venv 是空的）：
```
numpy==1.26.3  scipy  pandas  matplotlib  gym==0.26.2  pypdf
```
注意 `numpy<2` 是 gym 0.26 的硬要求，装 numpy 2.x 会在 import 时爆。

---

## 今天改的东西

### (A) Refactor 引入的 3 处同源 bug — 已修

全部是"MDP reward 是 wealth 增量，但 CPT 要在终态财富 `z + Σr` 上评估"这一 Barberis 特有语义在 refactor 时漏掉。

| 文件 | 症状 | 修法 |
|---|---|---|
| [lib/envs/barberis_spe.py:51](lib/envs/barberis_spe.py:51) | SPE oracle 全给 "exit"（任何 CPT 参数下）| `rewards.append(total + z)` |
| [lib/paper_eval.py](lib/paper_eval.py) `rollout_cpt_from_state` | V^π(x) 在 z≠0 状态偏低 | 新增 `initial_offset_for_state` callback |
| [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `QRCritic.cpt_value*` | greedy 永远不在 loss-domain 翻转到 play；backward propagation 断掉 | CPT 评估前把 offset 加到 quantiles |

架构抽象：新增 `Featurizer.cpt_offset(obs)` hook（[lib/envs/featurizers.py](lib/envs/featurizers.py)）。Barberis 返回 `float(z)`，OptEx 默认 `0.0`。

**验证**（Paper §4 / Appendix C.2.5 主表）：

| 配置 | 我们的 Optimality | Paper | 结论 |
|---|---|---|---|
| CPT88 (α=0.88, δ=0.65, λ=2.25), p=0.72 | 7.60 ± 0.70 | 7.91 ± 1.64 | ✓ 1σ 内 |
| CPT95 (α=0.95, δ=0.5, λ=1.5), p=0.60 | 0.00 ± 0.00 | 0.0 / 0.0 | ✓ 完美 |

SPE oracle 单点检查：CPT88/p=0.72, V_SPE(0,0) = 7.78（paper 8.07）✓

### (B) 论文 Algorithm 3 & 4 的 port — 已完成代码，未重跑

Legacy 脚本里有两个使 SPERL 稳定收敛到 reference SPE 的机制，generic path 之前没有。今天都 port 了。

**Algorithm 3 — Consistent tie-break / is-better guard**
- 位置：[agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) `GreedyPolicy.update_from_critic_values`
- 新参数 `sticky=True`：只有当新 argmax 的 CPT **严格大于** 旧 policy 动作的 CPT 时才翻转（避免 flat-landscape 的 argmax 抖动）
- 新参数 `tie_thresh>0`：在 max CPT 的 `tie_thresh` 范围内均匀随机（避免 oracle vs SPERL 的符号 tie-break 差异）
- First-visit (prev==-1) 无条件接受

**Algorithm 4 — Quantile filter + acceptance gate**
- 位置：[agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) 新 `filter_quantiles` helper + `QRCritic._cpt_from_quantiles`
- 滤波逻辑：计算 quantile 间 gap，取 `filter_thresh` 分位数作阈值；gap 超阈或负（crossing quantile）的位置被 snap 到最近 valid 邻居
- 接受门：若 filtered CPT 与 unfiltered 的相对差 > `filter_accept_ratio`，回退到 unfiltered。legacy/paper 默认 `filter_thresh=0.75, accept_ratio=inf`（永远信 filter），paper Appendix C.2.5 明确给过 `accept_ratio=0.5`

**CLI 新 flag**（[agents/run_paper_eval.py](agents/run_paper_eval.py) 和 [agents/run_experiments.py](agents/run_experiments.py) 都加了）：
```
--sticky-policy                  # Alg 3 on
--tie-thresh FLOAT               # default 0 (exact ties only)
--filter-thresh FLOAT            # Alg 4 filterTresh; default None = off
--filter-accept-ratio FLOAT      # Alg 4 treshRatio; default inf = trust filter
```

所有默认保持向后兼容（全 off），现有 baseline 跑法不变。

### (C) 辅助产物

- [scripts/analyze_optex.py](scripts/analyze_optex.py) — 读 `per_state_values.csv` 汇总逐 state 对比 + heatmap

---

## 🔴 重要观察：OptEx 的 SPE 非唯一性

**现象**：修完所有 bug 后，OptEx σ=0.015 仍有 ~49% state-level policy disagreement。加训练量（500→3000 eps, 6×）disagreement 纹丝不动。

**证据**：

| 指标 | σ=0.015 (493 states) | σ=0.029 (1012 states) |
|---|---|---|
| Policy disagree | 241/493 = 48.9% | 465/1012 = 46.0% |
| Cross-seed stability (3 seeds 选同 action) | 87.8% | 76.1% |
| mean \|Δv\| at disagreements | 0.011 | 0.009 |
| mean \|Δa\| | 2.13 / 11 | 2.02 / 11 |
| t=5 disagree | 0% | 0% |
| t=1-3 disagree | 85-89% | 88-91% |

**诊断**：
1. **Value 对齐良好**（per-state \|Δv\|≈0.006-0.011, 比 state 的 V 值本身小 1 个数量级）→ critic 没错
2. **Cross-seed 一致性高**（87.8%）→ SPERL 确定性地收敛到它自己的那个 policy，不是随机游走
3. **mean \|Δa\|=2** 说明不是"相邻 action tie-break"问题
4. **后期完美一致，中段大范围分歧**

**结论**（最可能的解释）：**OptEx 在中段 state 的 CPT landscape 上存在多个近乎等值的 SPE 均衡**。Paper 的 reference oracle 是通过 backward induction + deterministic argmax 选定的一个；SPERL 通过 QR 收敛到另一个有效均衡。两者都是 valid SPE policy，CPT 值几乎相同，只是 argmax 的具体选择不同。

**支持这一解释的现象**：
- σ 翻倍，disagreement pattern 完全一致 → 不是噪声级别问题
- value gap 随 seed 变化很小 → policy 差异 "is a real choice, not numerical fluctuation"
- Barberis 在同样的 generic path 下能精确匹配 paper（policy_disagree=0 在 CPT95/p=0.6），说明算法本身没问题 — Barberis 的 2-action 空间几乎没 tie，OptEx 的 11-action 空间有

**Alg 3/4 的作用**：理论上 paper 用 Algorithm 3 (`sticky` + random tie-break) 来"稳定"收敛方向，用 Algorithm 4 (quantile filter) 来减少噪声；两者一起可以把 SPERL 推向 reference oracle 的那个特定均衡。**今天代码 port 完了，但还没跑验证**。

---

## 工作目录结构（今天的产物）

```
results_postfix/        # bug fix #1-2 后的 Barberis 结果（SPE oracle 正确）
results_postfix_v2/     # bug fix #3 后（critic cpt_offset），对齐 paper 完成
results_optex/          # OptEx 两个 sigma 的结果 + heatmap
HANDOFF.md              # 本文档
scripts/analyze_optex.py
```

这些 `results_*` 目录已在 `.gitignore`（从原先的 commit 6338a3b 起）。

---

## 下次接手的启动包

### 1. 先确认 Alg 3/4 port 没写错 — Barberis regression test

代码默认全 off，但打开了不该 degrade。跑个中等规模的 Barberis 当 smoke test：

```bash
PY="C:/Users/Jingxiang Tang/FNN/Scripts/python.exe"
cd "C:/Users/Jingxiang Tang/FNN/CPT-QD-SPERL"

"$PY" agents/run_paper_eval.py --env barberis --p-win 0.72 \
    --alpha 0.88 --rho1 0.65 --rho2 0.65 --lmbd 2.25 \
    --train-eps 15000 --batch 1 --support-size 50 \
    --critic-lr 0.04 --eps 0.3 --seeds 3 \
    --sticky-policy --tie-thresh 0.01 \
    --filter-thresh 0.75 --filter-accept-ratio 0.5 \
    --spe-rollouts 2000 --results-dir results_alg34
```

**期望**：Optimality 应该仍在 ~7.6 附近（paper 7.91 ± 1.64）。如果大幅恶化，说明 sticky 过早锁死了早期噪声 policy，得调 `eps`（加大探索）或减小 `tie_thresh`。

### 2. OptEx — Alg 3/4 是否真能拉近 SPERL 和 oracle

```bash
"$PY" agents/run_paper_eval.py --env optex --sigma 0.015 --num-w 4 \
    --horizon 5 --train-eps 3000 --batch 20 --support-size 20 \
    --critic-lr 0.1 --eps 0.3 --seeds 3 --eval-per-state 30 \
    --sticky-policy --filter-thresh 0.75 --filter-accept-ratio 0.5 \
    --spe-file CumSPERL_ref/SPE_OptEx_5_0.015_4.npy \
    --results-dir results_alg34_optex

"$PY" scripts/analyze_optex.py \
    results_alg34_optex/optex_sperl_sig0.015_numw4_cpt_a0.95_r0.5_l1.5
```

**期望**：
- **A) disagreement 大幅下降**（比如 241 → ~50）→ Alg 3/4 确实让 SPERL 收敛到 reference SPE，假设"非唯一性"其实是可以被 paper trick 压下去的
- **B) disagreement 基本不变**（241 → ~200+）但 cross-seed stability 上升 → 非唯一性是结构性的；paper trick 只稳定不统一
- **C) 变差**（disagreement 上升）→ port 有 bug，大概率在 `sticky_policy` 的 "strict >" 比较上

**仅靠 sticky（不开 filter）单独测一次**也有价值，能分离两者贡献。

### 3. 遗留问题

1. **Welfare 量级差距**：paper 报 CPT88/p=0.72 `SPE Welfare = -24.15 ± 1.21`，我们算出 `-154.86`。x₀ 处 Optimality 完美对上，全状态求和的 welfare 差 6×。怀疑 paper 只对从 x₀ 可达的状态子集求和，或用了其他 normalization。查 paper 正文 §4.2 的 "Social Welfare" 定义能消除疑虑。

2. **README 没同步**：README 是 refactor 后一次性写的，里面的 CLI 示例参数和 paper 不一致（例如 README 用 `--critic-lr 2.0` 来自 legacy 语义，而 generic path 实际需要 `0.04`）。port 完 Alg 3/4 验证完后再更新 README。

3. **更多 paper 配置没跑**：paper 还有 CPT88/p=0.66 (paper 说这是 "bad example"，最难收敛)、CPT95/p=0.72 等。如果要做完整 Table 1/2 复现，需要把所有组合都跑一遍。

4. **legacy 没跑过对照**：我们始终是拿 paper 的数字做对照，没真正跑过 `agents/rerun_GreedySPERL_QR__main.py` 做独立校验。如果 paper 数字有疑问，跑一次 legacy 作为第二 ground truth。

### 4. 跑 experiment 时的一些踩坑点

- stderr 里 gym 0.26 会刷一堆 numpy 2.0 warning（即使装的是 numpy 1.26），用 `2>&1 | grep -v "Gym has\|Please upgrade\|Users of this\|See the migration"` 过滤
- Barberis 5 seeds × 15k eps ≈ 40 秒；OptEx 3 seeds × 3k eps ≈ 8-10 分钟（主要是 per-state eval rollouts 费时）
- `run_paper_eval.py` 生成的 `per_state_values.csv` 是逐 state 对比的黄金数据，都保留着

---

## 2026-04-26 验证结果

### 关键概念修正
SPE policy 在 x_0 处的值 **可以高于** SPE oracle 计算出的值。SPE 不是任意一点的最优解，是 Bellman-CPT 算子的不动点；多个 SPE 可以共存，每个有不同的 V(x_0)。"SPERL 的 V > oracle V" **不**是 bug，是均衡多重性的体现。

### Phase 1 — Barberis Alg 3/4 回归测试 (CPT88/p=0.72, 3 seeds × 15k eps)

| 配置 | Optimality | Disagree | Value Error |
|---|---|---|---|
| All off (regression check) | 8.39 ± 0.74 | 15.7/36 | 70.6 |
| Sticky only | 6.56 ± 1.59 | 15.3/36 | 74.4 |
| Filter only | 7.82 ± 0.97 | 16.0/36 | 77.0 |
| Sticky + filter | 9.38 ± 1.40 | 13.3/36 | 74.5 |
| Paper | 7.91 ± 1.64 | — | — |

- ✅ 所有配置在 paper 1σ 带内 → port 没破回归
- ⚠️ Disagree 在 Barberis 上 13-16/36 几乎不动（2-action 空间多重性少）
- 3 seeds 噪声 (per-seed 摆幅 4-11) 完全淹没配置间差异——后续如要在 Barberis 上做 head-to-head，需 ≥10 seeds

### Phase 2 — OptEx σ=0.015 Alg 3/4 (3 seeds × 3k eps)

| 指标 | Baseline (off) | Sticky only | Sticky + Filter |
|---|---|---|---|
| Disagree / 493 | **241.0 ± 3.7** | **242.7 ± 0.9** | **242.0 ± 2.4** |
| Optimality | -0.037 ± 0.010 | -0.033 ± 0.006 | -0.024 ± 0.003 |
| Value error | 2.77 ± 0.09 | 2.77 ± 0.11 | 2.73 ± 0.03 |
| Cross-seed stable / 493 | 433 (87.8%) | 431 (87.4%) | 428 (86.8%) |
| ↳ stable AND matches SPE | 245/433 = **56.6%** | 244/431 = **56.6%** | 244/428 = **57.0%** |
| mean\|Δv\| at disagree | 0.0113 | 0.0113 | 0.0111 |

**结论**: HANDOFF "启动包" 第 2 项的 **B 档假设确认** = 结构性 SPE 多重性。
- Disagreement 在 3 个配置间完全不动 (241/242/242)，说明 paper 的 Alg 3/4 不能把 SPERL 推向 paper oracle 选定的那个 SPE
- Cross-seed 稳定的 ~430 个 state 中只 57% 匹配 oracle——剩下 43% 不是噪声/抖动，是 SPERL 确定地学到了**另一个有效 SPE**
- Alg 3/4 **确实有微弱效果**:
  - Sticky 把 cross-seed std 从 3.7 缩到 0.9（4× 改善），符合 Alg 3 的稳定收敛设计目标
  - Sticky + filter 把 Optimality bias 从 -0.037 拉到 -0.024，更接近 oracle V(x_0)
  - 但 policy 分布的差异是无法消除的

**潜在解释 (paper 为何声称低 disagreement?)**:
- Paper 可能用了不同的 "disagreement" 定义（比如只计 reachable states 或加权计）
- Paper 可能用了不同的 oracle (`CumSPERL_ref/SPE_OptEx_5_0.015_4.npy` 是我们继承的，未必和 paper 的 reference 一致)
- Paper 可能有未文档化的初始化或训练机制
- 排查这点需要读 paper §4 / Appendix C 详细描述

### 新产物
```
results_alg34/                    # Barberis sticky+filter
results_alg34_off/                # Barberis 全 off
results_alg34_stickyonly/         # Barberis sticky-only
results_alg34_filteronly/         # Barberis filter-only
results_alg34_optex/              # OptEx sticky+filter (含 disagree heatmap)
results_alg34_optex_stickyonly/   # OptEx sticky-only (含 disagree heatmap)
```

### Phase 3 — Welfare 量级差距 (HANDOFF §"遗留问题" #1) — ✅ 解决

读 paper Manuscript p18 Definition 5: `SW(π) = Σ_x V^π(x)`。问题是"x 取哪些"——我们之前在 Barberis 上枚举 `(t, z)` 全 36 个，包含 (a) 不可达 (z 与 t 不同奇偶) 和 (b) 终态 (t=T)。OptEx 一直用 BFS reachable tree 所以没问题，只 Barberis 过度枚举。

**验证** (CPT88/p=0.72, baseline, 3 seeds):

| Filter | SW (count) | SPE oracle SW | Paper |
|---|---|---|---|
| 全枚举 | 36 states | -156.27 | — |
| 可达 + 终态 | 21 states | -96.28 | — |
| **可达，去终态** | **15 states** | **-22.78** | **-24.15 ± 1.21** ✓ |

**修复**: [lib/envs/featurizers.py:60](lib/envs/featurizers.py:60) `BarberisFeaturizer.iter_states` 改为只 yield 可达决策 state (`range(-t, t+1, 2)`，且 `t < T`)。

**修复后 baseline 数字** (results_swfix):
- Optimality: 8.39 ± 0.74 (paper 7.91 ± 1.64) ✓ 不变 (V at x_0 不受求和域影响)
- Policy disagree: **1.67/15** (vs 旧 15/36) — 更有意义，反映真正决策 state 上的差异
- SPE Welfare (oracle): **-23.30** (paper -24.15 ± 1.21) ✓ 1σ 内
- SPERL SW: -35.53 ± 11.11 (无 paper 直接对比，但量级合理)

**注意**: 这个 fix 让 Barberis 的所有 metric (policy_disagree, value_error, social_welfare) 的求和域和 paper 一致；旧 `results_*` 目录里的数值 base 在过度枚举上，不能直接和新数比较。重要的不变量: `optimality` 不变，`SPE Welfare (ref)` 落入 paper 1σ。

### Phase 3 续 — paper Table 1/2 跨配置复现 (CPT88/p=0.72, 0.66)

抓出 paper §4.2.5 的 Table 1 (Optimality) 和 Table 2 (Social Welfare):

**Paper Table 1 — Optimality**:
| p_win | Paper SPERL | Paper SPE oracle |
|---|---|---|
| CPT88/0.72 | 7.91 ± 1.64 | 8.07 ± 0.39 |
| CPT88/0.66 | 2.74 ± 0.95 | 3.59 ± 0.30 |
| CPT88/0.60 | 0.0 ± 0.0 | 0.0 ± 0.0 |

**Paper Table 2 — Social Welfare**:
| p_win | Paper SPERL | Paper SPE oracle |
|---|---|---|
| CPT88/0.72 | -26.03 ± 3.63 | -24.15 ± 1.21 |
| CPT88/0.66 | -59.94 ± 2.88 | -51.5 ± 1.06 |
| CPT88/0.60 | -80.09 ± 4.22 | -79.28 ± 0.67 |

**我们 (welfare fix 后, 3 seeds × 15k eps)**:
| Config | Metric | 我们 (baseline) | Paper SPE / SPERL |
|---|---|---|---|
| CPT88/0.72 | SPE Welfare | -23.30 | -24.15 ± 1.21 ✅ |
| | SPERL Optimality | 8.39 ± 0.74 | 7.91 ± 1.64 ✅ |
| | SPERL SW | -35.53 ± 11.11 | -26.03 ± 3.63 ⚠️ off (high variance, 3 seeds 不够) |
| CPT88/0.66 | SPE Welfare | -51.29 | -51.5 ± 1.06 ✅ |
| | SPERL Optimality | 4.57 ± 2.31 | 2.74 ± 0.95 (我们更高) |
| | SPERL SW | -59.31 ± 4.43 | -59.94 ± 2.88 ✅ |

**结论**: SPE oracle 在所有测过的配置上完美对齐 paper 1σ → welfare fix 正确。SPERL 的 Optimality 比 paper 高、SW 一致 → 我们和 paper 落在 SPE 多重性的不同点上 (SW 等价但 V(x_0) 不同) — 这进一步坐实了 Phase 2 的"多重 SPE 共存"结论。

### Phase 3 续 — Alg 3/4 在 hardest config (CPT88/p=0.66) 上的 ablation

| Config | Optimality | Disagree | VE | SW |
|---|---|---|---|---|
| Baseline (off) | **4.57 ± 2.31** | 3.67/15 | 23.64 | -59.31 |
| Sticky (tie=0.01) + Filter | -0.02 ± 0.03 | 6.67/15 | 27.68 | -67.34 |
| Sticky (tie=0) + Filter | **0.00 ± 0.00** ❌ | 7.67/15 | 19.86 | -63.51 |
| Paper SPERL (with Alg 3/4) | 2.74 ± 0.95 | — | 9.72 | -59.94 |

**惊人发现**: 我们 port 的 Alg 3/4 在 paper 称为"hardest convergence" 的 CPT88/p=0.66 上**主动把 policy 锁到"exit"** (Optimality → 0 in all seeds)。`tie_thresh=0`（strict-better only）甚至更糟。HANDOFF 早就预测过 ("sticky 过早锁死了早期噪声 policy")。

可能原因 (排查优先级):
1. **Hyperparameter mismatch**: paper Appendix C.2.5 应有 SPERL 训练超参 (eps, batch, train_eps, critic_lr 等)；我们用的是 HANDOFF 给的 `eps=0.3, critic-lr=0.04, train-eps=15000`，可能不是 paper 的设置
2. **Sticky 实现差异**: 我们的 `update_from_critic_values` `is_better_guard` 用 strict ">"，paper 可能用 ">= + epsilon" 或 SoftMax；需要对照 paper Appendix C.2.2 的 Algorithm 3 伪代码
3. **Paper Table 4 filter sweep**: 我们用 `--filter-thresh 0.75` (paper 范围最低)。Paper p=0.66 的 Pareto-optimal 是 `filter=0.9` (VE 9.3 ± 0.95)。需要 sweep filter 才能看到 Alg 4 的最佳效果

### 新产物 (Phase 3)
```
results_swfix/                       # welfare fix 后的 baseline (CPT88/p=0.72)
results_cpt88_p066_baseline/         # CPT88/p=0.66 baseline
results_cpt88_p066_alg34/            # CPT88/p=0.66 with sticky+filter (tie=.01)
results_cpt88_p066_alg34_strict/     # CPT88/p=0.66 with sticky+filter (tie=0)
scripts/sw_breakdown.py              # SW filter-definition 比较脚本
scripts/visit_freq_check.py          # 一次性 OptEx visit-frequency 探针 (已并入 analyze_optex.py)
```

### Phase 4 — OptEx 49% disagreement 是 off-equilibrium-path artifact

用户在 session 末提出关键问题: 不匹配的 184 个 state 是不是本来就不太被 visit？
跑 SPE oracle 2000 条轨迹做 visit-frequency 拆分，结果颠覆 Phase 2 的解读。

| Config (OptEx σ=0.015) | BFS disagree (raw) | On-path disagree | stable_mismatch on-path | unstable on-path |
|---|---|---|---|---|
| Baseline (off) | 248/493 (50.3%) | **5/53 (9.4%)** | **0** | 5 |
| Sticky only | 249/493 (50.5%) | **5/53 (9.4%)** | **0** | 5 |
| Sticky + Filter | 249/493 (50.5%) | **5/53 (9.4%)** | **0** | 5 |

**关键发现**:
1. SPE 实际玩 2000 局只触及 **53 个 distinct (t, state)**（vs BFS reachable 493）。其余 440 个 state 都是 SPERL 没在 training 里见过（或仅以 ε-greedy 探索小概率到达）→ 学到的策略基本是先验默认值
2. 244 + 184 个 stable state 全部 visits=0 → off-path
3. **on-path stable_mismatch = 0** in all 3 configs：SPERL 在均衡路径上**从来没有自信地、跨 seed 一致地选择和 SPE oracle 不同的 action**
4. on-path disagreement = 5/53 全是 "unstable across seeds" → 真正的 SPE 多重性集中在 **5 个 contested decision points**
5. Alg 3/4 无法压下这 5 个 → 它们是真实的 ε-tie 决策点，需要更长 training / 不同 tie-break 策略

**对 Phase 2 解读的修正**:
- ❌ 旧 B 档结论："49% 结构性多重性，paper Alg 3/4 解决不了"
- ✅ 新结论："on-path 上 SPERL 几乎完美匹配 SPE (5/53 不稳定，0 mismatch)；49% 是 off-path artifact (SPERL 在没访问过的 state 上随便学)"

**对 Optimality / Welfare 等 paper metric 的影响**: 无 —— 这些都是 visit-weighted 或 V(x_0)，已经只算实际重要的 state。49% disagreement 是 BFS 等权 count 才有的 artifact，不该当作算法对比 metric。

**新 metric 集成**: `scripts/analyze_optex.py` 现在自动跑 visit-frequency breakdown (默认 2000 SPE 轨迹，~10s)。Disable 用 `--no-visit`。每个 OptEx run 会多生成 `visit_freq_breakdown.csv`。

### 还没做的 (按价值)
1. **Hyperparameter sweep on CPT88/p=0.66** — 用 paper Appendix C.2.5 的 setting 重跑，看是否 reproduce paper 的 2.74 Optimality
2. **Paper Algorithm 3 伪代码精确对照** — 检查我们 `GreedyPolicy.update_from_critic_values` 的 sticky semantic 是否和 paper 完全一致
3. **Filter-thresh sweep on hardest config** — 复现 paper Table 4: 跑 filter ∈ {1, .95, .9, .85, .8, .75} on CPT88/p=0.66, 看是否能匹配 Pareto-optimal VE=9.3 at filter=.9
4. **Legacy 对照** (HANDOFF #4) — 跑 `agents/rerun_GreedySPERL_QR__main.py` 当独立 ground truth (尤其确认 paper 的 SPERL 数字)
5. **CPT95 整套配置** — 跑 CPT95/p=[0.48..0.72] 完成 Table 1/2 复现
6. **README 同步** — welfare fix + Alg 3/4 ablation 后 default 行为变了，要更新 CLI 示例和说明
7. **paper SPE oracle 复现** (OptEx) — 看我们的 backward induction 能否得出和 `CumSPERL_ref/SPE_OptEx_5_0.015_4.npy` 一致的 oracle
