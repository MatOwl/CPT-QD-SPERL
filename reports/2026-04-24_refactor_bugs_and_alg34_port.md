# 2026-04-24 — Refactor 引入的 bug 修复 + Algorithm 3/4 port

## 任务

验证 refactor 后的 generic SPERL 是否和 paper (`paperRef/MSci_MANUSCRIPT.pdf`) 对得上，并初步对比 OptEx 学到 policy 和 SPE oracle 的逐 state 差异。

## (A) Refactor 引入的 3 处同源 bug — 已修

全部是"MDP reward 是 wealth 增量，但 CPT 要在终态财富 `z + Σr` 上评估"这一 Barberis 特有语义在 refactor 时漏掉。

| 文件 | 症状 | 修法 |
|---|---|---|
| [lib/envs/barberis_spe.py:51](../lib/envs/barberis_spe.py:51) | SPE oracle 全给 "exit"（任何 CPT 参数下） | `rewards.append(total + z)` |
| [lib/paper_eval.py](../lib/paper_eval.py) `rollout_cpt_from_state` | V^π(x) 在 z≠0 状态偏低 | 新增 `initial_offset_for_state` callback |
| [agents/sperl_qr_generic.py](../agents/sperl_qr_generic.py) `QRCritic.cpt_value*` | greedy 永远不在 loss-domain 翻转到 play；backward propagation 断掉 | CPT 评估前把 offset 加到 quantiles |

架构抽象：新增 `Featurizer.cpt_offset(obs)` hook（[lib/envs/featurizers.py](../lib/envs/featurizers.py)）。Barberis 返回 `float(z)`，OptEx 默认 `0.0`。

**验证**（Paper §4 / Appendix C.2.5 主表）：

| 配置 | 我们的 Optimality | Paper | 结论 |
|---|---|---|---|
| CPT88 (α=0.88, δ=0.65, λ=2.25), p=0.72 | 7.60 ± 0.70 | 7.91 ± 1.64 | ✓ 1σ 内 |
| CPT95 (α=0.95, δ=0.5, λ=1.5), p=0.60 | 0.00 ± 0.00 | 0.0 / 0.0 | ✓ 完美 |

SPE oracle 单点检查：CPT88/p=0.72, V_SPE(0,0) = 7.78（paper 8.07）✓

## (B) 论文 Algorithm 3 & 4 的 port

Legacy 脚本里有两个使 SPERL 稳定收敛到 reference SPE 的机制，generic path 之前没有。今天都 port 了。

### Algorithm 3 — Consistent tie-break / is-better guard
- 位置：[agents/sperl_qr_generic.py](../agents/sperl_qr_generic.py) `GreedyPolicy.update_from_critic_values`
- 新参数 `sticky=True`：只有当新 argmax 的 CPT **严格大于** 旧 policy 动作的 CPT 时才翻转（避免 flat-landscape 的 argmax 抖动）
- 新参数 `tie_thresh>0`：在 max CPT 的 `tie_thresh` 范围内均匀随机（避免 oracle vs SPERL 的符号 tie-break 差异）
- First-visit (prev==-1) 无条件接受

### Algorithm 4 — Quantile filter + acceptance gate
- 位置：[agents/sperl_qr_generic.py](../agents/sperl_qr_generic.py) 新 `filter_quantiles` helper + `QRCritic._cpt_from_quantiles`
- 滤波逻辑：计算 quantile 间 gap，取 `filter_thresh` 分位数作阈值；gap 超阈或负（crossing quantile）的位置被 snap 到最近 valid 邻居
- 接受门：若 filtered CPT 与 unfiltered 的相对差 > `filter_accept_ratio`，回退到 unfiltered。legacy/paper 默认 `filter_thresh=0.75, accept_ratio=inf`（永远信 filter），paper Appendix C.2.5 明确给过 `accept_ratio=0.5`

### CLI 新 flag
（[agents/run_paper_eval.py](../agents/run_paper_eval.py) 和 [agents/run_experiments.py](../agents/run_experiments.py) 都加了）：
```
--sticky-policy                  # Alg 3 on
--tie-thresh FLOAT               # default 0 (exact ties only)
--filter-thresh FLOAT            # Alg 4 filterTresh; default None = off
--filter-accept-ratio FLOAT      # Alg 4 treshRatio; default inf = trust filter
```

所有默认保持向后兼容（全 off），现有 baseline 跑法不变。

## 当时的初步 OptEx 观察 (后被 Phase 4 修正)

> ⚠️ 此节结论已被 [2026-04-26_optex_visit_frequency.md](2026-04-26_optex_visit_frequency.md) 修正为 off-path artifact，保留作为历史记录。

修完所有 bug 后，OptEx σ=0.015 仍有 ~49% state-level policy disagreement。加训练量（500→3000 eps, 6×）disagreement 纹丝不动。

| 指标 | σ=0.015 (493 states) | σ=0.029 (1012 states) |
|---|---|---|
| Policy disagree | 241/493 = 48.9% | 465/1012 = 46.0% |
| Cross-seed stability | 87.8% | 76.1% |
| mean \|Δv\| at disagreements | 0.011 | 0.009 |
| mean \|Δa\| | 2.13 / 11 | 2.02 / 11 |

当时怀疑 OptEx 上有结构性 SPE 多重性。

## 当时遗留问题 (后续进展)

1. ~~**Welfare 量级差距**~~ — 已在 [2026-04-26_welfare_and_paper_tables.md](2026-04-26_welfare_and_paper_tables.md) 解决
2. **README 没同步** — 已在 2026-04-26 sync
3. **更多 paper 配置没跑** — 已在 [2026-04-26_welfare_and_paper_tables.md](2026-04-26_welfare_and_paper_tables.md) 跑了 CPT88/p=0.72, p=0.66
4. **legacy 没跑过对照** — 仍未做
