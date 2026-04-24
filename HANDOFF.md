# 交接文档 — 2026-04-24

验证 refactor 后的 generic SPERL 是否和 paper (`paperRef/MSci_MANUSCRIPT.pdf`) 对得上，并初步对比 OptEx 学到 policy 和 SPE oracle 的逐 state 差异。

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
