# 2026-04-26 — OptEx 49% disagreement 是 off-equilibrium-path artifact (Phase 4)

## 触发问题

Session 末用户提出关键问题:
> 不匹配的 184 个 state 是不是本来就不太被 visit？

如果 disagreement 集中在 SPE 路径下访问频率很低的 state，那 49% 就是 "off-equilibrium-path 的无关紧要选择"——SPERL 在那些 state 学不准是因为 visit 太少，但 SPE 实际玩的时候根本不去那里。

## 方法

跑 SPE oracle 2000 条轨迹做 visit-frequency 拆分:

1. 用 `lib.envs.optex_spe.SPEOracle` 加载 paper SPE
2. 从 x_0 跑 N=2000 条 trajectory，计每个 (t, state) 的 visit count
3. 把 [Phase 2](2026-04-26_alg34_validation.md) 的 per_state_values.csv 按 BFS 节点 id join 进来
4. Cross-tab: visit_count × disagreement category (stable_match / stable_mismatch / unstable)

## 结果

| Config (OptEx σ=0.015) | BFS disagree (raw) | On-path disagree | stable_mismatch on-path | unstable on-path |
|---|---|---|---|---|
| Baseline (off) | 248/493 (50.3%) | **5/53 (9.4%)** | **0** | 5 |
| Sticky only | 249/493 (50.5%) | **5/53 (9.4%)** | **0** | 5 |
| Sticky + Filter | 249/493 (50.5%) | **5/53 (9.4%)** | **0** | 5 |

## 关键发现

1. **SPE 实际玩 2000 局只触及 53 个 distinct (t, state)**（vs BFS reachable 493）。其余 440 个 state 都是 SPERL 没在 training 里见过（或仅以 ε-greedy 探索小概率到达）→ 学到的策略基本是先验默认值
2. **244 + 184 个 stable state 全部 visits=0** → off-path
3. **on-path stable_mismatch = 0** in all 3 configs：SPERL 在均衡路径上**从来没有自信地、跨 seed 一致地选择和 SPE oracle 不同的 action**
4. on-path disagreement = 5/53 全是 "unstable across seeds" → 真正的 SPE 多重性集中在 **5 个 contested decision points**
5. Alg 3/4 无法压下这 5 个 → 它们是真实的 ε-tie 决策点，需要更长 training / 不同 tie-break 策略

## 对 Phase 2 解读的修正

- ❌ 旧 B 档结论："49% 结构性多重性，paper Alg 3/4 解决不了"
- ✅ 新结论："on-path 上 SPERL 几乎完美匹配 SPE (5/53 不稳定，0 mismatch)；49% 是 off-path artifact (SPERL 在没访问过的 state 上随便学)"

## 对 Optimality / Welfare 等 paper metric 的影响

无 —— 这些都是 visit-weighted 或 V(x_0)，已经只算实际重要的 state。49% disagreement 是 BFS 等权 count 才有的 artifact，不该当作算法对比 metric。

## 工程产物

- 一次性探针 [scripts/visit_freq_check.py](../scripts/visit_freq_check.py)（独立脚本）
- 已并入 [scripts/analyze_optex.py](../scripts/analyze_optex.py): 自动跑 visit-frequency breakdown (默认 2000 SPE 轨迹，~10s)
  - Disable: `--no-visit`
  - 调整轨迹数: `--n-traj N`
  - 每个 OptEx run 多生成 `visit_freq_breakdown.csv` (每节点 cat × visit count)
