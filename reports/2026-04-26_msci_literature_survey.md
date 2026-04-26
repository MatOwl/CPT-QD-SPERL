# Management Science 候选论文 / 环境 综述

**生成日期**: 2026-04-26
**目的**: 回应 AE report 第 6 点 — "Scope and external validity beyond the casino example"。Reviewer 要求加一个 operations-flavored example, 或把 casino 显式映射到 managerial decision。本文档梳理 MS / OR / 同等期刊里, 可能拿来做 CPT-SPERL env 或在 paper 里引用 mapping 的候选论文。

> **使用说明**: 每篇论文标 **CPT/SPERL 相性** ✅/⚠/❌, **Tier** S/A/B/C/D。Tier S = 强烈建议优先精读; A = abstract 值得读; B = 单期/静态, 不能直接做 env 但可作 narrative mapping; C = 邻近主题, 备用; D = 方法论 backbone, 必引但不做 env。

> **CPT/SPERL 技术约束** (筛选论文用):
> 1. **多期 / 序贯** (≥ 2 时间步, 否则 time-inconsistency 不存在)
> 2. **State 可枚举或可离散化** (兼容现有 `Featurizer`)
> 3. **Action 离散** (binary / few-action 最理想; 连续需 discretize)
> 4. **Reward 含 reference / loss aversion / probability weighting**, 或可改写
> 5. **有 closed-form / numerical / experimental benchmark** 用于 cross-validate SPE oracle
> 6. **State space 不能太大** (现有 SPERL 是 tabular; OptEx 已是 ~50k states 极限)

---

## 目录 (按 MS 子领域)

1. [Behavioral OM — Newsvendor / Inventory](#1-behavioral-om--newsvendor--inventory)
2. [Project Management — Multistage Continue/Abandon](#2-project-management--multistage-continueabandon) ⭐
3. [Revenue Management — Dynamic Pricing](#3-revenue-management--dynamic-pricing)
4. [Consumption / Portfolio / Life-cycle](#4-consumption--portfolio--life-cycle) ⭐
5. [Optimal Stopping / Liquidation / Disposition](#5-optimal-stopping--liquidation--disposition)
6. [Healthcare Operations — Treatment / Screening / Transplant](#6-healthcare-operations--treatment--screening--transplant)
7. [Insurance / Annuities under PT](#7-insurance--annuities-under-pt)
8. [Capacity Expansion / Real Options](#8-capacity-expansion--real-options)
9. [Energy Markets / Demand Response (RL-side)](#9-energy-markets--demand-response-rl-side)
10. [Ridesharing / Online Platforms (RL-side)](#10-ridesharing--online-platforms-rl-side)
11. [Disaster / Humanitarian Logistics](#11-disaster--humanitarian-logistics)
12. [Theoretical — PT in Dynamic Context (必引)](#12-theoretical--pt-in-dynamic-context-必引)
13. [Methodology — CPT-RL (必引)](#13-methodology--cpt-rl-必引)
14. [推荐下载顺序](#推荐下载顺序)
15. [Paper → env 转化 checklist](#paper--env-转化-checklist)

---

## 1. Behavioral OM — Newsvendor / Inventory

### Schweitzer & Cachon (2000) — **Tier B**, 单期实验
- **Title**: Decision Bias in the Newsvendor Problem with a Known Demand Distribution: Experimental Evidence
- **Authors**: Maurice E. Schweitzer, Gérard P. Cachon
- **Year**: 2000
- **Venue**: *Management Science* 46(3):404–420
- **DOI**: `10.1287/mnsc.46.3.404.12070`
- **核心**: 行为 OM 开山之作。实验证明 newsvendor 系统性偏离 risk-neutral 最优, 但**不是** PT 偏好可解释 — 反而像是 anchor + ex-post inventory error aversion。
- **CPT/SPERL 相性**: ❌ 单期, 不能做 env。但**必须在 paper 里引用** — 是"managerial decision under behavioral preferences"的 textbook 例子。

### Long & Nasiry (2015) — **Tier B**, 单期理论
- **Title**: Prospect Theory Explains Newsvendor Behavior: The Role of Reference Points
- **Authors**: Xiaoyang Long, Javad Nasiry
- **Year**: 2015
- **Venue**: *Management Science* 61(12):3009–3012 (Note)
- **DOI**: `10.1287/mnsc.2014.2050`
- **核心**: 推翻 Schweitzer & Cachon 的"PT 不能解释"结论 — 关键在 reference point 取 "salient payoffs 的加权平均", 不是 status quo。**PT 可以解释 newsvendor 偏差**。
- **CPT/SPERL 相性**: ❌ 单期。但若我们想做"多期 newsvendor under CPT", 这篇给出 reference-point 的标准设法。可作 env 设计参考。

### Pricing and Inventory Decisions Facing Strategic Customers (2025)
- **Title**: Pricing and Inventory Decisions Facing Strategic Customers with Behavioral Preferences
- **Year**: 2025
- **Venue**: *Journal of Systems Science and Complexity*
- **核心**: 两期 inventory + pricing 问题, 客户有 reference-dependence + loss aversion + risk preferences。
- **CPT/SPERL 相性**: ⚠ 两期可能太短; 期刊偏弱。**Tier C**。

---

## 2. Project Management — Multistage Continue/Abandon ⭐

### Long, Nasiry, Wu (2020) — **Tier S** ⭐⭐⭐ 首推
- **Title**: A Behavioral Study on Abandonment Decisions in Multistage Projects
- **Authors**: Xiaoyang Long, Javad Nasiry, Yaozhong Wu
- **Year**: 2020
- **Venue**: *Management Science* 66(5):1999–2016
- **DOI**: `10.1287/mnsc.2018.3270`
- **核心**: T-stage 项目, 每期看 project value 演化 → 决定 continue / abandon。实验数据用 **reference point + sunk cost + status quo** 三因素模型解释。发现 abandonment 倾向延迟、路径依赖、middle-stage 最容易 abandon。
- **CPT/SPERL 相性**: ✅✅✅ 完美匹配。
  - 多期 ✓ (T 期)
  - Discrete action ✓ (continue/abandon)
  - Reference-dependent reward ✓ (论文显式建模)
  - State 简单 ✓ (project value + stage)
  - 有 experimental benchmark ✓
- **为什么 Tier S**: 这就是 Barberis-casino 的 OM 版本 — 同样 stop/continue, 同样 reference-dependent, 同样多期。直接套现有 SPERL 几乎零阻力。可作 paper 里**主推的"operations-flavored example"**。

---

## 3. Revenue Management — Dynamic Pricing

### Popescu & Wu (2007) — **Tier A**
- **Title**: Dynamic Pricing Strategies with Reference Effects
- **Authors**: Ioana Popescu, Yaozhi Wu
- **Year**: 2007
- **Venue**: *Operations Research* 55(3):413–429
- **DOI**: `10.1287/opre.1070.0393`
- **核心**: Foundational 的 reference-dependent 多期定价模型。Reference price 是过去价格的 EWMA, 客户对 gain/loss 不对称反应。
- **CPT/SPERL 相性**: ⚠ 连续 action (price), 需要 discretize。但很标准的多期 MDP, reward 显式 reference-dependent。**做成 env 可行, 适合作 Barberis 之外的第二个核心 env**。

### Nasiry & Popescu (2011) — **Tier A**
- **Title**: Dynamic Pricing with Loss-Averse Consumers and Peak-End Anchoring
- **Authors**: Javad Nasiry, Ioana Popescu
- **Year**: 2011
- **Venue**: *Operations Research* 59(6):1361–1368
- **DOI**: `10.1287/opre.1110.0952`
- **核心**: Loss-averse buyers + peak-end anchoring 的多期定价。卖家最优策略要么收敛到 steady-state, 要么**循环**。
- **CPT/SPERL 相性**: ⚠ 类似上篇。**循环结构**特别有意思 — 直接呼应 AE 第 1 点 (QDRL operator 是否 cycling) 的担心。**强烈建议引用**, 即使不做 env。

### den Boer & Keskin (2022) — **Tier A**
- **Title**: Dynamic Pricing with Demand Learning and Reference Effects
- **Authors**: Arnoud V. den Boer, N. Bora Keskin
- **Year**: 2022
- **Venue**: *Management Science* 68(10):7112–7130
- **DOI**: `10.1287/mnsc.2021.4234`
- **核心**: Reference-dependent demand + simultaneous demand-parameter learning。
- **CPT/SPERL 相性**: ⚠ 学习 + 控制双重, 比 SPERL 复杂; 但 AE references 已显式列了 den Boer & Keskin (2020), 这篇是其 dynamic 升级版, **必引**。

### Hu, Nasiry, Liu (2017) — **Tier A** (同领域)
- **Title**: Are Markets with Loss-Averse Consumers More Sensitive to Losses?
- **Authors**: Zhenyu Hu, Javad Nasiry
- **Venue**: *Management Science* 63(5):1372-1389 (估计, 待精确确认)
- **DOI 候选**: `10.1287/mnsc.2016.2678`
- **核心**: Aggregation 后 loss-averse 客户的市场需求未必更对 loss 敏感。
- **CPT/SPERL 相性**: ⚠ 偏理论, 不必做 env, 可作 narrative。

---

## 4. Consumption / Portfolio / Life-cycle ⭐

### van Bilsen, Laeven, Nijman (2020) — **Tier S** ⭐⭐
- **Title**: Consumption and Portfolio Choice Under Loss Aversion and Endogenous Updating of the Reference Level
- **Authors**: Servaas van Bilsen, Roger J. A. Laeven, Theo E. Nijman
- **Year**: 2020
- **Venue**: *Management Science* 66(9):3927–3955
- **DOI**: `10.1287/mnsc.2019.3393`
- **核心**: Life-cycle 消费 + portfolio 选择, **reference level 内生更新** (loss aversion 来源)。Loss-averse 个体: 跌了延后 cut consumption; conservative 在正常态、aggressive 在好/坏态。Welfare loss 相对 CRRA 超 10%。
- **CPT/SPERL 相性**: ✅✅ 几乎完美。
  - 多期 ✓ (life-cycle)
  - Reference-dependent ✓ (内生)
  - Time-inconsistency 来源清楚 ✓ (endogenous reference)
  - 有 closed-form / numerical ✓
  - ⚠ Continuous-time, 需 discretize; state 含 wealth (连续) 需 bin
- **为什么 Tier S**: Managerial 味最足 (退休规划 = 个人金融决策). MS 期刊。**比 casino 更"日常 managerial decision"**。

### He & Zhou (2011) — 已被 paper 引用
- **Title**: Portfolio Choice under Cumulative Prospect Theory: An Analytical Treatment
- **Venue**: *Management Science* 57(2):315–331
- **DOI**: `10.1287/mnsc.1100.1269`
- **核心**: CPT-portfolio 闭式分析。
- **相性**: 已是 reference, 可对照其 numerical example 做 env。**Tier B/C** 备用。

---

## 5. Optimal Stopping / Liquidation / Disposition

### Henderson (2012) — **Tier S** (backup for Long-Nasiry-Wu)
- **Title**: Prospect Theory, Liquidation, and the Disposition Effect
- **Authors**: Vicky Henderson
- **Year**: 2012
- **Venue**: *Management Science* 58(2):445–460
- **DOI**: `10.1287/mnsc.1110.1468`
- **核心**: PT 偏好下单 asset 的 optimal stopping → stop-loss + gain-take 阈值结构。
- **CPT/SPERL 相性**: ✅ 多期, 离散化容易, PT 显式。但和 casino 主题距离不远 (asset selling)。**作 Tier S 第三选择**。

### Henderson, Hobson, Tse (2017) — **Tier B** (theoretical mapping)
- **Title**: Randomized Strategies and Prospect Theory in a Dynamic Context
- **Authors**: Vicky Henderson, David Hobson, Alex S. L. Tse
- **Year**: 2017
- **Venue**: *Journal of Economic Theory* 168:287–300
- **DOI**: `10.1016/j.jet.2017.01.007`
- **核心**: 修正 Ebert-Strack — 加 randomization 后 PT agent 才会停。
- **相性**: 不做 env, 但 related-work 应引。

### Barberis & Xiong (2009) — **Tier C** (背景文献)
- **Title**: What Drives the Disposition Effect? An Analysis of a Long-Standing Preference-Based Explanation
- **Year**: 2009
- **Venue**: *Journal of Finance* 64(2):751–784
- **DOI**: `10.1111/j.1540-6261.2009.01448.x`
- **核心**: Annual gain/loss vs realized gain/loss 在解释 disposition effect 上的差异。
- **相性**: 不做 env, 但 PT-finance 经典。

---

## 6. Healthcare Operations — Treatment / Screening / Transplant

### Sandıkçı, Maillart, Schaefer, Alagoz, Roberts (2008) — **Tier A**
- **Title**: Estimating the Patient's Price of Privacy in Liver Transplantation
- **Year**: 2008
- **Venue**: *Operations Research* 56(6):1393–1410
- **DOI**: `10.1287/opre.1080.0648`
- **核心**: 患者面对 cadaveric liver offer 的 accept/reject MDP。State = (患者健康, liver 质量, waiting list rank)。最优策略是 control-limit (健康越差阈值越低)。
- **CPT/SPERL 相性**: ✅ 完美 MDP 结构 (discrete state + binary action), 但**原 paper 用 expected QALY, 不是 CPT**。需要我们 retrofit CPT 偏好 (e.g., "loss = 错过这次 offer 死掉"). 概念上很合理 — 患者面对 organ offer 时确有 loss aversion。
- **为什么 Tier A**: state-action 结构干净, 直接套 SPERL 容易; 但需要论证为什么把 CPT 加进来。

### Steimle & Denton (2017) — **Tier B**
- **Title**: Markov Decision Processes for Screening and Treatment of Chronic Diseases
- **Year**: 2017
- **Venue**: Book chapter in *Markov Decision Processes in Practice*
- **DOI**: `10.1007/978-3-319-47766-4_6`
- **核心**: 慢性病 (糖尿病、心血管) 的 screening + treatment MDP 综述。
- **相性**: 综述 → 找具体 case study。Tier B 备用。

### Alagoz et al. — Liver transplant 系列
- 多篇 *Management Science* / *Operations Research* paper, 主题 living-donor vs cadaveric liver acceptance。
- 例: Alagoz, Maillart, Schaefer, Roberts (2007) "Choosing Among Living-Donor and Cadaveric Livers", *MS* 53(11):1702–1715, DOI `10.1287/mnsc.1070.0726`
- **相性**: 同 Sandıkçı 系列, ✅ 结构。

---

## 7. Insurance / Annuities under PT

### Babcock (2020) — **Tier C**
- **Title**: Behavioral weather insurance: Applying cumulative prospect theory to agricultural insurance design under narrow framing
- **Authors**: Bruce A. Babcock
- **Year**: 2020
- **Venue**: *PLoS ONE* 15(5):e0232267
- **DOI**: `10.1371/journal.pone.0232267`
- **核心**: 多年农业保险合同, 显式 CPT 偏好 + narrow framing → 推导 multi-year 合同 + lump-sum premium 设计。
- **相性**: ⚠ Venue 偏弱 (PLoS ONE 不是 OM 顶刊), 但内容多期 + CPT。备用。

### Gottlieb (2012) — **Tier C**
- **Title**: Prospect Theory, Life Insurance, and Annuities (Working Paper)
- **Authors**: Daniel Gottlieb
- **Venue**: Wharton WP (后续可能发表; 检查最新版)
- **核心**: PT 解释为什么人们买太多寿险、太少年金 (annuity puzzle)。
- **相性**: 静态居多, Tier C。

---

## 8. Capacity Expansion / Real Options

### Anand & Girotra (2007) — **Tier B**
- **Title**: Capacity Investment Under Postponement Strategies, Market Competition, and Demand Uncertainty
- **Year**: 2007
- **Venue**: *Management Science*
- **DOI**: `10.1287/mnsc.1080.0940`
- **核心**: Capacity expansion timing, 含 postponement + 竞争。
- **相性**: ⚠ 模型偏 game theory + real options, state space 复杂; 不容易直接做 SPERL env。Tier B 仅作 narrative reference.

### Hagspiel et al. (2020) — **Tier C**
- **Title**: Time to build: A real options analysis of port capacity expansion investments under uncertainty
- **Venue**: *Transportation Research Part E*
- **核心**: 风险厌恶 + 不确定性下 port capacity 扩张 timing。
- **相性**: ⚠ 偏 transportation; 模型连续。Tier C。

---

## 9. Energy Markets / Demand Response (RL-side)

### Distributional RL for Energy Arbitrage (2024)
- **Title**: Distributional Reinforcement Learning-based Energy Arbitrage Strategies in Imbalance Settlement Mechanism
- **Year**: 2024
- **Venue**: arXiv:2401.00015
- **核心**: 直接用 distributional RL 做电力市场 arbitrage, 含 risk-averse 偏好。
- **相性**: ⚠ Venue 是 arxiv, 但**直接是 distributional RL + risk-averse + OM-flavored** application。可作 paper 里 "applications of distributional RL beyond gambling" 的引用。**Tier B**。

### 综述 Reinforcement Learning in Deregulated Energy Markets
- **Title**: Reinforcement learning in deregulated energy market: A comprehensive review
- **Venue**: *Applied Energy* (2023)
- **核心**: RL 在能源市场 bidding / DR 的综述。
- **相性**: 综述, 用来找具体 case。Tier C。

---

## 10. Ridesharing / Online Platforms (RL-side)

### Spatio-Temporal Pricing for Ridesharing Platforms (2022)
- **Title**: Spatio-Temporal Pricing for Ridesharing Platforms
- **Venue**: *Operations Research*
- **DOI**: `10.1287/opre.2021.2178`
- **核心**: Ridesharing 平台多期 + 空间动态定价。
- **相性**: ⚠ State 巨大 (空间分区), 现有 tabular SPERL 处理不了。但作为"RL 在 OM 里的应用"narrative 引用。**Tier C**。

### RL for Ridesharing 综述
- Qin et al. (2022) *Transportation Research Part C* — Reinforcement Learning for Ridesharing: An Extended Survey
- 用来找小规模 sub-problem 做 env。**Tier C**。

---

## 11. Disaster / Humanitarian Logistics

- 主流 humanitarian logistics 论文多是 **2-stage stochastic programming** 而非 MDP, **不适合**直接做 SPERL env。
- 但可在 paper 里以 "managerial decision under risk-aversion / CVaR" 引用一两篇, 作 mapping example。
- 候选: Inventory-Allocation Distribution Models for Postdisaster Humanitarian Logistics (Holguín-Veras et al. 2016 *Transportation Science*) DOI `10.1287/trsc.2014.0565`。
- **整体 Tier C/D**。

---

## 12. Theoretical — PT in Dynamic Context (必引)

### Ebert & Strack (2015) — **Tier S** (backbone, 不做 env 但必读必引)
- **Title**: Until the Bitter End: On Prospect Theory in a Dynamic Context
- **Authors**: Sebastian Ebert, Philipp Strack
- **Year**: 2015
- **Venue**: *American Economic Review* 105(4):1618–1633
- **DOI**: `10.1257/aer.20130896`
- **核心**: 证明 naive PT agent 在足够丰富的动态投资空间里**永不停止** — probability weighting + 朴素 (无 commitment) → 病态。
- **CPT/SPERL 相性**: ❌ 不做 env, 但**必引**。直接对应 AE 第 1, 2, 4 点 (cycling, internal consistency, equilibrium added value)。证明 "为什么需要 SPE / sophisticated agent"。

### Ebert & Strack — Never, Ever Getting Started
- **Title**: Never, Ever Getting Started: On Prospect Theory Without Commitment
- **Year**: SSRN WP, 2016 (后续发表待查)
- **SSRN ID**: 2765550
- **核心**: 上篇的姊妹篇 — naive PT + 无 commitment → 永不开始投资。

### Bjork & Murgoci (2010) — 已在 AE references
- **Title**: A general theory of Markovian time inconsistent stochastic control problems
- **Venue**: SSRN abstract 1694759 (后续 *Finance and Stochastics* 20(4):961–1005, 2016)
- **DOI**: `10.1007/s00780-016-0321-x`
- **核心**: Time-inconsistent stochastic control 的 SPE 一般理论。**必引**。

### He & Zhou (2022) — 已被 paper 引用
- **Title**: Who Are I: Time Inconsistency and Intrapersonal Conflict and Reconciliation
- **Venue**: 书章 *Stochastic Analysis, Filtering, and Stochastic Optimization* (Springer)
- **核心**: 综述 intrapersonal game / SPE / precommitment 概念。

---

## 13. Methodology — CPT-RL (必引)

### Prashanth, Jie, Fu, Marcus, Szepesvári (2016)
- **Title**: Cumulative Prospect Theory Meets Reinforcement Learning: Prediction and Control
- **Year**: 2016
- **Venue**: *ICML 2016* (PMLR 48)
- **arXiv**: `1506.02632`
- **核心**: CPT-RL 第一篇方法论文。**必引**, 检查 paper 是否已引。

### Lin et al. (2024 / 2025) — Policy Gradients for CPT in RL
- **Title**: Policy Gradients for Cumulative Prospect Theory in Reinforcement Learning
- **arXiv**: `2410.02605`
- **核心**: CPT 目标的 policy gradient 算法。
- **相性**: 方法论参考, 不做 env。**Tier D**。

### Risk-aware MDPs Using CPT (2025)
- **arXiv**: `2505.09514`
- **核心**: 最新 CPT-MDP 工作。
- **相性**: Future work / discussion 提到。

### Subgame-perfect equilibrium recursive stochastic control (2023)
- **Title**: Subgame-perfect equilibrium strategies for time-inconsistent recursive stochastic control problems
- **arXiv**: `2302.00471`
- **Venue**: *Journal of Mathematical Analysis and Applications* (2023)
- **DOI**: `10.1016/j.jmaa.2023.127268`
- **相性**: SPE 方法论, 可引。

### Lesmana & Pun (2025) — paper 已引
- **Title**: A Subgame Perfect Equilibrium Reinforcement Learning Approach to Time-Inconsistent Problems
- **Venue**: *SIAM Journal on Financial Mathematics*
- **DOI**: `10.1137/23M1594510`
- **相性**: 这就是 SPERL 原文。

---

## 推荐下载顺序

按 ROI 排序 (优先下 Tier S, 然后 Tier A 中和 SPERL 相性最高的):

| 优先级 | 论文 | 用途 |
|---|---|---|
| 1 ⭐ | Long, Nasiry, Wu (2020) MS — Multistage Abandonment | **首选 env candidate**, casino 的 OM 双胞胎 |
| 2 ⭐ | van Bilsen, Laeven, Nijman (2020) MS — Loss-averse Consumption | 第二个 env, managerial 味最浓 |
| 3 | Ebert & Strack (2015) AER — Until the Bitter End | 理论 backbone, 必引, 回应 AE 1/2/4 点 |
| 4 | Henderson (2012) MS — PT Liquidation | env backup |
| 5 | Sandıkçı, Maillart, Schaefer, Alagoz, Roberts (2008) OR — Liver Transplant | healthcare env candidate, 干净 MDP 结构 |
| 6 | Nasiry & Popescu (2011) OR — Loss-averse Pricing | 引用价值 (cycling); env 可选 |
| 7 | Popescu & Wu (2007) OR — Reference-effect Pricing | env candidate |
| 8 | den Boer & Keskin (2022) MS — Pricing + Learning | AE 已点名相关作者, 必引 |
| 9 | Long & Nasiry (2015) MS — PT explains newsvendor | 单期, 但可启发"多期 newsvendor" 自建 env |
| 10 | Schweitzer & Cachon (2000) MS — Newsvendor 实验 | 行为 OM 必引经典 |
| 11 | Henderson, Hobson, Tse (2017) JET — Randomized stopping | related work |
| 12 | Prashanth et al. (2016) ICML — CPT-RL | 方法论必引 |

## Paper → env 转化 checklist

下载到原文后, 对每篇执行以下 checklist:

- [ ] **多期 / 序贯**: ≥ 2 时间步, T 已知 (finite horizon) 或可截断 (infinite)
- [ ] **State 可枚举或可离散化**: 与现有 `Featurizer` 接口兼容; |S| ≤ ~50k (现 OptEx 上限)
- [ ] **Action 离散**: ≤ ~10 actions; 连续需 discretize
- [ ] **Reward 已含 reference 或 PT 结构**: 否则要论证为什么 retrofit CPT 合理
- [ ] **Time-inconsistency 来源清楚**: probability weighting / endogenous reference / hyperbolic / habit 任一
- [ ] **有 closed-form / numerical / experimental benchmark**: 用于 cross-validate SPERL 学到的策略
- [ ] **Backward induction SPE oracle 可写**: 否则没法跑 paper-eval 4 metrics

## 现有 codebase 复用点

新 env 要落到这几个文件:

- [lib/envs/registry.py](lib/envs/registry.py) — 注册新 env
- [lib/envs/featurizers.py](lib/envs/featurizers.py) — `Featurizer` (含 `cpt_offset` + state enumerator)
- [lib/envs/barberis_casino.py](lib/envs/barberis_casino.py) — 多期离散 env 最简模板
- [lib/envs/barberis_spe.py](lib/envs/barberis_spe.py) — backward induction SPE oracle 模板
- [lib/cpt.py](lib/cpt.py) — `compute_CPT` 直接复用
- [agents/sperl_qr_generic.py](agents/sperl_qr_generic.py) — env-agnostic, 不动

每个新 env 至少要: `env class + featurizer + SPE oracle (backward induction)`, 才能跑 `agents/run_paper_eval.py`。

## 文件位置

本文档位于: `C:\Users\Jingxiang Tang\.claude\plans\readme-md-management-snappy-panda.md`

Plan mode 退出后建议复制到 `C:\Users\Jingxiang Tang\FNN\CPT-QD-SPERL\reports\2026-04-26_msci_literature_survey.md` 以纳入 git。
