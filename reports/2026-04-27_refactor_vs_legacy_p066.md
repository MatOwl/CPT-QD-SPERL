# Refactor ↔ L-原 native 一致性确认 (CPT88/p=0.66/filter=0.9, 10 seeds)

**日期**: 2026-04-27
**结论**: 重构 generic SPERL 与原代码 (`agents/rerun_GreedySPERL_QR__main.py`) 在 e2e 分布上一致；四个 paper metric 全在 0.5σ 以内。Phase C 之前用 "L-仿" exec-slice driver 跑出的 −0.89 Optimality 数字被 fp-flip 污染，应作废。

## 背景

Phase A+B (unit isolation) 已经 bit-exact 验证：
- compute_CPT 三路径完全等价
- filter_quantiles：L-原 ≡ L-仿；重 与之差 cascading lb + signed dist (重忠于 paper §4 文字)
- QR critic update：L-原 ≡ L-仿；重 与 legacy 差仅 fp 累加噪声 (向量化更精确)

Phase C (e2e on CPT88/p=0.66/filter=0.9) 之前结论是 "重 (Opt 1.46) vs L-仿 (Opt −0.89)"，差距 1.4σ。

## L-仿 invalidation

单 seed sanity check (CPT88/p=0.42/seed=13/filter=0.85，对照 2023 年保存的 `agents/barberis/results/static/SPERL_27052023231700.csv` 历史 reference) 发现：
- L-原 native (current legacy code + S1+S2+S3 monitoring 加速) **bit-exactly** 复现 2023 ref
- L-仿 与 2023 ref 在 4/21 near-tie state 上 fp-flip 不一致
- 排除 S4 向量化嫌疑 (revert 后仍 4 个差异) → fp-flip 来自 driver-level (closure 注入 / namespace 重组)

→ Phase C 的 "L-仿 跑出 −0.89" 不能作为 "legacy 的真实数字"。需用 L-原 native 在同 config 重测。

## 实测 setup

L-原 native 通过 `scripts/run_legacy_native.py` exec `agents/rerun_GreedySPERL_QR__main.py` 自带 runner block。Active config 临时改为 refactor-matched：

```python
CPTParams = [(.88, .65, 2.25)]    # CPT88
pwinArr   = [.66]
p_filterArr = [.9]                 # paper §C.4 Pareto-optimal at this cell
treshRatio  = np.inf               # = refactor filter_accept_ratio=inf
ss_inverted = 1                    # sticky tie-break on
for n_batch in [5]:                # = refactor n_batch=5
    eps = .3
    train_num = 1500               # 15000 episodes via 2*n_batch*train_num
    seedArr = range(0, 10)         # 10 seeds
```

加速：S1 (eval_freq=2*n_batch*1500), S2 (n_eval_eps=100), S3 (record_quantiles → no-op)，仅影响 monitoring，不动训练。S4 已 revert (scalar QR loop)。

跑时 ~5 min/seed × 10 = ~50 min wall clock。

输出经 `scripts/parse_legacy_native.py` 提取 final policy 后，用 `lib/paper_eval.compute_paper_metrics` 对同一 SPE oracle 评估。

## 数据

| Metric | L-原 native (10s) | Refactor (10s) | Δ | Δ/σ_pool |
|---|---|---|---|---|
| Optimality | 0.45 ± 2.63 | 1.46 ± 1.71 | −1.01 | 0.45σ |
| Disagree | 5.60 ± 1.50 | 5.30 ± 1.55 | +0.30 | 0.20σ |
| VE | 27.00 ± 5.69 | 24.14 ± 5.45 | +2.86 | 0.51σ |
| SW | −67.46 ± 7.77 | −65.28 ± 6.78 | −2.18 | 0.30σ |

σ_pool = √((σ_legacy² + σ_refactor²) / 2)。

Refactor 数据来源：`runs/results_p066_10s_both_acceptInf/barberis_sperl_p0.66_cpt_a0.88_r0.65_l2.25/aggregate.json`。

## 结论

1. **重构与原代码 e2e 一致**。所有四个 metric 在 0.5σ 以内，远低于 1σ 阈值。重构对 fp 累加做的精度改进在 aggregate 层不显著。
2. **L-仿 −0.89 数字废弃**。Phase C 之前的 "重 vs legacy" 比较应替换为本次的 0.45 vs 1.46。
3. **Refactor verdict**: 重构忠于原算法。Tables 1/2 全 sweep 可继续用 refactor (`agents/sperl_qr_generic.py`) 推进，预期与 L-原 native 在 1σ 内一致。

## 与 paper 的差距

Refactor (1.46±1.71) 和 L-原 native (0.45±2.63) 都远低于 paper 在 CPT88/p=0.66/filter=0.9 上声称的 2.74±0.95。Paper Tables 数字大概率源于另一份 SPERL 实现 (legacy 自己也跑不出)；这一结论延续 Phase A+B+C 既有判断。

## 文件位置

- L-原 wrapper: `scripts/run_legacy_native.py`
- L-原 输出 parser: `scripts/parse_legacy_native.py`
- L-原 csv 输出 (10 seeds): `agents/barberis/results/static/SPERL_27042026{215842,...,223834}.csv`
- Refactor 数据: `runs/results_p066_10s_both_acceptInf/barberis_sperl_p0.66_cpt_a0.88_r0.65_l2.25/`
- 公共评估: `lib/paper_eval.py:compute_paper_metrics`
