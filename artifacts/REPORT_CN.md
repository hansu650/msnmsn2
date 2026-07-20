# EviPatch 实验结果报告

- Stage A verdict: **ABANDON**
- 本报告是实验、审计与可复现性汇总，不是论文正文。

## 完整性与审计

- 全量审计：**PASS**；训练 21/21，评估 63/63，差异项 0。
- 项目实验提交：`0bf46fc9d6cb00f70ffe110df8d48d4c3a592037`。
- APN upstream commit：`f0d6eeb7a2ee2d7c76475bf725b7ea25f98af3f4`。
- APN patch SHA-256：`00d8d59221d1580ee2b718365325bd69945dc2c103b0c23d7f93f9365e301746`。
- 审计覆盖 metric 重算、targets/target masks/sample IDs 不变、history-only shift、精确请求/实际删点、MCAR/burst 每患者变量匹配、跨变体 mask 一致、checkpoint/array hashes、CUDA 峰值与 provenance。

## Kill gate

- controlled-support：失败；APN MSE 0.249781，full MSE 0.250081，相对改善 -0.1199%，阈值 ≥ 5.0%。
- full vs raw_count macro：失败；full−raw MSE 0.0003818，95% CI [-0.0005169, 0.0013172]。
- full vs shuffled：失败；full−shuffled MSE 0.0013241，95% CI [0.0003639, 0.0023545]。
- full vs random_features：失败；full−random MSE 0.0011894，95% CI [-0.0000673, 0.0025224]。
- native 退化约束：通过；相对退化 0.3608%，上限 1.0%。
- 参数开销：通过；1.0745%，上限 < 5.0%。
- 100-step 时间开销：通过；3.1295%，上限 < 5.0%。

## 三种子汇总

下表数值为三种子 mean ± std；macro 对 native/MCAR/burst 等权平均。

| Variant | View | MSE | MAE |
|---|---|---:|---:|
| apn | burst | 0.335251 ± 0.00245 | 0.38256 ± 0.000656 |
| apn | mcar | 0.359313 ± 0.0009 | 0.401943 ± 0.00149 |
| apn | none | 0.312331 ± 0.000512 | 0.365334 ± 0.000863 |
| evipatch_full | burst | 0.336136 ± 0.00284 | 0.383445 ± 0.00192 |
| evipatch_full | mcar | 0.359115 ± 0.000407 | 0.401675 ± 0.00041 |
| evipatch_full | none | 0.313458 ± 0.000807 | 0.366317 ± 0.00115 |
| global_ratio | burst | 0.335051 ± 0.00306 | 0.383987 ± 0.000598 |
| global_ratio | mcar | 0.357698 ± 0.000472 | 0.401708 ± 0.00119 |
| global_ratio | none | 0.312529 ± 0.000534 | 0.366979 ± 0.000798 |
| random_features | burst | 0.334744 ± 0.00265 | 0.38238 ± 0.00171 |
| random_features | mcar | 0.358764 ± 0.00049 | 0.401845 ± 0.00142 |
| random_features | none | 0.311567 ± 0.000589 | 0.364417 ± 0.000776 |
| raw_count | burst | 0.33592 ± 0.00225 | 0.383891 ± 0.000657 |
| raw_count | mcar | 0.359028 ± 0.00157 | 0.401911 ± 0.00156 |
| raw_count | none | 0.313074 ± 0.00041 | 0.366791 ± 0.00125 |
| shuffled_evidence | burst | 0.334608 ± 0.00221 | 0.381539 ± 0.00132 |
| shuffled_evidence | mcar | 0.357636 ± 0.00112 | 0.39958 ± 0.00309 |
| shuffled_evidence | none | 0.311788 ± 0.000272 | 0.364508 ± 0.0019 |
| soft_mass | burst | 0.336498 ± 0.00316 | 0.383956 ± 0.00146 |
| soft_mass | mcar | 0.360298 ± 0.000577 | 0.402686 ± 0.000181 |
| soft_mass | none | 0.313162 ± 0.000265 | 0.366937 ± 0.000367 |
| apn | macro | 0.335632 ± 0.000656 | 0.383279 ± 0.00063 |
| evipatch_full | macro | 0.336236 ± 0.00105 | 0.383812 ± 0.000924 |
| global_ratio | macro | 0.335093 ± 0.00122 | 0.384225 ± 0.000464 |
| random_features | macro | 0.335025 ± 0.00103 | 0.382881 ± 0.000958 |
| raw_count | macro | 0.336008 ± 0.00102 | 0.384198 ± 0.000968 |
| shuffled_evidence | macro | 0.334677 ± 0.000483 | 0.381876 ± 0.00204 |
| soft_mass | macro | 0.336653 ± 0.00131 | 0.384526 ± 0.000323 |

## 失败原因分析

1. 受控 support 主指标没有达到机制成立所需的幅度：full 相对 APN 为 -0.1199%，不仅低于 +5%，方向还略为负。2,808 个冻结 pair 且三种子均远高于最小产量，因而不能归因于 pair 数不足。
2. full 没有显著优于简单 `raw_count`：macro 的 95% CI 跨 0。这不支持三维 evidence signature 相对单一计数带来稳定增益。
3. `shuffled_evidence` 反而显著优于 full（full−shuffled 的 CI 全部为正）。这说明当前收益不能归因于 evidence 与样本/变量的正确对应；额外投影宽度、随机正则化或优化噪声是更符合数据的解释，但本实验不能进一步区分这些机制。
4. `random_features` 也未被 full 显著击败；同时 full 的 native MSE 相对 APN 退化 0.3608%。因此 evidence quantity 在当前 APN/P12 设置中没有形成可重复的预测优势。
5. 全量审计为 PASS，排除了 target 被 shift 修改、不同变体删点不一致、计数不精确、artifact 损坏或 provenance 混杂等工程性解释。

## 决策

项目按预注册协议标记为 ABANDON；停止 HumanActivity、USHCN 与 t-PatchGNN 扩展，不新增或调参模型模块，只保留完整结果、失败分析与可复现包。
