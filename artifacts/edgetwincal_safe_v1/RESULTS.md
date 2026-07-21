# EdgeTwinCal-Safe 最终实验记录

最终结论：**ABANDON**。该路线未同时满足“两项新目标正向、任一目标退化不超过 1%、Safe 不劣于 Joint、两个必要模块均有消融支持”的预注册标准。因此停止修改方法，不运行 CPU/Jetson 延迟，不扩写论文，也不生成投稿 ZIP。

## 核心结论

- 北京空气质量：验证安全门通过，但 sealed test 上 Safe 的 MSE 比 APN 退化 0.484%；seed 2026 退化 1.282%，超过 1% 上限。
- Intel Lab：验证安全门失败，Safe 按协议精确回退 APN；五个 seed 的 Safe 预测与 APN 完全一致。
- 北京上 Safe 相对 Joint 的 MSE 收益为 -1.864%，95% CI [-3.033%, -0.542%]，即 Safe 显著差于 Joint。
- `SafeNoRobust` 和 `SafeNoBound` 两项必要模块均未得到 Holm 校正后的消融支持。

## 主结果

五种子均值 ± 样本标准差，train-normalized scale；末列为 pooled MSE 相对 APN 的收益。

| Dataset | Variant | MSE | MAE | MSE gain vs APN |
|---|---:|---:|---:|---:|
| Beijing Air | APN | 0.886283 ± 0.016151 | 0.614819 ± 0.007763 | 0.000% |
| Beijing Air | Joint | 0.874270 ± 0.010077 | 0.615354 ± 0.007603 | +1.355% |
| Beijing Air | Full | 0.874231 ± 0.010063 | 0.615297 ± 0.007592 | +1.360% |
| Beijing Air | Safe | 0.890569 ± 0.012421 | 0.612307 ± 0.008375 | -0.484% |
| Intel Lab | APN | 434.928686 ± 172.980359 | 18.427934 ± 4.035590 | 0.000% |
| Intel Lab | Joint | 427.355942 ± 119.068177 | 17.853691 ± 2.722263 | +1.741% |
| Intel Lab | Full | 445.485028 ± 148.556154 | 18.241124 ± 3.275408 | -2.427% |
| Intel Lab | Safe | 434.928686 ± 172.980359 | 18.427934 ± 4.035590 | 0.000% |

北京 Safe-vs-APN 的 MSE 收益 95% CI 为 [-1.289%, +0.208%]；MAE 收益为 +0.409%，95% CI [-0.043%, +1.045%]。Intel Safe-vs-APN 的 effect 和 CI 均精确为 0。

## 北京逐种子 APN/Safe MSE

| Seed | APN MSE | Safe MSE | Safe gain |
|---:|---:|---:|---:|
| 2024 | 0.874619 | 0.879667 | -0.577% |
| 2025 | 0.883606 | 0.886871 | -0.369% |
| 2026 | 0.877019 | 0.888259 | -1.282% |
| 2027 | 0.881718 | 0.886058 | -0.492% |
| 2028 | 0.914453 | 0.911991 | +0.269% |

完整的 2 × 5 × 8 逐种子结果见 `per_seed_results.csv`。

## 消融

| Comparator | Safe relative gain | 95% lower | Holm p | Role | Supported |
|---|---:|---:|---:|---|---|
| SafeNoBalance | -0.306% | -0.618% | 1.000 | diagnostic | no |
| SafeNoRobust | -0.424% | -0.766% | 1.000 | required | no |
| SafeNoBound | +0.553% | -0.734% | 1.000 | required | no |
| SafeNoGate | -0.380% | -0.708% | 1.000 | diagnostic | no |

## 交付文件

- `EdgeTwinCal-Safe_results.xlsx`：总览、主结果、逐种子、配对统计、消融。
- `aggregate_report.json`：50,000 次 shared crossed group × checkpoint paired bootstrap、Holm 校正和最终 gate。
- `summary_mean_std.csv`、`per_seed_results.csv`、`paired_statistics.csv`、`ablation_checks.csv`：机器可读结果。
- `gate_decision.json` 与两份 validation gate JSON：停止决策及验证审计。
- `SHA256SUMS.csv`：公开交付文件的大小和 SHA-256。

测试 campaign 已 sealed，80/80 manifests 完成且无 crash resume。公开目录不含原始数据、checkpoint、private predictions、test cache、样本 ID 或密钥。

## How to reproduce

```powershell
.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py train --dataset all --seed all --device cuda:0
.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py fit --dataset all --seed all --device cuda:0
.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py gate
.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py test --device cuda:0
.\.conda\envs\evipatch\python.exe code\scripts\run_edgetwincal_safe.py aggregate
```
