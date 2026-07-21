# EdgeTwinCal 论文讲解

## 一段话摘要

不规则传感器预测是边缘监控和数据驱动数字孪生的基础能力。APN
通过按通道独立的自适应时间 patch 和浅层共享解码器实现了高效预测，
但在模型冻结后，它没有显式路径利用两类剩余信息：不同传感器 latent
中尚未被共享解码器读出的局部残差，以及其他传感器预测中暴露的跨通道
关系。EdgeTwinCal 保留整个 APN 不变，并把修正拆成两个位置不同的模块：
Sensor Latent Residual Head (SLRH) 从冻结 latent 读取传感器和预测步特定的
局部修正，Cross-Forecast Graph (CFG) 再从其他传感器的中间预测中修正剩余
残差。三个既有 APN checkpoint 上，PhysioNet 2012 MSE 从
0.312331 降至 0.309058，相对改善 1.048%；每个 checkpoint 的闭式适配在
缓存特征上不超过 1.42 秒，并且不更新任何 APN 参数。

## 论文真正解决的问题

APN 的优势是简单高效。其公开架构先逐变量运行 TAPA 和 query
aggregation，再用浅层解码器预测。这个设计本身不是错误，也不能写成
“APN 作者承认的缺陷”。本文提出的是一个更窄的部署问题：

> 当一个训练好的 APN 必须保持冻结时，如何用极低成本修正它仍然存在的
> 局部传感器偏差和跨传感器残差？

这个问题贴合 **Edge Computing, IoT and Digital Twins** track，因为数字孪生
或边缘监控中的模型可能已经部署，重新训练完整 backbone 成本较高，而
传感器之间的统计关系仍可能需要快速校准。

## 两个模块为什么不是硬拆出来的

### 模块一：SLRH

位置：冻结 APN latent 与原共享 decoder 并行。

作用：针对每个传感器和预测步，用 ridge regression 从 APN latent 预测
原 APN 的残差。它补的是“同一共享 decoder 对不同变量缺少专门残差
readout”的问题。

### 模块二：CFG

位置：SLRH 得到中间预测之后。

作用：对每个目标传感器，仅使用其他传感器的中间预测拟合剩余残差。
图的对角线固定为零，因此 CFG 不能偷偷变成第二个 self-calibration
模块，消融能够较干净地检验跨传感器信息。

因此两者处理的是两个不同空间：

    冻结 APN latent --SLRH--> 局部校准预测 --CFG--> 跨传感器校准预测

SLRH 看 latent，CFG 看其他传感器的 forecast；一个在 decoder 旁，一个在
forecast 后。这就是论文的核心 insight：先消除局部残差，再解释跨传感器
剩余残差。

## 实验如何做

- 数据集：PhysioNet 2012，36 个临床变量。
- Backbone：三个预先存在、由 APN released implementation 在本项目中训练
  的 checkpoint，种子 2024/2025/2026。
- 本研究没有重新训练 APN，也没有复现其他 baseline。
- 四个严格配对变体：APN、SLRH、CFG、SLRH+CFG。
- 所有变体使用相同 checkpoint、样本、target 和 mask。
- 两个 ridge 系数只在训练数据拟合，强度只由 validation MSE 选择。
- 指标：masked MSE、MAE、逐患者误差和层次配对 bootstrap。

## 主实验和消融结果

| 变体 | MSE（mean +/- std） | MAE（mean +/- std） | 相对 APN 的 MSE 改善 |
|---|---:|---:|---:|
| APN | 0.312331 +/- 0.000512 | 0.365334 +/- 0.000863 | - |
| SLRH | 0.310604 +/- 0.000406 | 0.363217 +/- 0.000307 | 0.553% |
| CFG | 0.310486 +/- 0.000880 | 0.364285 +/- 0.000690 | 0.591% |
| SLRH + CFG | **0.309058 +/- 0.000494** | **0.362978 +/- 0.000335** | **1.048%** |

完整方法的逐种子 MSE 改善为 1.056%、1.005% 和 1.082%。SLRH 和 CFG
单独都有效，组合后优于任一单模块，因此当前证据支持“两种残差空间互补”。

主表是所有 observed targets 等权的 target-micro MSE。bootstrap 则先在每名
患者内部计算误差，再让 1,185 名有 target 的患者等权，因此其
full-minus-APN MSE 为 -0.003793，95% CI [-0.005619, -0.002130]，不会与
主表均值之差完全相等。

## 如何与 APN 论文对照

APN Table 2 的 PhysioNet 论文报告值只作为未配对背景：

| 方法 | 论文报告 MSE | 论文报告 MAE |
|---|---:|---:|
| Warpformer | 0.3056 +/- 0.0011 | 0.3661 +/- 0.0016 |
| GraFITi | 0.3075 +/- 0.0015 | 0.3637 +/- 0.0036 |
| APN | 0.3093 +/- 0.0011 | 0.3650 +/- 0.0026 |
| t-PatchGNN | 0.3133 +/- 0.0053 | 0.3697 +/- 0.0049 |

这些数字来自不同运行和五种子设置，不能与我们的三 checkpoint 结果做
配对提升计算，也不能据此宣称 SOTA。论文的 1.048% 只来自同 checkpoint
的 APN 与 EdgeTwinCal 比较。

## 当前证据边界

这是一项 exploratory pilot，而不是最终确认实验。原因是前五次结构尝试
曾使用同一 PhysioNet test set 做路线筛选，所以 bootstrap 没有校正方法选择
偏差。真正的确认需要一个完全未触碰的数据集或独立 holdout。

此外，所有配对变体共同继承 released P12 pipeline 的约 81/9/10 split、
train/validation drop-last 和在全数据上拟合 standardizer 的行为。因此当前
结论是“在 released-implementation-compatible 协议下，冻结 APN 可以被
快速校准”，不是无泄漏的通用性能结论。适配速度只在 RTX 4090 上测量，
还不能声称已经验证了真实边缘设备延迟。
