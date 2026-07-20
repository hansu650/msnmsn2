# MSN 2026 第二篇论文 Idea：EviPatch

> 结论先说：主 baseline 选 **APN（AAAI 2026）**，不再做视觉 TSAD，也不再碰坐标恢复。新问题是：**APN 的 soft weighted average 声称处理局部信息密度，却在归一化后把“这个 patch 到底有多少观测支撑”直接丢掉了。** 最小修复是在 TAPA 中保留 soft mass、effective support 与 temporal coverage，形成 Evidence-Preserving TAPA（EP-TAPA）。这是一个改动很小、能做结构性反例、单卡一天有希望完成“先杀后扩”验证、而且适合 MSN 传感/IoT 叙事的方向。当前结论是 **CAUTION，不是直接开写**：必须先证明 full method 显著胜过 raw count。

## 1. 最终建议

**暂定题目**

**Mass Matters: Evidence-Preserving Adaptive Patching for Irregular IoT Sensor Forecasting**

中文理解：**观测量也属于信息：面向不规则 IoT 传感预测的证据保留式自适应分块**。

**方法名**：EviPatch；核心模块名：Evidence-Preserving TAPA（EP-TAPA）。

**一句话 idea**：APN 对每个 patch 做归一化加权平均，保留了“平均内容”，却丢掉了 soft mass、有效支撑量与时间覆盖；EviPatch把这些本来已经计算过却被丢弃的统计量注入 patch token，使模型区分“内容相似但证据强度完全不同”的传感历史。

**第一投稿 Track**：Mobile & Wireless Sensing and Networking。

**备选 Track**：Edge Computing, IoT and Digital Twins。

[MSN 2026 CFP](https://ieee-msn.org/2026/cf-papers.php) 明确覆盖 sensing、IoT、smart healthcare、smart agriculture、digital twins，并允许上述两个 Track；regular paper 是双盲、含参考文献最多 8 页，截止时间为 2026-08-20 AoE。

## 2. 去年 MSN 论文在这里到底起什么作用

本地 `C:\Users\rober\Downloads\msn2025` **只用于判断会场口味，不参与 baseline 选择、缺陷发现和方法拼装**。

我从去年的论文得到的仅是以下 venue-fit 结论：

- `DLQP: A Dual-Branch Neural Network for Volatility-Aware LoRa Link Quality Prediction` 说明 MSN 接受“真实无线/传感问题 + 轻量预测模型 + 波动/可靠性分析”的故事。
- `Freshness-Aware and Cost-Efficient Continual Learning...` 说明准确率之外，把 freshness、cost、edge resource 放进评价是合适的。
- `Unsupervised Concept Drift Detection via Generative Distribution Alignment` 说明 distribution dynamics / drift 属于会场关注点。
- `A Reliable and Interpretable Data Augmentation Framework for Predicting Power Grid States` 说明传感预测中的 reliability 叙事可以成立。

因此 EviPatch 应写成“稀疏、突发丢包、不同采样密度下的可靠 IoT 预测”，而不是泛泛的 time-series architecture paper。上述论文**不是**技术 baseline，也不用于主张 novelty。

## 3. 为什么换成 APN，而不是继续 ViT4TS

现有 `submission.pdf` 的技术母题已经非常完整：冻结 ViT、patch-grid membership 修复、图像列到原始时间戳的投影，以及重叠窗口的坐标一致性拼接。EviPatch与它的对象和机制均分离：

| 维度 | ViTTrace | EviPatch |
|---|---|---|
| 任务 | time-series anomaly detection | irregular sensor forecasting |
| 输入 | 折线图 / image patches | 原生 `(timestamp, value, mask)` |
| 主 baseline | ViT4TS / OpenCLIP | APN |
| 核心问题 | image evidence 如何映回时间戳 | pooling 是否丢失观测证据量 |
| 方法 | IHP、NCTP、overlap stitching | evidence statistics 注入 TAPA |
| 训练 | 冻结视觉模型 | 约 1.97M 参数的轻量模型训练 |

不能复用 IHP、NCTP、列投影、patch-grid 修补或原先缓存。EviPatch里的 “patch” 是原生连续时间聚合窗口，不是视觉 patch，也不存在逆坐标恢复。

## 4. 主 baseline 审核

### 4.1 为什么 APN 合格

- 论文：[Rethinking Irregular Time Series Forecasting: A Simple Yet Effective Baseline](https://ojs.aaai.org/index.php/AAAI/article/view/39563)，AAAI 2026 main track；[CCF 官方页面将 AAAI 列为人工智能 A 类会议](https://www.ccf.org.cn/Academic_Evaluation/AI/zgjsjxhtjgjxshy/al/2017-04-25/592033.shtml)。
- 官方代码：[decisionintelligence/APN](https://github.com/decisionintelligence/APN)。2026-07-20 检查为 **60 stars、3 forks、18 commits**；项目页标注 AAAI 2026 Oral。
- HumanActivity、PhysioNet 2012、USHCN 可由官方代码自动下载；MIMIC 需要 credential，本计划直接跳过。
- 官方论文报告 APN 约 **1.97M 参数、0.19 GB 单步峰值显存、3.89 ms training-step time、1.46 ms inference-step time**（USHCN、batch 32、A800）。这些数字不能直接等同于 4090 总时长，但说明模型本身不是算力瓶颈。[APN paper PDF](https://ojs.aaai.org/index.php/AAAI/article/download/39563/43524)
- 代码已把 `sum_weights` 算出来再用于除法，因此 EP-TAPA 不需要新 backbone 或第二次前向，只是保留本来被丢弃的量。[APN `models/APN.py`](https://github.com/decisionintelligence/APN/blob/main/models/APN.py)

### 4.2 关于“star 多”的诚实说明

60★对 2026 年刚发表的细分 IMTS repo 算相对高，但它不是“数百星”。如果把硬门槛定义为 `>=100★`，APN 不满足，不能硬说满足。

我仍然选 APN，是因为目前候选中没有一个 `>=100★` 同时满足“结构问题清楚、代码可直接跑、与 ViTTrace 明显独立、4090 一天左右”：

| 2026 CCF-A 候选 | 检查时 stars | 淘汰理由 |
|---|---:|---|
| [SM3Det, AAAI 2026](https://github.com/zcablii/SM3Det) | 477 | 检测模型约 403G FLOPs，官方训练入口面向 8 GPU；环境、数据、训练均不适合一天闭环 |
| [VisualAD, CVPR 2026](https://github.com/7HHHHH/VisualAD) | 108 | checkpoint 完整且可跑，但第二篇仍是冻结 ViT + anomaly，和 ViTTrace 的技术气质太近 |
| [RAG-R1, AAAI 2026](https://github.com/inclusionAI/AWorld-RL/tree/main/RAG-R1) | 110 | 需要 Qwen-7B SFT/RL、Wikipedia/KILT 检索索引和网络任务数据构造，不是一天实验 |
| [TimeMosaic, AAAI 2026](https://github.com/BenchCouncil/TimeMosaic) | 78 | 代码完整，但“自适应粒度 + 分段预测”已是其核心，不适合再包装成 APN 的新模块 |
| [APN, AAAI 2026](https://github.com/decisionintelligence/APN) | 60 | stars 未过 100，但综合可复现性、独立性、结构缺口与算力约束最好 |

所以这里反驳一次：**用 star 作可信度筛选是合理的，但为跨过 100★而选一个多卡 baseline，反而违背“一天完成实验”的主约束。** 报告所有精确 star 数，让你决定门槛，而不模糊处理。

### 4.3 复现风险

- APN README 明确说明：论文写 AdamW，但官方实验实际使用 Adam；复现时应锁定 repo 的 Adam，并把这个差异写进实验设置。
- 仓库页面未检测到明确 LICENSE。可以先做研究复现，但若最终公开衍生代码，应先询问作者授权，或只发布独立重实现与 patch 说明。
- 论文主实验用 A800；以下 4090 时长只是根据模型规模、step time 和数据规模给出的预算估计，不是本轮实测。

## 5. 结构性问题：APN “适应密度”以后又把密度除掉了

APN 的 TAPA 对第 `p` 个 patch 计算：

\[
\mu_p=\frac{\sum_i \alpha_{ip}\,m_i\,z_i}
{\sum_i \alpha_{ip}\,m_i+\varepsilon},
\]

其中 `z_i=[value_i, TE(t_i)]`，`m_i` 是观测 mask，`alpha` 是 soft-window weight。

代码随后只把 `mu_p` 投影成 patch token；分母

\[
M_p=\sum_i \alpha_{ip}m_i
\]

在完成归一化以后被丢弃。[官方论文](https://ojs.aaai.org/index.php/AAAI/article/download/39563/43524)一方面强调 sparse/dense patch 和 local information density，另一方面其公开[实现](https://github.com/decisionintelligence/APN/blob/main/models/APN.py)中的边界参数是每个变量/patch 一套全局参数，并被 `expand(B, ...)` 复制给整个 batch；样本级密度最后主要只能通过归一化后的均值间接体现。

### 5.1 可写成 proposition 的结构性碰撞

对一个 patch 内所有观测复制 `k` 次，只要复制点拥有相同的值、时间和 mask，就有：

\[
\mu_p^{(k)}=
\frac{k\sum_i\alpha_{ip}m_i z_i}{k\sum_i\alpha_{ip}m_i}=\mu_p.
\]

因此 APN 后续 query 和 decoder 对这两个历史给出完全相同的预测。更一般地，任何拥有相同 weighted centroid、但支撑量不同的两组观测都会碰撞。

这不意味着“复制数据一定应该改变均值”，也不证明更多采样必然更可靠；它只证明下游无法再决定是否使用 total soft mass。潜在价值来自：

- 传感器测量有噪声时，1 个读数和 20 个独立读数的可信度不同；
- 采样/上报策略具有信息性时，观测密度本身可能反映活动强度、链路拥塞、电池节流或故障；
- APN声称处理 sparse/dense information density，却没有把 evidence quantity 交给下游模块决定是否使用。

这比“再加一个 attention”更适合作为论文起点，因为信息丢失可以由公式和代码确定，而不是先假设性能会涨。但“被丢了”仍不等于“对预测有用”，因此还需要下面的风险构造与 kill-test。

### 5.2 从表示碰撞到预测风险：必须补的统计模型

令一个 patch 内存在潜在真实状态 `theta`，传感读数满足

\[
x_i=\theta+\epsilon_i,\qquad
\epsilon_i\overset{iid}{\sim}\mathcal N(0,\sigma^2),
\qquad \theta\sim\mathcal N(0,\tau^2),
\]

未来目标为 `y=theta+eta`。考虑两个具有相同 feature centroid `x_bar`、相同有限维 time-embedding centroid，但有效支撑量分别为 `m_1` 与 `m_2` 的观测多重集。归一化 TAPA 可以令两者产生相同 `mu_p`；然而 MSE 下 Bayes 最优预测为

\[
\mathbb E[y\mid \bar x,m]
=\frac{m\tau^2}{\sigma^2+m\tau^2}\bar x,
\]

它显式依赖 `m`。因此，在独立测量噪声模型中，忽略 support 会把 Bayes 最优预测不同的历史压到同一表示。

这只是“存在性”模型，不代表真实数据一定满足独立高斯噪声。论文应进一步构造现实受控对：在 timestamp resolution、weighted centroid 与总删点率近似匹配时改变 patch 内 support，并验证预测确实变化；不能拿简单重复同一数据行冒充独立证据。

### 5.3 旁路审计

公开 `APN.py` 的完整前向中，`x_mask` 只在 TAPA 内乘到 `temporal_weights`；其后 query 与 decoder 只接收归一化后的 patch representations，padding length、mask count 或 `sum_weights` 没有独立旁路。因此 total soft mass 在当前实现中确实不可由后续模块直接读取。实现时仍应加入单元审计：固定 `mu_p`、改变 `sum_weights`，确认原 APN 的最终输出数值不变。

## 6. 方法：Evidence-Preserving TAPA

不改 APN 的 soft windows、query module 或 decoder，只为每个 `(sample, variable, patch)` 保留一个很小的 evidence signature。

### 6.1 三个无额外扫描的统计量

设 `a_i=alpha_ip * m_i`：

1. **Soft mass**

\[
s_p=\log(1+\sum_i a_i).
\]

2. **Effective support**

\[
n^{eff}_p=\frac{(\sum_i a_i)^2}{\sum_i a_i^2+\varepsilon},
\qquad e_p=\log(1+n^{eff}_p).
\]

3. **Normalized temporal coverage**

\[
c_p=\frac{\sqrt{\sum_i a_i(t_i-\bar t_p)^2/(\sum_i a_i+\varepsilon)}}
{t^{right}_p-t^{left}_p+\varepsilon}.
\]

`soft mass` 描述有多少加权支撑；`effective support` 区分一个强权重点与多个均匀支撑点；`coverage` 区分同样数量但挤在一个 burst 中和铺满窗口的观测。三者都可复用 TAPA 已有的 `temporal_weights` 和 timestamps。这里不把 mass 强行解释成 confidence，也不施加“越大越可靠”的单调约束。

### 6.2 最小融合

\[
h_p=\mathrm{MLP}\big([\mu_p,\operatorname{LN}(s_p,e_p,c_p)]\big).
\]

也就是把现有 projection 的输入从 `1 + D_te` 增加到 `1 + D_te + 3`，其余网络完全不变。不要同时加入动态边界、复杂 gate、cross-channel graph 或新 decoder，否则会破坏“单一缺陷—单一修复”的干净故事。

### 6.3 必须保留的简单对照

- `APN + raw count`：只拼真实观测数，排除“任何 count 都能涨”的质疑。
- `APN + patch soft mass`：只加 `s_p`。
- `APN + mass + effective support`。
- `EviPatch full`：三项全加。
- `Shuffled evidence`：在 batch 内打乱 evidence signature；若仍然涨，说明收益只是额外参数而非证据语义。
- `Equal-parameter random features`：加入同维随机但固定的特征，排除 projection 变宽本身带来的收益。

还必须量化 `coverage` 能否从 APN 的平均 time encoding 近似恢复：用冻结 APN patch token 训练一个线性 probe 预测 coverage。若 `R^2` 已很高，coverage 不能再作为“新信息”，full method 应退回 mass/support 两项。

## 7. 2024–2026 顶会论文如何使用

这里只把正式顶会论文用于结构诊断、机制边界和实验设计，不把它们改成主 baseline。

| 顶会论文 | 借用/约束的内容 | 与 EviPatch 的边界 |
|---|---|---|
| [t-PatchGNN, ICML 2024](https://proceedings.mlr.press/v235/zhang24bw.html) | 明确指出 irregular patch 可包含不同数量观测，说明 variable support 是 IMTS 的核心结构 | 它以 transformable patches + time-adaptive GNN 建模异步变量；EviPatch只修 APN 归一化池化丢失 evidence 的问题 |
| [ContiMask, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4eb5daabc45b45a9a312aa2c8fca8a74-Abstract-Conference.html) | 证明 irregular-time perturbation 不能忽略 observation structure / informative missingness | 借其“改变观测机制”的诊断思路，不借 NeuroEvolution 或解释器 |
| [Time-IMM, NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html) | 将真实 irregularity 分成 trigger、constraint、artifact 等 cause-driven 类型 | 可作为第二阶段真实机制压力集；第一天不强行接入全套 multimodal benchmark |
| [iTimER, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39545) | 提醒“未观测区域也携带可利用信号” | 它做 reconstruction-error self-supervised pretraining；EviPatch不做预训练或 pseudo-observation |
| [TimeMosaic, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39218) | 最重要的撞车红线：它已做 instance-wise adaptive granularity 和 horizon-specific segment decoding | 因此 EviPatch明确不改 patch 粒度、不改 decoder，只保存 APN 丢掉的 soft evidence |

### 7.1 有意放弃的旧 idea

“给 APN 做 sample-conditioned patch boundary + horizon-conditioned query”已经不建议。TimeMosaic 的公开代码会对每个样本/区域选择 patch length，并对不同 forecast segments 使用独立 prompts/heads；再把类似结构移植到 APN，审稿人很容易写成“TimeMosaic on irregular data”。

### 7.2 查新结论的证据边界

本轮查新严格限制在用户指定的 2024–2026 顶会正式论文。因此只能给出：

> 在这个**顶会受限检索范围**内，没有发现一篇直接以“APN/TAPA 归一化后丢弃 soft evidence mass”为结构缺陷，并用 mass/support/coverage signature 修复的论文。

这不是全球 novelty 证明。最大风险不是完全重复，而是审稿人认为“把 count 拼回去太 obvious”。所以论文成败必须依靠：精确的不可区分性命题、matched-rate 诊断、shuffled-evidence 控制，以及在 native 与 stress 两类场景都有效，而不是靠模块复杂度。

## 8. 一天版实验计划

### 8.1 数据集

只用 APN 官方可自动下载的三套：

- HumanActivity：移动/可穿戴 sensing；
- PhysioNet 2012：smart healthcare sensing；
- USHCN：环境/气象 sensing。

跳过 credentialed MIMIC。Time-IMM 只在主结果通过后作为可选加分项，不放进第一天的硬计划。

### 8.2 先杀后扩的两阶段评价

**Stage A：PhysioNet kill-test**

完全沿用 APN split 和 repo optimizer，只在 PhysioNet 做一个 native checkpoint 与一种 `30% burst loss` 压力测试。burst 与 30% MCAR 严格匹配总删点率；stress 只作用于 test history，不重训，因此这个协议明确叫 **observation-shift evaluation**，不再误称 matched training。

同时构造受控 support pairs：使全局缺失率、patch weighted centroid 和平均 time encoding 尽量匹配，只改变 patch 内 support allocation。它首先验证“丢掉的量对预测是否有用”。

**Stage B：通过 kill-test 后扩展**

三套数据都沿用 APN 的 80/10/10 native split、MSE/MAE 和 repo optimizer。每个 native checkpoint 在下列 test-only shift 上评价，forecast target 不变且不额外训练：

- 30% MCAR thinning；
- 30% matched-rate burst loss；
- 30% value-triggered thinning：保留概率只依赖截至当前的历史变化幅度；
- 30% battery throttling：后半段采样概率逐渐下降。

10%/50% 缺失率 sweep 只用单 seed 做补充趋势图，不进入一天主矩阵。若要声称“matched mechanism learning”，必须另行重训并放到第二天；第一天不作该 claim。

**自然采样变化证据**

不能只靠人工删点。Native test 中按原生 patch mass、最大 gap 与 patient/subject/station 观测率分四分位，报告 worst-quartile MAE；若主结果通过，再接入 [Time-IMM](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4199594d3c15736df2bf5274fa3155f4-Abstract-Datasets_and_Benchmarks_Track.html) 或真实设备丢包数据作为投稿前的第二阶段验证。

### 8.3 最小比较矩阵

Stage A 在 PhysioNet 跑 6 个配置：

| 配置 | 作用 |
|---|---|
| APN official | 主 baseline |
| APN + global observed ratio | 证明 patch-local 统计是否必要 |
| APN + raw patch count | **决定性 simple baseline** |
| APN + soft mass | 检验最小修复 |
| EviPatch full | 检验 support + coverage 是否超越 count |
| EviPatch + shuffled evidence / equal-parameter random features | 排除额外参数与伪相关 |

只有 full 显著胜过 raw count 才进入 Stage B。Stage B 为控制时间，只保留 `APN / raw count / full / shuffled` 四个配置；所有 stress 都复用 native checkpoint 推理。

t-PatchGNN 必须在 PhysioNet 的 native 与 30% burst 上做直接比较，或在论文中明确将“未能在预算内复现”列为局限；不能只靠 related-work 文字消除增量质疑。

### 8.4 指标

- 主指标：MSE、MAE；
- robustness：相对 native 的 error inflation；
- worst-quartile MAE：按观测 mass 或最大 gap 分层后的最差四分位；
- mechanism macro-average：四种 observation mechanisms 等权平均；
- efficiency：参数量、peak VRAM、训练 wall-clock、inference latency。

所有 threshold、normalization 和 evidence 标准化统计量只能由 training split 得到。

### 8.5 4090 时间预算

不再一开始承诺 45 个任务全跑完，而采用门控预算：

1. `100-step × 6 configs` smoke：实测每步、每 epoch、数据加载和验证时间；预计 0.5–1 小时。
2. Stage A：`PhysioNet × 6 configs × 3 seeds = 18` 次训练，加 native/burst/support-pair 推理；预计 4–8 小时。
3. 若 Stage A 通过，Stage B 复用其中 PhysioNet checkpoint，只补 `2 datasets × 4 configs × 3 seeds = 24` 次训练；四种 test-only shift 不重训；预计 6–12 小时。
4. t-PatchGNN 的 PhysioNet 对比、失败重跑与统计汇总预留 3–6 小时。

因此完整第一天是最多 42 次轻量训练，目标 14–27 小时，**靠近而不保证低于 24 小时**。只有 100-step 实测支持时才保留全部 Stage B；否则先删 10%/50% sweep，再把 t-PatchGNN 或第三数据集移到第二天。官方 A800 单步时间不能直接外推成 4090 端到端工时。**本轮没有运行 GPU。**

## 9. 论文 claim 与停止条件

### 9.1 可以争取的 claim

1. APN 的 normalized TAPA **严格丢弃 total soft mass**，并存在相同 weighted centroid、不同观测支撑量的表示碰撞；不声称它丢失全部 density information。
2. EP-TAPA以近零额外计算保留 soft mass、effective support 和 temporal coverage。
3. EviPatch在 native benchmarks 不退化，并在 matched-rate burst / trigger / throttle shift 下更稳。

不要声称“解决 MNAR”或“具有因果无偏性”；本方法只是让预测器看见 observation evidence，不等于识别不可观测的真实 missingness mechanism。

### 9.2 Go / Kill gates

**第一道 kill-test**：PhysioNet、3 seeds、相近 weighted centroid / 不同 support 的受控对上，比较 APN、raw count、full、shuffled。若 full 相对 APN 改善不足 5%、不显著优于 raw count，或 shuffle 后收益仍在，立即 ABANDON。

通过第一道以后，只有同时满足以下条件才继续写完整论文：

- full EviPatch 在至少 2/3 native datasets 改善 MSE 或 MAE，且第三个不显著退化；
- 在 matched-rate stress 的 mechanism macro-average 上，相对 APN 至少降低 5% error inflation；
- full 明显优于 `global observed ratio` 与 `raw count`，否则贡献只是“加一个计数”；
- shuffled evidence 失去收益，证明位置/语义对应关系重要；
- equal-parameter random features 不产生相同收益；
- 三 seed 的 paired bootstrap 或 seed-level CI 支持主要结论；
- 参数与训练时间开销均低于 5%。

出现以下任一情况就砍掉或降级为短文：

- 只在 50% 人工删点时有效，native 全部无提升；
- raw count 与 full 相同；
- 收益来自重新调参而非 evidence；
- standard APN 其实已能从 time embedding 稳定恢复相同信息；
- 需要再加动态边界、复杂 graph、conformal head 才能涨。

## 10. 预演审稿意见

### Attack 1：这不就是把 denominator 拼回去吗？太简单。

回应策略：承认实现简单，把贡献放在“被 normalized pooling 确定性抹去的信息”和可验证的 equivalence class；用 `raw count`、`soft mass`、`shuffled evidence` 三个控制证明并非参数增益。MSN 8 页反而适合一个窄而硬的结构故事。

### Attack 2：多观测不一定更可靠，informative sampling 甚至可能有偏。

回应策略：不施加 `mass 越大越可信` 的单调假设，只把 signature 交给 MLP学习；分别报告 MCAR、burst、trigger、throttle，避免把所有 missingness 混成一种。

### Attack 3：TimeMosaic 已经按信息密度自适应 patch。

回应策略：TimeMosaic改变 regular TS 的 patch granularity 和 segment decoder；EviPatch不重切 patch，不改 horizon decoder，而是修复 irregular soft pooling 的 evidence loss。实验中不把 TimeMosaic 的机制改名移植。

### Attack 4：APN 的时间编码均值已经包含采样模式。

回应策略：均值能编码 centroid，不能恢复被除掉的总质量；proposition 给出完全碰撞。另用同 centroid / 不同 support 的 controlled pairs 直接测试 representation 与预测。

### Attack 5：stress 是人工构造。

回应策略：native results 是硬门槛；stress 采用 matched deletion rate，并按 cause 分类；若主结果通过，再接 Time-IMM 的 cause-driven irregularity 作为外部验证。

## 11. 最小实现位置

只需改 `AttentionPatchAggregation.forward`：

1. 保留现有 `temporal_weights` 和 `sum_weights`；
2. 计算 `sum_sq_weights`、weighted timestamp variance；
3. 生成 3 维 evidence signature；
4. 与 `h_patches_avg` 拼接；
5. 把 `projection_layer` 输入维度增加 3。

预计核心改动少于 30 行。数据侧只新增可复现的 mask transform；不改 APN 数据 split、query、decoder 和 loss。

## 12. 独立反方审稿与最终判断

独立零上下文审稿结论为 **CAUTION（confidence 0.90）**。其最强拒稿理由是：

> 代数上 APN 确实丢弃 total soft mass，但非单射不证明该信息对预测有用；同值同时间复制不是独立证据，而 mass/support/coverage 容易被视为显然的计数与矩统计。若 full 不能稳定胜过 raw count，贡献不足。

该评审还指出原实验表实际有 6 个配置却按 5 个计算，并混淆了 matched training 与 checkpoint 复用。当前版本已加入 Bayes-risk 构造、旁路审计、raw-count/random-feature/linear-probe 控制，并改成 Stage A kill-test 后才扩展。

**最终建议：CAUTION / CONDITIONAL PROCEED。先做 kill-test，不要先写完整论文，更不要提前承诺 SOTA。**

这个 idea 最强的地方不是模块新奇，而是 APN 的论文叙事与公开计算路径之间存在一个可形式化的缺口：它要处理 sparse/dense information density，却在 weighted average 中丢掉了 evidence mass。它也满足第二篇与 ViTTrace 分家的要求。

最大风险同样清楚：审稿人可能认为修复过于 obvious。因此第一天实验真正要回答的不是“能不能涨一点”，而是：**在总删点率与 weighted centroid 受控时，support/coverage 是否产生 APN 和 raw count 都无法替代的稳定收益。** 如果答案是否，就应尽快止损。

---

## 本轮执行与证据说明

- 已读：MSN 2026 CFP、本地 MSN 2025 论文目录与若干代表论文、现有 `submission.pdf`、APN 正式论文/官方代码，以及上表 2024–2026 正式顶会近邻。
- MSN 2025 材料仅用于选题/Track fit；技术 baseline 与方法证据全部来自 2024–2026 顶会及官方代码。
- 2026-07-20 检查 GitHub stars；star 会变化，投稿前应重新记录 commit hash 与 star snapshot。
- 没有下载仓库、没有创建实验目录、没有运行 GPU。
- 查新范围受“只看 2024–2026 顶会”约束，结论是 venue-bounded provisional novelty，不是穷尽式查新。
- 独立审稿由同模型家族的新上下文 reviewer 完成，属于 provisional assurance；其 verdict 与最强反对意见已嵌入本文件，没有另建 trace。
- 本报告只保存在用户指定目录的这个 Markdown；未创建 manifest、trace、cache 或其他文件。
