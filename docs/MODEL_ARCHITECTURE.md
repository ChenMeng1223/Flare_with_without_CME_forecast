# 模型架构说明

本文档面向论文写作与答辩展示，对当前代码中实现的 `MultimodalTransformer` 模型进行正式化说明。内容依据当前工程实现整理，重点描述模型的任务定义、整体架构、核心模块、训练目标及其方法特点。

---

## 1. 方法定位

本研究的目标是面向太阳耀斑/CME 场景，构建一个能够同时利用多模态太阳观测序列信息的预测模型。与早期“整窗级编码后直接输出分类结果”的单阶段结构不同，当前实现采用了**两阶段多模态时空 Transformer 架构**：

1. 首先对不同模态的图像序列进行独立空间编码；
2. 然后在时间维与模态维上进行联合特征融合；
3. 接着从融合后的时空特征中生成固定数量的候选活动槽位；
4. 最后围绕每个候选槽位提取局部区域与上下文区域，并完成类别判别、时间预测、事件概率估计与边界框精修。

从建模形式上看，该模型具有以下特点：

- **多模态输入**：同时利用磁图、EUV、Hα 等观测信息；
- **时空联合建模**：在保留空间结构的基础上进行时序编码；
- **固定槽位检测式输出**：每个时间窗口输出固定数量的候选活动；
- **两阶段预测**：先生成 proposal，再进行 refinement；
- **多任务联合学习**：同时优化类别、位置、时间、事件概率与物理约束。

因此，当前模型可以概括为：

> 一个面向太阳耀斑/CME 预测任务的、多模态两阶段时空 Transformer 网络。

---

## 2. 任务定义

### 2.1 输入形式

模型输入为一个多模态时间窗口。对任一模态，其输入张量形状为：

- `(B, T, C, H, W)`

其中：

- `B` 表示 batch size；
- `T` 表示时间步长度；
- `C` 表示通道数；
- `H, W` 表示图像空间分辨率。

在当前默认配置下：

- `T = 9`
- `C = 1`
- `H = W = 256`

对于缺少显式通道维的输入，模型内部会自动扩展为单通道形式。

### 2.2 输出形式

模型并非只输出单个整窗类别，而是对每个输入窗口输出 `max_activities` 个候选槽位。默认配置为：

- `max_activities = 5`

每个槽位对应一个潜在活动实例，并输出如下预测结果：

- 类别预测 `class_logits / class_probs`
- 候选得分 `proposal_scores`
- 边界框预测 `bbox_pred`
- 时间预测 `time_pred`
- 事件概率 `event_prob`

### 2.3 标签定义

当前分类标签语义如下：

| 标签 | 含义 |
|------|------|
| 0 | 无事件 / padding 槽位 |
| 1 | 爆发耀斑（eruptive / CME-related） |
| 2 | 束缚耀斑（confined） |

需要特别说明的是：

1. 模型输出层保留 3 类结构，即 `num_classes = 3`；
2. 在训练分类损失时，仅对真实事件槽位进行类别 1 与类别 2 的判别；
3. 类别 0 不直接参与分类交叉熵，而是通过 proposal/objectness、事件概率与负槽位抑制机制进行学习。

这一设计使模型能够更好地区分“存在活动的槽位”与“空槽位”，并减少类别不平衡带来的干扰。

---

## 3. 配置与实现对应关系

当前模型的实际行为由 `model_config.yaml`、`data_config.yaml` 与训练脚本联合决定。

### 3.1 模型核心配置

当前关键参数可概括如下：

- `type = multimodal_transformer`
- `input_size = [256, 256]`
- `sequence_length = 9`
- `num_classes = 3`
- `max_activities = 5`
- `hidden_dim = 256`
- `num_heads = 4`
- `num_layers = 3`
- `dropout = 0.1`
- `encoder_downsample_factor = 4`
- `fusion.method = attention`
- `stage2.context_scale = 2.0`

### 3.2 数据配置对模型的影响

训练脚本会从数据配置中补齐或覆盖以下字段：

- `modalities`
- `input_size`
- `sequence_length`
- `max_activities`

因此，模型在训练时的真实输入分辨率、模态集合和时间窗口长度，最终以合并后的配置为准。

---

## 4. 整体架构

模型整体流程如下：

```text
多模态输入 {modality: (B, T, C, H, W)}
        │
        ▼
SpatialModalityEncoder × N
        │
        ▼
MultimodalSpatiotemporalFusion
  ├─ 跨模态注意力
  └─ 时序 Transformer 编码
        │
        ▼
融合时空特征
  ├─ fused_maps
  └─ global_feature
        │
        ▼
ProposalDecoder
  ├─ proposal_boxes
  ├─ proposal_scores
  └─ proposal_features
        │
        ▼
ROI 提取
  ├─ local ROI
  └─ context ROI
        │
        ▼
RoiTemporalEncoder × 2
        │
        ▼
StageTwoPredictor
  ├─ 类别预测
  ├─ 时间预测
  ├─ 事件概率估计
  └─ 边界框精修
        │
        ▼
最终槽位级输出
```

从功能上可以划分为两个阶段：

### 第一阶段：候选槽位生成

第一阶段从融合后的全局时空特征中生成定长 proposal，输出：

- `proposal_boxes`
- `proposal_scores`
- `proposal_features`

其作用是从全局视角粗略定位潜在活动区域，并为后续精修提供初始化框与候选特征。

### 第二阶段：局部精修与多任务预测

第二阶段围绕每个 proposal 提取局部与上下文 ROI 特征，并进一步输出：

- `class_logits / class_probs`
- `bbox_pred`
- `time_pred`
- `event_prob`

因此，当前模型不是简单的窗口级分类器，也不是标准 DETR，而是更接近**定长查询驱动的两阶段时空实例预测框架**。

---

## 5. 核心模块设计

## 5.1 模态空间编码器 `SpatialModalityEncoder`

每个模态均对应一个独立的空间编码器。该模块的主要目标是：

- 提取单模态图像序列的局部空间结构；
- 在降低分辨率的同时保留空间布局；
- 为后续跨模态融合和 proposal 生成提供共享维度的特征图。

其结构为两层卷积块堆叠：

1. `Conv2d(input_channels, mid_dim, 3, padding=1)`
2. `BatchNorm2d`
3. `ReLU`
4. `MaxPool2d(2)`
5. `Conv2d(mid_dim, hidden_dim, 3, padding=1)`
6. `BatchNorm2d`
7. `ReLU`
8. `MaxPool2d(2)`
9. `Dropout2d`

其中：

- `mid_dim = max(hidden_dim // 2, 32)`
- 默认 `hidden_dim = 256`

若输入大小为 `256×256`，则经过两次 `MaxPool2d(2)` 后，输出空间特征图大小约为：

- `64×64`

故该模块输出形状为：

- `(B, T, D, H', W')`

其中默认 `D = 256`，`H' = W' = 64`。

这一设计相较于直接全局池化的做法，能够为后续 proposal 生成保留更充分的空间定位信息。

## 5.2 跨模态与时序融合模块 `MultimodalSpatiotemporalFusion`

该模块用于完成跨模态信息交互与时序依赖建模，是整个网络的核心特征融合部分。

### （1）模态内空间压缩

对于每个模态的空间特征图 `(B, T, D, H', W')`，首先对空间维做平均池化，得到：

- `(B, T, D)`

这一步得到的是每个时间步上的模态摘要向量，用于后续跨模态注意力计算。

### （2）跨模态注意力融合

在多模态场景下，当前模态作为 Query，其余模态的均值作为 Key/Value，通过 `MultiheadAttention` 进行增强。增强后的各模态结果再求平均，得到：

- `fused_sequence: (B, T, D)`

与此同时，各模态空间特征图直接求平均，得到：

- `fused_maps: (B, T, D, H', W')`

### （3）时序 Transformer 编码

`fused_sequence` 随后送入 `TransformerEncoder`，以捕获长程时间依赖关系。输出为：

- `temporal_features: (B, T, D)`

### （4）全局特征与时空回注入

为了让空间特征图显式获得时间上下文，模型将时间特征回注入至空间图：

```text
fused_maps = fused_maps + temporal_features.unsqueeze(-1).unsqueeze(-1)
```

同时，对时间维取平均，得到全局序列表征：

- `global_feature: (B, D)`

该特征用于第二阶段预测头中的全局上下文建模。

## 5.3 候选解码器 `ProposalDecoder`

该模块构成第一阶段的核心，其目标是在融合后的时空特征图上生成固定数量的候选活动槽位。

### （1）时空 token 展平

首先将 `fused_maps` 从 `(B, T, D, H', W')` 重排为：

- `(B, T×H'×W', D)`

即将时间与空间位置统一视作一个时空 token 序列。

### （2）位置编码与查询读取

模型引入可学习位置编码 `LearnablePositionalEncoding`，随后利用 `max_activities` 个可学习查询向量，从全部时空 token 中读取候选特征。

在当前默认设置下：

- 查询数 `num_queries = max_activities = 5`

### （3）候选框与候选得分预测

对每个查询输出：

- `proposal_scores: (B, A, 1)`
- `proposal_boxes: (B, A, 4)`
- `proposal_features: (B, A, D)`

其中 proposal 框采用“中心 + 尺寸”的隐式预测方式，再转换为归一化 `xyxy` 形式。这种设计相较于直接回归角点坐标，数值更加稳定。

## 5.4 ROI 提取与上下文建模

为了提升第二阶段的定位与分类能力，模型并不直接使用 proposal 特征完成最终预测，而是围绕 proposal 框进一步提取局部区域与上下文区域。

### （1）第二阶段输入框混合机制

训练时，第二阶段所使用的输入框并不完全来自预测 proposal，而是按照一定比例混合：

- ground truth 框
- jittered ground truth 框
- predicted proposal 框

这一策略的目的是减小训练与推理阶段 ROI 分布差异，提高第二阶段预测头对真实推理条件的适应能力。

### （2）上下文框构建

模型会将 proposal 或 stage2 输入框按比例扩张，得到 context box：

- `context_scale = 2.0`

这使模型在进行局部目标判断时，同时感知周边太阳活动环境。

### （3）ROI 采样方式

当前实现没有使用标准 ROIAlign，而是通过归一化采样网格与 `grid_sample` 从 `fused_maps` 中抽取 ROI 特征。默认输出大小为：

- `roi_output_size = 4`

因此，局部 ROI 与上下文 ROI 的形状分别为：

- `local_roi_maps: (B, A, T, D, 4, 4)`
- `context_roi_maps: (B, A, T, D, 4, 4)`

## 5.5 ROI 时序编码器 `RoiTemporalEncoder`

该模块对 ROI 序列特征进行时间维建模。

其过程包括：

1. 对每一帧 ROI 进行自适应平均池化，得到 ROI 向量；
2. 加入时间位置编码；
3. 通过单层 `TransformerEncoder` 建模 ROI 内部的时间演化。

输出为：

- `sequence: (B, A, T, D)`
- `pooled_feature: (B, A, D)`

模型分别对 local ROI 与 context ROI 各执行一次该模块，以获得局部信息与上下文信息的时序表示。

## 5.6 第二阶段预测头 `StageTwoPredictor`

第二阶段预测头融合以下多源信息：

- 局部 ROI 特征 `local_feature`
- 上下文 ROI 特征 `context_feature`
- proposal 特征 `proposal_feature`
- 全局特征 `global_feature`
- proposal 分数 `proposal_score`

拼接后通过线性层进行特征统一，形成：

- `slot_features: (B, A, D)`

随后分成多个任务分支：

### （1）分类分支

输出：

- `class_logits: (B, A, 3)`
- `class_probs: (B, A, 3)`

用于完成爆发耀斑、束缚耀斑与空槽位语义空间下的类别建模。

### （2）事件概率分支

输出：

- `event_prob: (B, A, 1)`
- `event_gate = sigmoid(event_prob)`

其中 `event_gate` 在后续时间预测与 bbox 精修中起到门控作用，用于抑制负槽位产生不合理预测。

### （3）时间预测分支

时间预测并非只依赖 pooled 向量，而是进一步利用 local/context ROI 的时序序列做注意力读取，得到更具动态信息的时间特征。最终输出：

- `time_pred: (B, A, 3)`

其语义对应：

- 起始时刻偏移 `start`
- 峰值时刻偏移 `peak`
- 结束时刻偏移 `end`

此外，模型还派生：

- `time_center`
- `time_duration`
- `time_params`

以便于训练损失的构建与后续分析。

### （4）边界框精修分支

该分支以 proposal box 为初始框，进一步预测：

- 中心偏移量 `delta_center`
- 对数尺度偏移量 `delta_log_size`

由此得到 refined bbox，即：

- `bbox_pred: (B, A, 4)`

同时，为避免无事件槽位输出过大的虚假框，模型通过 `event_gate` 对框尺寸进行门控抑制。

---

## 6. 模型输出

当前 `forward()` 返回的不仅包括最终任务输出，也包括较丰富的中间特征，便于后处理、可视化和误差分析。核心输出如下：

| 字段 | 形状 | 含义 |
|------|------|------|
| `class_logits` | `(B, A, 3)` | 槽位类别 logits |
| `class_probs` | `(B, A, 3)` | 槽位类别概率 |
| `bbox_pred` | `(B, A, 4)` | 第二阶段精修后的边界框 |
| `time_pred` | `(B, A, 3)` | 槽位级时间预测 |
| `event_prob` | `(B, A, 1)` | 事件概率 logits |
| `event_gate` | `(B, A, 1)` | 事件门控系数 |
| `proposal_boxes` | `(B, A, 4)` | 第一阶段 proposal 框 |
| `proposal_scores` | `(B, A, 1)` | proposal/objectness 分数 |
| `proposal_features` | `(B, A, D)` | 候选特征 |
| `global_feature` | `(B, D)` | 全局时序特征 |
| `temporal_features` | `(B, T, D)` | 融合后的时间特征 |
| `fused_maps` | `(B, T, D, H', W')` | 融合后的时空特征图 |
| `local_roi_feature` | `(B, A, D)` | 局部 ROI 特征 |
| `context_roi_feature` | `(B, A, D)` | 上下文 ROI 特征 |
| `physics_loss` | 标量 | 物理约束项 |

这表明当前模型既是一个预测模型，也具备较强的可解释性分析接口。

---

## 7. 训练目标与损失函数

当前训练器将优化目标划分为 proposal 阶段、refinement 阶段和负槽位抑制三部分。

### 7.1 分类损失

分类损失采用 `CrossEntropyLoss`，但并非对所有槽位和全部三类同时监督，而是：

- 仅对正样本槽位计算；
- 仅对类别 1 与类别 2 做判别。

因此，该项损失本质上是“真实事件槽位上的 eruptive/confined 二分类损失”。

### 7.2 Proposal 分数损失

proposal 分数损失采用 `BCEWithLogitsLoss`，用于监督每个槽位是否包含真实事件，即 objectness 学习。

### 7.3 Proposal 边界框损失

对正样本槽位上的 proposal 框计算：

- `L1 loss`
- `GIoU loss`

二者之和构成第一阶段定位损失。

### 7.4 Refined 边界框损失

对第二阶段精修框同样计算：

- `L1 loss`
- `GIoU loss`

该项用于提升最终定位精度。

### 7.5 时间预测损失

时间预测损失仅对正样本槽位计算，包含三部分：

1. 中心时刻损失；
2. 持续时间损失；
3. 起始-峰值-结束边界损失。

其组合形式为：

```text
time_loss = 0.25 * center_loss + 0.25 * duration_loss + 0.50 * boundary_loss
```

这一设计兼顾了整体时间区间的一致性与关键时刻预测精度。

### 7.6 事件概率损失

事件概率分支采用 `BCEWithLogitsLoss`，用于学习每个槽位的事件存在概率。

### 7.7 负槽位抑制损失

为减少空槽位产生虚假检测，模型额外对负槽位施加：

- `bbox_suppression_loss`
- `time_suppression_loss`

其作用是促使无事件槽位输出更小的框、更短或接近零的时间跨度。这是当前实现中区别于简单多头回归模型的关键设计之一。

### 7.8 物理约束损失

当开启 `use_physical_constraints` 且输入中提供物理量时，模型会附加 `physics_loss`，用于将一定的物理先验纳入训练过程。

### 7.9 总损失函数

总体损失由上述多项加权构成，可表示为：

```text
L = L_cls + λ1 L_prop-box + λ2 L_refine-box + λ3 L_time + λ4 L_event + λ5 L_prop-score + λ6 L_suppress + λ7 L_phys
```

在当前默认训练配置中，对应权重大致为：

- 分类损失：`1.0`
- proposal bbox 损失：`2.0`
- refined bbox 损失：`1.0`
- 时间损失：`0.5`
- 事件概率与 proposal score：`0.5`
- 物理约束损失：`0.1`
- 负槽位抑制：默认 `0.2`

---

## 8. 数据组织与训练样本构造

模型训练数据来源于事件级 HDF5 文件，图像数据按如下层级组织：

- `events/<event_id>/data/<modality>/images`

同时配有：

- 边界框标注 `bbox`
- 类别标注 `label`
- 时间标签 `time_features`
- 槽位有效性掩码 `activity_mask`

训练时，`SolarFlareDataset` 会基于事件级序列构造滑动时间窗口，因此当前模型的训练单位是：

- **窗口级样本**，而非整事件端到端输入。

在默认配置下：

- `sequence_length = 9`
- `stride` 由数据配置控制

---

## 9. 方法特点与答辩可强调点

若从论文贡献或答辩展示角度概括，当前架构的亮点主要体现在以下几个方面：

### 9.1 多模态互补建模

不同波段和观测模态反映太阳活动的不同物理属性。当前模型通过独立编码 + 跨模态注意力融合，使多源观测信息能够在统一特征空间中交互，从而提升事件判别能力。

### 9.2 时空联合表示学习

模型不是将图像先完全压缩再处理时间信息，而是先保留空间结构，再通过 Transformer 建模时间演化，因此更适合处理太阳活动区域随时间发展的动态模式。

### 9.3 两阶段槽位预测机制

通过“proposal 生成 + ROI refinement”的设计，模型兼顾了：

- 全局搜索能力；
- 局部精细定位能力；
- 固定输出长度下的多活动建模能力。

### 9.4 空槽位显式抑制

相较于仅依赖分类头区分背景与事件，当前实现额外引入 event/objectness 监督与负槽位抑制损失，使空槽位更容易收缩到“低分、短时、小框”的合理状态。

### 9.5 物理先验可扩展

模型支持引入物理约束项，使其不仅是纯数据驱动网络，也为后续构建“数据驱动 + 物理先验”联合框架提供了接口。

---

## 10. 结论性表述

综上，当前 `MultimodalTransformer` 并不是简单的多头分类回归模型，而是一个针对太阳耀斑/CME 场景设计的、具有明确层级结构的两阶段多模态时空预测框架。其核心思想是：

- 先在全局时空特征上完成候选活动发现；
- 再在局部 ROI 与上下文信息支持下完成细粒度预测；
- 并通过多任务损失共同约束类别、位置、时间和事件存在性。

这一架构兼顾了多模态融合、时空建模、目标定位与任务可扩展性，适合作为论文与答辩中的核心方法部分进行展示。

---

## 11. 对应代码入口

- 模型定义：`models/multimodal_transformer.py`
- 训练器：`training/trainer.py`
- 训练脚本：`scripts/f_train_model.py`
- 推理器：`inference/predictor.py`
- 模型配置：`configs/model_config.yaml`
- 数据配置：`configs/data_config.yaml`
- 训练配置：`configs/training_config.yaml`
