下面给你一份**尽量贴你当前代码结构**、同时又足够具体可落地的两阶段改造蓝图。

我会按这几个部分展开：

1. **先定义目标**
2. **现有模型哪里要保留，哪里要拆**
3. **第一阶段：时空多模态检测器**
4. **第二阶段：局部 + 上下文 + 全局 的分类/时间预测器**
5. **跨模态融合怎么做**
6. **时序信息怎么保留进去**
7. **输入输出 tensor shape 逐模块写清楚**
8. **训练流程怎么改**
9. **如何减小级联误差**
10. **建议的分步实施顺序**

---

# 1. 目标先明确

你现在要的不是普通“两阶段检测”而已，而是一个：

- **阶段 1**：在多模态时序数据里定位 active region / bbox
- **阶段 2**：基于 bbox 对应区域，再结合**周边上下文**和**全局背景**，做：
  - eruptive/confined 分类
  - 时间回归
  - event probability 预测

同时还要满足：

- **时序不能丢**
- **跨模态不能丢**
- **尽量减小级联误差**
- **局部与全局都要保留**

所以最合适的形式不是“先裁图，再做普通分类”，而是：

## 最终推荐框架
**Spatiotemporal Multimodal Two-Stage Model**

### Stage 1
`多模态时空检测器`
输出：
- `K` 个候选框
- 每个框的置信度
- 每个框的对象级 embedding
- 全局时空特征

### Stage 2
`ROI级时空多模态预测器`
输入：
- local ROI
- context ROI
- global feature
- detection confidence / box embedding

输出：
- 每个 ROI 的类别
- 时间回归
- event probability

---

# 2. 现有模型哪里保留，哪里重构

你当前最关键的问题是：

```267:279:models/multimodal_transformer.py
temporal_features = self.transformer_encoder(fused_features)  # (B, T, D)  
pooled_features = temporal_features.mean(dim=1)  # (B, D)

class_logits = self.classifier(pooled_features)
bbox_raw = self.bbox_predictor(pooled_features)
```

bbox 是从全局 pooled vector 直接出来的，所以没有空间定位能力。

---

## 建议保留的东西
你现在这些部分还可以保留思路，但要升级：

- 多模态编码器的整体框架
- GRU / Transformer 的时序建模思路
- 多任务输出思路
- `max_activities` 多对象上限的设定

---

## 必须改掉的东西
### 改掉 1：过早的空间池化
这里要改：

```29:37:models/multimodal_transformer.py
self.input_projection = nn.Sequential(
    ...
    nn.AdaptiveAvgPool2d((1, 1))
)
```

这一步必须拆掉，否则 bbox 永远学不到空间。

### 改掉 2：bbox 不再从全局向量直接回归
bbox 必须来自：
- 空间 feature map
- 或 query 对时空 token 的 cross-attention

我推荐你用：
- **Stage 1：query-based proposal detector**
- 因为你已经有 `max_activities` 设定，很适合 query 方案

---

# 3. 第一阶段：时空多模态检测器蓝图

---

## 3.1 第一阶段的核心职责
不是只输出一个框，而是输出：

- `K=max_activities` 个候选对象
- 每个对象：
  - `bbox`
  - `objectness/confidence`
  - `object embedding`
- 同时输出全局时空语义特征供第二阶段使用

---

## 3.2 第一阶段总体结构

## 模块 A：Per-modality spatial encoder
每个模态单独编码每一帧，保留空间图。

### 输入
每个模态：
\[
x_m \in \mathbb{R}^{B \times T \times C_m \times H \times W}
\]

### 输出
\[
f_m^{spatial} \in \mathbb{R}^{B \times T \times D \times H' \times W'}
\]

其中：
- `B` batch size
- `T` 序列长度
- `D` hidden_dim
- `H', W'` 例如 `H/8, W/8` 或 `H/16, W/16`

### 说明
这里**不要池化到 1x1**。  
只做卷积下采样，保留空间分辨率。

---

## 模块 B：Cross-modal spatial fusion
每个时刻，把多个模态的空间图融合。

### 输入
多个模态的：
\[
f_m^{spatial} \in \mathbb{R}^{B \times T \times D \times H' \times W'}
\]

### 输出
\[
f^{fused} \in \mathbb{R}^{B \times T \times D \times H' \times W'}
\]

### 推荐做法
先简单做，不要一开始过度复杂：

#### v1 推荐
- 各模态 feature stack 后做 mean / weighted sum
- 再接一个 `1x1 conv + norm + relu`

即：
\[
f^{fused}_{t} = Conv1x1(\text{Mean}_m(f_{m,t}^{spatial}))
\]

#### v2 升级
如果后面要增强，可以做：
- cross-modal attention on spatial tokens

但第一版不用这么复杂。

---

## 模块 C：Temporal fusion on spatial maps
这里是时序重点，不能丢。

你要让每个空间位置都能看见整个时间序列。  
最稳妥的方法不是先全局池化，而是做**时空特征聚合**。

### 推荐方式 1：Temporal Conv/GRU on each spatial cell
把每个空间位置 `(h,w)` 的时间序列拿出来做 GRU：

输入：
\[
f^{fused} \in \mathbb{R}^{B \times T \times D \times H' \times W'}
\]

reshape 成：
\[
(B \cdot H' \cdot W', T, D)
\]

送入 temporal encoder，得到：

\[
f^{st} \in \mathbb{R}^{B \times H' \times W' \times D}
\]

再转成：
\[
f^{st} \in \mathbb{R}^{B \times D \times H' \times W'}
\]

### 推荐方式 2：Temporal attention pooling
更简单一点：

- 对每个空间位置做时间 attention pooling
- 学出一个该位置的时序汇总表示

我建议第一版用 **GRU 或 temporal attention pooling**，因为你现在已经有 GRU 经验。

---

## 模块 D：Global spatiotemporal feature branch
同时保留全局分支，避免你后面分类又丢掉全局背景。

从 `f^{st}` 走一个全局池化：

\[
g \in \mathbb{R}^{B \times D}
\]

这个 `g` 后面给第二阶段做 global prior。

---

## 模块 E：Proposal decoder / object queries
这是两阶段里非常关键的一步。

你当前有 `max_activities=K`，所以非常适合：

- 定义 `K` 个 learnable object queries

\[
Q \in \mathbb{R}^{K \times D}
\]

扩展成 batch：

\[
Q_b \in \mathbb{R}^{B \times K \times D}
\]

然后用这些 queries 去 attend 时空空间 token。

### 输入 token
把 `f^{st}` 展平：

\[
tokens \in \mathbb{R}^{B \times (H'W') \times D}
\]

### decoder 输出
\[
z^{obj} \in \mathbb{R}^{B \times K \times D}
\]

每个 query 对应一个候选 active region。

---

## 模块 F：Stage-1 detection heads
对每个 `object query feature` 输出：

### 1. bbox
\[
bbox^{(1)} \in \mathbb{R}^{B \times K \times 4}
\]
格式：归一化 `xyxy`

### 2. objectness
\[
score^{(1)} \in \mathbb{R}^{B \times K \times 1}
\]

### 3. object embedding
直接保留：
\[
z^{obj} \in \mathbb{R}^{B \times K \times D}
\]

### 4. global feature
\[
g \in \mathbb{R}^{B \times D}
\]

---

## Stage 1 输出总结
第一阶段最终输出建议是：

```python
{
    "proposal_boxes":      (B, K, 4),
    "proposal_scores":     (B, K, 1),
    "proposal_features":   (B, K, D),
    "spatial_feature_map": (B, D, H', W'),
    "global_feature":      (B, D)
}
```

---

# 4. 第二阶段：局部 + 上下文 + 全局 的分类/时间预测器

这里是你最关心的：  
**不要只把定位框内部塞给分类头，而是要把周边和全局也融合进去。**

这一步我建议直接做成 3 路甚至 4 路。

---

## 4.1 第二阶段输入来源

对每个 proposal `k`，拿到：

- `bbox_k`
- `score_k`
- `proposal_feature_k`
- `global_feature`
- `spatial_feature_map`

然后抽 3 类 ROI。

---

## 4.2 Local ROI branch
原始 proposal 框：

\[
B_k
\]

从时空特征图里抽局部区域。

### 更推荐的输入
不要只从单帧图像抽，而是从**时空特征图对应的多时刻表示**抽。

你有两种做法：

### 做法 A：先时间聚合，再抽 ROI
从 `f^{st} \in (B,D,H',W')` 上做 ROIAlign：
\[
r_k^{local} \in \mathbb{R}^{B \times K \times D \times P \times P}
\]

### 做法 B：先每个时刻抽 ROI，再时间编码
从 `f^{fused} \in (B,T,D,H',W')` 上，每个时刻都 ROIAlign：
\[
r_k^{local,time} \in \mathbb{R}^{B \times K \times T \times D \times P \times P}
\]

然后再对 ROI 序列做时间建模。

### 我建议
**第二阶段尽量保留时序**，所以更推荐做法 B。

---

## 4.3 Context ROI branch
构造放大框：

\[
B_k^{ctx} = Expand(B_k, s)
\]

例如：
- `s = 2.0`

从同一组时空特征图里抽上下文区域：

\[
r_k^{ctx,time} \in \mathbb{R}^{B \times K \times T \times D \times P \times P}
\]

它用来看：
- 周边磁场
- 近邻结构
- 活动区环境

---

## 4.4 Global branch
保留第一阶段的全局特征：

\[
g \in \mathbb{R}^{B \times D}
\]

广播到每个 proposal：

\[
g_k \in \mathbb{R}^{B \times K \times D}
\]

---

## 4.5 Proposal branch
第一阶段 decoder 自身输出的 proposal feature 也很重要：

\[
z_k^{obj} \in \mathbb{R}^{B \times K \times D}
\]

这个表示“这个 query 认为自己找到了什么对象”。

---

# 5. 第二阶段内部怎么处理时序与跨模态

这里是你特别要求的重点：  
**时序不要丢，跨模态也要考虑。**

---

## 5.1 ROI 级时序建模
假设你从每个 proposal 抽到了：

\[
r_k^{local,time} \in \mathbb{R}^{B \times K \times T \times D \times P \times P}
\]

先做空间池化：

\[
u_k^{local,time} \in \mathbb{R}^{B \times K \times T \times D}
\]

对 `context ROI` 同样得到：

\[
u_k^{ctx,time} \in \mathbb{R}^{B \times K \times T \times D}
\]

然后对时间维做编码，可以用：

- ROI-level GRU
- ROI-level temporal transformer
- temporal attention pooling

输出：

\[
u_k^{local} \in \mathbb{R}^{B \times K \times D}
\]
\[
u_k^{ctx} \in \mathbb{R}^{B \times K \times D}
\]

---

## 5.2 跨模态怎么保留到第二阶段
这里我建议**阶段 1 先融合，阶段 2 也保留轻量模态感知**。

有两种版本。

### 版本 1：先融合后 ROI
最简单：
- 跨模态先在 stage 1 融成统一 feature map
- stage 2 从 fused map 上抽 ROI

优点：
- 实现最简单

缺点：
- 模态细节可能有损失

### 版本 2：每模态分别 ROI，再做 ROI-level multimodal fusion
更强一些：
对每个模态单独抽：

\[
r_{k,m}^{local,time} \in \mathbb{R}^{B \times K \times T \times D \times P \times P}
\]

再编码成：

\[
u_{k,m}^{local} \in \mathbb{R}^{B \times K \times D}
\]

然后做跨模态 fusion：

\[
u_k^{local,fused} = Fusion_m(u_{k,m}^{local})
\]

context 同理。

### 我的建议
如果你第一版想可落地，先做：

- **Stage 1 做跨模态融合**
- **Stage 2 用 fused ROI**

以后如果 bbox 和分类稳定了，再升级到 per-modality ROI。

---

# 6. 第二阶段融合头的具体设计

---

## 6.1 Proposal-level fusion
对每个 proposal k，拿到这几路：

- `u_k^{local}`: ROI 局部时序特征
- `u_k^{ctx}`: ROI 上下文时序特征
- `z_k^{obj}`: 第一阶段 object query 特征
- `g_k`: 全局特征
- `score_k`: proposal confidence

融合成：

\[
h_k = Fusion([u_k^{local}, u_k^{ctx}, z_k^{obj}, g_k, score_k])
\]

### 建议 fusion 形式
第一版推荐：

```python
concat -> linear -> relu -> layernorm -> dropout
```

更强一点可以做 gate：

\[
\alpha,\beta,\gamma,\delta = \text{softmax}(MLP([...]))
\]
\[
h_k = \alpha u_k^{local} + \beta u_k^{ctx} + \gamma z_k^{obj} + \delta g_k
\]

这样模型会自己学：
- proposal 很准时，多信 local
- proposal 不稳时，多信 context/global

这就是减轻级联误差的关键机制之一。

---

## 6.2 第二阶段输出头
对每个 proposal 输出：

### 分类
\[
class\_logits^{(2)} \in \mathbb{R}^{B \times K \times C}
\]

### 时间回归
\[
time\_pred^{(2)} \in \mathbb{R}^{B \times K \times 3}
\]

### event probability
\[
event\_prob^{(2)} \in \mathbb{R}^{B \times K \times 1}
\]

也可以保留 refinement bbox：

### bbox refinement（推荐）
\[
\Delta bbox^{(2)} \in \mathbb{R}^{B \times K \times 4}
\]

得到 refined box：

\[
bbox^{(final)} = Refine(bbox^{(1)}, \Delta bbox^{(2)})
\]

这样第二阶段还能回头修正第一阶段 bbox，进一步减轻级联误差。

---

# 7. 全模型 tensor shape 总结

下面给你一条完整的 shape 流。

假设：

- `B` = batch size
- `T` = 序列长度，比如 48
- `M` = 模态数
- `H,W` = 输入尺寸，比如 512,512
- `D` = hidden_dim，比如 256
- `H',W'` = 下采样后尺寸，比如 64,64
- `K` = max_activities，比如 5
- `P` = ROIAlign 输出大小，比如 7

---

## 输入
每个模态：
\[
x_m: (B, T, C_m, H, W)
\]

---

## Stage 1
### per-modality spatial encoder
\[
f_m^{spatial}: (B, T, D, H', W')
\]

### cross-modal fusion
\[
f^{fused}: (B, T, D, H', W')
\]

### temporal fusion on spatial cells
\[
f^{st}: (B, D, H', W')
\]

### global pooling
\[
g: (B, D)
\]

### flatten spatial tokens
\[
tokens: (B, H'W', D)
\]

### object queries
\[
Q: (B, K, D)
\]

### decoder output
\[
z^{obj}: (B, K, D)
\]

### detection heads
\[
bbox^{(1)}: (B, K, 4)
\]
\[
score^{(1)}: (B, K, 1)
\]

---

## Stage 2
### ROIAlign on fused temporal map
如果先对每个时刻抽 ROI：

\[
r^{local}: (B, K, T, D, P, P)
\]
\[
r^{ctx}: (B, K, T, D, P, P)
\]

### spatial pooling per ROI-frame
\[
u^{local,time}: (B, K, T, D)
\]
\[
u^{ctx,time}: (B, K, T, D)
\]

### ROI temporal encoder
\[
u^{local}: (B, K, D)
\]
\[
u^{ctx}: (B, K, D)
\]

### global repeat
\[
g_k: (B, K, D)
\]

### fusion
\[
h: (B, K, D)
\]

### stage-2 heads
\[
class\_logits^{(2)}: (B, K, C)
\]
\[
time\_pred^{(2)}: (B, K, 3)
\]
\[
event\_prob^{(2)}: (B, K, 1)
\]
\[
\Delta bbox^{(2)}: (B, K, 4)
\]

---

# 8. 训练流程怎么变

这里是最关键的落地部分。

---

# 8.1 训练总思路
建议分成 **三阶段训练**，不要一上来端到端全开。

---

## Phase A：先单独训 Stage 1 检测器
目标：
- proposal box 能靠谱
- score 有区分度
- 不再塌缩到中心大框

### 输入
全量多模态时序

### 监督
- `bbox loss`：GIoU / L1 + GIoU
- `objectness loss`
- proposal matching

### matching
推荐用 Hungarian matching 或稳定排序匹配。

如果你不想第一版就上 Hungarian，也可以先按已有 activity 槽位顺序监督，但长期建议上 matching。

### 输出
得到一个稳定的 proposal detector。

---

## Phase B：冻结/半冻结 Stage 1，训练 Stage 2
目标：
- 让 ROI-level 分类/时间预测学会利用 local/context/global
- 建立对 bbox 偏差的鲁棒性

### 训练输入来源
Stage 2 的 ROI 不要只来自一种框，而要混合：

- `GT bbox`
- `Stage1 predicted bbox`
- `GT + random jitter`

### 比例建议
初期：
- 60% GT
- 20% jittered GT
- 20% predicted

中期：
- 30% GT
- 20% jittered GT
- 50% predicted

后期：
- 10% GT
- 20% jittered GT
- 70% predicted

这就是 scheduled sampling / anti-cascade bias。

---

## Phase C：端到端联合微调
目标：
- 检测器和第二阶段协同优化
- second-stage refinement 反过来帮助 bbox

### loss 组成
\[
L = \lambda_{det}L_{stage1\_det} + \lambda_{cls}L_{cls} + \lambda_{time}L_{time} + \lambda_{event}L_{event} + \lambda_{refine}L_{bbox\_refine}
\]

建议：
- 初期 `det loss` 大一点
- 后期均衡

---

# 8.2 每阶段 loss 设计建议

---

## Stage 1 loss
### proposal classification/objectness
- BCE 或 focal loss

### proposal bbox
- `L1 + GIoU`
不要只用 GIoU，建议：
\[
L_{bbox1} = \lambda_{l1}L1 + \lambda_{giou}L_{giou}
\]

因为纯 GIoU 有时不够稳定。

---

## Stage 2 loss
### 分类
- 只对 matched / valid proposals 算 cross entropy

### 时间
- 只对正样本 proposal 算 MSE / SmoothL1

### event probability
- BCEWithLogits

### bbox refinement
- 对 matched 正样本 proposal 算 refined bbox loss

---

# 9. 如何减小级联误差：这套蓝图里的具体机制

你特意关心这个，我单列一下。

---

## 机制 1：local + context + global 三路融合
不是只依赖 bbox 内部内容。

---

## 机制 2：训练时混合 GT / pred / jittered bbox
避免训练-推理不一致。

---

## 机制 3：proposal confidence-aware fusion
proposal score 低时，自动更依赖 context/global。

---

## 机制 4：第二阶段做 bbox refinement
不是被动吃第一阶段框，而是能修正它。

---

## 机制 5：保留全局支路
即使 ROI 偏了，模型还有整图时空背景信息兜底。

---

## 机制 6：时序信息在两阶段都保留
这很重要，因为如果 stage 2 只看单时刻 patch，也会放大误检风险。  
保留 ROI 的时间演化，可以提升鲁棒性。

---

# 10. 按你当前代码，建议改哪些模块

我不直接写实现，但给你文件级蓝图。

---

## `models/multimodal_transformer.py`
建议拆成几个类：

### 新类 1：`SpatialModalityEncoder`
职责：
- 输入 `(B,T,C,H,W)`
- 输出 `(B,T,D,H',W')`

### 新类 2：`MultimodalSpatiotemporalFusion`
职责：
- 输入多个模态空间特征
- 输出 fused 时空特征 `(B,T,D,H',W')`

### 新类 3：`ProposalDecoder`
职责：
- 输入 `(B,D,H',W')` 或 spatial tokens
- 输出：
  - `proposal_boxes`
  - `proposal_scores`
  - `proposal_features`
  - `global_feature`

### 新类 4：`RoiTemporalEncoder`
职责：
- 输入 `(B,K,T,D,P,P)`
- 输出 `(B,K,D)`

### 新类 5：`StageTwoPredictor`
职责：
- 输入 local/context/global/proposal features
- 输出 class/time/event/refined box

### 顶层新模型
比如：
- `TwoStageMultimodalForecaster`

---

## `training/trainer.py`
要拆出两阶段 loss：

### 新增
- stage1 matching
- stage1 det loss
- ROI source mixing（GT/pred/jitter）
- stage2 loss
- joint training schedule

---

## `data/dataset.py`
当前数据已经有：
- `bbox`
- `label`
- `time_features`

可能还需要补：
- 更稳定的 activity 排序策略
- 如果要 Hungarian，可以不强依赖固定顺序

---

# 11. 我建议的最小可落地版本

如果你想最小风险实现，我建议你先做这个版本，不要一步到顶。

---

## Version 1
### Stage 1
- 保留空间特征
- fused 时空特征图
- `K` 个 query proposals
- 输出 bbox + score + proposal feature + global feature

### Stage 2
- 只做两个 ROI：
  - local
  - context(2x)
- ROI temporal encoder
- 与 global/proposal feature concat
- 输出 class/time/event
- 暂时不做 refinement 或 top-k proposal

### 训练
- 先训 stage1
- 再训 stage2（混 GT/pred bbox）
- 最后联合微调

这是成功率最高、复杂度可控的版本。

---

# 12. 最后的结构示意图

你可以把整个模型理解成这样：

```text
多模态时序输入
    ↓
Per-modality Spatial Encoder
    ↓
Cross-modal Spatial Fusion
    ↓
Temporal Fusion on Spatial Maps
    ├── Global Pooling → global_feature
    └── Spatial Tokens + Object Queries → K proposals
                                ↓
                proposal_boxes / proposal_scores / proposal_features
                                ↓
                     ROI Extractor (local + context)
                                ↓
                    ROI Temporal Encoder (keep time)
                                ↓
     Fuse(local_roi_feature, context_roi_feature, proposal_feature, global_feature, score)
                                ↓
            ├── classification head
            ├── time regression head
            ├── event probability head
            └── bbox refinement head(optional)
```

---

# 一句话总结

如果你要一份最核心的设计原则，那就是：

> **阶段 1 用“保留空间分辨率的时空多模态检测器”找 proposal，阶段 2 不只吃框内 patch，而是同时融合“局部 ROI + 扩张上下文 ROI + 全局时空特征 + proposal feature”，并且训练时混合 GT/pred bbox 来降低级联误差。**

这套结构既保留了：
- **时序**
- **跨模态**
- **局部区域**
- **周边上下文**
- **全局背景**

也符合你对太阳活动区任务的物理直觉。

---

如果你愿意，我下一步可以继续给你两种更具体的输出之一：

1. **把这份蓝图直接转成“代码模块清单 + 类名/函数名/文件修改清单”**  
2. **直接给你画出一个“Version 1 最小可落地实现”的详细 shape 流和伪代码**

我建议下一步做第 1 个，这样你就能开始真正改代码。