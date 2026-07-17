# 推理与可视化指南

## 1. Event Probability (事件概率) 计算

### 计算流程

```
输入: 融合的特征向量 (B, D)
  ↓
线性层: hidden_dim → hidden_dim/2 (ReLU激活)
  ↓
线性层: hidden_dim/2 → 1 (无激活)
  ↓
输出: logits (B, 1)
  ↓
Sigmoid激活函数: sigmoid(logits) → [0, 1]
  ↓
输出: event_prob (B, 1)
```

### 代码实现

**模型输出** (`models/multimodal_transformer.py`):
```python
# 事件概率预测头（没有Sigmoid，使用BCE loss处理）
self.event_prob_predictor = nn.Sequential(
    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
    nn.ReLU(),
    nn.Linear(self.hidden_dim // 2, 1)  # 输出logits
)

# 前向传播
event_prob = self.event_prob_predictor(pooled_features)  # (B, 1)
```

**后处理** (`inference/post_processing.py`):
```python
def _process_event_probability(self, predictions):
    event_prob = predictions['event_prob_mean']
    
    # 应用Sigmoid将logits转换为概率
    if isinstance(event_prob, torch.Tensor):
        event_prob = torch.sigmoid(event_prob).item()
    else:
        # numpy版本
        event_prob = 1.0 / (1.0 + np.exp(-float(event_prob)))
    
    # 确保在[0, 1]范围内
    event_prob = max(0.0, min(1.0, float(event_prob)))
    
    predictions['processed_event_prob'] = event_prob
    return predictions
```

### 意义解释

- **event_prob = 0.0**: 该时间窗口内**不太可能**发生太阳耀斑
- **event_prob = 0.5**: **中等可能性**发生太阳耀斑
- **event_prob = 1.0**: **非常可能**发生太阳耀斑

---

## 2. Time Prediction 细节

### 时间预测单位: **小时 (Hours)**

### 三个分量的含义

| 分量 | 名称 | 范围 | 含义 |
|------|------|------|------|
| `time_pred[0]` | 开始时间偏移 (Start Offset) | ≥ 0 | 相对于窗口开始的耀斑开始时间（小时） |
| `time_pred[1]` | 峰值时间偏移 (Peak Offset) | ≥ min_interval | 相对于窗口开始的峰值时间（小时） |
| `time_pred[2]` | 持续时间 (Duration) | 1～48 | 耀斑持续的时间长度（小时） |

### 配置约束

在 `inference_config.yaml` 中配置时间预测约束:

```yaml
inference:
  time_processing:
    min_prediction_interval: 1.0  # 小时
    max_prediction_horizon: 48.0  # 小时
```

### 时间转换示例

如果一个窗口的时间戳从 `2023-01-01 10:00:00` 开始：
- `time_pred[0] = 0.5` → 耀斑开始于 `2023-01-01 10:30:00`
- `time_pred[1] = 2.0` → 耀斑峰值于 `2023-01-01 12:00:00`
- `time_pred[2] = 1.5` → 耀斑持续 1.5 小时 （10:30～12:00）

---

## 3. 可视化功能

### 3.1 可视化模块结构

**主要类**: `inference/visualization.py::PredictionVisualizer`

```python
class PredictionVisualizer:
    def visualize_prediction(self, modality_images, predictions, event_id, ...)
        # 在原图上绘制单个预测的bbox
    
    def visualize_batch_predictions(self, predictions_list, output_dir, ...)
        # 生成批量预测的网格可视化
    
    def create_summary_report(self, predictions_list, output_dir, event_id)
        # 生成事件摘要报告（4合1可视化）
```

### 3.2 单预测可视化 (visualize_prediction)

**输出**：在原图上绘制出预测的边界框，并标注：
- 预测类别（颜色编码）
- 分类概率
- 事件概率
- 时间预测（开始、峰值、持续时间）
- 边界框索引号

**颜色编码**：
- 🟢 **绿色** → No Event (类别 0)
- 🔴 **红色** → Eruptive/CME (类别 1)
- 🟡 **黄色** → Confined (类别 2)

**示例输出**:
```
predictions/direct_run/test/EVT_XXXXXXX/visualizations/
├── prediction_window_0.png      # 窗口0的预测可视化
├── prediction_window_6.png      # 窗口6的预测可视化
└── summary_report_EVT_XXXXX.png # 事件摘要报告
```

### 3.3 摘要报告 (create_summary_report)

生成包含4个子图的摘要报告：

1. **类别分布图** (左上)
   - 显示各个类别的窗口数量
   - 条形图 + 百分比

2. **事件概率分布** (右上)
   - 直方图显示所有窗口的事件概率分布
   - 红色虚线标注平均值

3. **时间预测散点图** (左下)
   - X轴：开始时间偏移
   - Y轴：持续时间
   - 点的颜色：峰值时间偏移

4. **文本统计信息** (右下)
   - 事件ID
   - 总窗口数
   - 各类别统计及百分比
   - 事件概率统计（平均、最大、最小）

### 3.4 使用方法

#### 方式1：直接运行脚本（推荐）

编辑 `DIRECT_RUN_CONFIG`：

```python
USE_DIRECT_RUN_CONFIG = True
DIRECT_RUN_CONFIG = {
    'run_mode': 'test_set',  # 或 'single_event'
    'model_path': 'logs/checkpoints/solar_flare_model_best.pth',
    'data_path': 'data/Solar_Flares_CME_dataset.h5',
    'event_id': '',  # 单事件模式留空则自动选择
    'output_dir': 'predictions/direct_run',
    'config_path': 'configs/inference_config.yaml',
    'disable_visualization': False,  # 启用可视化
}
```

然后运行：
```bash
python scripts/g_inference_pipeline.py
```

#### 方式2：命令行参数

```bash
python scripts/g_inference_pipeline.py \
    --run_mode single_event \
    --model_path logs/checkpoints/solar_flare_model_best.pth \
    --data_path data/Solar_Flares_CME_dataset.h5 \
    --event_id EVT_20230101_001 \
    --output_dir predictions/manual \
    --config_path configs/inference_config.yaml
```

#### 方式3：批量测试整个数据集

```bash
python scripts/g_inference_pipeline.py \
    --run_mode test_set \
    --model_path logs/checkpoints/solar_flare_model_best.pth \
    --data_path data/Solar_Flares_CME_dataset.h5 \
    --split_name test \
    --output_dir predictions/batch_test
```

### 3.5 输出文件结构

#### 单事件模式输出：
```
predictions/direct_run/
├── EVT_20230101_001_predictions.json           # 所有窗口的JSON预测
├── EVT_20230101_001_predictions_detailed.csv   # 详细的表格格式
├── EVT_20230101_001_alerts.json                # 行动警报
└── visualizations/                             # 可视化输出
    ├── prediction_window_0.png
    ├── prediction_window_6.png
    └── summary_report_EVT_20230101_001.png
```

#### 批量测试模式输出：
```
predictions/direct_run/test/
├── EVT_20230101_001/
│   ├── predictions.json
│   ├── predictions_detailed.csv
│   ├── alerts.json
│   └── visualizations/
│       ├── prediction_window_0.png
│       └── summary_report_EVT_20230101_001.png
├── EVT_20230102_002/
│   └── ...
├── test_predictions.json              # 全部事件的汇总
├── test_predictions_detailed.csv      # 全部事件的详细表格
└── test_event_summary.csv             # 事件级别统计
```

---

## 4. 关键配置参数

### 推理配置 (configs/inference_config.yaml)

```yaml
inference:
  windowing:
    window_size: 9          # 滑动窗口大小（帧数）
    stride: 6               # 滑动步长（帧数）
    use_uncertainty: true   # 是否使用MC Dropout
    modalities: ['magnetogram', 'euv_94', 'euv_171', 'euv_193', 'halpha']

  thresholds:
    class0: 0.3   # 无事件的分类概率阈值
    class1: 0.5   # 爆发耀斑的分类概率阈值
    class2: 0.7   # 束缚耀斑的分类概率阈值

  time_processing:
    min_prediction_interval: 1.0   # 最小时间间隔（小时）
    max_prediction_horizon: 48.0   # 最大预测范围（小时）

  bbox_processing:
    min_area: 10              # 最小边界框面积
    nms_threshold: 0.5        # NMS IoU阈值
    max_detections: 10        # 最大检测数量

model:
  sequence_length: 9   # 输入序列长度
  num_classes: 3       # 分类类别数
  max_activities: 5    # 每个事件的最大活动数
```

---

## 5. 常见问题 (FAQ)

### Q1: event_prob 为什么总是 0-1 之间的小数？
**A:** 这是通过 Sigmoid 函数处理后的结果。模型输出的是 logits（无界实数），通过 Sigmoid 将其映射到 [0, 1] 范围。

### Q2: time_pred 的单位是什么？
**A:** 所有时间预测的单位都是**小时**。

### Q3: 如何调整时间预测范围？
**A:** 修改 `configs/inference_config.yaml` 中的：
```yaml
time_processing:
  min_prediction_interval: 1.0      # 改为需要的最小值
  max_prediction_horizon: 48.0       # 改为需要的最大值
```

### Q4: 可视化中的 bbox 坐标是什么格式？
**A:** 使用归一化坐标 [x1, y1, x2, y2]，范围在 [0, 1]，其中：
- x1, y1 = 左上角
- x2, y2 = 右下角

在绘制时会自动转换为像素坐标。

### Q5: 如何只生成预测而不生成可视化？
**A:** 使用命令行参数：
```bash
python scripts/g_inference_pipeline.py ... --disable_visualization
```

---

## 6. 性能优化建议

1. **减少窗口数量**：在 `DIRECT_RUN_CONFIG` 中设置 `max_events` 来限制测试事件数
2. **禁用不确定性估计**：设置 `disable_uncertainty=True` 可加速推理
3. **关闭可视化**：设置 `disable_visualization=True` 可大幅减少内存使用
4. **使用GPU**：确保 `configs/inference_config.yaml` 中的模型在可用GPU上运行

---

## 7. 故障排除

### 问题：matplotlib 导入失败
**解决**：
```bash
pip install matplotlib --upgrade
pip install numpy --upgrade  # 解决NumPy版本冲突
```

### 问题：可视化图片不生成
**解决**：检查输出目录是否可写，或尝试重新创建输出目录

### 问题：内存不足
**解决**：
- 减少 `max_events`
- 使用 `--disable_uncertainty`
- 减少 `window_size`
