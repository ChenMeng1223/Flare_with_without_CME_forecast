# 推理与可视化改进总结

## 本次改进概览

### 1️⃣ **event_prob 计算修复**
- ✅ **问题**：原代码直接将 logits 映射到 [0, 1]，没有应用 Sigmoid 函数
- ✅ **修复**：在 `post_processing.py` 中添加 `torch.sigmoid()` 将 logits 正确转换为概率
- ✅ **结果**：event_prob 现在是真正的概率值，范围 [0, 1]

### 2️⃣ **时间预测单位确认**
- ✅ **单位**：**小时（Hours）**
- ✅ **三个分量**：
  - `[0]` = 耀斑开始时间（相对于窗口开始）
  - `[1]` = 耀斑峰值时间
  - `[2]` = 耀斑持续时间

### 3️⃣ **新增详细可视化功能**
- ✅ **创建模块**：`inference/visualization.py::PredictionVisualizer`
- ✅ **单预测可视化**：在原图上绘制 bbox 及详细标注
- ✅ **批量预测可视化**：网格视图展示多个事件
- ✅ **摘要报告**：4合1统计图表（类别分布、概率分布、时间预测、统计信息）

---

## 🎯 使用示例

### 示例 1：预测单个事件并生成可视化

```python
# 编辑 DIRECT_RUN_CONFIG 或使用命令行
DIRECT_RUN_CONFIG = {
    'run_mode': 'single_event',
    'event_id': 'EVT_20230101_001',
    'disable_visualization': False,  # 启用可视化
}

# 运行
python scripts/g_inference_pipeline.py
```

**输出**：
```
predictions/direct_run/
├── EVT_20230101_001_predictions.json
├── EVT_20230101_001_predictions_detailed.csv
├── EVT_20230101_001_alerts.json
└── visualizations/
    ├── prediction_window_0.png        ← 原图 + bbox 标注
    ├── prediction_window_6.png        ← 原图 + bbox 标注
    └── summary_report_EVT_XXXX.png   ← 4合1统计图表
```

### 示例 2：批量测试并生成所有可视化

```bash
python scripts/g_inference_pipeline.py \
    --run_mode test_set \
    --model_path logs/checkpoints/solar_flare_model_best.pth \
    --data_path data/Solar_Flares_CME_dataset.h5 \
    --split_name test \
    --output_dir predictions/batch_test \
    --max_events 10  # 只测试前10个事件
```

---

## 📊 可视化输出说明

### 单预测可视化 (prediction_window_X.png)

```
┌─────────────────────────────────────────────┐
│           Original Image (Magnetogram)      │
│                                             │
│  ┌──────────────────────────────────────┐  │
│  │  ╔════════════════════╗  (Red Box)   │  │
│  │  ║ Eruptive (CME)     ║ ← Class Name │  │
│  │  ║ Class: 85%         ║ ← Prob       │  │
│  │  ║ Event: 92%         ║ ← Event Prob │  │
│  │  ║ Start: 0.5h        ║ ← Time Info  │  │
│  │  ║ Peak: 2.0h         ║   (in hours) │  │
│  │  ║ Dur: 1.5h          ║              │  │
│  │  ╚════════════════════╝ ①            │  │ ← Box Index
│  │                                      │  │
│  └──────────────────────────────────────┘  │
│                                             │
│  Title: Event: EVT_20230101_001            │
│         Class: Eruptive (CME)             │
│         Class Prob: 0.850 | Event Prob: 0.920
└─────────────────────────────────────────────┘
```

**颜色编码**：
- 🟢 绿色 = No Event (类别 0)
- 🔴 红色 = Eruptive/CME (类别 1)  
- 🟡 黄色 = Confined (类别 2)

### 摘要报告 (summary_report_XXX.png)

```
┌─────────────────────────────────────────────────────────────┐
│              Event Summary Report                           │
├──────────────────────────┬──────────────────────────────────┤
│  Class Distribution      │  Event Probability Distribution  │
│  ░░░░ (50 windows)       │  ▇▇▇▇                            │
│  No Event   ░░░░░░░ (30) │  ╱_╲ Mean                        │
│  Eruptive   ░░░░░ (15)   │  ╱   ╲   ┐                       │
│  Confined   ░░░░░░ (5)   │  │ Freq│   │                     │
│             ▤▤▤ 100%     │  │   │   │ │                     │
├──────────────────────────┼──────────────────────────────────┤
│  Time Predictions        │  Text Statistics                │
│  ▲ Duration (h)          │  Event Summary Report            │
│  │      ●(3.0h)          │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━   │
│  │   ●  ●(2.5h)  ●       │  Event ID: EVT_20230101_001     │
│  │  ●   ●  ●     ●       │  Total Windows: 50              │
│  │                       │                                 │
│  │  ●   ●   ●    ●       │  Class Statistics:              │
│  │ ●    ●    ●(1.0h)     │  • No Event: 30 (60.0%)         │
│  ├─────────────────────→ │  • Eruptive: 15 (30.0%)         │
│  Peak Offset → Start     │  • Confined: 5 (10.0%)          │
│                          │                                 │
│  ◆ Warm = High Peak      │  Event Probability:             │
│                          │  • Mean: 0.650                  │
│                          │  • Max: 0.920                   │
│                          │  • Min: 0.120                   │
└──────────────────────────┴──────────────────────────────────┘
```

---

## 🔧 核心代码改进

### 修复 1：event_prob Sigmoid 处理

**位置**：`inference/post_processing.py`

```python
def _process_event_probability(self, predictions):
    event_prob = predictions['event_prob_mean']
    
    # ✅ 添加 Sigmoid 将 logits 转换为概率
    if isinstance(event_prob, torch.Tensor):
        event_prob = torch.sigmoid(event_prob).item()
    else:
        import numpy as np
        event_prob = 1.0 / (1.0 + np.exp(-float(event_prob)))
    
    event_prob = max(0.0, min(1.0, float(event_prob)))
    predictions['processed_event_prob'] = event_prob
    return predictions
```

### 修复 2：融合方法配置支持

**位置**：`models/multimodal_transformer.py`

```python
# ✅ 添加配置驱动的融合选择
fusion_config = config.get('fusion', {})
fusion_method = fusion_config.get('method', 'attention')
self.fusion_method = fusion_method

if fusion_method == 'attention':
    # 使用跨模态注意力融合
    self.cross_attention_layers = nn.ModuleList([...])
elif fusion_method in ['concatenation', 'mean']:
    # 使用简单融合模块
    self.fusion_module = SimpleFusion(...)
```

### 新增：可视化集成

**位置**：`scripts/g_inference_pipeline.py`

```python
# ✅ 在推理后自动生成可视化
if visualizer:
    visualizer.visualize_prediction(
        modality_images=modality_images,
        predictions=pred,
        event_id=event_id,
        output_dir=vis_dir
    )
    visualizer.create_summary_report(predictions, vis_dir, event_id)
```

---

## 📁 新增文件清单

| 文件 | 功能 | 状态 |
|------|------|------|
| `inference/visualization.py` | 预测可视化模块 | ✅ 新增 |
| `docs/INFERENCE_VISUALIZATION_GUIDE.md` | 详细使用指南 | ✅ 新增 |
| `inference/__init__.py` | 更新导出 | ✅ 修改 |

---

## 🚀 快速开始

### 单事件预测 + 可视化

```bash
python scripts/g_inference_pipeline.py \
    --run_mode single_event \
    --model_path logs/checkpoints/solar_flare_model_best.pth \
    --data_path data/Solar_Flares_CME_dataset.h5 \
    --event_id EVT_20230101_001 \
    --output_dir predictions/demo
```

### 批量测试 + 可视化

```bash
python scripts/g_inference_pipeline.py \
    --run_mode test_set \
    --model_path logs/checkpoints/solar_flare_model_best.pth \
    --data_path data/Solar_Flares_CME_dataset.h5 \
    --split_name test \
    --output_dir predictions/batch \
    --max_events 20
```

### 直接运行配置

编辑 `scripts/g_inference_pipeline.py` 的 `DIRECT_RUN_CONFIG`，然后：

```bash
python scripts/g_inference_pipeline.py
```

---

## ✨ 主要特性

| 功能 | 详情 |
|------|------|
| **event_prob 修复** | 正确应用 Sigmoid，概率范围 [0, 1] |
| **时间单位确认** | 所有时间预测单位为小时（hours） |
| **bbox 可视化** | 在原图上绘制预测的活动区域 |
| **详细标注** | 标注类别、概率、时间预测等信息 |
| **摘要报告** | 统计图表：类别分布、概率分布、时间预测、统计信息 |
| **批量处理** | 支持整个数据集的批量预测和可视化 |
| **灵活配置** | 支持命令行参数和配置文件 |

---

## 📝 注意事项

1. **matplotlib 依赖**：如果导入失败，运行：
   ```bash
   pip install matplotlib --upgrade
   pip install "numpy<2"  # 解决NumPy版本冲突
   ```

2. **内存占用**：生成可视化会增加内存占用，大批量测试时建议：
   - 使用 `--disable_visualization` 关闭可视化
   - 使用 `--max_events` 限制事件数量

3. **模型依赖**：必须提供有效的 `.pth` 模型文件

---

## 🎓 理解关键概念

### Event Probability (事件概率)

```
Model Output (logits) → Sigmoid → Probability [0, 1]
      任意实数          σ(x)    0% 到 100% 的概率

例：
 logits = -2    → sigmoid = 0.12  (12% 可能性)
 logits = 0     → sigmoid = 0.50  (50% 可能性)
 logits = 2     → sigmoid = 0.88  (88% 可能性)
 logits = 10    → sigmoid ≈ 1.00  (99% 可能性)
```

### Time Prediction Units (时间单位)

所有时间都以**小时**表示：
- 例：`start_offset = 0.5` 表示 30 分钟
- 例：`duration = 1.5` 表示 1 小时 30 分钟
- 例：`peak_offset = 2.0` 表示 2 小时

---

## 📞 故障排除

**问题**：可视化不生成
- 检查 matplotlib 是否安装
- 检查输出目录权限
- 查看日志输出是否有错误提示

**问题**：内存溢出
- 减少 `max_events`
- 启用 `--disable_visualization`
- 减少 `batch_size`

**问题**：event_prob 仍然不正确
- 确保已重新启动 Python（清除导入缓存）
- 检查 `post_processing.py` 是否包含 Sigmoid 处理

---

祝您使用愉快！如有任何问题，请参考详细指南 `docs/INFERENCE_VISUALIZATION_GUIDE.md`。
