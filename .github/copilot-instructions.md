# Copilot 指南（中文）

## 项目概览
本项目基于多模态 Transformer，目标是预测太阳耀斑及其是否伴随 CME（日冕物质抛射）。输入为多时间步、多模态的卫星影像与辅助物理量，输出包含分类、边界框回归与时间预测。

## 核心架构要点

### 1. 数据管线（以 HDF5 为中心）
- 入口脚本/模块：`data/hdf5_creator.py`（从事件 CSV 生成 HDF5）
- 流程：原始 CSV → HDF5（每个事件为一组）→ `HDF5DatasetReader` → `SolarFlareDataset` → DataLoader
````instructions
# Copilot 指南（中文）

## 项目概览
本项目基于多模态 Transformer，目标是预测太阳耀斑及其是否伴随 CME（日冕物质抛射）。输入为多时间步、多模态的卫星影像与辅助物理量，输出包含分类、边界框回归与时间预测。

## 核心架构要点

### 1. 数据管线（以 HDF5 为中心）
- 入口脚本：`data/hdf5_creator.py`（从事件 CSV 生成 HDF5）
- 流程：原始 CSV → HDF5（每个事件为一组）→ `HDF5DatasetReader` → `SolarFlareDataset` → DataLoader
- 关键模块：`data/hdf5_reader.py`（读取元数据并加载事件索引表）
- 数据集类：`data/dataset.py`（滑动窗口逻辑，默认 `sequence_length=48, stride=6`）
- HDF5 约定：事件存为 `events/{event_id}` 组，组属性包含 `num_frames`，帧数组与 `timestamps`、元数据并列存储。
- 常用加载模式：`HDF5DatasetReader` → `SolarFlareDataset` → `create_data_loaders()`，便于快速索引与批处理

### 2. 多模态模型结构
- 主要实现：`models/multimodal_transformer.py`（各模态并行编码后融合）
- 流程：每个模态 → `ModalityEncoder`（空间-时间 CNN + GRU/编码器）→ 跨模态注意力融合 → Transformer 解码与预测头
- 输出：多任务（分类、边界框回归、时间预测）
- 模态定义：在 HDF5 元数据或数据生成脚本中配置，常见模态包括磁场图（magnetogram）、多普勒、EUV、连续谱等
- 缺失模态处理：见 `data/dataset.py` 的 `handle_missing`（默认 `interpolate`）

### 3. 配置管理
- 集中管理：`utils/config_utils.py` 中的 `ConfigManager` 负责 YAML 加载、深度合并与校验
- 配置文件：`configs/` 下分为数据、模型、训练、推理若干 YAML（加载后通过 `load_config()` 传入训练器与预测器）
- 示例：`training/trainer.py` 从配置读取优化器、scheduler、loss 权重与 checkpoint 策略

### 4. 训练循环组织
- 训练器类：`training/trainer.py` 负责训练流程组织（训练/验证循环、指标记录、W&B 集成）
- 组成组件：
	- `CheckpointManager`（`training/checkpoint_manager.py`）— 保存最佳模型并管理回滚
	- `EarlyStopping`（`training/early_stopping.py`）— 基于验证指标早停
	- `MetricsTracker`（`training/metrics_tracker.py`）— 聚合并跟踪 epoch 指标
- 入口脚本：`run_training.py`（适合快速迭代）或通过 `main.py` 的 `train` 子命令
- 损失构成：按配置加权组合分类损失、bbox 回归损失、时间预测损失及物理约束项

### 5. 推理与后处理
- 预测器：`inference/predictor.py`（加载 checkpoint，支持批量与单事件推理）
- 后处理：`inference/post_processing.py`（阈值过滤、NMS 等）
- 不确定性估计：`inference/uncertainty_estimation.py`（用于计算置信度/不确定性）
- 输出格式：JSON（包含 `event_id`、`predicted_label`、`confidence`、`bbox`、`onset_time`）

## 开发工作流

### 快速训练示例
```bash
# 编辑训练配置：configs/training_config.yaml（修改 batch_size、learning_rate 等）
# 运行训练（保留 stdout/stderr）：
python run_training.py --config configs/training_config.yaml --data data/solar_flares_dataset.h5
# 或使用 CLI：
python main.py train --config configs/training_config.yaml --data data/solar_flares_dataset.h5
```

### 创建新数据集
```bash
# 准备事件元数据 CSV（示例字段）：event_id, start_time, end_time, flare_class, cme_associated, peak_time, peak_flux, duration, active_region
# 生成 HDF5 数据集：
python scripts/a_create_hdf5_dataset.py --events_csv data/events_example.csv --output data/solar_flares_dataset.h5
```

### 推理示例（Python）
```python
from inference.predictor import SolarFlarePredictor
predictor = SolarFlarePredictor(model_path='outputs/best_model.pth', config_path='configs/inference_config.yaml')
# 单事件预测：
res = predictor.predict_from_hdf5('data/solar_flares_dataset.h5', event_id='EVT_20230101_001', start_idx=0, end_idx=48)
# 批量/时间序列预测可用 predictor.predict_time_series()
```

## 代码模式与约定

### 张量形状约定
- 时序空间数据使用 `(B, T, C, H, W)`（Batch × Time × Channels × Height × Width）。
- 编码后常见表示为 `(B, T, D)`（Batch × Time × Embedding_dim）。
- 所有时空数据遵循该约定；在函数/类的 docstring 中注明示例形状。

### 数据加载时的异常处理
- 缺失/损坏帧：使用 `handle_missing='interpolate'`（默认）或 `forward_fill`。
- 整个事件缺失：可通过 `available_only=True` 在读取时跳过。
- 配置校验：若不满足约定会抛出 `ConfigError`（详见 `utils/config_utils.py`）。

### 检查点 & 恢复模式
- `CheckpointManager.save_checkpoint()` 持久化模型、优化器、epoch 与指标
- 总是使用 `load_checkpoint(path)` 恢复完整的训练状态
- 使用 `restore_best_weights=true` 在早停时回滚至最佳模型

### 日志约定
- 通过 `utils/logging_utils.py` 的 `setup_logging()` 设置日志
- 所有模块在顶部使用 `logger = logging.getLogger(__name__)`
- 日志级别：DEBUG（详细）、INFO（里程碑）、WARNING（可恢复问题）、ERROR（失败）

### 物理约束
- 模块：`models/physical_constraints.py`
- 作为辅助损失项使用（权重在 training_config.yaml 中配置）
- 强制执行领域知识（例如，事件开始时间必须早于峰值时间）

## 常见任务示例

### 添加新模态
1. 在 HDF5 创建脚本中更新模态列表：`data/hdf5_creator.py`
2. 确保编码器处理新输入通道：`models/multimodal_transformer.py` 的 ModalityEncoder
3. 如需要，更新数据增强：`data/transforms.py`

### 性能调优
- 在 `configs/training_config.yaml` 中调整 `loss_weights` 以平衡分类与预测精度
- 在数据集创建时修改 sequence_length 与 stride 以权衡时间分辨率
- 尝试不同调度器（cosine、step、plateau）— 参见训练器初始化

### 调试模型问题
- 使用 `inference/post_processing.py` 检查预测的 NMS 阈值
- 通过 `inference/uncertainty_estimation.py` 检查不确定性分数
- 通过数据集的 `__getitem__()` 输出验证输入数据形状
- 在配置中启用 wandb 日志以可视化训练曲线

## 关键文件索引
- 配置模板：`configs/*.yaml`
- 主要入口：`main.py`（CLI）、`run_training.py`（直接运行）
- 数据流：`data/hdf5_reader.py` → `data/dataset.py` → `data/data_loader.py`
- 模型：`models/multimodal_transformer.py`（主要架构）
- 训练：`training/trainer.py`（编排器）
- 推理：`inference/predictor.py`（批量与单事件）
