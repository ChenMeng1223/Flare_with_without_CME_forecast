# Copilot Instructions (精简版)

这是一个基于多模态 Transformer 的太阳耀斑 + CME 预测项目。下面列出对 AI 代码助手立刻有用的要点、精确命令和代码位置。

- **快速命令**:
  - 训练（快速迭代）: `python run_training.py --config configs/training_config.yaml --data data/solar_flares_dataset.h5`
  - CLI（所有子命令）: `python main.py train --config configs/training_config.yaml --data data/solar_flares_dataset.h5`
  - 生成 HDF5 数据集: `python scripts/a_create_hdf5_dataset.py --events_csv data/events_example.csv --output data/solar_flares_dataset.h5`

- **数据流 & 存储**:
  - HDF5 组织：事件存为 `events/{event_id}` 组，组属性包含 `num_frames`，帧数组与 `timestamps` 和元数据并列。
  - 读取路径：`data/hdf5_reader.py` → `data/dataset.py` (滑动窗口逻辑，默认 `sequence_length=48, stride=6`) → `data/data_loader.py`。

- **模型与输入约定**:
  - 主要模型：`models/multimodal_transformer.py`，每种模态单独编码后做融合（注意 `ModalityEncoder` 与 `CrossModalAttention`）。
  - 张量形状：`(B, T, C, H, W)`（Batch, Time, Channels, H, W）。
  - 缺失模态处理：见 `SolarFlareDataset(handle_missing='interpolate')`。

- **配置与运行时约定**:
  - 集中配置：`utils/config_utils.py` 的 `ConfigManager`（使用 `load_config(path)` 快速加载）。
  - 常用配置文件：`configs/{data_config.yaml,model_config.yaml,training_config.yaml,inference_config.yaml}`。

- **训练 & 检查点**:
  - 训练器：`training/trainer.py`（包含优化器、scheduler、loss 权重、W&B 支持）。
  - 检查点管理：`training/checkpoint_manager.py`（保存 model/optimizer/epoch/metrics），`run_training.py` 支持 `--resume`。

- **推理与产出**:
  - 推理器：`inference/predictor.py`（加载 checkpoint、支持不确定性估计和滑动窗口时序预测）。

- 如需扩展说明（HDF5 元字段、checkpoint 命名规范等），告诉我你最想补充的点。
