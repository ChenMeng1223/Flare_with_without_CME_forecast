# 数据集创建指南（已更新）

> 已依据当前代码方案更新，包含 event_id 生成、时间采样、HDF5 路径

## 快速开始

### 1. 准备事件元数据 `data/raw/events.xlsx`

当前流程使用 Excel 事件表 `data/raw/events.xlsx`。典型字段如下（列名需与下载脚本保持一致）：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `DATE` | string | 观测日期 | `2024.12.23` |
| `start_time` | string | 耀斑开始时间（可含 `(+1day)` 标记） | `18:20`, `23:50(+1day)` |
| `peak_time` | string | 峰值时间（同样支持 `(+1day)`） | `18:35`, `0:30:00(+1day)` |
| `end_time` | string | 耀斑结束时间 | `19:10`, `01:00(+1day)` |
| `flare_class` | string | 耀斑等级 | `M1.5`, `X2.1`, `C3.2` |
| `cme_associated` / `CME_asociated` | string/bool | 是否伴随CME | `yes` / `no` / `True` / `False` |
| `position of active region` | string/number | 活动区位置标记 | `AR12345` / `S15E10` |
| （可选）`peak_flux` | float | 峰值通量 | `1.5e-4` |
| （可选）`duration` | float | 持续时间（分钟） | `90` |

在创建 HDF5 时，`scripts/d_create_hdf5_dataset.py` 会自动完成：

- 使用 `DATE + start_time / peak_time / end_time` 解析出完整的 ISO 时间（支持 `(+1day)` / `(+2day)`）。
- 若没有 `peak_flux` 列或值无法解析，则自动置为 `0.0`。
- `duration` **总是根据 `start_time` 和 `end_time` 重新计算**（单位：分钟），不依赖表中的原值。
- 若不存在 `active_region` 列，会尝试从 `position of active region` 复制，否则默认填 `'0'`。

### 2. 生成与数据文件夹一致的 `event_id`

如果 `events.xlsx` 中还没有 `event_id` 列，可以运行：

```bash
python scripts/c1_add_event_id_to_events_xlsx.py
```

该脚本会按 `scripts/a_download_sdo_data.py` 中相同的规则，为每行生成
形如 `EVT_YYYYMMDD_HHMMSS_序号` 的 `event_id`，并写回 `data/raw/events.xlsx` 的最后一列。

之后：

- `data/raw/downloaded/EVT_.../`
- `data/processed/EVT_.../`
- HDF5 中的 `events/EVT_.../`
- `events.xlsx` 中的 `event_id`

将保持完全一致。

---

## 创建数据集方法

### 使用脚本 `scripts/d_create_hdf5_dataset.py`（推荐）

```bash
python scripts/d_create_hdf5_dataset.py \
    --config configs/data_config.yaml \
    --events_csv data/raw/events.xlsx \
    --output data/Solar_Flares_CME_dataset.h5
```

**参数说明：**

- `--config`：数据配置文件（默认 `configs/data_config.yaml`）。
- `--events_csv` / `--events`：事件元数据文件（xlsx 或 csv，不指定则使用配置中的 `data.events_file`）。
- `--output`：输出 HDF5 文件路径。
- `--log_dir`：日志目录（默认 `logs`）。
- `--debug`：调试模式（可选）。

---

## 关键配置（`configs/data_config.yaml`）

与当前代码一致的关键片段示例：

```yaml
data:
  # 事件数据文件路径
  events_file: "D:/.../Flare_with_without_CME_forecast/data/raw/events.xlsx"

  # HDF5 文件路径
  hdf5_path: "data/Solar_Flares_CME_dataset.h5"
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"

  # 模态配置
  modalities:
    magnetogram:
      channels: 1
      resolution: [256, 256]
      wavelength: 6173.0
      unit: "Gauss"
      instrument: "SDO/HMI"
      normalization: [-2000, 2000]
      required: true
      cadence: "720s"
      series: "hmi.M_720s"

    euv_94:
      channels: 1
      resolution: [256, 256]
      wavelength: 94.0
      unit: "DN/s"
      instrument: "SDO/AIA"
      normalization: [0, 50000]
      required: true
      cadence: "12s"

    euv_171:
      channels: 1
      resolution: [256, 256]
      wavelength: 171.0
      unit: "DN/s"
      instrument: "SDO/AIA"
      normalization: [0, 30000]
      required: true
      cadence: "12s"

    euv_193:
      channels: 1
      resolution: [256, 256]
      wavelength: 193.0
      unit: "DN/s"
      instrument: "SDO/AIA"
      normalization: [0, 40000]
      required: false
      cadence: "12s"

    halpha:
      channels: 1
      resolution: [256, 256]
      wavelength: 6562.8
      unit: "counts"
      instrument: "CHASE/Hα"
      normalization: [0, 255]
      required: false
      cadence: "1200s"
      url_file: "data/raw/Halpha_download.txt"

  # 时间配置（决定时间轴）
  time:
    pre_event_hours: 6
    post_event_hours: 3
    cadence_minutes: 60

  # 窗口与目标配置
  sequence_length: 9
  max_activities: 5

  # 预处理配置（缺失处理与缓存）
  preprocessing:
    handle_missing: "nearest"   # nearest / zero_fill / interpolate
    interpolation_order: 1
    cache_processed: true
    num_workers: 4
```

---

## 数据收集逻辑（当前实现）

`HDF5DatasetCreator` 通过 `scripts/d_create_hdf5_dataset.py` 中的 `collect_event_data` 完成真实数据收集，逻辑为：

- 从 `events.xlsx` 中读取 `event_id` 与标准化后的 `start_time` / `peak_time` / `end_time`。
- 根据 `pre_event_hours` / `post_event_hours` / `cadence_minutes` 生成统一长度时间轴 `timestamps`。
- 对每个时间点、每个模态，在 `data/processed/EVT_.../<modality>/*.png` 中：
  - 解析文件名时间；
  - 找到**时间最近**的一张图像；
  - 按 `preprocessing.handle_missing` 决定：
    - `nearest`：总是使用最近图像；
    - `zero_fill`：若时间差超过阈值（采样间隔/2 与模态 cadence/2 的组合）则本帧填 0。
- 从 PNG 像素值按 `normalization` 映射回物理范围，组成形如 `[T, H, W]` 的时间序列并写入 HDF5。
- 若某个模态完全无图像，则为该模态填入全 0 的占位数据。

你无需再手写 `collect_event_data` 函数，真实采样和缺失补齐已内置在脚本中。

### HDF5 事件组 vs 训练滑动窗口

当前 HDF5 创建阶段**按事件组保存完整时间轴**：每个 `events/<event_id>` 下包含完整 `timestamps` 与各模态 `images[T, H, W]`。真正供模型训练的长度为 `sequence_length=9` 的样本，是在 `data.dataset.SolarFlareDataset` 读取 HDF5 后再按滑动窗口切出的，而不是在 HDF5 中预先分块保存。

这意味着：

- 修改 `pre_event_hours` / `post_event_hours` / `cadence_minutes` 会影响每个事件在 HDF5 中的总帧数；
- 修改 `sequence_length` / `stride` 会影响训练样本数量，但**不需要改变 HDF5 的存储结构**；
- 当前仓库常用设置为 `sequence_length=9`、`max_activities=5`，`stride` 由训练入口决定。

---

## 多事件 bbox 与主/非主事件

为支持“主事件 + 同一时间窗内其他事件”的多目标预测，项目中引入了 `bboxes.json` 与 HDF5 中的多 bbox 存储：

1. **生成假 bbox（后续可被人工标注替换）**

   ```bash
   python scripts/c2_generate_json_with_fake_bboxes.py --config configs/data_config.yaml
   ```

   - 读取 `events.xlsx`，按时间规则为每个主事件构造时间窗；
   - 查找时间窗内所有重叠事件（主事件 + 其他事件）；
   - 按图像分辨率随机生成若干 bbox：
     - `event_id`：该 bbox 对应的事件；
     - `is_primary`：是否为当前事件组的主事件；
     - `label`：由 `cme_associated` / `CME_asociated` 映射，`yes/True → 1`（爆发耀斑，有 CME），`no/False → 2`（束缚耀斑，无 CME）。
   - 将结果写入：

     - `data/processed/EVT_.../bboxes.json`：
       - `primary_event_id`
       - `bboxes`: 每个元素包含 `event_id`, `is_primary`, `label`, `bbox=[xmin,ymin,xmax,ymax]`

2. **在 HDF5 中存储 bbox 信息**

   `data/hdf5_creator.py` 在为每个 `event_id` 创建事件组时，会：

   - 从 `data/processed/EVT_.../bboxes.json` 读取多事件 bbox；
   - 写入：
     - 数据集 `events/<event_id>/bboxes`，形状 `[N, 4]`；
     - 属性：
       - `bbox_event_ids`: 所有框对应的事件 ID 列表；
       - `bbox_is_primary`: 是否为当前主事件；
       - `bbox_labels`: 每个 bbox 的 CME 标签（1/2）；
       - `num_bboxes`: bbox 数量。

这样，模型训练时既可以针对主事件预测框和标签，也可以利用同一时间窗内其他事件的 bbox 作为附加预测目标或上下文信息。

---

## 事件时间轴 + 采样点 + 主/非主事件 bbox 示意图

下面是一个简化的文字示意图，展示时间轴、采样点以及主/非主事件 bbox 的关系：

```text
时间轴（以 start_time 为中心，向前/向后扩展）

     t_start - 6h                      t_start                  t_start + 3h
────────────┬───────────────┬───────────────┬───────────────┬───────────────>
           S0              S1              S2              ...             S_T
         （采样点）     （采样点）      （采样点）                      （采样点）

示例：
- 主事件 A: [t_A_start, t_A_end]，事件组 ID = EVT_..._A
- 其他事件 B: [t_B_start, t_B_end]，部分时间落在 A 的时间窗内

在 EVT_..._A 这个事件组中：
- 时间轴由上面的 S0, S1, ..., S_T 决定（所有模态共用同一时间轴）。
- 对于每个采样点 S_k，脚本在 data/processed/EVT_..._A/<modality>/ 中
  选择最接近 S_k 时间戳的一张 PNG。
- bboxes.json 中会包含：
  - 一条 bbox，event_id = A，is_primary = true  （主事件）
  - 一条 bbox，event_id = B，is_primary = false （同窗其他事件）

在 EVT_..._B 自己的事件组中，同样会以 B 为主事件构造时间窗和采样点，
并将 B 作为主事件（is_primary = true），其他事件作为 is_primary = false。
```

---

## HDF5 文件结构（当前版本）

整体结构示意：

```text
Solar_Flares_CME_dataset.h5
├── (全局属性)
│   ├── dataset_name: "Solar_Flares_CME_dataset"
│   ├── creation_date: "2026-03-10T..."
│   ├── modalities: ["magnetogram", "euv_94", "euv_171", "euv_193", "halpha", ...]
│   └── time_config: {...}
│
├── index_table (数据集)
│   └── 每行一个事件：
│       event_id, start_time, end_time, flare_class, cme_associated,
│       label, peak_time, peak_flux, duration, active_region,
│       data_available, num_frames, bbox_xmin/ymin/xmax/ymax
│
└── events/ (组)
    ├── EVT_YYYYMMDD_HHMMSS_1/ (事件组)
    │   ├── (属性)
    │   │   ├── flare_class, start_time, peak_time, end_time
    │   │   ├── cme_associated, label, active_region
    │   │   ├── peak_flux, duration_minutes
    │   │   ├── num_frames
    │   │   └── 其他兼容属性
    │   ├── timestamps (数据集: [T])
    │   ├── data/ (组，每个模态一个子组)
    │   │   ├── magnetogram/
    │   │   │   ├── images (数据集: [T, 256, 256])
    │   │   │   └── (属性) wavelength, unit, instrument
    │   │   ├── euv_94/
    │   │   ├── euv_171/
    │   │   ├── euv_193/    (可选)
    │   │   ├── halpha/     (可选)
    │   │   └── ...
    │   ├── bboxes (数据集: [N, 4]，如存在 bboxes.json)
    │   │   └── (属性) bbox_event_ids, bbox_is_primary, bbox_labels, num_bboxes
    │   └── auxiliary/ (可选)
    │       └── ...
    └── EVT_YYYYMMDD_HHMMSS_2/
        └── ...
```

---

## 故障排除与下一步

常见问题：

- **时间解析异常（日期变成今天）**：检查 `DATE` 与 `start_time/peak_time/end_time` 是否为本说明中的格式；确保使用最新版本的 `add_event_id_to_events_xlsx.py`。
- **缺少 `peak_flux` / `duration` / `active_region` 列**：当前脚本会自动填充，不再阻塞数据集创建。
- **某模态完全无图像**：该模态在 HDF5 中会填入全 0，占位用；可在训练数据加载时筛除或保留。

创建数据集后，可以继续使用 `scripts/e_preprocess_and_split.py` 进行数据划分，并通过 `run_training.py`、`scripts/f_train_model.py` 或 `main.py` 的 `train` 子命令启动训练。