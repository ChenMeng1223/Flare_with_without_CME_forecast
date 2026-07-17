# SDO 数据处理与数据集创建流程（已更新）

> 已融合当前代码调整并验证：a_download_sdo_data.py / a1_download_halpha_only.py / a2_download_halpha_addition_GONG.py

本项目提供了从原始事件表到 HDF5 训练数据集的完整流水线，包括：

- 事件表标准化与 `event_id` 生成
- SDO / CHASE 数据自动批量下载
- 多波段 FITS → PNG 图像处理
- 多事件假 bbox 生成（主事件 + 同窗其他事件）
- 基于时间配置的图像采样与缺失处理
- HDF5 数据集构建（多模态、多事件 bbox）

---

## 功能特性

- **自动批量下载**：根据事件时间范围自动下载多波段 SDO/CHASE 数据。
- **批量图像处理**：将 FITS 文件统一转换为规范的 PNG 图像。
- **HDF5 数据集创建**：从处理好的图像和事件元数据创建可直接用于模型训练的 HDF5 文件。
- **多模态 & 多事件支持**：同时支持 `magnetogram`、`euv_94/171/193`、`halpha` 等模态；每个事件组中可以包含主事件和多个同窗其他事件的 bbox。
- **精确时间匹配与缺失处理**：根据 `pre_event_hours` / `post_event_hours` / `cadence_minutes` 严格采样，每个采样点选取时间最近的图像，并提供可配置的缺失策略。

---

## 支持的波段（当前配置）

| 波段        | 仪器    | 描述             | 默认分辨率（采样后） |
|-------------|---------|------------------|----------------------|
| magnetogram | SDO/HMI | 视向磁场图       | 256×256              |
| euv_94      | SDO/AIA | 94Å 极紫外      | 256×256              |
| euv_171     | SDO/AIA | 171Å 极紫外     | 256×256              |
| euv_193     | SDO/AIA | 193Å 极紫外     | 256×256（可选）      |
| halpha      | CHASE   | Hα 线宽         | 256×256（由脚本重采样） |

---

## 安装依赖

```bash
pip install -r requirements.txt
```

建议使用支持 NumPy 2.x 的环境，并根据需要安装 `sunpy`、`astropy` 等天文数据相关库。

---

## 数据准备与事件表

1. **准备事件元数据文件 `data/raw/events.xlsx`**

   典型字段（与你当前 Excel 一致）：

   - `DATE`：观测日期（例如 `2024.12.23`）。
   - `start_time` / `peak_time` / `end_time`：时间字符串，可包含 `(+1day)` / `(+2day)` 标记。
   - `flare_class`：耀斑等级。
   - `cme_associated` / `CME_asociated`：是否伴随 CME。
   - `position of active region`：活动区位置标记。
   - （可选）`peak_flux` / `duration`：若缺失，创建 HDF5 时会自动填补/重计算。

2. **根据 DATE + 时间列标准化时间并生成 `event_id`**

   ```bash
   python scripts/c1_add_event_id_to_events_xlsx.py
   ```

   - 将 `DATE + start_time / peak_time / end_time` 组合为完整 ISO 时间；
   - 生成 `EVT_YYYYMMDD_HHMMSS_序号` 形式的 `event_id`，写入 `events.xlsx` 最后一列；
   - 保证 `event_id` 与后续 `downloaded/` 和 `processed/` 目录中的事件文件夹名一致。

---

## 使用方法

### 完整流程（可选）

> `scripts/full_data_pipeline.py` 仍提供下载 → 图像处理 → HDF5 创建的串联入口；若本地使用的是带 GONG 的图像处理脚本，实践中更建议按下面的“分步执行”分别运行 `b1/b2/b3` 或 `b_batch_process_images(correct_version_with_gong).py`。

一键执行完整的数据处理与 HDF5 创建流程：

```bash
python scripts/full_data_pipeline.py \
    --config configs/data_config.yaml \
    --output data/Solar_Flares_CME_dataset.h5 \
    --max_workers 4
```

### 分步执行

#### 1. 下载数据（FITS）

```bash
python scripts/a_download_sdo_data.py \
    --config configs/data_config.yaml \
    --max_workers 2
```

输出示例：

```text
data/raw/downloaded/EVT_YYYYMMDD_HHMMSS_1/magnetogram/*.fits
data/raw/downloaded/EVT_YYYYMMDD_HHMMSS_1/euv_171/*.fits
...
```

#### 2. 处理图像（FITS → PNG）

- 磁图：

  ```bash
  python scripts/b1_batch_plot_magnetogram.py --config configs/data_config.yaml
  ```

- EUV：

  ```bash
  python scripts/b2_batch_plot_euv.py --config configs/data_config.yaml
  ```

- Hα：

  ```bash
  python scripts/b3_batch_plot_halpha.py --config configs/data_config.yaml
  ```

输出示例：

```text
data/processed/EVT_YYYYMMDD_HHMMSS_1/magnetogram/*.png
data/processed/EVT_YYYYMMDD_HHMMSS_1/euv_171/*.png
data/processed/EVT_YYYYMMDD_HHMMSS_1/halpha/*.png
```

#### 3. 生成假 bbox（主事件 + 非主事件）

```bash
python scripts/c2_generate_json_with_fake_bboxes.py --config configs/data_config.yaml
```

为每个 `EVT_...` 生成：

```text
data/processed/EVT_YYYYMMDD_HHMMSS_1/bboxes.json
```

结构示例：

```json
{
  "primary_event_id": "EVT_20241224_182000_1",
  "bboxes": [
    {
      "event_id": "EVT_20241224_182000_1",
      "is_primary": true,
      "label": 1,
      "bbox": [50, 60, 120, 140]
    },
    {
      "event_id": "EVT_20241224_201500_1",
      "is_primary": false,
      "label": 2,
      "bbox": [150, 80, 210, 160]
    }
  ]
}
```

之后可以用人工标注工具将这些假框替换为真实框，只要保持 JSON 结构不变即可。

#### 3.x 批量人工标注（替换假 bbox）

我们提供了一个 GUI 工具用于批量人工标注，每个事件组只需要标一次 bbox（该组内所有图像共享同一组 bbox）。

- **脚本**：`scripts/c3_annotate_bboxes_gui(correct_version).py`
- **输入**：`data/processed/<event_id>/bboxes.json` + `data/processed/<event_id>/<modality>/*.png`
- **输出**：覆盖写回同路径的 `bboxes.json`（增加 `annotated=true`、`annotated_at`）
- **关键点**：
  - **event_id 选择**：只从该事件组 `bboxes.json` 里的候选 event_id 列表中点选（缩小范围，避免输错）
  - **主事件**：勾选 `is_primary`（保存时会自动将 `primary_event_id` 同步为该项）

运行：

```bash
python "scripts/c3_annotate_bboxes_gui(correct_version).py"
```

#### 4. 创建 HDF5 数据集

```bash
python scripts/d_create_hdf5_dataset.py \
    --config configs/data_config.yaml \
    --events_csv data/raw/events.xlsx \
    --output data/Solar_Flares_CME_dataset.h5
```

---

## 目录结构（当前实现）

```text
data/
├── raw/
│   ├── events.xlsx              # 事件元数据
│   ├── downloaded/              # 下载的 FITS 文件
│   │   └── EVT_YYYYMMDD_HHMMSS_1/
│   │       ├── magnetogram/
│   │       ├── euv_94/
│   │       ├── euv_171/
│   │       ├── euv_193/
│   │       └── halpha/
│   └── Halpha_download.txt      # Hα 下载 URL 记录
│
├── processed/                   # 预处理后的 PNG 与 bbox
│   └── EVT_YYYYMMDD_HHMMSS_1/
│       ├── magnetogram/
│       ├── euv_94/
│       ├── euv_171/
│       ├── euv_193/
│       ├── halpha/
│       └── bboxes.json
│
└── Solar_Flares_CME_dataset.h5  # 最终 HDF5 数据集
```

---

## 事件时间轴 + 采样点 + 主/非主事件 bbox 示意图

```text
时间轴（以 start_time 为中心，向前/向后扩展）

     t_start - pre_event_hours                t_start                     t_start + post_event_hours
───────────────┬───────────────┬───────────────┬───────────────┬───────────────────────────────>
              S0              S1              S2              ...                            S_T
            （采样点）     （采样点）      （采样点）                                  （采样点）

示例：
- 主事件 A: [t_A_start, t_A_end]，事件组 ID = EVT_..._A
- 其他事件 B: [t_B_start, t_B_end]，部分时间落在 A 的时间窗内

在 EVT_..._A 事件组中：
- 时间轴由 S0, S1, ..., S_T 决定（所有模态共用同一时间轴）。
- 对每个采样点 S_k，脚本在 data/processed/EVT_..._A/<modality>/ 中
  选择时间最近的一张 PNG。
- bboxes.json 中包含：
  - event_id = A, is_primary = true 的 bbox（主事件）
  - event_id = B, is_primary = false 的 bbox（同窗其他事件）

在 EVT_..._B 事件组中：
- 以 B 为主事件重新构造自己的时间窗与采样点；
- B 的 bbox 在该组中 is_primary = true，A 若落入 B 的窗内则作为 is_primary = false 出现。
```

---

## 配置说明（时间与模态）

### 时间范围配置（`configs/data_config.yaml`）

```yaml
data:
  events_file: "data/raw/events.xlsx"

  time:
    pre_event_hours: 6      # 当前配置：事件前扩展小时数
    post_event_hours: 3    # 当前配置：事件后扩展小时数
    cadence_minutes: 60       # 当前配置：采样间隔（分钟）

  sequence_length: 9          # 训练阶段从 HDF5 再切滑动窗口
  max_activities: 5
```

### 模态配置（示例）

```yaml
data:
  modalities:
    magnetogram:
      channels: 1
      resolution: [256, 256]
      normalization: [-2000, 2000]
      required: true
      cadence: "720s"
```

---

## 故障排除与日志

### 常见问题

1. **SunPy 不可用**

   ```bash
   pip install sunpy
   ```

2. **下载失败**
   - 检查网络连接；
   - 确认时间范围合理；
   - 查看日志中的具体错误信息。

3. **内存不足**
   - 减少 `max_workers` 参数；
   - 处理较小的时间范围；
   - 降低图像分辨率或采样帧数。

4. **FITS 文件损坏**
   - 脚本会自动跳过损坏的文件；
   - 检查日志中的警告信息。

### 日志查看

所有操作都会记录详细日志，例如：

```bash
# 查看最新日志（Windows 可使用 PowerShell 的 Get-Content -Wait）
tail -f logs/__main___*.log
```

---

## 性能优化与扩展

- **并发下载**：使用 `--max_workers` 控制并发数；
- **内存管理**：图像处理使用多进程，避免内存累积；
- **增量处理**：可只重新运行部分步骤（例如只重建 HDF5）。

要扩展新波段：

1. 在 `configs/data_config.yaml` 中添加新的模态配置；
2. 在 `scripts/a_download_sdo_data.py` 中添加对应下载逻辑；
3. 在 `scripts/b*_batch_plot_*.py` 中添加对应绘图/保存逻辑；
4. 运行完整流水线重新生成 `processed` 与 HDF5。

本项目遵循 MIT 许可证。