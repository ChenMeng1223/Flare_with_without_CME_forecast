# HDF5 数据集生成问题修复总结

## 问题1：最终生成的文件名错误

**问题**：最终生成的文件是 `solar_flares_dataset.h5` 而非配置文件中设置的 `Solar_Flares_CME_dataset.h5`

**原因**：在 `scripts/c_create_hdf5_dataset.py` 中，`parse_args()` 函数第89行硬编码了默认输出路径：
```python
parser.add_argument('--output', type=str,
                    default='data/solar_flares_dataset.h5',  # 硬编码！
                    help='输出HDF5文件路径')
```

这导致即使配置文件中设置了 `hdf5_path: "data/Solar_Flares_CME_dataset.h5"`，也会被这个硬编码的默认值覆盖。

**解决**：
- 将默认值改为 `None`
- 在 `main()` 函数中，仅当显式指定 `--output` 参数时才覆盖配置文件值
- 如果不指定 `--output`，则使用配置文件中的 `hdf5_path`

**修改后的调用方式**：
```bash
# 使用配置文件中的 hdf5_path
python scripts/d_create_hdf5_dataset.py --config configs/data_config.yaml

# 或明确指定输出路径
python scripts/d_create_hdf5_dataset.py --output data/Solar_Flares_CME_dataset.h5
```

---

## 问题2：CME 关联列全为1

**问题**：HDF5文件中的 `cme_associated` 列全是1，应该根据 events.xlsx 中的相关列生成，yes为1，no为0

**原因**：原始代码没有对 `CME_associated` 列进行值的转换，直接使用：
```python
bool(row['cme_associated'])  # "yes" 和 "no" 都被转换为 True
```

任何非空字符串都会被 `bool()` 转换为 `True`，导致所有值都变成 1。

**解决**：在 `load_events_metadata()` 函数中添加了规范化函数：

```python
# 规范化 cme_associated 列: yes/True->1, no/False->0
def _to_cme_bool(v):
    if pd.isna(v):
        return 0  # 缺失默认为0（不伴随CME）
    s = str(v).strip().lower()
    if s in ('yes', 'true', '1', 'y'):
        return 1
    elif s in ('no', 'false', '0', 'n'):
        return 0
    else:
        return 0  # 其他值默认为0

df['cme_associated'] = df['cme_associated'].apply(_to_cme_bool)
```

---

## 额外改进：label 列的正确生成

同时修改了 `label` 列的生成逻辑，使其基于 `cme_associated` 的值：
- CME_associated = yes (值为1) → label = 1（爆发伴CME）
- CME_associated = no (值为0) → label = 2（爆发无CME）
- 其他情况 → label = 0

**修改的文件**：
- `scripts/c_create_hdf5_dataset.py`

**测试命令**：
```bash
python scripts/d_create_hdf5_dataset.py --output data/Solar_Flares_CME_dataset.h5
```

这样生成的 HDF5 文件将：
1. ✅ 使用正确的文件名（`Solar_Flares_CME_dataset.h5`）
2. ✅ CME_associated 列正确显示 0 或 1
3. ✅ Label 列正确显示 0、1 或 2
