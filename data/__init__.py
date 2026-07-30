"""
数据模块 - 统一的数据处理接口
"""
import logging
from pathlib import Path

# 设置模块级别的日志
logger = logging.getLogger(__name__)

# 导出核心类（延迟/容错导入，避免在环境缺少二进制依赖时导入失败）
try:
    from .hdf5_reader import HDF5DatasetReader
except Exception as e:
    logger.warning(f"延迟导入 HDF5DatasetReader 失败: {e}")
    HDF5DatasetReader = None

try:
    from .dataset import SolarFlareDataset
except Exception as e:
    logger.warning(f"延迟导入 SolarFlareDataset 失败: {e}")
    SolarFlareDataset = None

# Avoid importing training/augmentation helpers eagerly. Some inference
# environments only need the dataset reader and can fail here because of
# optional imaging dependencies pulled in by transforms.
create_data_loaders = None
get_dataset_statistics = None
create_subset_data_loaders = None
default_collate_fn = None
collate_fn = None

# 尝试导入 data_creator 中的类
try:
    from .hdf5_creator import HDF5DatasetCreator
except ImportError as e:
    logger.warning(f"导入 data_creator 失败: {e}")
    HDF5DatasetCreator = None

# transforms 模块按需加载
create_transforms = None
ComposeSampleTransforms = None
NormalizeTransform = None
RandomSynchronizedAffine = None
RandomRotate = None
RandomFlip = None
RandomModalityIntensity = None
ResizeTransform = None

_LAZY_COMPONENTS = {
    'create_data_loaders',
    'get_dataset_statistics',
    'create_subset_data_loaders',
    'default_collate_fn',
    'collate_fn',
    'create_transforms',
    'ComposeSampleTransforms',
    'NormalizeTransform',
    'RandomSynchronizedAffine',
    'RandomRotate',
    'RandomFlip',
    'RandomModalityIntensity',
    'ResizeTransform',
}

# 定义 __all__ 以便于导入
__all__ = [
    # 核心类
    'HDF5DatasetReader',
    'SolarFlareDataset',
    'HDF5DatasetCreator',

    # 数据加载函数
    'create_data_loaders',
    'get_dataset_statistics',
    'create_subset_data_loaders',

    # 数据处理函数
    'default_collate_fn',
    'collate_fn',

    # 数据变换（可选）
    'create_transforms',
    'ComposeSampleTransforms',
    'NormalizeTransform',
    'RandomSynchronizedAffine',
    'RandomRotate',
    'RandomFlip',
    'RandomModalityIntensity',
    'ResizeTransform',

    # 工具函数
    'load_sample_data',
    'validate_dataset'
]


# ==================== 工具函数 ====================

def load_sample_data(hdf5_path: str, event_id: str = None):
    """
    加载示例数据用于测试

    Args:
        hdf5_path: HDF5文件路径
        event_id: 事件ID（可选）

    Returns:
        数据示例
    """
    try:
        reader = HDF5DatasetReader(hdf5_path)

        if event_id:
            # 加载指定事件
            return reader.get_event_data(event_id)
        else:
            # 加载第一个可用事件
            event_ids = reader.get_event_ids(available_only=True)
            if event_ids:
                return reader.get_event_data(event_ids[0])
            else:
                logger.error("没有可用的事件")
                return None
    except Exception as e:
        logger.error(f"加载示例数据失败: {e}")
        return None


def validate_dataset(hdf5_path: str) -> dict:
    """
    验证数据集完整性

    Args:
        hdf5_path: HDF5文件路径

    Returns:
        验证结果字典
    """
    validation_result = {
        'path': hdf5_path,
        'exists': False,
        'valid': False,
        'events_count': 0,
        'available_events': 0,
        'modalities': [],
        'issues': []
    }

    try:
        # 检查文件是否存在
        if not Path(hdf5_path).exists():
            validation_result['issues'].append(f"文件不存在: {hdf5_path}")
            return validation_result

        validation_result['exists'] = True

        # 尝试打开文件
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            # 检查必需的结构
            required_groups = ['index_table', 'events']
            for group in required_groups:
                if group not in f:
                    validation_result['issues'].append(f"缺少必需组: {group}")

            # 检查索引表
            if 'index_table' in f:
                index_table = f['index_table']
                validation_result['events_count'] = len(index_table)

                # 统计可用事件
                if 'data_available' in index_table.dtype.names:
                    available_count = sum(index_table['data_available'][:])
                    validation_result['available_events'] = available_count

            # 获取模态信息
            if 'modalities' in f.attrs:
                import json
                modalities = json.loads(f.attrs['modalities'])
                validation_result['modalities'] = modalities

            # 检查事件组
            if 'events' in f:
                events_group = f['events']
                event_keys = list(events_group.keys())

                # 检查每个事件的完整性
                for event_id in event_keys[:5]:  # 只检查前5个事件
                    event_group = events_group[event_id]

                    # 检查必需的数据集
                    required_datasets = ['timestamps']
                    for ds in required_datasets:
                        if ds not in event_group:
                            validation_result['issues'].append(
                                f"事件 {event_id} 缺少数据集: {ds}"
                            )

                    # 检查数据组
                    if 'data' in event_group:
                        data_group = event_group['data']
                        # 检查每个模态
                        for modality in validation_result['modalities']:
                            if modality not in data_group:
                                validation_result['issues'].append(
                                    f"事件 {event_id} 缺少模态: {modality}"
                                )

        # 如果没有发现问题，标记为有效
        if len(validation_result['issues']) == 0:
            validation_result['valid'] = True

        return validation_result

    except Exception as e:
        validation_result['issues'].append(f"验证过程中出错: {str(e)}")
        return validation_result


# ==================== 模块初始化 ====================

def init_module():
    """初始化数据模块"""
    logger.info("初始化数据模块")

    # 检查关键组件
    missing_components = []

    if HDF5DatasetReader is None:
        missing_components.append("HDF5DatasetReader")

    if SolarFlareDataset is None:
        missing_components.append("SolarFlareDataset")

    if missing_components:
        logger.warning(f"数据模块缺少组件: {missing_components}")
    else:
        logger.info("数据模块初始化完成")

    return len(missing_components) == 0


# 自动初始化
_MODULE_INITIALIZED = init_module()


def __getattr__(name):
    """按需导入可选组件，避免影响只读 HDF5 / 推理场景。"""
    global create_data_loaders, get_dataset_statistics, create_subset_data_loaders
    global default_collate_fn, collate_fn
    global create_transforms, ComposeSampleTransforms, NormalizeTransform
    global RandomSynchronizedAffine, RandomRotate, RandomFlip
    global RandomModalityIntensity, ResizeTransform

    if name in {
        'create_data_loaders',
        'get_dataset_statistics',
        'create_subset_data_loaders',
        'default_collate_fn',
        'collate_fn',
    }:
        from .data_loader import (
            create_data_loaders as _create_data_loaders,
            get_dataset_statistics as _get_dataset_statistics,
            create_subset_data_loaders as _create_subset_data_loaders,
            default_collate_fn as _default_collate_fn,
            collate_fn as _collate_fn,
        )
        create_data_loaders = _create_data_loaders
        get_dataset_statistics = _get_dataset_statistics
        create_subset_data_loaders = _create_subset_data_loaders
        default_collate_fn = _default_collate_fn
        collate_fn = _collate_fn
        return globals()[name]

    if name in {
        'create_transforms',
        'ComposeSampleTransforms',
        'NormalizeTransform',
        'RandomSynchronizedAffine',
        'RandomRotate',
        'RandomFlip',
        'RandomModalityIntensity',
        'ResizeTransform',
    }:
        from .transforms import (
            create_transforms as _create_transforms,
            ComposeSampleTransforms as _ComposeSampleTransforms,
            NormalizeTransform as _NormalizeTransform,
            RandomSynchronizedAffine as _RandomSynchronizedAffine,
            RandomRotate as _RandomRotate,
            RandomFlip as _RandomFlip,
            RandomModalityIntensity as _RandomModalityIntensity,
            ResizeTransform as _ResizeTransform,
        )
        create_transforms = _create_transforms
        ComposeSampleTransforms = _ComposeSampleTransforms
        NormalizeTransform = _NormalizeTransform
        RandomSynchronizedAffine = _RandomSynchronizedAffine
        RandomRotate = _RandomRotate
        RandomFlip = _RandomFlip
        RandomModalityIntensity = _RandomModalityIntensity
        ResizeTransform = _ResizeTransform
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
