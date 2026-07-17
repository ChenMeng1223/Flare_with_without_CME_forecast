"""
配置管理工具模块 - 整合版
提供统一的配置加载、保存、验证和管理功能
"""
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import copy
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """配置错误异常"""
    pass


class ConfigManager:
    """
    配置管理器 - 统一管理所有配置操作
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（可选）
        """
        self.config = {}
        self.config_path = Path(config_path) if config_path else None

        if config_path and Path(config_path).exists():
            self.config = self.load(config_path)

    def load(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            配置字典

        Raises:
            ConfigError: 配置文件加载失败
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"配置文件不存在: {config_path}")

        try:
            if config_path.suffix in ['.yaml', '.yml']:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                # 确保数值类型正确
                config = self._ensure_numeric_types(config)
            elif config_path.suffix == '.json':
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                raise ConfigError(f"不支持的配置文件格式: {config_path.suffix}")

            self.config = config
            self.config_path = config_path

            logger.info(f"加载配置文件: {config_path}")
            return config

        except Exception as e:
            raise ConfigError(f"加载配置文件失败 {config_path}: {e}")

    def save(self, config_path: Optional[Union[str, Path]] = None,
             format: str = 'yaml') -> None:
        """
        保存配置文件

        Args:
            config_path: 配置文件路径（如果为None，则使用原始路径）
            format: 格式 ('yaml' 或 'json')

        Raises:
            ConfigError: 配置文件保存失败
        """
        if config_path is None and self.config_path is None:
            raise ConfigError("未指定配置文件路径")

        save_path = Path(config_path) if config_path else self.config_path

        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == 'yaml':
                with open(save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            elif format.lower() == 'json':
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
            else:
                raise ConfigError(f"不支持的配置文件格式: {format}")

            logger.info(f"保存配置文件: {save_path}")

        except Exception as e:
            raise ConfigError(f"保存配置文件失败 {save_path}: {e}")

    def merge(self, override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置（深度合并）

        Args:
            override_config: 覆盖配置

        Returns:
            合并后的配置字典
        """
        result = copy.deepcopy(self.config)

        def recursive_merge(base: Dict, override: Dict) -> None:
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    recursive_merge(base[key], value)
                else:
                    base[key] = copy.deepcopy(value)

        recursive_merge(result, override_config)
        self.config = result
        return result

    def update_from_args(self, args: Dict[str, Any],
                        arg_mappings: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
        """
        使用参数更新配置

        Args:
            args: 参数字典（如命令行参数）
            arg_mappings: 参数到配置路径的映射

        Returns:
            更新后的配置
        """
        if arg_mappings is None:
            # 默认映射
            arg_mappings = {
                'batch_size': ['training', 'batch_size'],
                'learning_rate': ['training', 'learning_rate'],
                'epochs': ['training', 'epochs'],
                'model_type': ['model', 'type'],
                'data_dir': ['data', 'processed_data_dir'],
                'log_dir': ['logging', 'log_dir'],
                'device': ['training', 'device'],
                'num_workers': ['data', 'num_workers']
            }

        for arg_name, arg_value in args.items():
            if arg_value is not None and arg_name in arg_mappings:
                path = arg_mappings[arg_name]
                self.set_value('.'.join(path), arg_value)

        return self.config

    def get_value(self, key_path: str, default: Any = None) -> Any:
        """
        获取嵌套配置值

        Args:
            key_path: 点分隔的键路径 (如 'data.modalities.magnetogram.channels')
            default: 默认值

        Returns:
            配置值或默认值
        """
        keys = key_path.split('.')
        current = self.config

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def _ensure_numeric_types(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        确保配置中的数值类型正确

        Args:
            config: 配置字典

        Returns:
            处理后的配置字典
        """
        def convert_value(value: Any) -> Any:
            if isinstance(value, str):
                # 尝试转换为数值
                try:
                    # 检查是否是科学计数法
                    if 'e' in value.lower() or '.' in value:
                        return float(value)
                    # 检查是否是整数
                    return int(value)
                except ValueError:
                    pass
            elif isinstance(value, dict):
                return {k: convert_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [convert_value(item) for item in value]
            return value

        return convert_value(config)

    def set_value(self, key_path: str, value: Any) -> None:
        """
        设置嵌套配置值

        Args:
            key_path: 点分隔的键路径
            value: 要设置的值
        """
        keys = key_path.split('.')
        current = self.config

        for i, key in enumerate(keys[:-1]):
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def validate(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置的完整性

        Args:
            schema: 验证模式（可选）

        Returns:
            是否有效
        """
        # 如果没有提供schema，使用默认验证
        if schema is None:
            required_sections = ['training', 'model']

            for section in required_sections:
                if section not in self.config:
                    logger.error(f"配置缺少必要部分: {section}")
                    return False

            # 验证训练配置
            training_config = self.config.get('training', {})
            required_training_keys = ['batch_size', 'epochs', 'learning_rate']
            for key in required_training_keys:
                if key not in training_config:
                    logger.error(f"训练配置缺少 '{key}'")
                    return False

            return True

        # 如果有schema，按照schema验证
        try:
            import jsonschema
            jsonschema.validate(self.config, schema)
            return True
        except ImportError:
            logger.warning("未安装jsonschema，跳过schema验证")
            return True
        except jsonschema.exceptions.ValidationError as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def print(self, indent: int = 0,
              filter_sections: Optional[List[str]] = None) -> str:
        """
        格式化打印配置信息

        Args:
            indent: 缩进级别
            filter_sections: 只显示指定的部分

        Returns:
            格式化的配置字符串
        """
        lines = []

        def recursive_print(cfg: Dict, level: int, prefix: str = ''):
            for key, value in cfg.items():
                # 过滤部分
                if filter_sections and level == 0 and key not in filter_sections:
                    continue

                current_prefix = prefix + '  ' * level

                if isinstance(value, dict):
                    lines.append(f"{current_prefix}{key}:")
                    recursive_print(value, level + 1, prefix)
                else:
                    lines.append(f"{current_prefix}{key}: {value}")

        lines.append("=" * 60)
        lines.append("配置信息:")
        lines.append("=" * 60)
        recursive_print(self.config, 0)
        lines.append("=" * 60)

        return '\n'.join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """返回配置字典的副本"""
        return copy.deepcopy(self.config)

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """支持字典式设置"""
        self.config[key] = value

    def __contains__(self, key: str) -> bool:
        """支持in操作符"""
        return key in self.config


# ==================== 工具函数 ====================

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    快速加载配置（兼容旧代码）

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    return ConfigManager(config_path).config


def save_config(config: Dict[str, Any],
                config_path: Union[str, Path],
                format: str = 'yaml') -> None:
    """
    快速保存配置（兼容旧代码）

    Args:
        config: 配置字典
        config_path: 配置文件路径
        format: 格式 ('yaml' 或 'json')
    """
    mgr = ConfigManager()
    mgr.config = config
    mgr.save(config_path, format)


def merge_configs(base_config: Dict[str, Any],
                  override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并两个配置字典（深度合并）（兼容旧代码）

    Args:
        base_config: 基础配置
        override_config: 覆盖配置

    Returns:
        合并后的配置字典
    """
    mgr = ConfigManager()
    mgr.config = base_config
    return mgr.merge(override_config)


def validate_config(config: Dict[str, Any],
                    schema: Optional[Dict[str, Any]] = None) -> bool:
    """
    验证配置文件（兼容旧代码）

    Args:
        config: 配置字典
        schema: 验证模式（可选）

    Returns:
        是否有效
    """
    mgr = ConfigManager()
    mgr.config = config
    return mgr.validate(schema)


def create_default_config() -> Dict[str, Any]:
    """
    创建默认配置

    Returns:
        默认配置字典
    """
    config = {
        'project': {
            'name': 'solar_flare_prediction',
            'version': '1.0.0',
            'description': 'Solar flare prediction using multimodal data'
        },
        'data': {
            'hdf5_path': 'data/solar_flares_dataset.h5',
            'raw_data_dir': 'data/raw',
            'processed_data_dir': 'data/processed',
            'modalities': {
                'magnetogram': {
                    'channels': 1,
                    'resolution': [512, 512],
                    'wavelength': 6173.0,
                    'unit': 'Gauss',
                    'instrument': 'SDO/HMI'
                }
            },
            'time': {
                'pre_event_hours': 24,
                'post_event_hours': 12,
                'cadence_minutes': 12,
                'sequence_length': 48,
                'forecast_horizon': 24
            },
            'splits': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'random_seed': 42
            },
            'augmentation': {
                'enabled': True,
                'rotation_range': 30,
                'zoom_range': 0.2,
                'flip_horizontal': True,
                'flip_vertical': False
            }
        },
        'model': {
            'type': 'multimodal_transformer',
            'name': 'SolarFlareTransformer',
            'input_size': [512, 512],
            'num_classes': 3,
            'use_physical_constraints': True,
            'transformer': {
                'hidden_dim': 512,
                'num_heads': 8,
                'num_layers': 6,
                'dropout': 0.1,
                'feedforward_dim': 2048
            }
        },
        'training': {
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'gradient_clip': 1.0,
            'device': 'cuda',
            'num_workers': 4,
            'pin_memory': True,
            'optimizer': {
                'type': 'adamw',
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'type': 'cosine',
                'warmup_epochs': 5,
                'min_lr': 1e-6,
                'step_size': 10,
                'gamma': 0.5
            },
            'loss_weights': {
                'classification': 1.0,
                'bbox_regression': 0.5,
                'time_prediction': 0.3,
                'event_probability': 0.2,
                'physics_constraint': 0.1
            },
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.001,
                'restore_best_weights': True,
                'mode': 'min'
            },
            'checkpoint': {
                'save_frequency': 5,
                'save_best_only': True,
                'max_checkpoints': 10,
                'monitor': 'val_loss'
            }
        },
        'inference': {
            'threshold_class1': 0.7,
            'threshold_class2': 0.6,
            'use_uncertainty': True,
            'num_mc_samples': 10,
            'batch_size': 32
        },
        'logging': {
            'log_dir': 'logs',
            'level': 'INFO',
            'capture_exceptions': True,
            'tensorboard': True,
            'wandb': False,
            'log_frequency': 10,
            'save_plots': True
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
            'save_predictions': True,
            'save_confusion_matrix': True,
            'visualize_results': True
        }
    }

    return config


def get_config_value(config: Dict[str, Any],
                     key_path: str,
                     default: Any = None) -> Any:
    """
    获取嵌套配置值（兼容旧代码）

    Args:
        config: 配置字典
        key_path: 点分隔的键路径
        default: 默认值

    Returns:
        配置值或默认值
    """
    mgr = ConfigManager()
    mgr.config = config
    return mgr.get_value(key_path, default)


def set_config_value(config: Dict[str, Any],
                     key_path: str,
                     value: Any) -> Dict[str, Any]:
    """
    设置嵌套配置值（兼容旧代码）

    Args:
        config: 配置字典
        key_path: 点分隔的键路径
        value: 要设置的值

    Returns:
        更新后的配置
    """
    mgr = ConfigManager()
    mgr.config = config
    mgr.set_value(key_path, value)
    return mgr.config


def update_config_with_args(config: Dict[str, Any],
                            args: Dict[str, Any],
                            arg_mappings: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    使用参数更新配置（兼容旧代码）

    Args:
        config: 配置字典
        args: 参数字典（如命令行参数）
        arg_mappings: 参数到配置路径的映射

    Returns:
        更新后的配置字典
    """
    mgr = ConfigManager()
    mgr.config = config
    return mgr.update_from_args(args, arg_mappings)


def print_config(config: Dict[str, Any],
                 indent: int = 0,
                 logger: Optional[logging.Logger] = None,
                 filter_sections: Optional[List[str]] = None) -> None:
    """
    打印配置信息（兼容旧代码）

    Args:
        config: 配置字典
        indent: 缩进级别
        logger: Logger对象
        filter_sections: 只显示指定的部分
    """
    mgr = ConfigManager()
    mgr.config = config
    output = mgr.print(indent, filter_sections)

    if logger:
        for line in output.split('\n'):
            logger.info(line)
    else:
        print(output)


# ==================== 预定义配置 ====================

def get_training_config() -> Dict[str, Any]:
    """获取训练专用配置"""
    default_config = create_default_config()
    return {
        'data': default_config['data'],
        'model': default_config['model'],
        'training': default_config['training'],
        'logging': default_config['logging']
    }


def get_inference_config() -> Dict[str, Any]:
    """获取推理专用配置"""
    default_config = create_default_config()
    return {
        'model': default_config['model'],
        'inference': default_config['inference'],
        'logging': {
            'log_dir': 'logs/inference',
            'level': 'INFO'
        }
    }


def get_evaluation_config() -> Dict[str, Any]:
    """获取评估专用配置"""
    default_config = create_default_config()
    return {
        'model': default_config['model'],
        'evaluation': default_config['evaluation'],
        'logging': {
            'log_dir': 'logs/evaluation',
            'level': 'INFO'
        }
    }


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 测试代码
    print("测试配置管理器...")

    # 1. 创建默认配置
    config = create_default_config()
    print("\n1. 默认配置结构:")
    print_config(config, filter_sections=['training', 'model'])

    # 2. 使用配置管理器
    mgr = ConfigManager()
    mgr.config = config

    # 3. 获取配置值
    lr = mgr.get_value('training.learning_rate')
    print(f"\n2. 学习率: {lr}")

    # 4. 设置配置值
    mgr.set_value('training.batch_size', 32)
    mgr.set_value('model.hidden_dim', 1024)

    # 5. 打印修改后的配置
    print("\n3. 修改后的配置:")
    print(mgr.print(filter_sections=['training', 'model']))

    # 6. 保存配置
    mgr.save('test_config.yaml')

    # 7. 加载配置
    mgr2 = ConfigManager('test_config.yaml')
    print("\n4. 加载的配置:")
    print(mgr2.print(filter_sections=['training', 'model']))

    # 8. 验证配置
    is_valid = mgr2.validate()
    print(f"\n5. 配置验证: {'通过' if is_valid else '失败'}")

    # 9. 合并配置
    override = {
        'training': {
            'epochs': 50,
            'learning_rate': 0.001
        }
    }
    merged = mgr2.merge(override)
    print("\n6. 合并后的配置:")
    print_config(merged, filter_sections=['training'])

    # 清理测试文件
    import os
    if os.path.exists('test_config.yaml'):
        os.remove('test_config.yaml')
        print("\n测试文件已清理")