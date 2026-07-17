"""
文件工具模块
"""
import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Union, Optional, Tuple, Dict, Any
import json
import pickle
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    确保目录存在

    Args:
        path: 目录路径

    Returns:
        目录的Path对象
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def find_files(directory: Union[str, Path],
               pattern: str = "**/*",
               recursive: bool = True) -> List[Path]:
    """
    查找匹配模式的文件

    Args:
        directory: 目录路径
        pattern: 文件模式
        recursive: 是否递归搜索

    Returns:
        文件路径列表
    """
    directory = Path(directory)

    if recursive:
        files = list(directory.glob(pattern))
    else:
        files = list(directory.glob(pattern.split('/')[-1]))

    # 过滤掉目录
    files = [f for f in files if f.is_file()]

    return sorted(files)


def copy_files(source_files: List[Union[str, Path]],
               target_dir: Union[str, Path],
               overwrite: bool = False) -> List[Path]:
    """
    复制文件列表到目标目录

    Args:
        source_files: 源文件列表
        target_dir: 目标目录
        overwrite: 是否覆盖已存在的文件

    Returns:
        复制的文件路径列表
    """
    target_dir = ensure_directory(target_dir)
    copied_files = []

    for source_file in source_files:
        source_path = Path(source_file)

        if not source_path.exists():
            logger.warning(f"源文件不存在: {source_path}")
            continue

        target_path = target_dir / source_path.name

        if target_path.exists() and not overwrite:
            logger.warning(f"目标文件已存在: {target_path}")
            continue

        try:
            shutil.copy2(source_path, target_path)
            copied_files.append(target_path)
            logger.debug(f"复制文件: {source_path} -> {target_path}")
        except Exception as e:
            logger.error(f"复制文件失败 {source_path}: {e}")

    logger.info(f"复制了 {len(copied_files)}/{len(source_files)} 个文件到 {target_dir}")
    return copied_files


def remove_files(file_patterns: List[Union[str, Path]],
                 recursive: bool = False) -> int:
    """
    删除匹配模式的文件

    Args:
        file_patterns: 文件模式列表
        recursive: 是否递归删除

    Returns:
        删除的文件数量
    """
    removed_count = 0

    for pattern in file_patterns:
        pattern_path = Path(pattern)

        if pattern_path.exists() and pattern_path.is_file():
            try:
                pattern_path.unlink()
                removed_count += 1
                logger.debug(f"删除文件: {pattern_path}")
            except Exception as e:
                logger.error(f"删除文件失败 {pattern_path}: {e}")

        elif recursive:
            # 如果是目录，查找所有文件
            if pattern_path.is_dir():
                files = find_files(pattern_path, "**/*", recursive=True)
                for file_path in files:
                    try:
                        file_path.unlink()
                        removed_count += 1
                        logger.debug(f"删除文件: {file_path}")
                    except Exception as e:
                        logger.error(f"删除文件失败 {file_path}: {e}")
            else:
                # 使用glob模式
                for file_path in Path(".").glob(str(pattern_path)):
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                            removed_count += 1
                            logger.debug(f"删除文件: {file_path}")
                        except Exception as e:
                            logger.error(f"删除文件失败 {file_path}: {e}")

    logger.info(f"删除了 {removed_count} 个文件")
    return removed_count


def get_file_size(file_path: Union[str, Path],
                  human_readable: bool = True) -> Union[int, str]:
    """
    获取文件大小

    Args:
        file_path: 文件路径
        human_readable: 是否返回人类可读格式

    Returns:
        文件大小
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    size_bytes = file_path.stat().st_size

    if human_readable:
        # 转换为人类可读格式
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    else:
        return size_bytes


def get_file_hash(file_path: Union[str, Path],
                  algorithm: str = 'sha256',
                  chunk_size: int = 8192) -> str:
    """
    计算文件哈希值

    Args:
        file_path: 文件路径
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256', 'sha512')
        chunk_size: 读取块大小

    Returns:
        文件哈希值
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 获取哈希算法
    hash_func = hashlib.new(algorithm)

    # 计算哈希
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hash_func.update(chunk)

    return hash_func.hexdigest()


def read_text_file(file_path: Union[str, Path],
                   encoding: str = 'utf-8') -> str:
    """
    读取文本文件

    Args:
        file_path: 文件路径
        encoding: 文件编码

    Returns:
        文件内容
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()

    return content


def write_text_file(file_path: Union[str, Path],
                    content: str,
                    encoding: str = 'utf-8',
                    mode: str = 'w') -> None:
    """
    写入文本文件

    Args:
        file_path: 文件路径
        content: 内容
        encoding: 文件编码
        mode: 写入模式 ('w' 或 'a')
    """
    file_path = Path(file_path)

    # 确保目录存在
    ensure_directory(file_path.parent)

    with open(file_path, mode, encoding=encoding) as f:
        f.write(content)

    logger.debug(f"写入文本文件: {file_path}")


def read_json_file(file_path: Union[str, Path]) -> Dict:
    """
    读取JSON文件

    Args:
        file_path: 文件路径

    Returns:
        JSON数据
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def write_json_file(file_path: Union[str, Path],
                    data: Dict,
                    indent: int = 2,
                    ensure_ascii: bool = False) -> None:
    """
    写入JSON文件

    Args:
        file_path: 文件路径
        data: JSON数据
        indent: 缩进
        ensure_ascii: 是否确保ASCII编码
    """
    file_path = Path(file_path)

    # 确保目录存在
    ensure_directory(file_path.parent)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)

    logger.debug(f"写入JSON文件: {file_path}")


def read_yaml_file(file_path: Union[str, Path]) -> Dict:
    """
    读取YAML文件

    Args:
        file_path: 文件路径

    Returns:
        YAML数据
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return data


def write_yaml_file(file_path: Union[str, Path],
                    data: Dict,
                    default_flow_style: bool = False) -> None:
    """
    写入YAML文件

    Args:
        file_path: 文件路径
        data: YAML数据
        default_flow_style: YAML流风格
    """
    file_path = Path(file_path)

    # 确保目录存在
    ensure_directory(file_path.parent)

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=default_flow_style)

    logger.debug(f"写入YAML文件: {file_path}")


def read_pickle_file(file_path: Union[str, Path]) -> Any:
    """
    读取Pickle文件

    Args:
        file_path: 文件路径

    Returns:
        反序列化的数据
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    return data


def write_pickle_file(file_path: Union[str, Path],
                      data: Any,
                      protocol: int = pickle.HIGHEST_PROTOCOL) -> None:
    """
    写入Pickle文件

    Args:
        file_path: 文件路径
        data: 要序列化的数据
        protocol: pickle协议版本
    """
    file_path = Path(file_path)

    # 确保目录存在
    ensure_directory(file_path.parent)

    with open(file_path, 'wb') as f:
        pickle.dump(data, f, protocol=protocol)

    logger.debug(f"写入Pickle文件: {file_path}")


def backup_file(file_path: Union[str, Path],
                backup_dir: Optional[Union[str, Path]] = None,
                timestamp_format: str = "%Y%m%d_%H%M%S") -> Path:
    """
    备份文件

    Args:
        file_path: 要备份的文件路径
        backup_dir: 备份目录（默认与源文件同一目录）
        timestamp_format: 时间戳格式

    Returns:
        备份文件路径
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    # 确定备份目录
    if backup_dir is None:
        backup_dir = file_path.parent / "backups"
    else:
        backup_dir = Path(backup_dir)

    ensure_directory(backup_dir)

    # 生成备份文件名
    timestamp = datetime.now().strftime(timestamp_format)
    backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
    backup_path = backup_dir / backup_name

    # 复制文件
    shutil.copy2(file_path, backup_path)

    logger.info(f"文件备份: {file_path} -> {backup_path}")
    return backup_path


def get_file_info(file_path: Union[str, Path]) -> Dict:
    """
    获取文件详细信息

    Args:
        file_path: 文件路径

    Returns:
        文件信息字典
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    stat_info = file_path.stat()

    info = {
        'path': str(file_path.absolute()),
        'name': file_path.name,
        'stem': file_path.stem,
        'suffix': file_path.suffix,
        'size_bytes': stat_info.st_size,
        'size_human': get_file_size(file_path, human_readable=True),
        'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
        'accessed': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
        'is_file': file_path.is_file(),
        'is_dir': file_path.is_dir(),
        'is_symlink': file_path.is_symlink()
    }

    # 如果是文件，计算哈希
    if file_path.is_file():
        try:
            info['md5'] = get_file_hash(file_path, 'md5')
            info['sha256'] = get_file_hash(file_path, 'sha256')
        except Exception as e:
            logger.warning(f"计算文件哈希失败 {file_path}: {e}")

    return info


class FileUtils:
    """
    文件工具类（面向对象接口）
    """

    def __init__(self, base_dir: Union[str, Path] = "."):
        """
        初始化文件工具

        Args:
            base_dir: 基础目录
        """
        self.base_dir = Path(base_dir).absolute()
        logger.info(f"初始化FileUtils，基础目录: {self.base_dir}")

    def ensure_dir(self, relative_path: Union[str, Path]) -> Path:
        """
        确保目录存在

        Args:
            relative_path: 相对于基础目录的路径

        Returns:
            目录的绝对路径
        """
        abs_path = self.base_dir / relative_path
        return ensure_directory(abs_path)

    def find(self, pattern: str = "**/*", recursive: bool = True) -> List[Path]:
        """
        查找文件

        Args:
            pattern: 文件模式
            recursive: 是否递归搜索

        Returns:
            文件路径列表
        """
        return find_files(self.base_dir, pattern, recursive)

    def copy_to(self, source_files: List[Union[str, Path]],
                target_relative_path: Union[str, Path],
                overwrite: bool = False) -> List[Path]:
        """
        复制文件到目标目录

        Args:
            source_files: 源文件列表
            target_relative_path: 目标相对路径
            overwrite: 是否覆盖

        Returns:
            复制的文件路径列表
        """
        target_dir = self.base_dir / target_relative_path
        return copy_files(source_files, target_dir, overwrite)

    def get_info(self, relative_path: Union[str, Path]) -> Dict:
        """
        获取文件信息

        Args:
            relative_path: 相对路径

        Returns:
            文件信息字典
        """
        abs_path = self.base_dir / relative_path
        return get_file_info(abs_path)

    def backup(self, relative_path: Union[str, Path],
               backup_relative_dir: Optional[Union[str, Path]] = None) -> Path:
        """
        备份文件

        Args:
            relative_path: 相对路径
            backup_relative_dir: 备份相对目录

        Returns:
            备份文件路径
        """
        abs_path = self.base_dir / relative_path

        if backup_relative_dir:
            backup_dir = self.base_dir / backup_relative_dir
        else:
            backup_dir = None

        return backup_file(abs_path, backup_dir)

    def read_text(self, relative_path: Union[str, Path],
                  encoding: str = 'utf-8') -> str:
        """
        读取文本文件

        Args:
            relative_path: 相对路径
            encoding: 编码

        Returns:
            文件内容
        """
        abs_path = self.base_dir / relative_path
        return read_text_file(abs_path, encoding)

    def write_text(self, relative_path: Union[str, Path],
                   content: str,
                   encoding: str = 'utf-8',
                   mode: str = 'w') -> None:
        """
        写入文本文件

        Args:
            relative_path: 相对路径
            content: 内容
            encoding: 编码
            mode: 写入模式
        """
        abs_path = self.base_dir / relative_path
        write_text_file(abs_path, content, encoding, mode)


if __name__ == '__main__':
    # 测试代码
    test_dir = Path("test_files")

    # 清理旧测试文件
    if test_dir.exists():
        shutil.rmtree(test_dir)

    # 创建测试文件
    file1 = test_dir / "test1.txt"
    file2 = test_dir / "test2.json"
    file3 = test_dir / "subdir/test3.txt"

    ensure_directory(test_dir)
    ensure_directory(test_dir / "subdir")

    # 写入测试文件
    write_text_file(file1, "Hello, World!")
    write_json_file(file2, {"name": "test", "value": 123})
    write_text_file(file3, "Another test file")

    # 测试查找文件
    files = find_files(test_dir, "**/*.txt")
    print(f"找到 {len(files)} 个txt文件")

    # 测试文件信息
    info = get_file_info(file1)
    print(f"文件信息: {info}")

    # 测试文件哈希
    file_hash = get_file_hash(file1)
    print(f"文件哈希: {file_hash}")

    # 测试备份
    backup = backup_file(file1)
    print(f"备份文件: {backup}")

    # 测试FileUtils类
    utils = FileUtils(test_dir)
    utils_files = utils.find("**/*.txt")
    print(f"FileUtils找到 {len(utils_files)} 个文件")

    # 清理测试文件
    shutil.rmtree(test_dir)
    print("测试完成")