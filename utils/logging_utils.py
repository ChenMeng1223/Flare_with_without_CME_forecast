"""
日志工具模块
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


def setup_logging(log_dir: str,
                  log_name: Optional[str] = None,
                  level: int = logging.INFO,
                  debug: bool = False,
                  console_level: Optional[int] = None) -> logging.Logger:
    """
    设置日志记录

    Args:
        log_dir: 日志目录
        log_name: 日志名称（默认使用调用模块名称）
        level: 日志级别
        debug: 是否启用调试模式
        console_level: 控制台日志级别（默认与level相同）

    Returns:
        配置好的Logger对象
    """
    # 确定日志名称
    if log_name is None:
        # 获取调用者的模块名称
        import inspect
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        log_name = module.__name__ if module else 'unknown'

    # 确保日志目录存在
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 生成日志文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{log_name}_{timestamp}.log"
    log_filepath = log_dir_path / log_filename

    # 设置日志级别
    if debug:
        level = logging.DEBUG
        console_level = logging.DEBUG

    if console_level is None:
        console_level = level

    # 创建格式化器
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # 创建日志记录器
    logger = logging.getLogger(log_name)
    logger.setLevel(level)

    # 避免重复添加处理器
    if logger.handlers:
        return logger

    # 文件处理器
    file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 防止日志向上传播到 root logger（避免重复输出）
    logger.propagate = False

    # 记录日志配置信息
    logger.info(f"日志系统初始化完成")
    logger.info(f"日志级别: {logging.getLevelName(level)}")
    logger.info(f"日志文件: {log_filepath}")
    logger.info(f"调试模式: {debug}")

    return logger


def setup_exception_logging(log_dir: str, log_name: str = 'exceptions') -> None:
    """
    设置异常日志记录

    Args:
        log_dir: 日志目录
        log_name: 异常日志名称
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # 异常日志文件
    timestamp = datetime.now().strftime('%Y%m%d')
    exception_log = log_dir_path / f"{log_name}_{timestamp}.log"

    def handle_exception(exc_type, exc_value, exc_traceback):
        """
        全局异常处理器
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # 忽略键盘中断
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # 记录异常
        logger = logging.getLogger('exception_handler')

        # 确保logger有处理器
        if not logger.handlers:
            logger.setLevel(logging.ERROR)
            handler = logging.FileHandler(exception_log, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        # 防止向上传播，避免重复打印到根日志
        logger.propagate = False

        logger.error("未捕获的异常:", exc_info=(exc_type, exc_value, exc_traceback))

        # 同时打印到控制台
        print(f"\n❌ 未捕获的异常: {exc_type.__name__}: {exc_value}", file=sys.stderr)

    # 设置全局异常处理器
    sys.excepthook = handle_exception


class JsonFormatter(logging.Formatter):
    """
    JSON格式的日志格式化器
    """

    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """
        将日志记录格式化为JSON字符串

        Args:
            record: 日志记录

        Returns:
            JSON格式的日志字符串
        """
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'process': record.process,
            'thread': record.threadName if record.threadName else record.thread
        }

        # 添加异常信息
        if record.exc_info:
            log_data['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.formatException(record.exc_info) if record.exc_info[2] else None
            }

        # 添加额外字段
        if self.include_extra and hasattr(record, 'extra'):
            log_data['extra'] = record.extra

        return json.dumps(log_data, ensure_ascii=False)


def setup_json_logging(log_dir: str,
                       log_name: str = 'application',
                       level: int = logging.INFO) -> logging.Logger:
    """
    设置JSON格式的日志记录

    Args:
        log_dir: 日志目录
        log_name: 日志名称
        level: 日志级别

    Returns:
        配置好的Logger对象
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)

    # JSON日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_log_file = log_dir_path / f"{log_name}_{timestamp}.jsonl"

    logger = logging.getLogger(f"{log_name}_json")
    logger.setLevel(level)

    # JSON文件处理器
    json_handler = logging.FileHandler(json_log_file, encoding='utf-8')
    json_handler.setLevel(level)
    json_handler.setFormatter(JsonFormatter())
    logger.addHandler(json_handler)

    # 控制台处理器（普通格式）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # 防止日志向上传播到 root logger（避免重复输出）
    logger.propagate = False

    logger.info(f"JSON日志系统初始化完成，日志文件: {json_log_file}")

    return logger


def get_logger(name: str,
               log_dir: Optional[str] = None,
               level: int = logging.INFO) -> logging.Logger:
    """
    获取或创建日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志目录（可选，如果提供则创建文件处理器）
        level: 日志级别

    Returns:
        Logger对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 如果已经配置过处理器，直接返回
    if logger.handlers:
        return logger

    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 如果需要文件处理器
    if log_dir is not None:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # 创建日志文件
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = log_dir_path / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # 防止日志向上传播到 root logger（避免重复输出）
    logger.propagate = False

    return logger


def add_file_handler(logger: logging.Logger,
                     log_file: str,
                     level: int = logging.INFO,
                     formatter: Optional[logging.Formatter] = None) -> None:
    """
    为日志记录器添加文件处理器

    Args:
        logger: Logger对象
        log_file: 日志文件路径
        level: 日志级别
        formatter: 格式化器（可选）
    """
    # 确保目录存在
    log_file_path = Path(log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建文件处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)

    # 设置格式化器
    if formatter is None:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"添加文件处理器: {log_file_path}")


class LoggingContext:
    """
    日志上下文管理器，用于临时修改日志级别
    """

    def __init__(self, logger: logging.Logger, level: int = logging.DEBUG):
        """
        初始化上下文管理器

        Args:
            logger: Logger对象
            level: 临时日志级别
        """
        self.logger = logger
        self.level = level
        self.old_level = logger.level
        self.old_handlers_level = []

    def __enter__(self):
        """进入上下文"""
        # 保存旧级别
        self.old_level = self.logger.level

        # 保存所有处理器的旧级别
        for handler in self.logger.handlers:
            self.old_handlers_level.append(handler.level)
            handler.setLevel(self.level)

        # 设置logger级别
        self.logger.setLevel(self.level)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        # 恢复logger级别
        self.logger.setLevel(self.old_level)

        # 恢复处理器级别
        for handler, old_level in zip(self.logger.handlers, self.old_handlers_level):
            handler.setLevel(old_level)

        return False  # 不处理异常


def log_execution_time(logger: logging.Logger):
    """
    记录函数执行时间的装饰器

    Args:
        logger: Logger对象

    Returns:
        装饰器函数
    """
    import time
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"开始执行 {func.__name__}")

            try:
                result = func(*args, **kwargs)
                elapsed_time = time.time() - start_time
                logger.debug(f"执行 {func.__name__} 完成，耗时: {elapsed_time:.3f}秒")
                return result
            except Exception as e:
                elapsed_time = time.time() - start_time
                logger.error(f"执行 {func.__name__} 失败，耗时: {elapsed_time:.3f}秒，错误: {e}")
                raise

        return wrapper

    return decorator


def log_function_call(logger: logging.Logger, level: int = logging.DEBUG):
    """
    记录函数调用的装饰器

    Args:
        logger: Logger对象
        level: 日志级别

    Returns:
        装饰器函数
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 记录函数调用
            func_name = func.__name__
            arg_str = ', '.join([repr(arg) for arg in args])
            kwarg_str = ', '.join([f"{k}={repr(v)}" for k, v in kwargs.items()])

            if arg_str and kwarg_str:
                call_str = f"{func_name}({arg_str}, {kwarg_str})"
            elif arg_str:
                call_str = f"{func_name}({arg_str})"
            elif kwarg_str:
                call_str = f"{func_name}({kwarg_str})"
            else:
                call_str = f"{func_name}()"

            logger.log(level, f"调用: {call_str}")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"返回: {func_name} -> {repr(result)}")
                return result
            except Exception as e:
                logger.error(f"异常: {func_name} -> {type(e).__name__}: {e}")
                raise

        return wrapper

    return decorator


def setup_logging_from_config(config_file: str) -> Dict[str, logging.Logger]:
    """
    从配置文件设置日志系统

    Args:
        config_file: 配置文件路径

    Returns:
        字典：{logger_name: logger_object}
    """
    import yaml

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    loggers = {}
    log_config = config.get('logging', {})

    # 全局设置
    log_dir = log_config.get('log_dir', 'logs')
    default_level = getattr(logging, log_config.get('level', 'INFO'))

    # 为每个logger配置
    for logger_config in log_config.get('loggers', []):
        name = logger_config['name']
        level = getattr(logging, logger_config.get('level', 'INFO'))

        # 创建logger
        logger = get_logger(name, log_dir, level)

        # 添加额外的处理器
        for handler_config in logger_config.get('handlers', []):
            handler_type = handler_config['type']

            if handler_type == 'file':
                filename = handler_config['filename']
                log_file = os.path.join(log_dir, filename)
                add_file_handler(logger, log_file, level)

            elif handler_type == 'json':
                json_logger = setup_json_logging(log_dir, name, level)
                loggers[f"{name}_json"] = json_logger

        loggers[name] = logger

    # 设置异常日志
    if log_config.get('capture_exceptions', True):
        setup_exception_logging(log_dir)

    return loggers


if __name__ == '__main__':
    # 测试代码
    logger = setup_logging('test_logs', 'test_logger', debug=True)
    logger.debug("这是一条调试信息")
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
    logger.error("这是一条错误")

    try:
        raise ValueError("测试异常")
    except ValueError as e:
        logger.exception("捕获到异常")

    # 测试JSON日志
    json_logger = setup_json_logging('test_logs', 'json_test')
    json_logger.info("这是一条JSON格式的日志")

    # 测试上下文管理器
    with LoggingContext(logger, logging.DEBUG):
        logger.debug("在上下文中的调试信息")

    logger.debug("恢复后的调试信息（可能不显示）")

    print("日志测试完成")