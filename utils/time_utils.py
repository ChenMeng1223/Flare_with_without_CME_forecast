"""
时间工具模块
"""
import time
from datetime import datetime, timedelta
from typing import Optional, Union, Any
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def parse_timestamp(timestamp_str: str,
                    format: str = "%Y-%m-%dT%H:%M:%S") -> datetime:
    """
    解析时间戳字符串

    Args:
        timestamp_str: 时间戳字符串
        format: 时间格式

    Returns:
        datetime对象
    """
    try:
        return datetime.strptime(timestamp_str, format)
    except ValueError:
        # 尝试其他常见格式
        formats = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y%m%d%H%M%S"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析时间戳: {timestamp_str}")


def format_timedelta(delta: timedelta,
                     format: str = "auto") -> str:
    """
    格式化时间间隔

    Args:
        delta: timedelta对象
        format: 格式 ('auto', 'hh:mm:ss', 'days', 'hours', 'minutes', 'seconds')

    Returns:
        格式化后的时间间隔字符串
    """
    total_seconds = int(delta.total_seconds())

    if format == "auto":
        if total_seconds < 60:
            return f"{total_seconds}秒"
        elif total_seconds < 3600:
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            return f"{minutes}分{seconds}秒"
        elif total_seconds < 86400:
            hours = total_seconds // 3600
            minutes = (total_seconds % 3600) // 60
            return f"{hours}小时{minutes}分"
        else:
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            return f"{days}天{hours}小时"

    elif format == "hh:mm:ss":
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    elif format == "days":
        return f"{delta.days}天"

    elif format == "hours":
        return f"{total_seconds / 3600:.2f}小时"

    elif format == "minutes":
        return f"{total_seconds / 60:.2f}分钟"

    elif format == "seconds":
        return f"{total_seconds}秒"

    else:
        raise ValueError(f"不支持的格式: {format}")


class Timer:
    """
    计时器类
    """

    def __init__(self, name: str = "计时器", logger: Optional[logging.Logger] = None):
        """
        初始化计时器

        Args:
            name: 计时器名称
            logger: Logger对象
        """
        self.name = name
        self.logger = logger
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed: Optional[float] = None

    def __enter__(self):
        """进入上下文，开始计时"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，停止计时并记录"""
        self.stop()

        if self.logger:
            self.logger.info(f"{self.name} 耗时: {self.format()}")
        else:
            print(f"{self.name} 耗时: {self.format()}")

        return False  # 不处理异常

    def start(self) -> None:
        """开始计时"""
        self.start_time = time.time()
        self.end_time = None
        self.elapsed = None

    def stop(self) -> float:
        """停止计时并返回耗时（秒）"""
        if self.start_time is None:
            raise RuntimeError("计时器未启动")

        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        return self.elapsed

    def lap(self) -> float:
        """
        记录单圈时间

        Returns:
            从开始到当前的时间
        """
        if self.start_time is None:
            raise RuntimeError("计时器未启动")

        current = time.time()
        lap_time = current - self.start_time
        return lap_time

    def format(self, format: str = "auto") -> str:
        """
        格式化耗时

        Args:
            format: 格式

        Returns:
            格式化后的时间字符串
        """
        if self.elapsed is None:
            if self.start_time is None:
                return "未开始"
            else:
                # 仍在运行中
                current_elapsed = time.time() - self.start_time
                return f"运行中: {format_timedelta(timedelta(seconds=current_elapsed), format)}"

        return format_timedelta(timedelta(seconds=self.elapsed), format)

    def reset(self) -> None:
        """重置计时器"""
        self.start_time = None
        self.end_time = None
        self.elapsed = None


@contextmanager
def timeit(name: str = "操作", logger: Optional[logging.Logger] = None):
    """
    计时上下文管理器

    Args:
        name: 操作名称
        logger: Logger对象
    """
    timer = Timer(name, logger)
    with timer:
        yield timer


def timeit_decorator(name: Optional[str] = None,
                     logger: Optional[logging.Logger] = None):
    """
    计时装饰器

    Args:
        name: 操作名称（默认使用函数名）
        logger: Logger对象

    Returns:
        装饰器函数
    """
    import functools

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            with Timer(timer_name, logger) as timer:
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


def get_current_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    获取当前时间字符串

    Args:
        format: 时间格式

    Returns:
        当前时间字符串
    """
    return datetime.now().strftime(format)


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取时间戳字符串

    Args:
        format: 时间格式

    Returns:
        时间戳字符串
    """
    return datetime.now().strftime(format)


def calculate_time_window(start_time: Union[str, datetime],
                          end_time: Union[str, datetime],
                          window_size: timedelta,
                          stride: Optional[timedelta] = None) -> list:
    """
    计算时间窗口

    Args:
        start_time: 开始时间
        end_time: 结束时间
        window_size: 窗口大小
        stride: 步长（默认等于窗口大小）

    Returns:
        时间窗口列表 [(window_start, window_end), ...]
    """
    # 解析时间
    if isinstance(start_time, str):
        start_time = parse_timestamp(start_time)
    if isinstance(end_time, str):
        end_time = parse_timestamp(end_time)

    # 设置步长
    if stride is None:
        stride = window_size

    windows = []
    current_start = start_time

    while current_start + window_size <= end_time:
        current_end = current_start + window_size
        windows.append((current_start, current_end))
        current_start += stride

    return windows


def time_since(start_time: Union[datetime, str],
               end_time: Optional[Union[datetime, str]] = None) -> str:
    """
    计算从开始时间到现在的时间间隔（人类可读格式）

    Args:
        start_time: 开始时间
        end_time: 结束时间（默认现在）

    Returns:
        时间间隔字符串
    """
    # 解析时间
    if isinstance(start_time, str):
        start_time = parse_timestamp(start_time)

    if end_time is None:
        end_time = datetime.now()
    elif isinstance(end_time, str):
        end_time = parse_timestamp(end_time)

    # 计算时间间隔
    delta = end_time - start_time

    # 格式化
    if delta.days > 365:
        years = delta.days // 365
        return f"{years}年"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months}个月"
    elif delta.days > 0:
        return f"{delta.days}天"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours}小时"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes}分钟"
    else:
        return f"{delta.seconds}秒"


class TimeSeriesGenerator:
    """
    时间序列生成器
    """

    def __init__(self, start_time: Union[str, datetime],
                 end_time: Union[str, datetime],
                 interval: timedelta):
        """
        初始化时间序列生成器

        Args:
            start_time: 开始时间
            end_time: 结束时间
            interval: 时间间隔
        """
        if isinstance(start_time, str):
            self.start_time = parse_timestamp(start_time)
        else:
            self.start_time = start_time

        if isinstance(end_time, str):
            self.end_time = parse_timestamp(end_time)
        else:
            self.end_time = end_time

        self.interval = interval

    def generate(self) -> list:
        """
        生成时间序列

        Returns:
            时间点列表
        """
        times = []
        current = self.start_time

        while current <= self.end_time:
            times.append(current)
            current += self.interval

        return times

    def generate_str(self, format: str = "%Y-%m-%dT%H:%M:%S") -> list:
        """
        生成时间序列字符串

        Args:
            format: 时间格式

        Returns:
            时间字符串列表
        """
        times = self.generate()
        return [t.strftime(format) for t in times]


def estimate_remaining_time(start_time: float,
                            current_progress: float,
                            total_items: int,
                            processed_items: int) -> str:
    """
    估计剩余时间

    Args:
        start_time: 开始时间戳
        current_progress: 当前进度（0-1）
        total_items: 总项目数
        processed_items: 已处理项目数

    Returns:
        剩余时间字符串
    """
    if current_progress <= 0 or processed_items <= 0:
        return "未知"

    # 计算已用时间
    elapsed_time = time.time() - start_time

    # 估计总时间
    estimated_total_time = elapsed_time / current_progress

    # 计算剩余时间
    remaining_time = estimated_total_time - elapsed_time

    # 格式化
    if remaining_time < 60:
        return f"{remaining_time:.0f}秒"
    elif remaining_time < 3600:
        return f"{remaining_time / 60:.1f}分钟"
    else:
        return f"{remaining_time / 3600:.1f}小时"


if __name__ == '__main__':
    # 测试代码
    print("当前时间:", get_current_time())
    print("时间戳:", get_timestamp())

    # 测试计时器
    with Timer("测试操作") as timer:
        time.sleep(1.5)

    print(f"耗时: {timer.format()}")

    # 测试时间窗口
    start = "2024-01-01T00:00:00"
    end = "2024-01-01T02:00:00"
    windows = calculate_time_window(start, end, timedelta(hours=1), timedelta(minutes=30))

    print(f"\n时间窗口:")
    for w_start, w_end in windows:
        print(f"  {w_start} - {w_end}")

    # 测试时间序列生成器
    generator = TimeSeriesGenerator(start, end, timedelta(minutes=15))
    times = generator.generate_str()
    print(f"\n时间序列 (15分钟间隔): {len(times)} 个时间点")
    print(f"前5个: {times[:5]}")

    # 测试时间间隔计算
    print(f"\n从 {start} 到现在的时间间隔: {time_since(start)}")


    # 测试装饰器
    @timeit_decorator("装饰器测试")
    def test_function():
        time.sleep(0.5)
        return "完成"


    result = test_function()
    print(f"装饰器测试结果: {result}")