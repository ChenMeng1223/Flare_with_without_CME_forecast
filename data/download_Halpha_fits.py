"""
下载CHASE Hα数据
修改日期：2026-03-06
修改人：陈蒙
"""
import os
import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Optional
import requests
from urllib.parse import urlparse


def download_halpha_fits(start_time: datetime,
                         end_time: datetime,
                         url_file: Optional[str] = None,
                         save_dir: Optional[str] = None,
                         dry_run: bool = False,
                         fetch: bool = True,
                         max_retries: int = 3,
                         retry_backoff_base_seconds: float = 5.0,
                         retry_backoff_max_seconds: float = 30.0) -> List[str]:
    """封装的 CHASE Hα 下载/查询函数

    Args:
        start_time, end_time: datetime 范围
        url_file: 包含URL列表的TXT文件路径，若为None则使用默认路径
        save_dir: 保存目录，若为 None 会自动生成
        dry_run: 如果为 True，仅返回将被下载的条目占位列表，不执行网络下载
        fetch: 如果为 False，只进行查询并返回记录列表
        max_retries: 最大重试次数
        retry_backoff_base_seconds: 重试基础等待时间
        retry_backoff_max_seconds: 重试最大等待时间

    Returns:
        已下载文件路径列表（dry_run=True 时为占位列表）或记录列表（fetch=False）
    """
    logger = logging.getLogger(__name__)

    # 默认URL文件路径
    if url_file is None:
        url_file = os.path.join(os.path.dirname(__file__), 'raw', 'Halpha_download.txt')

    # 构建保存目录
    if save_dir is None:
        save_dir = "CHASE_HA_LINEWIDTH_fits"
    os.makedirs(save_dir, exist_ok=True)

    # dry run - 不发起网络请求，返回估算条目占位
    if dry_run:
        return [f"CHASE_HA_item_{i+1}" for i in range(1)]

    # 读取URL列表
    if not os.path.exists(url_file):
        logger.error(f"URL文件不存在: {url_file}")
        return []

    try:
        with open(url_file, 'r', encoding='utf-8') as f:
            urls = f.readlines()
    except Exception as e:
        logger.error(f"读取URL文件失败: {e}")
        return []

    # 解析URL并提取时间信息
    url_time_pairs = []
    for url in urls:
        url = url.strip()
        if url.endswith(','):
            url = url[:-1]
        if not url:
            continue

        try:
            parsed_url = urlparse(url)
            file_name = os.path.basename(parsed_url.path)
            # 假设文件名格式为 "RSM20241222T010022_0000_HA.fits"
            # 从文件名中提取时间信息
            if len(file_name) > 15 and file_name.startswith('RSM'):  # 确保文件名足够长
                date_str = file_name[3:11]  # 提取日期部分 YYYYMMDD
                time_str = file_name[12:18]  # 提取时间部分 HHMMSS
                try:
                    obs_datetime = datetime.strptime(f"{date_str}{time_str}", '%Y%m%d%H%M%S')
                    url_time_pairs.append((url, obs_datetime, file_name))
                except ValueError:
                    logger.warning(f"无法解析文件名中的时间: {file_name}")
                    continue
        except Exception as e:
            logger.warning(f"解析URL失败: {url}, 错误: {e}")
            continue

    if not url_time_pairs:
        logger.warning("未找到有效的URL")
        return []

    # 查找时间范围内最接近的观测数据并直接打印结果
    target_time = start_time + (end_time - start_time) / 2  # 使用时间窗口中心作为目标时间
    best_match = None
    min_time_diff = timedelta.max
    candidates = []

    for url, obs_time, file_name in url_time_pairs:
        # 检查观测时间是否在指定范围内
        if start_time <= obs_time <= end_time:
            time_diff = abs(obs_time - target_time)
            candidates.append((obs_time, file_name, time_diff))
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = (url, obs_time, file_name)

    if not best_match:
        logger.info(f"Hα: 在时间窗口 [{start_time} ~ {end_time}] 中未找到合适的观测数据")
        return []

    url, obs_time, file_name = best_match
    logger.info(f"Hα: 时间窗口 [{start_time} ~ {end_time}] 找到 {len(candidates)} 个候选观测，选择最接近的观测 {obs_time} (文件: {file_name})，时间差: {min_time_diff}，开始下载")

    # 即便fetch=False，也仅仅返回信息，不需要上层再调用一次
    if not fetch:
        return [{
            'url': url,
            'obs_time': obs_time,
            'file_name': file_name
        }]

    # 执行下载
    file_path = os.path.join(save_dir, file_name)
    last_exc = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"下载Hα文件: {file_name} (尝试 {attempt}/{max_retries})")
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(file_path, 'wb') as f:
                last_update_ts = time.time()
                total_written = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_written += len(chunk)
                        now_ts = time.time()
                        # 1min 刷新一次下载进度
                        if now_ts - last_update_ts >= 60:
                            logger.info(f"Hα文件 {file_name} 下载进度: {total_written} bytes")
                            last_update_ts = now_ts

            logger.info(f"Hα文件下载成功: {file_path} (共 {total_written} bytes)")
            return [file_path]

        except Exception as e:
            last_exc = e
            if attempt >= max_retries:
                logger.error(f"Hα文件下载失败（已重试 {max_retries} 次）: {type(e).__name__}: {e}")
                break

            backoff = min(
                float(retry_backoff_max_seconds),
                float(retry_backoff_base_seconds) * (2 ** (attempt - 1)),
            )
            sleep_seconds = max(0.0, backoff + random.random() * min(2.0, backoff))
            logger.warning(f"Hα下载第 {attempt}/{max_retries} 次失败，{sleep_seconds:.1f}s 后重试: {type(e).__name__}: {e}")
            time.sleep(sleep_seconds)

    return []


if __name__ == '__main__':
    # 示例运行方式，便于单独测试
    start_time = datetime(2024, 12, 23, 0, 0, 0)
    end_time = datetime(2024, 12, 23, 23, 59, 59)

    print("开始示例 Hα 下载...")
    files = download_halpha_fits(start_time, end_time, dry_run=True)
    