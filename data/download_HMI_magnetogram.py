"""
下载HMI的磁场数据
修改日期：2025-12-03
修改人：陈蒙
"""
import os
import time
import random
import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd
# 延迟导入 sunpy/astropy 在函数内部进行，避免模块导入时触发环境兼容问题
# 所有导入都在函数内部执行


# 根据series自动生成对应的保存文件夹名称
def get_save_dir(series_name: str) -> str:
    """根据数据类型生成对应的保存文件夹名称"""
    type_map = {
        'M': 'LOS_MAGNETIC_FIELD',
        'B': 'VECTOR_MAGNETIC_FIELD'
    }

    if '.' in series_name:
        _, suffix = series_name.split('.', 1)
        data_type = suffix[0]
        time_res = suffix[2:]
        if data_type in type_map:
            time_res_formatted = time_res.upper() if time_res == '45s' else time_res
            return f"HMI_{type_map[data_type]}_{time_res_formatted}_fits"

    return "HMI_MAGNETOGRAMS_DEFAULT"


def download_hmi_magnetogram(start_time: datetime,
                             end_time: datetime,
                             series: Optional[str] = None,
                             save_dir: Optional[str] = None,
                             notify_email: Optional[str] = "cm13616210865@163.com",
                             dry_run: bool = False,
                             max_files: Optional[int] = None,
                             fetch: bool = True,
                             max_retries: int = 5,
                             retry_backoff_base_seconds: float = 30.0,
                             retry_backoff_max_seconds: float = 300.0) -> List[str]:
    """封装的 HMI 磁图下载/查询函数

    Args:
        start_time, end_time: datetime
        series: JSOC 系列，如 'hmi.M_720s' 或 'hmi.M_45s'
        save_dir: 保存目录，若为 None 基于 series 生成
        notify_email: JSOC 通知邮箱
        dry_run: True 时不执行下载，返回将要下载的条目占位
        max_files: 限制下载条目数（可选）
        fetch: 如果为 False，只执行查询并返回 record 列表，不执行实际下载

    Returns:
        已下载文件路径列表或占位列表；当 fetch=False 时返回 sunpy.net.jsoc.Records 列表
    """
    # 延迟导入 heavy 依赖
    from sunpy.net import Fido, attrs as a
    from parfive import Downloader, SessionConfig
    import aiohttp
    logger = logging.getLogger(__name__)

    if series is None:
        series = 'hmi.M_720s'

    if save_dir is None:
        save_dir = get_save_dir(series)
    os.makedirs(save_dir, exist_ok=True)

    start_time = pd.to_datetime(start_time).to_pydatetime()
    end_time = pd.to_datetime(end_time).to_pydatetime()

    # dry run - 不发起网络请求，返回估算条目占位
    if dry_run:
        seconds = max(1, int((end_time - start_time).total_seconds()))
        est = max(1, int(seconds / 3600) + 1)
        if max_files:
            est = min(est, max_files)
        # 如果只是查询，不加前缀
        if not fetch:
            return [f"REC_HMI_{series}_{i+1}" for i in range(est)]
        return [f"HMI_{series}_item_{i+1}" for i in range(est)]

    # 构建查询并执行下载（带自动重试，避免 JSOC/网络偶发卡死导致必须手动重跑）
    last_exc: Optional[Exception] = None
    effective_retries = max(1, int(max_retries))

    for attempt in range(1, effective_retries + 1):
        try:
            # 构建查询参数列表，避免将 None 传给 Fido.search（否则 SunPy 内部会对 None 调用 .collides 导致
            # “'NoneType' object has no attribute 'collides'” 错误）
            search_attrs = [
                a.Time(start_time, end_time),
                a.jsoc.Series(series),
            ]
            if notify_email:
                search_attrs.append(a.jsoc.Notify(notify_email))

            query = Fido.search(*search_attrs)
            if len(query[0]) == 0:
                return []

            # 限制条目数量
            records = query[0]
            if max_files and len(records) > max_files:
                records = records[:max_files]

            if not fetch:
                return records

            if len(records) == 0:
                return []

            # 显示稳定的进度条：只显示总体进度（不为每个文件单独开进度条，避免刷屏）
            os.environ.pop('SUNPY_DISABLE_PROGRESS_BARS', None)
            timeouts = aiohttp.ClientTimeout(total=0, sock_read=1800)
            session_config = SessionConfig(file_progress=False, notebook=False, timeouts=timeouts)
            downloader = Downloader(max_conn=5, progress=True, overwrite=False, config=session_config)

            fetched = Fido.fetch(records, path=f"{save_dir}/{'{file}'}", downloader=downloader)
            return [str(f) for f in fetched]

        except Exception as e:
            last_exc = e
            if attempt >= effective_retries:
                logger.error(
                    "HMI (%s) 下载失败（已重试 %s 次）: %s: %s",
                    series,
                    effective_retries,
                    type(e).__name__,
                    e,
                )
                raise

            backoff = min(
                float(retry_backoff_max_seconds),
                float(retry_backoff_base_seconds) * (2 ** (attempt - 1)),
            )
            sleep_seconds = max(0.0, backoff + random.random() * min(5.0, backoff))
            logger.warning(
                "HMI (%s) 第 %s/%s 次下载失败，%.1fs 后重试: %s: %s",
                series,
                attempt,
                effective_retries,
                sleep_seconds,
                type(e).__name__,
                e,
            )
            time.sleep(sleep_seconds)

    # 理论上不会到这里
    if last_exc is not None:
        raise last_exc
    return []

if __name__ == '__main__':
    # 示例运行（dry run）
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = datetime(2024, 1, 1, 0, 1, 0)
    items = download_hmi_magnetogram(start_time, end_time, series='hmi.M_45s', notify_email='cm13616210865@163.com', dry_run=True)
    print(f"示例查询返回 {len(items)} 条目")