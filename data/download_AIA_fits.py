import os
import time
import random
import logging
from datetime import datetime
from typing import List, Optional
import pandas as pd
# 注意：延迟导入 sunpy/astropy，以避免模块在非下载场景下导入失败
# 所有导入都在函数内部执行


def download_aia_fits(start_time: datetime,
                      end_time: datetime,
                      wavelength: float,
                      series: Optional[str] = None,
                      save_dir: Optional[str] = None,
                      notify_email: Optional[str] = "cm13616210865@163.com",
                      dry_run: bool = False,
                      fetch: bool = True,
                      max_retries: int = 5,
                      retry_backoff_base_seconds: float = 30.0,
                      retry_backoff_max_seconds: float = 300.0) -> List[str]:
    """封装的 AIA 下载/查询函数

    Args:
        start_time, end_time: datetime 范围
        wavelength: 波长，单位 Å（数值）
        series: 可选的 JSOC 系列名（例如 'aia.lev1_euv_12s'），若提供将使用 jsoc.Series 查询
        save_dir: 保存目录，若为 None 会基于 series/wavelength 自动生成
        notify_email: JSOC 通知邮箱（可选）
        dry_run: 如果为 True，仅返回将被下载的条目占位列表，不执行网络下载
        fetch: 如果为 False，只进行查询并返回记录列表

    Returns:
        已下载文件路径列表（dry_run=True 时为占位列表）或记录列表（fetch=False）
    """
    # 构建保存目录
    if save_dir is None:
        if series:
            series_suffix = series.split('.', 1)[1] if '.' in series else series
            save_dir = f"AIA_{series_suffix.upper()}_{int(wavelength)}A_fits"
        else:
            save_dir = f"AIA_{int(wavelength)}A_fits"
    os.makedirs(save_dir, exist_ok=True)

    # 延迟导入 SunPy（避免在不需要时触发 heavy imports）
    import astropy.units as u
    from sunpy.net import Fido, attrs as a
    from parfive import Downloader, SessionConfig
    import aiohttp

    wavelength_q = wavelength * u.angstrom
    logger = logging.getLogger(__name__)

    start_time = pd.to_datetime(start_time).to_pydatetime()
    end_time = pd.to_datetime(end_time).to_pydatetime()

    # dry run - 不发起网络请求，返回估算条目占位
    if dry_run:
        seconds = max(1, int((end_time - start_time).total_seconds()))
        # 简单估算：每小时1个观测点
        est = max(1, int(seconds / 3600) + 1)
        if not fetch:
            return [f"REC_AIA_{int(wavelength)}A_{i+1}" for i in range(est)]
        return [f"AIA_{int(wavelength)}A_item_{i+1}" for i in range(est)]

    # 构建查询并执行下载（带自动重试，避免 JSOC/网络偶发卡死导致必须手动重跑）
    last_exc: Optional[Exception] = None
    effective_retries = max(1, int(max_retries))

    for attempt in range(1, effective_retries + 1):
        try:
            if series:
                # 同样避免将 None 作为 Attr 传入 Fido.search
                search_attrs = [
                    a.Time(start_time, end_time),
                    a.jsoc.Series(series),
                    a.jsoc.Wavelength(wavelength_q),
                ]
                if notify_email:
                    search_attrs.append(a.jsoc.Notify(notify_email))
                query = Fido.search(*search_attrs)
            else:
                query = Fido.search(
                    a.Time(start_time, end_time),
                    a.Instrument('AIA'),
                    a.Wavelength(wavelength_q),
                )

            if len(query[0]) == 0:
                return []

            records = query[0]
            if not fetch:
                return records

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
                    "AIA %sÅ 下载失败（已重试 %s 次）: %s: %s",
                    int(wavelength),
                    effective_retries,
                    type(e).__name__,
                    e,
                )
                raise

            backoff = min(
                float(retry_backoff_max_seconds),
                float(retry_backoff_base_seconds) * (2 ** (attempt - 1)),
            )
            # 加一点抖动，避免多进程/多任务同时重试导致“同步拥塞”
            sleep_seconds = max(0.0, backoff + random.random() * min(5.0, backoff))
            logger.warning(
                "AIA %sÅ 第 %s/%s 次下载失败，%.1fs 后重试: %s: %s",
                int(wavelength),
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
    # 保留示例运行方式，便于单独测试
    # 示例参数
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    end_time = datetime(2024, 1, 1, 0, 0, 30)
    series = "aia.lev1_euv_12s"
    wavelength = 193
    notify_email = "cm13616210865@163.com"

    print("开始示例 AIA 下载（dry run）...")
    items = download_aia_fits(start_time, end_time, wavelength, series=series, notify_email=notify_email, dry_run=True)
    print(f"示例查询返回 {len(items)} 条目")