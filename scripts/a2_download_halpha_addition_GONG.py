#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从 GONG 站点下载 Halpha 文件来补齐 CHASE 缺失数据。"""

import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.b_1_add_event_id_to_events_xlsx import EventIdGenerator


BASE_URL = "https://gong2.nso.edu/ftp/HA/haf/"

# ===== GONG补充下载范围配置（直接改这里即可） =====
EVENTS_XLSX_PATH = Path("data/raw/events.xlsx")
ONLY_PROCESS_EVENTS_IN_TABLE = True  # True: 仅处理事件表中的事件组；False: 处理 downloaded 下全部 EVT_* 目录
# ==========================================


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = create_session()


def get_links(url: str) -> List[str]:
    try:
        resp = SESSION.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        return [a["href"].rstrip("/") for a in soup.find_all("a", href=True) if a["href"] != "../"]
    except Exception as e:
        print(f"[WARN] get_links({url}) failed: {e}")
        return []


def parse_iso_datetime(dt_str: str) -> Optional[datetime]:
    if not dt_str or dt_str.strip() == "":
        return None
    try:
        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%dT%H:%M:%S"]:
            try:
                return datetime.strptime(dt_str.strip(), fmt)
            except ValueError:
                continue
        return datetime.fromisoformat(dt_str.strip())
    except Exception:
        return None


def parse_event_id_to_datetime(event_id: str) -> Optional[datetime]:
    match = re.match(r"EVT_(\d{8})_(\d{6})", event_id)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y%m%d %H%M%S")
    except ValueError:
        return None


def parse_event_table(path: Path) -> Dict[str, Dict[str, Optional[datetime]]]:
    result: Dict[str, Dict[str, Optional[datetime]]] = {}
    print(f"[INFO] 读取事件表: {path}")

    df = pd.read_excel(path)
    if df.empty:
        print("[WARN] 事件表为空")
        return result

    generator = EventIdGenerator()
    df = generator.combine_date_time(df)

    missing_before = 0
    if "event_id" not in df.columns:
        missing_before = len(df)
        df["event_id"] = generator.generate_event_ids(df)
        print(f"[INFO] 事件表缺少 event_id 列，已自动为 {missing_before} 行生成 event_id")
    else:
        existing_event_ids = df["event_id"]
        missing_mask = (
            existing_event_ids.isna()
            | (existing_event_ids.astype(str).str.strip() == "")
            | (existing_event_ids.astype(str).str.lower() == "nan")
        )
        missing_before = int(missing_mask.sum())
        if missing_before > 0:
            generated_event_ids = generator.generate_event_ids(df)
            df.loc[missing_mask, "event_id"] = generated_event_ids[missing_mask]
            print(f"[INFO] 事件表中有 {missing_before} 行缺少 event_id，已自动补齐")

    def _read_dt(row: pd.Series, col: str) -> Optional[datetime]:
        if col not in df.columns:
            return None
        value = row.get(col)
        if pd.isna(value) or value is None:
            return None
        return parse_iso_datetime(str(value))

    valid_count = 0
    skipped_count = 0
    for _, row in df.iterrows():
        event_id = str(row.get("event_id", "")).strip()
        if not event_id or event_id.lower() == "nan":
            skipped_count += 1
            continue

        result[event_id] = {
            "start_time": _read_dt(row, "start_time"),
            "peak_time": _read_dt(row, "peak_time"),
            "end_time": _read_dt(row, "end_time"),
        }
        valid_count += 1

    print(f"[INFO] 事件表解析完成: 有效事件 {valid_count} 个，跳过 {skipped_count} 行")
    return result


def load_config() -> Dict:
    config_path = Path("configs/data_config.yaml")
    default = {"pre_event_hours": 6, "post_event_hours": 3, "cadence_minutes": 60}
    if not config_path.exists():
        print(f"[WARN] 配置文件不存在: {config_path}，使用默认配置")
        return default

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        time_config = config.get("data", {}).get("time", {})
        return {
            "pre_event_hours": time_config.get("pre_event_hours", 6),
            "post_event_hours": time_config.get("post_event_hours", 3),
            "cadence_minutes": time_config.get("cadence_minutes", 60),
        }
    except Exception as e:
        print(f"[WARN] 加载配置失败: {e}，使用默认配置")
        return default


def calculate_sampling_points(event_info: Dict[str, Optional[datetime]], config: Dict) -> List[datetime]:
    start_time = event_info.get("start_time")
    if start_time is None:
        return []

    sampling_points: List[datetime] = []
    current = start_time - timedelta(hours=config.get("pre_event_hours", 6))
    end_time = start_time + timedelta(hours=config.get("post_event_hours", 3))
    cadence_minutes = config.get("cadence_minutes", 60)

    while current <= end_time:
        sampling_points.append(current)
        current += timedelta(minutes=cadence_minutes)
    return sampling_points


def parse_missing_entries(path: Path) -> Tuple[List[datetime], List[Tuple[datetime, datetime]]]:
    if not path.exists():
        return [], []

    points: List[datetime] = []
    intervals: List[Tuple[datetime, datetime]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) == 1:
                point = parse_iso_datetime(parts[0])
                if point is not None:
                    points.append(point)
                continue
            if len(parts) >= 2:
                begin = parse_iso_datetime(parts[0])
                end = parse_iso_datetime(parts[1])
                if begin is not None and end is not None:
                    intervals.append((begin, end))
    return sorted(set(points)), intervals


def _iter_dates(start_dt: datetime, end_dt: datetime) -> List[datetime]:
    dates: List[datetime] = []
    cur = datetime(start_dt.year, start_dt.month, start_dt.day)
    end_day = datetime(end_dt.year, end_dt.month, end_dt.day)
    while cur <= end_day:
        dates.append(cur)
        cur += timedelta(days=1)
    return dates


def _get_search_window(sample_time: datetime, config: Dict) -> Tuple[datetime, datetime]:
    half_window_min = max(10, int(config.get("cadence_minutes", 60)) // 2)
    return sample_time - timedelta(minutes=half_window_min), sample_time + timedelta(minutes=half_window_min)


def _build_missing_sampling_points(event_info: Dict[str, Optional[datetime]], config: Dict, missing_file: Path) -> List[datetime]:
    points, intervals = parse_missing_entries(missing_file)
    if intervals:
        for sample_time in calculate_sampling_points(event_info, config):
            for begin, end in intervals:
                if begin <= sample_time <= end:
                    points.append(sample_time)
                    break
    return sorted(set(points))


def find_gong_candidate_file(sample_time: datetime, config: Dict, ym_links: List[str]) -> Optional[Tuple[str, datetime, str]]:
    window_start, window_end = _get_search_window(sample_time, config)
    candidates: List[Tuple[str, datetime, str]] = []

    for day_dt in _iter_dates(window_start, window_end):
        ym = day_dt.strftime("%Y%m")
        day = day_dt.strftime("%Y%m%d")
        if ym not in ym_links:
            continue

        day_url = f"{BASE_URL}{ym}/{day}/"
        day_links = get_links(day_url)
        if not day_links:
            continue

        for fn in [l for l in day_links if l.endswith(".fits.fz") or l.endswith(".fits")]:
            m = re.match(r"^(\d{14})", fn)
            if not m:
                continue
            try:
                file_dt = datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
            except Exception:
                continue
            if window_start <= file_dt <= window_end:
                candidates.append((fn, file_dt, f"{day_url}{fn}"))

    if not candidates:
        return None
    return min(candidates, key=lambda x: abs((x[1] - sample_time).total_seconds()))


def download_file(url: str, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    dst = save_dir / os.path.basename(url)
    if dst.exists():
        print(f"[SKIP] 已存在: {dst}")
        return
    try:
        print(f"[DOWNLOAD] {url} -> {dst}")
        r = SESSION.get(url, stream=True, timeout=90)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[OK] {dst}")
    except Exception as e:
        print(f"[ERR] 下载失败: {url} : {e}")


def process_event(event_id: str, event_dir: Path, event_info: Dict[str, Optional[datetime]], config: Dict) -> None:
    halpha_dir = event_dir / "halpha"
    missing_file = halpha_dir / "missing_halpha.txt"
    if not missing_file.exists():
        print(f"[INFO] {event_id}: missing_halpha.txt 不存在，跳过")
        return

    missing_points = _build_missing_sampling_points(event_info, config, missing_file)
    if not missing_points:
        print(f"[INFO] {event_id}: 无缺失采样点，跳过")
        return

    ym_links = get_links(BASE_URL)
    if not ym_links:
        print(f"[WARN] {event_id}: 无法访问 GONG 根目录，跳过")
        return

    print(f"[INFO] {event_id}: 缺失采样点 {len(missing_points)} 个，开始按采样点搜索最近 GONG 文件")
    downloaded_count = 0
    for sample_time in missing_points:
        window_start, window_end = _get_search_window(sample_time, config)
        candidate = find_gong_candidate_file(sample_time, config, ym_links)
        if candidate is None:
            print(f"[WARN] {event_id}: 采样点 {sample_time} 在窗口 [{window_start} ~ {window_end}] 没有可用 GONG 文件")
            continue

        filename, file_time, url = candidate
        print(f"[INFO] {event_id}: 采样点={sample_time}, 搜索窗口=[{window_start} ~ {window_end}], 最近文件时间={file_time}, 文件={filename}")
        download_file(url, halpha_dir)
        downloaded_count += 1

    print(f"[INFO] {event_id}: 下载完成，共 {downloaded_count} 个文件")


def main() -> None:
    config = load_config()
    evt_table = parse_event_table(EVENTS_XLSX_PATH)
    raw_root = Path("data/raw/downloaded")
    if not raw_root.exists():
        print("[ERROR] data/raw/downloaded 不存在")
        return

    event_dirs = [p for p in raw_root.iterdir() if p.is_dir() and p.name.startswith("EVT_")]
    if not event_dirs:
        print("[WARN] 没有发现事件目录")
        return

    print(f"[INFO] 使用时间配置: pre_event_hours={config['pre_event_hours']}, post_event_hours={config['post_event_hours']}, cadence_minutes={config['cadence_minutes']}")
    print(f"[INFO] 事件表路径: {EVENTS_XLSX_PATH}")
    mode_text = "仅处理事件表中的事件组" if ONLY_PROCESS_EVENTS_IN_TABLE else "处理 downloaded 下全部事件组"
    print(f"[INFO] 处理模式: {mode_text}")

    if ONLY_PROCESS_EVENTS_IN_TABLE:
        selected_event_ids = set(evt_table.keys())
        event_dirs = [ev for ev in event_dirs if ev.name in selected_event_ids]
        print(f"[INFO] 事件表中共有 {len(selected_event_ids)} 个事件，downloaded 中匹配到 {len(event_dirs)} 个事件目录")
        if not event_dirs:
            print("[WARN] 没有发现与事件表匹配的事件目录")
            return

    for ev in sorted(event_dirs, key=lambda x: x.name):
        event_id = ev.name
        event_info = evt_table.get(event_id, {"start_time": None, "peak_time": None, "end_time": None})
        if event_id not in evt_table:
            print(f"[WARN] 事件 {event_id} 未在事件表中找到，尝试从事件ID解析start_time")
            parsed_start_time = parse_event_id_to_datetime(event_id)
            if parsed_start_time:
                event_info["start_time"] = parsed_start_time
                print(f"[INFO] 从事件ID解析得start_time: {parsed_start_time}")
            else:
                print("[WARN] 无法从事件ID解析start_time")

        process_event(event_id, ev, event_info, config)


if __name__ == "__main__":
    main()
