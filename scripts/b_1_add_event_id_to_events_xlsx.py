#!/usr/bin/env python3
"""
为 data/raw/events.xlsx 自动补充 event_id 列

- 使用与 scripts/a_download_sdo_data.py 中 _generate_event_ids 完全一致的规则
- event_id 形式: EVT_YYYYMMDD_HHMMSS_计数器
- 写回到原 xlsx 文件中，并放在最后一列
- 如果 event_id 列已存在，则只补缺失值，不覆盖原有 event_id

--注意创建时要关掉要写入的文件，否则会permission denied
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_XLSX = PROJECT_ROOT / "data" / "raw" / "events.xlsx"
# EVENTS_XLSX = PROJECT_ROOT / "data" / "raw" / "events_plot.xlsx"


class EventIdGenerator:
    """复刻 a_download_sdo_data.py 中的时间组合与 event_id 生成逻辑"""

    @staticmethod
    def _is_row_empty(row: pd.Series) -> bool:
        """检查一行是否完全为空（与原脚本一致逻辑）"""
        for value in row.values:
            if pd.notna(value) and str(value).strip():
                return False
        return True

    @staticmethod
    def _parse_time_column(time_value, base_date: datetime.date) -> datetime:
        """解析时间列，支持 datetime.time 和 'HH:MM', 'HH:MM(+1day)' 形式"""
        if pd.isna(time_value) or str(time_value).lower() == "nan" or time_value is None:
            raise ValueError(f"时间值无效: {time_value}")

        if isinstance(time_value, str):
            s = time_value.strip()

            # 先处理 (+1day) / (+2day) 这种标记（兼容 HH:MM 和 HH:MM:SS）
            day_offset = 0
            if "(+1day)" in s:
                day_offset = 1
                s = s.replace("(+1day)", "").strip()
            elif "(+2day)" in s:
                day_offset = 2
                s = s.replace("(+2day)", "").strip()

            # 优先按纯时间解析，并与 DATE 组合
            for fmt in ("%H:%M:%S", "%H:%M"):
                try:
                    time_obj = datetime.strptime(s, fmt).time()
                    actual_date = base_date + timedelta(days=day_offset)
                    return datetime.combine(actual_date, time_obj)
                except Exception:
                    continue

            # 若包含完整日期信息（例如已经是 ISO 字符串），直接让 pandas 解析
            try:
                dt = pd.to_datetime(s)
                # 若前面已经识别出 day_offset，则在解析出的日期上再加偏移
                if day_offset != 0:
                    dt = dt + pd.Timedelta(days=day_offset)
                return dt.to_pydatetime()
            except Exception:
                # 最后兜底：仍按 HH:MM 处理
                time_obj = datetime.strptime(s[:5], "%H:%M").time()
                actual_date = base_date + timedelta(days=day_offset)
                return datetime.combine(actual_date, time_obj)
        else:
            # datetime.time 或类似对象
            return datetime.combine(base_date, time_value)

    def combine_date_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """将 DATE + start_time / peak_time / end_time 组合成 ISO 字符串（与下载脚本一致）"""
        df = df.copy()

        if "DATE" not in df.columns:
            # 如果已经是 ISO 格式，这里直接返回
            return df

        for idx, row in df.iterrows():
            if self._is_row_empty(row):
                continue

            date_str = str(row["DATE"])
            if date_str.lower() == "nan" or not date_str or date_str == "None":
                continue

            # 解析 DATE，如 2024.12.23
            try:
                parts = date_str.split(".")
                if len(parts) != 3:
                    continue
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
                base_date = datetime(year, month, day).date()
            except Exception:
                continue

            # 依次处理 start_time / peak_time / end_time 三列，只要存在就加上日期
            for col in ("start_time", "peak_time", "end_time"):
                if col not in df.columns:
                    continue
                try:
                    dt = self._parse_time_column(row[col], base_date)
                except Exception:
                    # 对于 start_time / end_time，如果解析失败则整行跳过，以避免 event_id 时间不一致
                    if col in ("start_time", "end_time"):
                        dt = None
                    else:
                        continue
                if dt is None:
                    break
                df.at[idx, col] = dt.isoformat()

        return df

    def generate_event_ids(self, df: pd.DataFrame) -> pd.Series:
        """
        根据时间生成唯一的 event_id
        完全仿照 a_download_sdo_data.SDODataDownloader._generate_event_ids
        """
        event_ids = []
        time_count: Dict[str, int] = {}

        for idx, row in df.iterrows():
            start_time_str = str(row["start_time"])
            end_time_str = str(row["end_time"])

            time_key = f"{start_time_str}_{end_time_str}"

            if time_key in time_count:
                time_count[time_key] += 1
            else:
                time_count[time_key] = 1

            try:
                start_dt = pd.to_datetime(start_time_str)
                event_id = f"EVT_{start_dt.strftime('%Y%m%d_%H%M%S')}_{time_count[time_key]}"
            except Exception:
                event_id = f"EVT_{idx+1:04d}_{time_count[time_key]}"

            event_ids.append(event_id)

        return pd.Series(event_ids)


def main():
    if not EVENTS_XLSX.exists():
        raise FileNotFoundError(f"找不到事件文件: {EVENTS_XLSX}")

    print(f"读取事件文件: {EVENTS_XLSX}")
    df = pd.read_excel(EVENTS_XLSX)

    gen = EventIdGenerator()
    # 先将 DATE + start_time/peak_time/end_time 组合成 ISO 字符串
    df = gen.combine_date_time(df)

    # 检查必须的列
    for col in ("start_time", "end_time"):
        if col not in df.columns:
            raise ValueError(f"事件表缺少必要列: {col}")

    # 按完整规则为整张表生成 event_id，再仅填充缺失项，保留已有值
    generated_event_ids = gen.generate_event_ids(df)

    if "event_id" not in df.columns:
        df["event_id"] = generated_event_ids
        filled_count = int(generated_event_ids.notna().sum())
        print(f"事件表原本没有 event_id 列，已新增并写入 {filled_count} 个 event_id。")
    else:
        existing_event_ids = df["event_id"].copy()
        missing_mask = (
            existing_event_ids.isna()
            | (existing_event_ids.astype(str).str.strip() == "")
            | (existing_event_ids.astype(str).str.lower() == "nan")
        )
        filled_count = int(missing_mask.sum())
        if filled_count == 0:
            print("事件表中 event_id 列已存在，且所有行都有值，无需修改。")
            return
        df.loc[missing_mask, "event_id"] = generated_event_ids[missing_mask]
        print(f"event_id 列已存在，仅补充了 {filled_count} 个缺失 event_id，原有值保持不变。")

    # 确保 event_id 在最后一列
    cols = list(df.columns)
    cols.remove("event_id")
    cols.append("event_id")
    df = df[cols]

    # 写回原 xlsx
    backup_path = EVENTS_XLSX.with_suffix(".backup_before_add_event_id.xlsx")
    if not backup_path.exists():
        df_orig = pd.read_excel(EVENTS_XLSX)
        df_orig.to_excel(backup_path, index=False)
        print(f"已创建备份: {backup_path}")

    df.to_excel(EVENTS_XLSX, index=False)
    print(f"已写回 event_id 到: {EVENTS_XLSX}")
    print("完成。")


if __name__ == "__main__":
    main()
