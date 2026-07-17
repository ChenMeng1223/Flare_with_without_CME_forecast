import re
from typing import Any, Dict

import pandas as pd


ACTIVE_REGION_SOURCE_COLUMNS = ("position of active region", "active_region")


def parse_active_region_info(raw_value: Any) -> Dict[str, str]:
    """从 `position of active region` / `active_region` 字段中解析活动区号与括号内位置。

    Returns:
        {
            'raw': 原始字符串,
            'region_id': 活动区号或 UNKNOWN,
            'region_position': 括号内坐标或空串,
        }
    """
    raw = "" if raw_value is None else str(raw_value).strip()
    if not raw or raw.lower() == "nan":
        return {"raw": raw, "region_id": "UNKNOWN", "region_position": ""}

    pos_match = re.search(r"\(([^()]*)\)", raw)
    region_position = pos_match.group(1).strip() if pos_match else ""

    id_match = re.search(r"(\d{4,6})", raw)
    if id_match:
        region_id = id_match.group(1)
    else:
        prefix = raw.split("(")[0].strip()
        prefix = re.sub(r"\b(NOAA|AR)\b", "", prefix, flags=re.IGNORECASE).strip()
        region_id = prefix if prefix else "UNKNOWN"

    return {
        "raw": raw,
        "region_id": region_id or "UNKNOWN",
        "region_position": region_position,
    }


def is_unknown_active_region(region_id: Any) -> bool:
    region_id_str = "" if region_id is None else str(region_id).strip()
    return not region_id_str or region_id_str.lower() == "nan" or region_id_str.upper() == "UNKNOWN"


def attach_active_region_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为事件表补齐活动区解析列。"""
    df = df.copy()
    source_col = next((col for col in ACTIVE_REGION_SOURCE_COLUMNS if col in df.columns), None)

    if source_col is None:
        df["active_region_source"] = ""
        df["active_region_id"] = "UNKNOWN"
        df["active_region_position"] = ""
        return df

    parsed = df[source_col].apply(parse_active_region_info)
    df["active_region_source"] = parsed.apply(lambda x: x["raw"])
    df["active_region_id"] = parsed.apply(lambda x: x["region_id"])
    df["active_region_position"] = parsed.apply(lambda x: x["region_position"])
    return df


def find_unknown_active_region_events(df: pd.DataFrame) -> pd.DataFrame:
    """返回活动区编号未知的事件。"""
    if "active_region_id" not in df.columns:
        df = attach_active_region_columns(df)
    unknown_mask = df["active_region_id"].apply(is_unknown_active_region)
    return df.loc[unknown_mask].copy()


def format_unknown_active_region_summary(df: pd.DataFrame, max_rows: int = 10) -> str:
    """将未知活动区事件格式化为可读摘要。"""
    if df.empty:
        return ""

    lines = []
    for _, row in df.head(max_rows).iterrows():
        parts = []
        if "event_id" in row.index:
            parts.append(f"event_id={row['event_id']}")
        if "DATE" in row.index:
            parts.append(f"DATE={row['DATE']}")
        if "start_time" in row.index:
            parts.append(f"start_time={row['start_time']}")
        if "active_region_source" in row.index:
            raw_value = row["active_region_source"]
        elif "position of active region" in row.index:
            raw_value = row["position of active region"]
        elif "active_region" in row.index:
            raw_value = row["active_region"]
        else:
            raw_value = ""
        parts.append(f"active_region={raw_value!r}")
        lines.append("- " + ", ".join(str(part) for part in parts))

    remaining = len(df) - min(len(df), max_rows)
    if remaining > 0:
        lines.append(f"- ... 另外还有 {remaining} 个未知活动区事件")

    return "\n".join(lines)


def raise_if_unknown_active_regions(
    df: pd.DataFrame,
    *,
    context: str,
    max_rows: int = 10,
) -> None:
    """若存在未知活动区事件则抛出异常，阻止继续处理。"""
    unknown_df = find_unknown_active_region_events(df)
    if unknown_df.empty:
        return

    summary = format_unknown_active_region_summary(unknown_df, max_rows=max_rows)
    raise ValueError(
        f"{context}发现 {len(unknown_df)} 个活动区编号未知的事件，无法继续处理。\n"
        "请先在事件表中补全 `position of active region` 或 `active_region`，再重新运行。\n"
        f"{summary}"
    )


def build_region_key(region_id: Any, fallback_event_id: str = "") -> str:
    region_id_str = "" if region_id is None else str(region_id).strip()
    if not is_unknown_active_region(region_id_str):
        return region_id_str
    return f"UNKNOWN_{fallback_event_id}" if fallback_event_id else "UNKNOWN"
