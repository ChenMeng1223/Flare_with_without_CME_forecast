import re
from datetime import date, datetime, timedelta
from typing import Iterable, Optional

import pandas as pd


ISO_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
_DAY_OFFSET_RE = re.compile(r"^\s*(.*?)\s*\(\+(\d+)day\)\s*$", re.IGNORECASE)


def is_missing_event_id(value) -> bool:
    if pd.isna(value):
        return True
    text = str(value).strip()
    return (not text) or text.lower() in {"nan", "none"}


def parse_event_date_value(value) -> Optional[date]:
    if pd.isna(value) or value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.to_pydatetime().date()

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    for fmt in ("%Y.%m.%d", "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().date()


def parse_event_time_value(value, base_date: Optional[date] = None) -> Optional[datetime]:
    if pd.isna(value) or value is None:
        return None

    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        dt = value.to_pydatetime()
        if dt.year == 1900 and base_date is not None:
            return datetime.combine(base_date, dt.time())
        return dt

    if isinstance(value, datetime):
        if value.year == 1900 and base_date is not None:
            return datetime.combine(base_date, value.time())
        return value

    if hasattr(value, "hour") and hasattr(value, "minute") and not isinstance(value, str):
        if base_date is None:
            return None
        seconds = getattr(value, "second", 0)
        microseconds = getattr(value, "microsecond", 0)
        return datetime.combine(
            base_date,
            value.replace(second=seconds, microsecond=microseconds),
        )

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None

    day_offset = 0
    day_match = _DAY_OFFSET_RE.match(text)
    if day_match:
        text = day_match.group(1).strip()
        day_offset = int(day_match.group(2))

    for fmt in ("%H:%M:%S", "%H:%M"):
        try:
            time_obj = datetime.strptime(text, fmt).time()
            if base_date is None:
                return None
            return datetime.combine(base_date + timedelta(days=day_offset), time_obj)
        except ValueError:
            continue

    normalized = text.replace("Z", "").replace("+00:00", "").strip()
    parsed = pd.to_datetime(normalized, errors="coerce")
    if pd.isna(parsed):
        return None

    dt = parsed.to_pydatetime()
    if dt.year == 1900 and base_date is not None:
        return datetime.combine(base_date + timedelta(days=day_offset), dt.time())
    if day_offset:
        dt = dt + timedelta(days=day_offset)
    return dt


def normalize_event_time_columns(
    df: pd.DataFrame,
    *,
    time_columns: Iterable[str] = ("start_time", "peak_time", "end_time"),
    date_column: str = "DATE",
) -> pd.DataFrame:
    result = df.copy()
    base_dates = (
        result[date_column].apply(parse_event_date_value)
        if date_column in result.columns
        else pd.Series([None] * len(result), index=result.index, dtype=object)
    )

    for col in time_columns:
        if col not in result.columns:
            continue

        parsed_values = [
            parse_event_time_value(result.at[idx, col], base_dates.at[idx])
            for idx in result.index
        ]
        dt_series = pd.to_datetime(pd.Series(parsed_values, index=result.index), errors="coerce")
        result[col] = dt_series.dt.strftime(ISO_TIME_FORMAT)
        result.loc[dt_series.isna(), col] = pd.NA

        if col == "start_time":
            result["start_dt"] = dt_series
        elif col == "end_time":
            result["end_dt"] = dt_series
        elif col == "peak_time":
            result["peak_dt"] = dt_series

    return result


def generate_event_ids_like_downloader(df: pd.DataFrame) -> pd.Series:
    event_ids = []
    time_count = {}

    for idx, row in df.iterrows():
        start_dt = row.get("start_dt")
        end_dt = row.get("end_dt")

        if pd.notna(start_dt) and pd.notna(end_dt):
            start_dt = pd.Timestamp(start_dt).to_pydatetime()
            end_dt = pd.Timestamp(end_dt).to_pydatetime()
            time_key = f"{start_dt.strftime(ISO_TIME_FORMAT)}_{end_dt.strftime(ISO_TIME_FORMAT)}"
            stem = start_dt.strftime("%Y%m%d_%H%M%S")
        else:
            time_key = f"row_{idx}"
            stem = f"{idx + 1:04d}"

        time_count[time_key] = time_count.get(time_key, 0) + 1
        event_ids.append(f"EVT_{stem}_{time_count[time_key]}")

    return pd.Series(event_ids, index=df.index)


def ensure_event_ids_like_downloader(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "start_dt" not in result.columns or "end_dt" not in result.columns:
        result = normalize_event_time_columns(result)

    generated_event_ids = generate_event_ids_like_downloader(result)

    if "event_id" not in result.columns:
        result["event_id"] = generated_event_ids
        return result

    missing_mask = result["event_id"].apply(is_missing_event_id)
    if missing_mask.any():
        result.loc[missing_mask, "event_id"] = generated_event_ids[missing_mask]

    result["event_id"] = result["event_id"].astype(str).str.strip()
    return result
