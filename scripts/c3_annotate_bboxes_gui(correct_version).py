"""Region级BBox人工标注器（含主activity同步与历史推荐）。"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
PREFERRED_MODALITIES = ['magnetogram', 'euv_94', 'euv_171', 'euv_193', 'halpha']
ANNO_W = ANNO_H = 2048
LEGACY_BBOX_W = LEGACY_BBOX_H = 256
DRIFT_PX_PER_HOUR = 8.0
MAX_RECOMMEND_HOURS = 24 * 8

@dataclass
class RegionItem:
    region_id: str
    region_position: str
    is_primary_region: bool
    bbox: List[int]
    annotation_frame_timestamp: str = ''
    annotation_frame_name: str = ''
    history_copy_enabled: bool = True
    history_drift_enabled: bool = True

@dataclass
class ActivityItem:
    event_id: str
    is_primary_activity: bool
    label: int
    region_id: str
    active_region_source: str = ''
    active_region_position: str = ''


def _load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _list_event_dirs(processed_dir: Path) -> List[Path]:
    return sorted([p for p in processed_dir.iterdir() if p.is_dir() and p.name.startswith('EVT_')], key=lambda p: p.name) if processed_dir.exists() else []


def _event_id_to_datetime(event_id: str) -> Optional[datetime]:
    m = re.match(r'^EVT_(\d{8})_(\d{6})', str(event_id))
    if not m:
        return None
    try:
        return datetime.strptime(f'{m.group(1)}_{m.group(2)}', '%Y%m%d_%H%M%S')
    except Exception:
        return None


def _region_base_id(region_id: str) -> str:
    rid = '' if region_id is None else str(region_id).strip()
    if rid.startswith('UNKNOWN_'):
        return 'UNKNOWN'
    return rid


def _scale_bbox(bbox: List[int], sw: int, sh: int, dw: int, dh: int) -> List[int]:
    if min(sw, sh, dw, dh) <= 0:
        return list(bbox)
    x1, y1, x2, y2 = [int(x) for x in bbox]
    return [int(round(x1 * dw / sw)), int(round(y1 * dh / sh)), int(round(x2 * dw / sw)), int(round(y2 * dh / sh))]


def _clamp(b: List[int]) -> List[int]:
    x1, y1, x2, y2 = b
    x1, x2 = sorted([max(0, min(int(x1), ANNO_W - 1)), max(0, min(int(x2), ANNO_W - 1))])
    y1, y2 = sorted([max(0, min(int(y1), ANNO_H - 1)), max(0, min(int(y2), ANNO_H - 1))])
    return [x1, y1, x2, y2]


def _shift_bbox(bbox: List[int], dx: float, dy: float = 0.0) -> List[int]:
    x1, y1, x2, y2 = bbox
    return _clamp([round(x1 + dx), round(y1 + dy), round(x2 + dx), round(y2 + dy)])


def _fmt_region(r: RegionItem) -> str:
    pos = f' ({r.region_position})' if r.region_position else ''
    tag = 'PRIMARY_REGION' if r.is_primary_region else 'REGION'
    ts = f' | ts={r.annotation_frame_timestamp}' if r.annotation_frame_timestamp else ''
    hist = f" | copy={'Y' if r.history_copy_enabled else 'N'} drift={'Y' if r.history_drift_enabled else 'N'}"
    return f'{r.region_id}{pos} | {tag} | bbox={r.bbox}{hist}{ts}'


def _fmt_activity(a: ActivityItem) -> str:
    pos = f' ({a.active_region_position})' if a.active_region_position else ''
    tag = 'PRIMARY_ACTIVITY' if a.is_primary_activity else 'ACTIVITY'
    return f'{a.event_id} | {tag} | label={a.label} | region={a.region_id}{pos}'


def _parse_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    try:
        return datetime.fromisoformat(s.replace('Z', ''))
    except Exception:
        pass
    for fmt in ('%Y%m%d_%H%M%S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _frame_timestamp_from_name(name: str) -> Optional[datetime]:
    if not name:
        return None
    name = str(name)
    patterns = [
        r'(\d{14})',
        r'(\d{8}_\d{6})',
        r'(\d{4}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})',
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
    ]
    for pattern in patterns:
        m = re.search(pattern, name)
        if not m:
            continue
        token = m.group(1)
        for fmt in ('%Y%m%d%H%M%S', '%Y%m%d_%H%M%S', '%Y_%m_%dT%H_%M_%S', '%Y-%m-%dT%H:%M:%S'):
            try:
                return datetime.strptime(token, fmt)
            except Exception:
                continue
    return None


def _frame_timestamp_from_path(path: Path) -> Optional[datetime]:
    return _frame_timestamp_from_name(path.name)


def _parse_bboxes_json(path: Path):
    raw = _load_json(path)
    res = raw.get('bbox_resolution') or {}
    sw = int(res.get('width', raw.get('anno_width', LEGACY_BBOX_W)))
    sh = int(res.get('height', raw.get('anno_height', LEGACY_BBOX_H)))
    peid = str(raw.get('primary_event_id', path.parent.name))
    prid = str(raw.get('primary_region_id', ''))
    regions, activities = [], []
    if raw.get('regions') is not None or raw.get('activities') is not None:
        for r in raw.get('regions', []):
            regions.append(RegionItem(
                str(r.get('region_id', 'UNKNOWN')),
                str(r.get('region_position', '') or ''),
                bool(r.get('is_primary_region', False)),
                _clamp(_scale_bbox([int(x) for x in r.get('bbox', [0, 0, 0, 0])], sw, sh, ANNO_W, ANNO_H)),
                str(r.get('annotation_frame_timestamp', '') or ''),
                str(r.get('annotation_frame_name', '') or ''),
                bool(r.get('history_copy_enabled', True)),
                bool(r.get('history_drift_enabled', True)),
            ))
        for a in raw.get('activities', []):
            activities.append(ActivityItem(
                str(a.get('event_id', '')),
                bool(a.get('is_primary_activity', False)),
                int(a.get('label', 2)),
                str(a.get('region_id', 'UNKNOWN')),
                str(a.get('active_region_source', '') or ''),
                str(a.get('active_region_position', '') or ''),
            ))
    else:
        for i, b in enumerate(raw.get('bboxes', []), start=1):
            rid = str(b.get('event_id', f'R{i}'))
            is_p = bool(b.get('is_primary', False))
            regions.append(RegionItem(rid, '', is_p, _clamp(_scale_bbox([int(x) for x in b.get('bbox', [0, 0, 0, 0])], sw, sh, ANNO_W, ANNO_H))))
            activities.append(ActivityItem(str(b.get('event_id', '')), is_p, int(b.get('label', 2)), rid))
            if is_p:
                prid = rid
    if not regions:
        regions.append(RegionItem('R1', '', True, [0, 0, 0, 0]))
    if not any(r.is_primary_region for r in regions):
        regions[0].is_primary_region = True
    if not any(a.is_primary_activity for a in activities) and activities:
        for a in activities:
            if a.event_id == peid:
                a.is_primary_activity = True
                break
    return peid, prid, regions, activities, raw


def _raw_region_bbox_to_anno_space(raw_region: Dict, raw_json: Dict) -> List[int]:
    """Convert a bbox stored in its source resolution into the 2048x2048 GUI canvas space."""
    res = raw_json.get('bbox_resolution') or {}
    sw = int(res.get('width', raw_json.get('anno_width', LEGACY_BBOX_W)))
    sh = int(res.get('height', raw_json.get('anno_height', LEGACY_BBOX_H)))
    bbox = [int(x) for x in raw_region.get('bbox', [0, 0, 0, 0])]
    return _clamp(_scale_bbox(bbox, sw, sh, ANNO_W, ANNO_H))


def _region_id_matches(a: str, b: str) -> bool:
    return a == b or _region_base_id(a) == _region_base_id(b)


class AnnotatorApp:
    def __init__(self, processed_dir: Path):
        self.processed_dir = processed_dir
        self.event_dirs = _list_event_dirs(processed_dir)
        self.idx = 0
        self.current_event_dir: Optional[Path] = None
        self.current_bbox_path: Optional[Path] = None
        self.primary_event_id = ''
        self.primary_region_id = ''
        self.regions: List[RegionItem] = []
        self.activities: List[ActivityItem] = []
        self.raw_json: Dict = {}
        self.modality = ''
        self.frame_paths: List[Path] = []
        self.frame_idx = 0
        self.selected_region_index: Optional[int] = None
        self.selected_activity_index: Optional[int] = None
        self._pending_bbox: Optional[List[int]] = None
        self.user_zoom = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.drawing = False
        self._panning = False
        self._drag_start = None
        self._pan_start_xy = None
        self._pan_start_offset = None
        self._tk_img = None
        self._pil_cache = {}
        self._render_after_id = None
        self._is_refreshing_ui = False
        self._suspend_selection_callbacks = False

        self.root = tk.Tk()
        self.root.title('Region级BBox标注器')
        self.root.geometry('1280x840')
        self.root.bind('<Configure>', self._on_configure)

        self.var_event = tk.StringVar()
        self.var_modality = tk.StringVar()
        self.var_skip_done = tk.BooleanVar(value=False)
        self.var_label = tk.IntVar(value=2)
        self.var_primary_activity = tk.BooleanVar(value=False)
        self.var_region_primary = tk.BooleanVar(value=False)
        self.var_region_id = tk.StringVar()
        self.var_region_pos = tk.StringVar()
        self.var_activity_region = tk.StringVar()
        self.var_auto_sync_primary = tk.BooleanVar(value=True)
        self.var_auto_apply_history_copy = tk.BooleanVar(value=True)
        self.var_auto_apply_history_drift = tk.BooleanVar(value=True)
        self.var_region_history_copy = tk.BooleanVar(value=True)
        self.var_region_history_drift = tk.BooleanVar(value=True)
        self.var_status = tk.StringVar(value='就绪')
        self.history_copy_enabled = True
        self.history_drift_enabled = True
        self._build_ui()
        self._load_current_event()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)
        paned = ttk.Panedwindow(main, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        ttk.Label(left, text='事件组').pack(anchor='w')
        self.cbo_event = ttk.Combobox(left, textvariable=self.var_event, state='readonly', values=[d.name for d in self.event_dirs])
        self.cbo_event.pack(fill=tk.X)
        self.cbo_event.bind('<<ComboboxSelected>>', lambda _e: self._on_event_selected())

        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text='模态').pack(side=tk.LEFT)
        self.cbo_mod = ttk.Combobox(row, textvariable=self.var_modality, state='readonly')
        self.cbo_mod.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cbo_mod.bind('<<ComboboxSelected>>', lambda _e: self._reload_frames(True))
        ttk.Button(row, text='上一张', command=self._prev_frame).pack(side=tk.LEFT)
        ttk.Button(row, text='下一张', command=self._next_frame).pack(side=tk.LEFT)

        ttk.Label(left, textvariable=self.var_status, wraplength=360).pack(anchor='w', pady=4)
        ttk.Checkbutton(left, text='主activity自动同步主region', variable=self.var_auto_sync_primary).pack(anchor='w')
        ttk.Checkbutton(left, text='本事件组一键勾选历史复制', variable=self.var_auto_apply_history_copy, command=self._on_toggle_history_mode).pack(anchor='w')
        ttk.Checkbutton(left, text='本事件组一键勾选历史偏移', variable=self.var_auto_apply_history_drift, command=self._on_toggle_history_mode).pack(anchor='w')

        ttk.Label(left, text='Regions').pack(anchor='w', pady=(8, 0))
        self.region_list = tk.Listbox(left, height=8)
        self.region_list.pack(fill=tk.X)
        self.region_list.bind('<<ListboxSelect>>', self._on_select_region)

        rf = ttk.Frame(left)
        rf.pack(fill=tk.X, pady=4)
        ttk.Entry(rf, textvariable=self.var_region_id, width=12).pack(side=tk.LEFT)
        ttk.Entry(rf, textvariable=self.var_region_pos, width=12).pack(side=tk.LEFT, padx=4)
        ttk.Button(rf, text='新增', command=self._add_region).pack(side=tk.LEFT)
        ttk.Button(rf, text='删除', command=self._delete_region).pack(side=tk.LEFT, padx=4)

        ttk.Button(left, text='应用画框到选中region', command=self._apply_bbox_to_region).pack(fill=tk.X)
        ttk.Checkbutton(left, text='当前region启用历史复制', variable=self.var_region_history_copy, command=self._toggle_region_history_flags).pack(anchor='w')
        ttk.Checkbutton(left, text='当前region启用历史偏移', variable=self.var_region_history_drift, command=self._toggle_region_history_flags).pack(anchor='w')
        ttk.Checkbutton(left, text='设为primary_region', variable=self.var_region_primary, command=self._toggle_primary_region).pack(anchor='w')

        ttk.Label(left, text='Activities').pack(anchor='w', pady=(8, 0))
        self.activity_list = tk.Listbox(left, height=10)
        self.activity_list.pack(fill=tk.X)
        self.activity_list.bind('<<ListboxSelect>>', self._on_select_activity)
        af = ttk.Frame(left)
        af.pack(fill=tk.X, pady=4)
        ttk.Radiobutton(af, text='1爆发', variable=self.var_label, value=1, command=self._apply_activity).pack(side=tk.LEFT)
        ttk.Radiobutton(af, text='2束缚', variable=self.var_label, value=2, command=self._apply_activity).pack(side=tk.LEFT)
        self.cbo_activity_region = ttk.Combobox(left, textvariable=self.var_activity_region, state='readonly')
        self.cbo_activity_region.pack(fill=tk.X)
        self.cbo_activity_region.bind('<<ComboboxSelected>>', lambda _e: self._apply_activity())
        ttk.Checkbutton(left, text='设为primary_activity', variable=self.var_primary_activity, command=self._apply_activity).pack(anchor='w')

        br = ttk.Frame(left)
        br.pack(fill=tk.X, pady=8)
        ttk.Button(br, text='保存', command=self._save).pack(side=tk.LEFT)
        ttk.Button(br, text='保存并向后批量应用', command=self._save_and_batch_apply_forward).pack(side=tk.LEFT, padx=4)
        ttk.Button(br, text='上一组', command=self._prev_event).pack(side=tk.LEFT, padx=4)
        ttk.Button(br, text='下一组', command=self._next_event).pack(side=tk.LEFT)
        ttk.Checkbutton(left, text='跳过已标注', variable=self.var_skip_done).pack(anchor='w')

        self.canvas = tk.Canvas(right, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind('<ButtonPress-1>', self._on_mouse_down)
        self.canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.canvas.bind('<Control-MouseWheel>', self._on_zoom)

    def _set_status(self, text: str):
        self.var_status.set(text)

    def _safe_selection_set(self, listbox: tk.Listbox, index: int):
        self._suspend_selection_callbacks = True
        try:
            listbox.selection_clear(0, tk.END)
            listbox.selection_set(index)
            listbox.activate(index)
        finally:
            self._suspend_selection_callbacks = False

    def _on_configure(self, _evt=None):
        if self._is_refreshing_ui:
            return
        if self._render_after_id is not None:
            try:
                self.root.after_cancel(self._render_after_id)
            except Exception:
                pass
        self._render_after_id = self.root.after(30, self._render)

    def _on_toggle_history_mode(self):
        self.history_copy_enabled = bool(self.var_auto_apply_history_copy.get())
        self.history_drift_enabled = bool(self.var_auto_apply_history_drift.get())
        for region in self.regions:
            region.history_copy_enabled = self.history_copy_enabled
            region.history_drift_enabled = self.history_drift_enabled
        mode = []
        if self.history_copy_enabled:
            mode.append('历史复制')
        if self.history_drift_enabled:
            mode.append('历史偏移')
        self._refresh_lists()
        self._set_status(f"当前事件组一键设置: {', '.join(mode) if mode else '全部关闭'}")

    def _toggle_region_history_flags(self):
        region = self._get_selected_region()
        if not region:
            return
        region.history_copy_enabled = bool(self.var_region_history_copy.get())
        region.history_drift_enabled = bool(self.var_region_history_drift.get())
        self._refresh_lists()
        self._set_status(
            f"region={region.region_id} 历史设置: copy={'开' if region.history_copy_enabled else '关'} | drift={'开' if region.history_drift_enabled else '关'}"
        )

    def _current_frame_path(self) -> Optional[Path]:
        return self.frame_paths[self.frame_idx] if self.frame_paths else None

    def _current_frame_timestamp(self) -> Optional[datetime]:
        path = self._current_frame_path()
        return _frame_timestamp_from_path(path) if path else None

    def _load_current_event(self):
        if not self.event_dirs:
            return
        self._is_refreshing_ui = True
        self._suspend_selection_callbacks = True
        self._set_status('正在切换事件组，请稍候...')
        self.root.config(cursor='watch')
        start = self.idx
        scheduled_auto_apply = False
        try:
            while True:
                self.current_event_dir = self.event_dirs[self.idx]
                self.current_bbox_path = self.current_event_dir / 'bboxes.json'
                if self.current_bbox_path.exists():
                    self.primary_event_id, self.primary_region_id, self.regions, self.activities, self.raw_json = _parse_bboxes_json(self.current_bbox_path)
                    history_settings = self.raw_json.get('history_auto_apply', {}) or {}
                    self.history_copy_enabled = bool(history_settings.get('copy', True))
                    self.history_drift_enabled = bool(history_settings.get('drift', True))
                    self.var_auto_apply_history_copy.set(self.history_copy_enabled)
                    self.var_auto_apply_history_drift.set(self.history_drift_enabled)
                    if not (self.var_skip_done.get() and bool(self.raw_json.get('annotated', False))):
                        break
                self.idx = (self.idx + 1) % len(self.event_dirs)
                if self.idx == start:
                    break
            self._reload_frames(True)
            self._refresh_lists()
            self._render()
            scheduled_auto_apply = self.history_copy_enabled or self.history_drift_enabled
        finally:
            self.root.config(cursor='')
            self._suspend_selection_callbacks = False
            self._is_refreshing_ui = False
        if scheduled_auto_apply:
            self.root.after(10, self._auto_apply_history_for_event)

    def _reload_frames(self, reset_idx=False):
        if not self.current_event_dir:
            return
        mods = [m for m in PREFERRED_MODALITIES if (self.current_event_dir / m).is_dir()] or [p.name for p in self.current_event_dir.iterdir() if p.is_dir()]
        self.modality = self.var_modality.get() if self.var_modality.get() in mods else (mods[0] if mods else '')
        self.var_modality.set(self.modality)
        self.cbo_mod['values'] = mods
        self.frame_paths = sorted((self.current_event_dir / self.modality).glob('*.png')) if self.modality else []
        self.frame_idx = len(self.frame_paths) // 2 if reset_idx and self.frame_paths else min(self.frame_idx, max(0, len(self.frame_paths) - 1))
        ts = self._current_frame_timestamp()
        self._set_status(f"{self.current_event_dir.name if self.current_event_dir else '-'} | frame {self.frame_idx + 1}/{max(1, len(self.frame_paths))} | ts={ts.isoformat(sep=' ') if ts else '未知'}")
        self._render()

    def _refresh_lists(self):
        self._suspend_selection_callbacks = True
        try:
            self.var_event.set(self.current_event_dir.name if self.current_event_dir else '')
            self.region_list.delete(0, tk.END)
            for r in self.regions:
                self.region_list.insert(tk.END, _fmt_region(r))
            self.activity_list.delete(0, tk.END)
            for a in self.activities:
                self.activity_list.insert(tk.END, _fmt_activity(a))
            vals = [r.region_id for r in self.regions]
            self.cbo_activity_region['values'] = vals
            if self.selected_region_index is None and self.regions:
                self.selected_region_index = next((i for i, r in enumerate(self.regions) if r.is_primary_region), 0)
            if self.selected_region_index is not None and self.regions:
                self.selected_region_index = min(self.selected_region_index, len(self.regions) - 1)
                self._safe_selection_set(self.region_list, self.selected_region_index)
                selected_region = self.regions[self.selected_region_index]
                self.var_region_primary.set(selected_region.is_primary_region)
                self.var_region_history_copy.set(selected_region.history_copy_enabled)
                self.var_region_history_drift.set(selected_region.history_drift_enabled)
            if self.selected_activity_index is None and self.activities:
                self.selected_activity_index = next((i for i, a in enumerate(self.activities) if a.is_primary_activity), 0)
            if self.selected_activity_index is not None and self.activities:
                self.selected_activity_index = min(self.selected_activity_index, len(self.activities) - 1)
                self._safe_selection_set(self.activity_list, self.selected_activity_index)
                self._sync_activity_vars(self.activities[self.selected_activity_index])
        finally:
            self._suspend_selection_callbacks = False

    def _sync_activity_vars(self, a: ActivityItem):
        self.var_label.set(a.label)
        self.var_primary_activity.set(a.is_primary_activity)
        self.var_activity_region.set(a.region_id)

    def _get_selected_region(self) -> Optional[RegionItem]:
        if self.selected_region_index is None or not self.regions:
            return None
        return self.regions[self.selected_region_index]

    def _sync_primary_region_to_activity(self, region_id: str):
        for r in self.regions:
            r.is_primary_region = (r.region_id == region_id)
        self.primary_region_id = region_id
        self.selected_region_index = next((i for i, r in enumerate(self.regions) if r.region_id == region_id), self.selected_region_index)

    def _on_event_selected(self):
        if self._is_refreshing_ui:
            return
        for i, d in enumerate(self.event_dirs):
            if d.name == self.var_event.get():
                if i == self.idx:
                    return
                if not self._autosave_before_event_change():
                    if self.event_dirs and 0 <= self.idx < len(self.event_dirs):
                        self.var_event.set(self.event_dirs[self.idx].name)
                    return
                self.idx = i
                self.selected_region_index = None
                self.selected_activity_index = None
                self.root.after_idle(self._load_current_event)
                return

    def _on_select_region(self, _e=None):
        if self._suspend_selection_callbacks or self._is_refreshing_ui:
            return
        sel = self.region_list.curselection()
        if sel:
            self.selected_region_index = int(sel[0])
            region = self.regions[self.selected_region_index]
            self.var_region_primary.set(region.is_primary_region)
            self.var_region_history_copy.set(region.history_copy_enabled)
            self.var_region_history_drift.set(region.history_drift_enabled)
            self._render()
            self._maybe_auto_apply_history()

    def _on_select_activity(self, _e=None):
        if self._suspend_selection_callbacks or self._is_refreshing_ui:
            return
        sel = self.activity_list.curselection()
        if sel:
            self.selected_activity_index = int(sel[0])
            activity = self.activities[self.selected_activity_index]
            self._sync_activity_vars(activity)
            self.selected_region_index = next((i for i, r in enumerate(self.regions) if r.region_id == activity.region_id), self.selected_region_index)
            self._render()

    def _add_region(self):
        rid = self.var_region_id.get().strip() or f'R{len(self.regions) + 1}'
        if any(r.region_id == rid for r in self.regions):
            return
        self.regions.append(RegionItem(rid, self.var_region_pos.get().strip(), len(self.regions) == 0, [0, 0, 0, 0]))
        self.selected_region_index = len(self.regions) - 1
        self._refresh_lists()
        self._render()

    def _delete_region(self):
        if self.selected_region_index is None or len(self.regions) <= 1:
            return
        removed = self.regions.pop(self.selected_region_index)
        fallback = self.regions[0].region_id
        for a in self.activities:
            if a.region_id == removed.region_id:
                a.region_id = fallback
        if removed.is_primary_region:
            self._sync_primary_region_to_activity(fallback)
        self.selected_region_index = 0
        self._refresh_lists()
        self._render()

    def _toggle_primary_region(self):
        region = self._get_selected_region()
        if not region:
            return
        self._sync_primary_region_to_activity(region.region_id)
        self._refresh_lists()
        self._render()

    def _apply_activity(self):
        if self.selected_activity_index is None:
            return
        a = self.activities[self.selected_activity_index]
        a.label = int(self.var_label.get())
        a.region_id = self.var_activity_region.get() or a.region_id
        a.is_primary_activity = bool(self.var_primary_activity.get())
        if a.is_primary_activity:
            for i, other in enumerate(self.activities):
                if i != self.selected_activity_index:
                    other.is_primary_activity = False
            self.primary_event_id = a.event_id
            if self.var_auto_sync_primary.get():
                self._sync_primary_region_to_activity(a.region_id)
        self._refresh_lists()
        self._render()

    def _load_img(self):
        path = self._current_frame_path()
        if not path:
            return None
        cache_key = str(path)
        if cache_key in self._pil_cache:
            return self._pil_cache[cache_key]
        img = Image.open(path).convert('RGB')
        ow, oh = img.size
        s = min(ANNO_W / ow, ANNO_H / oh)
        rw, rh = max(1, int(ow * s)), max(1, int(oh * s))
        bg = Image.new('RGB', (ANNO_W, ANNO_H), 'white')
        bg.paste(img.resize((rw, rh)), ((ANNO_W - rw) // 2, (ANNO_H - rh) // 2))
        self._pil_cache[cache_key] = bg
        if len(self._pil_cache) > 12:
            oldest_key = next(iter(self._pil_cache))
            if oldest_key != cache_key:
                self._pil_cache.pop(oldest_key, None)
        return bg

    def _view(self):
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(cw / ANNO_W, ch / ANNO_H) * self.user_zoom
        dw, dh = int(ANNO_W * scale), int(ANNO_H * scale)
        ox, oy = (cw - dw) / 2 + self.pan_offset_x, (ch - dh) / 2 + self.pan_offset_y
        return scale, ox, oy, dw, dh

    def _render(self, *_):
        self._render_after_id = None
        img = self._load_img()
        if img is None:
            return
        scale, ox, oy, dw, dh = self._view()
        self.canvas.delete('all')
        self._tk_img = ImageTk.PhotoImage(img.resize((max(1, dw), max(1, dh))))
        self.canvas.create_image(ox, oy, image=self._tk_img, anchor='nw')
        for i, r in enumerate(self.regions):
            x1, y1, x2, y2 = r.bbox
            color = 'yellow' if i == self.selected_region_index else ('lime' if r.is_primary_region else 'cyan')
            self.canvas.create_rectangle(ox + x1 * scale, oy + y1 * scale, ox + x2 * scale, oy + y2 * scale, outline=color, width=2)
            self.canvas.create_text(ox + x1 * scale + 4, oy + y1 * scale + 10, text=r.region_id, fill=color, anchor='w')

    def _on_mouse_down(self, evt):
        if evt.state & 0x4:
            if self.selected_region_index is None:
                return
            self.drawing = True
            self._drag_start = (evt.x, evt.y)
        else:
            self._panning = True
            self._pan_start_xy = (evt.x, evt.y)
            self._pan_start_offset = (self.pan_offset_x, self.pan_offset_y)

    def _on_mouse_drag(self, evt):
        if self._panning and self._pan_start_xy:
            sx, sy = self._pan_start_xy
            ox, oy = self._pan_start_offset
            self.pan_offset_x, self.pan_offset_y = ox + evt.x - sx, oy + evt.y - sy
            self._render()
            return
        if self.drawing and self._drag_start:
            self._render()
            x0, y0 = self._drag_start
            self.canvas.create_rectangle(x0, y0, evt.x, evt.y, outline='red', dash=(4, 2))

    def _on_mouse_up(self, evt):
        if self._panning:
            self._panning = False
            return
        if not (self.drawing and self._drag_start):
            return
        self.drawing = False
        x0, y0 = self._drag_start
        scale, ox, oy, _, _ = self._view()
        self._pending_bbox = _clamp([round((x0 - ox) / scale), round((y0 - oy) / scale), round((evt.x - ox) / scale), round((evt.y - oy) / scale)])
        self._set_status(f'已画框，待应用: {self._pending_bbox}')

    def _apply_bbox_to_region(self):
        region = self._get_selected_region()
        if not region or not self._pending_bbox:
            return
        region.bbox = list(self._pending_bbox)
        ts = self._current_frame_timestamp()
        region.annotation_frame_timestamp = ts.isoformat(timespec='seconds') if ts else ''
        region.annotation_frame_name = self._current_frame_path().name if self._current_frame_path() else ''
        self._refresh_lists()
        self._render()
        saved_path = self._save_current_event(show_message=False)
        self._set_status(
            f'已应用并自动保存 region={region.region_id} | 标注时间戳={region.annotation_frame_timestamp or "未知"}'
            f' | 文件={saved_path.name if saved_path else "未保存"}'
        )

    def _maybe_auto_apply_history(self):
        region = self._get_selected_region()
        if not region:
            return
        if region.history_drift_enabled:
            self._apply_history_recommendation(use_drift=True, silent=True)
        elif region.history_copy_enabled:
            self._apply_history_recommendation(use_drift=False, silent=True)

    def _auto_apply_history_for_event(self):
        if self._is_refreshing_ui:
            return
        applied_copy = 0
        applied_drift = 0
        original_index = self.selected_region_index
        for idx, region in enumerate(self.regions):
            self.selected_region_index = idx
            if region.history_drift_enabled:
                ok = self._apply_history_recommendation(use_drift=True, silent=True)
                if ok:
                    applied_drift += 1
            elif region.history_copy_enabled:
                ok = self._apply_history_recommendation(use_drift=False, silent=True)
                if ok:
                    applied_copy += 1
            else:
                ok = False
            if ok and not region.annotation_frame_name:
                region.annotation_frame_name = self._current_frame_path().name if self._current_frame_path() else ''
        if self.regions:
            self.selected_region_index = original_index if original_index is not None else min(len(self.regions) - 1, 0)
        self._refresh_lists()
        self._render()
        self._set_status(f'自动套用完成：copy={applied_copy} | drift={applied_drift} | total={len(self.regions)}')

    def _find_history_region(self, region_id: str):
        target_ts = self._current_frame_timestamp()
        current_event_ts = _event_id_to_datetime(self.current_event_dir.name) if self.current_event_dir else None
        region_candidates = [_region_base_id(region_id)]
        if region_id not in region_candidates:
            region_candidates.append(region_id)

        grouped_with_time = defaultdict(list)
        grouped_without_time = defaultdict(list)

        for event_dir in self.event_dirs:
            if self.current_event_dir and event_dir == self.current_event_dir:
                continue
            path = event_dir / 'bboxes.json'
            if not path.exists():
                continue
            try:
                raw = _load_json(path)
            except Exception:
                continue
            if not bool(raw.get('annotated', False)):
                continue

            source_event_ts = _event_id_to_datetime(event_dir.name)
            for region in raw.get('regions', []):
                source_region_id = str(region.get('region_id', '') or '')
                if _region_base_id(source_region_id) not in region_candidates and source_region_id not in region_candidates:
                    continue

                src_bbox = _raw_region_bbox_to_anno_space(region, raw)
                src_ts = _parse_dt(str(region.get('annotation_frame_timestamp', '') or ''))
                if not src_ts:
                    src_ts = _frame_timestamp_from_name(str(region.get('annotation_frame_name', '') or ''))

                entry = {
                    'event_id': event_dir.name,
                    'bbox': src_bbox,
                    'source_ts': src_ts,
                    'source_event_ts': source_event_ts,
                }
                key = _region_base_id(source_region_id)
                if src_ts is not None and target_ts is not None:
                    grouped_with_time[key].append(entry)
                else:
                    grouped_without_time[key].append(entry)

        def _pick_best(candidates: List[Dict]):
            if not candidates:
                return None
            if target_ts is not None:
                earlier = []
                later = []
                for item in candidates:
                    src_ts = item['source_ts']
                    if src_ts is None:
                        continue
                    delta_h = abs((target_ts - src_ts).total_seconds()) / 3600.0
                    if delta_h > MAX_RECOMMEND_HOURS:
                        continue
                    payload = (item['event_id'], item['bbox'], src_ts, target_ts, delta_h)
                    if src_ts <= target_ts:
                        earlier.append(payload)
                    else:
                        later.append(payload)
                if earlier:
                    return min(earlier, key=lambda x: (x[4], -x[2].timestamp()))
                if later:
                    return min(later, key=lambda x: (x[4], x[2].timestamp()))

            no_time_candidates = [item for item in candidates if item.get('source_ts') is None]
            if current_event_ts is not None:
                ordered = sorted(
                    no_time_candidates or candidates,
                    key=lambda item: (
                        abs((item['source_event_ts'] - current_event_ts).total_seconds()) if item['source_event_ts'] else float('inf'),
                        -(item['source_event_ts'].timestamp()) if item['source_event_ts'] else float('-inf'),
                    )
                )
                first = ordered[0]
            else:
                first = (no_time_candidates or candidates)[0]
            return (first['event_id'], first['bbox'], first['source_ts'], target_ts, None)

        for candidate_key in region_candidates:
            best = _pick_best(grouped_with_time.get(candidate_key, []))
            if best is not None:
                return best

        for candidate_key in region_candidates:
            fallback = grouped_without_time.get(candidate_key, [])
            if fallback:
                chosen = _pick_best(fallback)
                if chosen is not None:
                    return chosen
        return None

    def _apply_history_recommendation(self, use_drift: bool, silent: bool = False):
        region = self._get_selected_region()
        if not region:
            return False
        hist = self._find_history_region(region.region_id)
        if not hist:
            if not silent:
                messagebox.showinfo('提示', f'未找到 region {region.region_id} 的历史已标注记录。')
            return False
        source_event_id, source_bbox, source_ts, target_ts, delta_h = hist
        if use_drift and source_ts is not None and target_ts is not None:
            signed_delta_h = (target_ts - source_ts).total_seconds() / 3600.0
            bbox = _shift_bbox(source_bbox, signed_delta_h * DRIFT_PX_PER_HOUR)
            mode = '历史偏移自动套用'
        else:
            bbox = list(source_bbox)
            mode = '历史复制自动套用' if not use_drift else '历史偏移回退为复制'
        region.bbox = bbox
        if target_ts is not None:
            region.annotation_frame_timestamp = target_ts.isoformat(timespec='seconds')
        region.annotation_frame_name = self._current_frame_path().name if self._current_frame_path() else ''
        self._refresh_lists()
        self._render()
        src_ts_text = source_ts.isoformat(sep=' ') if source_ts is not None else '未知'
        delta_text = f'{delta_h:.2f}h' if delta_h is not None else '未知'
        direction = ''
        if source_ts is not None and target_ts is not None:
            direction = ' | source<=target' if source_ts <= target_ts else ' | source>target(反向参考)'
        self._set_status(f'{mode}: region={region.region_id} | src={source_event_id} | dt={delta_text} | src_ts={src_ts_text}{direction}')
        return True

    def _on_zoom(self, evt):
        self.user_zoom = max(0.5, min(5.0, self.user_zoom * (1.1 if evt.delta > 0 else 1 / 1.1)))
        self._render()

    def _prev_frame(self):
        if self.frame_paths:
            self.frame_idx = (self.frame_idx - 1) % len(self.frame_paths)
            self._reload_frames(False)

    def _next_frame(self):
        if self.frame_paths:
            self.frame_idx = (self.frame_idx + 1) % len(self.frame_paths)
            self._reload_frames(False)

    def _prev_event(self):
        if not self.event_dirs or self._is_refreshing_ui:
            return
        if not self._autosave_before_event_change():
            return
        self.idx = (self.idx - 1) % len(self.event_dirs)
        self.selected_region_index = None
        self.selected_activity_index = None
        self.root.after_idle(self._load_current_event)

    def _next_event(self):
        if not self.event_dirs or self._is_refreshing_ui:
            return
        if not self._autosave_before_event_change():
            return
        self.idx = (self.idx + 1) % len(self.event_dirs)
        self.selected_region_index = None
        self.selected_activity_index = None
        self.root.after_idle(self._load_current_event)

    def _build_output_payload(self) -> Dict:
        if not any(r.is_primary_region for r in self.regions):
            self.regions[0].is_primary_region = True
        if self.activities and not any(a.is_primary_activity for a in self.activities):
            self.activities[0].is_primary_activity = True
        pe = next((a.event_id for a in self.activities if a.is_primary_activity), self.primary_event_id or self.current_event_dir.name)
        pr = next((r.region_id for r in self.regions if r.is_primary_region), self.regions[0].region_id)
        out = dict(self.raw_json)
        out['primary_event_id'] = pe
        out['primary_region_id'] = pr
        out['history_auto_apply'] = {
            'copy': bool(self.history_copy_enabled),
            'drift': bool(self.history_drift_enabled),
        }
        out['regions'] = [{
            'region_id': r.region_id,
            'region_position': r.region_position,
            'is_primary_region': r.is_primary_region,
            'bbox': _clamp(r.bbox),
            'annotation_frame_timestamp': r.annotation_frame_timestamp,
            'annotation_frame_name': r.annotation_frame_name,
            'history_copy_enabled': r.history_copy_enabled,
            'history_drift_enabled': r.history_drift_enabled,
        } for r in self.regions]
        out['activities'] = [{
            'event_id': a.event_id,
            'is_primary_activity': a.is_primary_activity,
            'label': a.label,
            'region_id': a.region_id,
            'active_region_source': a.active_region_source,
            'active_region_position': a.active_region_position,
        } for a in self.activities]
        out['bbox_resolution'] = {'width': ANNO_W, 'height': ANNO_H}
        out['anno_width'] = ANNO_W
        out['anno_height'] = ANNO_H
        out['annotated'] = True
        out['annotated_at'] = datetime.now().isoformat(timespec='seconds')
        out['annotator'] = os.environ.get('USERNAME') or os.environ.get('USER') or ''
        return out

    def _save_current_event(self, show_message: bool = True) -> Optional[Path]:
        if not self.current_bbox_path:
            return None
        out = self._build_output_payload()
        _save_json(self.current_bbox_path, out)
        if show_message:
            messagebox.showinfo('已保存', f'已保存到 {self.current_bbox_path}')
        return self.current_bbox_path

    def _autosave_before_event_change(self) -> bool:
        try:
            saved_path = self._save_current_event(show_message=False)
        except Exception as exc:
            messagebox.showerror('保存失败', f'切换事件组前自动保存失败：{exc}')
            return False
        if saved_path is not None:
            self._set_status(f'切换事件组前已自动保存: {saved_path.name}')
        return True

    def _restore_event_view(self, event_idx: int, modality: str, frame_name: str, region_id: str, activity_event_id: str) -> None:
        self.idx = event_idx
        self.selected_region_index = None
        self.selected_activity_index = None
        self._load_current_event()
        if modality and modality in [m for m in PREFERRED_MODALITIES if (self.current_event_dir / m).is_dir()] + [p.name for p in self.current_event_dir.iterdir() if p.is_dir()]:
            self.var_modality.set(modality)
            self._reload_frames(False)
        if frame_name and self.frame_paths:
            for i, path in enumerate(self.frame_paths):
                if path.name == frame_name:
                    self.frame_idx = i
                    break
            self._reload_frames(False)
        if region_id:
            self.selected_region_index = next((i for i, r in enumerate(self.regions) if _region_id_matches(r.region_id, region_id)), self.selected_region_index)
        if activity_event_id:
            self.selected_activity_index = next((i for i, a in enumerate(self.activities) if a.event_id == activity_event_id), self.selected_activity_index)
        self._refresh_lists()
        self._render()

    def _save_and_batch_apply_forward(self):
        region = self._get_selected_region()
        if not region or not self.current_event_dir:
            messagebox.showinfo('提示', '请先选中一个 region。')
            return

        current_event_idx = self.idx
        current_modality = self.var_modality.get()
        current_frame_name = self._current_frame_path().name if self._current_frame_path() else ''
        current_region_id = region.region_id
        current_activity_event_id = ''
        if self.selected_activity_index is not None and 0 <= self.selected_activity_index < len(self.activities):
            current_activity_event_id = self.activities[self.selected_activity_index].event_id

        self._save_current_event(show_message=False)

        saved_count = 1
        updated_count = 0
        skipped_count = 0
        batch_errors: List[str] = []

        try:
            for next_idx in range(current_event_idx + 1, len(self.event_dirs)):
                event_dir = self.event_dirs[next_idx]
                bbox_path = event_dir / 'bboxes.json'
                if not bbox_path.exists():
                    skipped_count += 1
                    continue

                try:
                    peid, prid, regions, activities, raw_json = _parse_bboxes_json(bbox_path)
                except Exception as exc:
                    batch_errors.append(f'{event_dir.name}: {exc}')
                    skipped_count += 1
                    continue

                matched_idx = next((i for i, r in enumerate(regions) if _region_id_matches(r.region_id, current_region_id)), None)
                if matched_idx is None:
                    skipped_count += 1
                    continue

                self.idx = next_idx
                self.current_event_dir = event_dir
                self.current_bbox_path = bbox_path
                self.primary_event_id = peid
                self.primary_region_id = prid
                self.regions = regions
                self.activities = activities
                self.raw_json = raw_json

                history_settings = self.raw_json.get('history_auto_apply', {}) or {}
                self.history_copy_enabled = bool(history_settings.get('copy', True))
                self.history_drift_enabled = bool(history_settings.get('drift', True))
                self.var_auto_apply_history_copy.set(self.history_copy_enabled)
                self.var_auto_apply_history_drift.set(self.history_drift_enabled)

                mods = [m for m in PREFERRED_MODALITIES if (self.current_event_dir / m).is_dir()] or [p.name for p in self.current_event_dir.iterdir() if p.is_dir()]
                target_modality = current_modality if current_modality in mods else (mods[0] if mods else '')
                self.var_modality.set(target_modality)
                self._reload_frames(True)

                self.selected_region_index = matched_idx
                self.selected_activity_index = next(
                    (i for i, a in enumerate(self.activities) if _region_id_matches(a.region_id, self.regions[matched_idx].region_id)),
                    None,
                )
                self._refresh_lists()

                target_region = self.regions[matched_idx]
                if target_region.history_drift_enabled:
                    ok = self._apply_history_recommendation(use_drift=True, silent=True)
                elif target_region.history_copy_enabled:
                    ok = self._apply_history_recommendation(use_drift=False, silent=True)
                else:
                    ok = False

                if not ok:
                    skipped_count += 1
                    continue

                self._save_current_event(show_message=False)
                saved_count += 1
                updated_count += 1
        finally:
            self._restore_event_view(
                event_idx=current_event_idx,
                modality=current_modality,
                frame_name=current_frame_name,
                region_id=current_region_id,
                activity_event_id=current_activity_event_id,
            )

        msg = (
            f'已保存当前事件，并向后批量应用同 region。\n'
            f'region={current_region_id}\n'
            f'保存事件数={saved_count}\n'
            f'向后更新数={updated_count}\n'
            f'跳过数={skipped_count}'
        )
        if batch_errors:
            msg += '\n错误样本:\n' + '\n'.join(batch_errors[:10])
        messagebox.showinfo('批量应用完成', msg)

    def _save(self):
        self._save_current_event(show_message=True)

    def run(self):
        self.root.mainloop()


def main():
    AnnotatorApp(DEFAULT_PROCESSED_DIR).run()


if __name__ == '__main__':
    main()
