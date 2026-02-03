from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from typing import Any

import pygetwindow as gw


@dataclass
class WindowInfo:
    hwnd: int
    title: str
    left: int
    top: int
    width: int
    height: int


def list_windows() -> list[WindowInfo]:
    items: list[WindowInfo] = []
    for w in gw.getAllWindows():
        title = (w.title or "").strip()
        if not title or w.width <= 0 or w.height <= 0:
            continue
        hwnd = int(getattr(w, "_hWnd", 0) or 0)
        if hwnd <= 0:
            continue
        items.append(
            WindowInfo(
                hwnd=hwnd,
                title=title,
                left=int(w.left),
                top=int(w.top),
                width=int(w.width),
                height=int(w.height),
            )
        )
    return items


def find_window_by_hwnd(hwnd: int) -> Any | None:
    for w in gw.getAllWindows():
        if int(getattr(w, "_hWnd", 0) or 0) == int(hwnd):
            return w
    return None


def find_hwnd_by_title_keyword(keyword: str) -> int | None:
    kw = (keyword or "").strip().lower()
    if not kw:
        return None
    for w in gw.getAllWindows():
        title = (w.title or "").strip().lower()
        if kw in title and w.width > 0 and w.height > 0:
            hwnd = int(getattr(w, "_hWnd", 0) or 0)
            if hwnd > 0:
                return hwnd
    return None


def get_window_bbox(hwnd: int) -> dict[str, int] | None:
    w = find_window_by_hwnd(hwnd)
    if w is None or w.width <= 0 or w.height <= 0:
        return None
    return {
        "left": int(w.left),
        "top": int(w.top),
        "width": int(w.width),
        "height": int(w.height),
    }


def activate_window(hwnd: int) -> bool:
    w = find_window_by_hwnd(hwnd)
    if w is None:
        return False
    try:
        if getattr(w, "isMinimized", False):
            w.restore()
        w.activate()
        time.sleep(0.08)
        return True
    except Exception:
        return False


def force_activate_window(hwnd: int, retries: int = 4) -> bool:
    try:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        SW_RESTORE = 9
        HWND_TOPMOST = -1
        HWND_NOTOPMOST = -2
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_SHOWWINDOW = 0x0040

        for _ in range(max(1, retries)):
            user32.ShowWindow(hwnd, SW_RESTORE)

            # Topmost toggle often helps bypass foreground restrictions.
            user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)
            user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW)

            fg = int(user32.GetForegroundWindow())
            cur_tid = int(kernel32.GetCurrentThreadId())
            fg_tid = int(user32.GetWindowThreadProcessId(fg, 0))
            tgt_tid = int(user32.GetWindowThreadProcessId(hwnd, 0))

            # Temporarily join input queues to increase success rate.
            if fg_tid and cur_tid and fg_tid != cur_tid:
                user32.AttachThreadInput(fg_tid, cur_tid, True)
            if tgt_tid and cur_tid and tgt_tid != cur_tid:
                user32.AttachThreadInput(tgt_tid, cur_tid, True)

            user32.BringWindowToTop(hwnd)
            user32.SetForegroundWindow(hwnd)
            user32.SetFocus(hwnd)
            user32.SetActiveWindow(hwnd)

            if fg_tid and cur_tid and fg_tid != cur_tid:
                user32.AttachThreadInput(fg_tid, cur_tid, False)
            if tgt_tid and cur_tid and tgt_tid != cur_tid:
                user32.AttachThreadInput(tgt_tid, cur_tid, False)

            time.sleep(0.06)
            if int(user32.GetForegroundWindow()) == int(hwnd):
                return True
        return False
    except Exception:
        return False


def get_foreground_hwnd() -> int | None:
    try:
        user32 = ctypes.windll.user32
        fg = int(user32.GetForegroundWindow())
        return fg if fg > 0 else None
    except Exception:
        return None
