from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import messagebox

import mss
from PIL import Image, ImageTk

from .window_control import WindowInfo, list_windows


@dataclass
class WindowChoice:
    hwnd: int
    title: str


def _capture_preview(window: WindowInfo, max_side: int = 380) -> Image.Image | None:
    if window.width <= 0 or window.height <= 0:
        return None
    bbox = {
        "left": max(0, window.left),
        "top": max(0, window.top),
        "width": max(1, window.width),
        "height": max(1, window.height),
    }
    try:
        with mss.mss() as sct:
            shot = sct.grab(bbox)
        img = Image.frombytes("RGB", shot.size, shot.rgb)
        img.thumbnail((max_side, max_side))
        return img
    except Exception:
        return None


def pick_window_ui() -> WindowChoice:
    root = tk.Tk()
    root.title("Select Target Window")
    root.geometry("940x620")

    selected: WindowChoice | None = None
    preview_ref: ImageTk.PhotoImage | None = None

    left = tk.Frame(root, padx=8, pady=8)
    left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    right = tk.Frame(root, padx=8, pady=8)
    right.pack(side=tk.RIGHT, fill=tk.BOTH)

    tk.Label(left, text="Choose one target window (capture + action + control)").pack(anchor="w")
    listbox = tk.Listbox(left, width=80, height=28)
    listbox.pack(fill=tk.BOTH, expand=True, pady=6)

    preview_label = tk.Label(right, text="Preview", width=50, height=28, anchor="center")
    preview_label.pack()
    detail_label = tk.Label(right, text="", justify="left", wraplength=380, anchor="w")
    detail_label.pack(fill=tk.X, pady=8)

    windows: list[WindowInfo] = []

    def refresh() -> None:
        nonlocal windows
        windows = list_windows()
        listbox.delete(0, tk.END)
        for i, w in enumerate(windows):
            line = f"[{i}] {w.title}   ({w.width}x{w.height} @ {w.left},{w.top})  hwnd={w.hwnd}"
            listbox.insert(tk.END, line)
        if windows:
            listbox.selection_set(0)
            update_preview()

    def update_preview(_event: object | None = None) -> None:
        nonlocal preview_ref
        if not listbox.curselection():
            return
        idx = int(listbox.curselection()[0])
        if idx < 0 or idx >= len(windows):
            return
        w = windows[idx]
        detail_label.config(
            text=f"Title: {w.title}\nHWND: {w.hwnd}\nPos: ({w.left}, {w.top})\nSize: {w.width} x {w.height}"
        )
        img = _capture_preview(w)
        if img is None:
            preview_label.config(text="Preview unavailable", image="")
            preview_ref = None
            return
        preview_ref = ImageTk.PhotoImage(img)
        preview_label.config(image=preview_ref, text="")

    def confirm() -> None:
        nonlocal selected
        if not listbox.curselection():
            messagebox.showwarning("No selection", "Please choose one window.")
            return
        idx = int(listbox.curselection()[0])
        if idx < 0 or idx >= len(windows):
            messagebox.showwarning("Invalid selection", "Please choose a valid window.")
            return
        w = windows[idx]
        selected = WindowChoice(hwnd=w.hwnd, title=w.title)
        root.destroy()

    def cancel() -> None:
        root.destroy()

    btn_row = tk.Frame(left)
    btn_row.pack(fill=tk.X, pady=6)
    tk.Button(btn_row, text="Refresh", command=refresh, width=12).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Use Selected", command=confirm, width=14).pack(side=tk.LEFT, padx=4)
    tk.Button(btn_row, text="Cancel", command=cancel, width=10).pack(side=tk.LEFT, padx=4)

    listbox.bind("<<ListboxSelect>>", update_preview)
    listbox.bind("<Double-Button-1>", lambda _e: confirm())
    root.protocol("WM_DELETE_WINDOW", cancel)

    refresh()
    root.mainloop()

    if selected is None:
        raise RuntimeError("Window selection canceled.")
    return selected
