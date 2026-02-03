from __future__ import annotations

import ctypes
import threading
import time
from typing import Any

import pyautogui
import pyttsx3

from .async_log import log
from .window_control import (
    activate_window,
    find_hwnd_by_title_keyword,
    force_activate_window,
    get_foreground_hwnd,
    get_window_bbox,
)


class Actuator:
    def __init__(self, chat_mode: str = "auto", chat_open_key: str = "y", osc_host: str = "127.0.0.1", osc_port: int = 9000) -> None:
        self._backend_name = "pyautogui"
        self._kb = pyautogui
        self._mouse = pyautogui
        self._osc_client = None
        self._chat_mode = (chat_mode or "auto").lower()
        self._chat_open_key = (chat_open_key or "y").lower()
        self._osc_control_enabled = self._chat_mode in {"auto", "osc"}
        self._osc_held_buttons: set[str] = set()
        try:
            import pydirectinput  # type: ignore

            self._backend_name = "pydirectinput"
            self._kb = pydirectinput
            self._mouse = pydirectinput
        except Exception:
            pass
        try:
            from pythonosc.udp_client import SimpleUDPClient  # type: ignore

            self._osc_client = SimpleUDPClient(osc_host, int(osc_port))
        except Exception:
            self._osc_client = None

    @staticmethod
    def _set_clipboard_text(text: str) -> bool:
        # Reliable way to input Chinese text: paste from clipboard.
        CF_UNICODETEXT = 13
        GMEM_MOVEABLE = 0x0002
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        if not user32.OpenClipboard(0):
            return False
        try:
            user32.EmptyClipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            h_global = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data))
            if not h_global:
                return False
            p_global = kernel32.GlobalLock(h_global)
            if not p_global:
                return False
            ctypes.memmove(p_global, data, len(data))
            kernel32.GlobalUnlock(h_global)
            if not user32.SetClipboardData(CF_UNICODETEXT, h_global):
                return False
            return True
        finally:
            user32.CloseClipboard()

    def _hotkey(self, k1: str, k2: str) -> None:
        self._kb.keyDown(k1)
        time.sleep(0.02)
        self._kb.press(k2)
        time.sleep(0.02)
        self._kb.keyUp(k1)

    @staticmethod
    def _paste_from_clipboard() -> None:
        # Try both paste shortcuts because some game UI only accepts one of them.
        pyautogui.hotkey("ctrl", "v")
        time.sleep(0.05)
        pyautogui.hotkey("shift", "insert")

    @staticmethod
    def _ui_press(key: str) -> None:
        # Use pyautogui for UI text box interactions (more reliable for chat input).
        pyautogui.press(key)

    def _try_osc_chat(self, text: str) -> bool:
        if self._chat_mode not in {"auto", "osc"}:
            return False
        if self._osc_client is None:
            return False
        try:
            text = (text or "").replace("\r", " ").replace("\n", " ").strip()
            if len(text) > 144:
                text = text[:144]
            if not text:
                return False
            # VRChat OSC Chatbox: /chatbox/input [message, send, addToHistory]
            self._osc_client.send_message("/chatbox/input", [text, True, False])
            return True
        except Exception:
            return False

    def _try_osc_axis(self, name: str, value: float, hold_sec: float = 0.08) -> bool:
        if not self._osc_control_enabled or self._osc_client is None:
            return False
        try:
            v = max(-1.0, min(1.0, float(value)))
            log(f"[osc] /input/{name}={v:.2f} hold={hold_sec:.2f}s")
            self._osc_client.send_message(f"/input/{name}", v)
            time.sleep(max(0.02, hold_sec))
            self._osc_client.send_message(f"/input/{name}", 0.0)
            return True
        except Exception:
            return False

    def _try_osc_button(self, name: str) -> bool:
        if not self._osc_control_enabled or self._osc_client is None:
            return False
        try:
            addr = f"/input/{name}"
            log(f"[osc] {addr}=1->0")
            self._osc_client.send_message(addr, 1)
            time.sleep(0.03)
            self._osc_client.send_message(addr, 0)
            return True
        except Exception:
            return False

    def _try_osc_button_state(self, name: str, pressed: bool) -> bool:
        if not self._osc_control_enabled or self._osc_client is None:
            return False
        try:
            addr = f"/input/{name}"
            self._osc_client.send_message(addr, 1 if pressed else 0)
            if pressed:
                self._osc_held_buttons.add(name)
            else:
                self._osc_held_buttons.discard(name)
            return True
        except Exception:
            return False

    @staticmethod
    def _osc_button_for_key(key: str) -> str | None:
        k = (key or "").strip().lower()
        mapping = {
            "w": "MoveForward",
            "s": "MoveBackward",
            "a": "MoveLeft",
            "d": "MoveRight",
            "space": "Jump",
            "shift": "Run",
            "left": "LookLeft",
            "right": "LookRight",
        }
        return mapping.get(k)

    def execute(
        self,
        actions: list[dict[str, Any]],
        dry_run: bool = True,
        target_hwnd: int | None = None,
        target_title_keyword: str = "",
    ) -> None:
        effective_hwnd = target_hwnd
        allow_local_input = True
        if effective_hwnd is None and target_title_keyword.strip():
            effective_hwnd = find_hwnd_by_title_keyword(target_title_keyword)

        def _ensure_focus() -> bool:
            if not effective_hwnd:
                return True
            fg = get_foreground_hwnd()
            if fg == effective_hwnd:
                return True
            # Try to recover focus right before sending input.
            ok = force_activate_window(effective_hwnd, retries=3)
            if not ok:
                ok = activate_window(effective_hwnd)
            fg2 = get_foreground_hwnd()
            return bool(ok and fg2 == effective_hwnd)

        log(f"[act] backend={self._backend_name} target_hwnd={effective_hwnd}")
        if dry_run:
            if effective_hwnd:
                log(f"[dry-run] target_hwnd={effective_hwnd}")
        else:
            if effective_hwnd:
                ok = force_activate_window(effective_hwnd, retries=5)
                if not ok:
                    ok = activate_window(effective_hwnd)
                if not ok:
                    log(f"[warn] target window hwnd={effective_hwnd} not focusable")
            if effective_hwnd:
                fg = get_foreground_hwnd()
                if fg != effective_hwnd:
                    log(
                        f"[warn] foreground hwnd={fg}, target hwnd={effective_hwnd}. "
                        "Local input disabled to avoid operating on wrong window; OSC-only actions still allowed."
                    )
                    allow_local_input = False

            # For deterministic relative mouse operations, place cursor in the target window.
            if allow_local_input and effective_hwnd and any(str(a.get("type", "")).startswith("mouse_") for a in actions):
                bbox = get_window_bbox(effective_hwnd)
                if bbox:
                    cx = int(bbox["left"] + bbox["width"] / 2)
                    cy = int(bbox["top"] + bbox["height"] / 2)
                    self._mouse.moveTo(cx, cy, duration=0.05)

        try:
            for action in actions:
                kind = action.get("type", "")
                if dry_run:
                    log(f"[dry-run] action: {action}")
                    continue
                if effective_hwnd and allow_local_input and not _ensure_focus():
                    log("[warn] skip action because target window is not foreground.")
                    continue

                if kind == "move":
                    direction = str(action.get("direction", "w")).lower()
                    seconds = float(action.get("seconds", 0.2))
                    if direction in {"w", "a", "s", "d"}:
                        osc_ok = False
                        if direction == "w":
                            osc_ok = self._try_osc_axis("Vertical", 1.0, hold_sec=seconds)
                        elif direction == "s":
                            osc_ok = self._try_osc_axis("Vertical", -1.0, hold_sec=seconds)
                        elif direction == "d":
                            osc_ok = self._try_osc_axis("Horizontal", 1.0, hold_sec=seconds)
                        elif direction == "a":
                            osc_ok = self._try_osc_axis("Horizontal", -1.0, hold_sec=seconds)
                        if osc_ok:
                            continue
                        if not allow_local_input:
                            log("[warn] skip local move fallback because target window is not foreground.")
                            continue
                        self._kb.keyDown(direction)
                        time.sleep(max(0.0, seconds))
                        self._kb.keyUp(direction)
                elif kind == "toggle_crouch":
                    if not allow_local_input:
                        log("[warn] skip toggle_crouch because target window is not foreground.")
                        continue
                    self._kb.press("c")
                elif kind == "toggle_prone":
                    if not allow_local_input:
                        log("[warn] skip toggle_prone because target window is not foreground.")
                        continue
                    self._kb.press("z")
                elif kind == "jump":
                    if self._try_osc_button("Jump"):
                        continue
                    if not allow_local_input:
                        log("[warn] skip local jump fallback because target window is not foreground.")
                        continue
                    self._kb.press("space")
                elif kind == "chat_send":
                    text = str(action.get("text", "")).strip()
                    if text:
                        log(f"[chat] send text len={len(text)}")
                        if self._try_osc_chat(text):
                            log("[chat] sent via OSC")
                            continue
                        if self._chat_mode == "osc":
                            log("[warn] chat mode is osc, but OSC send failed.")
                            continue
                        if not allow_local_input:
                            log("[warn] skip local chat fallback because target window is not foreground.")
                            continue
                        # Make sure game canvas has focus before opening chat box.
                        bbox = get_window_bbox(effective_hwnd) if effective_hwnd else None
                        if bbox:
                            cx = int(bbox["left"] + bbox["width"] / 2)
                            cy = int(bbox["top"] + bbox["height"] / 2)
                            self._mouse.moveTo(cx, cy, duration=0.03)
                            self._mouse.click(button="left")
                            time.sleep(0.06)
                        self._ui_press(self._chat_open_key)  # open chat box
                        time.sleep(0.3)
                        # Clean input box first to avoid IME/composition leftovers.
                        pyautogui.hotkey("ctrl", "a")
                        time.sleep(0.03)
                        self._ui_press("backspace")
                        time.sleep(0.03)
                        pasted = self._set_clipboard_text(text)
                        if pasted:
                            self._paste_from_clipboard()
                        else:
                            # Fallback for environments where clipboard API is unavailable.
                            pyautogui.write(text, interval=0.01)
                        time.sleep(0.08)
                        self._ui_press("enter")  # send
                        time.sleep(0.12)
                        # Ensure chat UI is closed even if game keeps focus in text box.
                        self._ui_press("esc")
                        time.sleep(0.05)
                        self._ui_press("esc")
                elif kind == "key_tap":
                    key = action.get("key", "")
                    duration = float(action.get("duration", 0.05))
                    osc_btn = self._osc_button_for_key(str(key))
                    if osc_btn and self._try_osc_button_state(osc_btn, True):
                        time.sleep(max(0.02, duration))
                        self._try_osc_button_state(osc_btn, False)
                        continue
                    if not allow_local_input:
                        log("[warn] skip key_tap because target window is not foreground.")
                        continue
                    self._kb.keyDown(key)
                    time.sleep(max(0.0, duration))
                    self._kb.keyUp(key)
                elif kind == "key_down":
                    key = str(action.get("key", ""))
                    osc_btn = self._osc_button_for_key(key)
                    if osc_btn and self._try_osc_button_state(osc_btn, True):
                        continue
                    if not allow_local_input:
                        log("[warn] skip key_down because target window is not foreground.")
                        continue
                    self._kb.keyDown(key)
                elif kind == "key_up":
                    key = str(action.get("key", ""))
                    osc_btn = self._osc_button_for_key(key)
                    if osc_btn and self._try_osc_button_state(osc_btn, False):
                        continue
                    if not allow_local_input:
                        log("[warn] skip key_up because target window is not foreground.")
                        continue
                    self._kb.keyUp(key)
                elif kind == "mouse_move":
                    dx = int(action.get("dx", 0))
                    dy = int(action.get("dy", 0))
                    look_mode = bool(action.get("look", True))
                    if look_mode:
                        osc_h_done = False
                        osc_v_done = False
                        if abs(dx) >= 2:
                            amount = max(-1.0, min(1.0, dx / 35.0))
                            hold = max(0.03, min(0.22, abs(dx) / 120.0))
                            osc_h_done = self._try_osc_axis("LookHorizontal", amount, hold_sec=hold)
                        if abs(dy) >= 2:
                            amount_v = max(-1.0, min(1.0, -dy / 35.0))
                            hold_v = max(0.03, min(0.22, abs(dy) / 120.0))
                            osc_v_done = self._try_osc_axis("LookVertical", amount_v, hold_sec=hold_v)
                        if (abs(dx) <= 1 or osc_h_done) and (abs(dy) <= 1 or osc_v_done):
                            continue
                        if not allow_local_input:
                            log("[warn] skip local look fallback because target window is not foreground.")
                            continue
                        self._mouse.mouseDown(button="right")
                        time.sleep(0.02)
                        self._mouse.moveRel(dx, dy, duration=0.05)
                        time.sleep(0.02)
                        self._mouse.mouseUp(button="right")
                    else:
                        if not allow_local_input:
                            log("[warn] skip local mouse_move because target window is not foreground.")
                            continue
                        self._mouse.moveRel(dx, dy, duration=0.05)
                elif kind == "mouse_click":
                    btn = action.get("button", "left")
                    if str(btn).lower() == "left" and self._try_osc_button("UseRight"):
                        continue
                    if str(btn).lower() == "right" and self._try_osc_button("GrabRight"):
                        continue
                    if not allow_local_input:
                        log("[warn] skip local mouse_click because target window is not foreground.")
                        continue
                    self._mouse.click(button=btn)
                elif kind == "wait":
                    time.sleep(float(action.get("seconds", 0.2)))
                # Cooperative yield between actions to reduce perceived stutter.
                time.sleep(0)
        finally:
            # Prevent stuck movement if a key_down mapped to OSC was not released.
            for btn in list(self._osc_held_buttons):
                self._try_osc_button_state(btn, False)


class Speaker:
    def speak(self, text: str, dry_run: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if dry_run:
            log(f"[dry-run] speak: {text}")
            return

        def _run_tts() -> None:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        t = threading.Thread(target=_run_tts, daemon=True)
        t.start()
        t.join(timeout=10.0)
        if t.is_alive():
            log("[warn] TTS timeout (>10s), skip waiting for voice engine.")
