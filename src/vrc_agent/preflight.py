from __future__ import annotations

import ctypes
import socket
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

import httpx

from .async_log import log
from .config import AgentConfig
from .window_control import find_hwnd_by_title_keyword, find_window_by_hwnd, get_foreground_hwnd

Status = Literal["GREEN", "YELLOW", "RED"]


@dataclass
class CheckResult:
    status: Status
    detail: str
    suggestion: str = ""


@dataclass
class PreflightResult:
    osc: Status
    window: Status
    audio: Status
    api: Status
    notes: list[str] = field(default_factory=list)
    details: dict[str, CheckResult] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = asdict(self)
        # asdict() keeps nested dataclasses already converted.
        return out


def _pick_worse(a: Status, b: Status) -> Status:
    order = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    return a if order[a] >= order[b] else b


def _check_osc(cfg: AgentConfig) -> CheckResult:
    host = cfg.chat.osc_host
    port = int(cfg.chat.osc_port)
    try:
        socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_DGRAM)
    except Exception as exc:
        return CheckResult(
            status="RED",
            detail=f"resolve failed: {host}:{port} ({exc})",
            suggestion="检查 [chat].osc_host / osc_port。",
        )

    # UDP is connectionless; send success means local stack accepted it.
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.sendto(b"/preflight/ping\0\0\0,\0\0\0", (host, port))
        sock.close()
    except Exception as exc:
        status: Status = "YELLOW" if cfg.chat.mode == "hotkey" else "RED"
        return CheckResult(
            status=status,
            detail=f"udp send failed: {host}:{port} ({exc})",
            suggestion="在 VRChat 中启用 OSC，或切换 [chat].mode=\"hotkey\"。",
        )

    return CheckResult(
        status="GREEN",
        detail=f"udp send ok: {host}:{port}",
        suggestion="",
    )


def _check_window(cfg: AgentConfig, target_hwnd: Optional[int]) -> CheckResult:
    hwnd = target_hwnd or find_hwnd_by_title_keyword(cfg.window.title_keyword)
    if not hwnd:
        return CheckResult(
            status="RED",
            detail="target hwnd not resolved",
            suggestion="使用窗口选择器启动，或检查 [window].title_keyword。",
        )

    w = find_window_by_hwnd(hwnd)
    if w is None:
        return CheckResult(
            status="RED",
            detail=f"hwnd not found: {hwnd}",
            suggestion="目标窗口可能已关闭；重新选择窗口。",
        )

    user32 = ctypes.windll.user32
    visible = bool(user32.IsWindowVisible(int(hwnd)))
    enabled = bool(user32.IsWindowEnabled(int(hwnd)))
    minimized = bool(getattr(w, "isMinimized", False))
    fg = get_foreground_hwnd()
    matched = fg == int(hwnd)

    status: Status = "GREEN"
    if not visible or not enabled or minimized:
        status = _pick_worse(status, "YELLOW")
    if not matched:
        status = _pick_worse(status, "YELLOW")

    suggestion = ""
    if minimized:
        suggestion = "请先还原目标窗口。"
    elif not matched:
        suggestion = "运行时尽量保持 VRChat 在前台。"

    return CheckResult(
        status=status,
        detail=f"hwnd={hwnd}, visible={visible}, enabled={enabled}, fg_match={matched}",
        suggestion=suggestion,
    )


def _check_audio(cfg: AgentConfig) -> CheckResult:
    try:
        import sounddevice as sd
    except Exception as exc:
        return CheckResult(
            status="RED",
            detail=f"sounddevice import failed ({exc})",
            suggestion="安装 sounddevice 并确认 PortAudio 可用。",
        )

    try:
        default_dev = sd.default.device
        input_dev = None
        if hasattr(default_dev, "__getitem__"):
            try:
                input_dev = default_dev[0]
            except Exception:
                input_dev = None
        if input_dev is None:
            input_dev = getattr(default_dev, "input", None)
        if input_dev is None and isinstance(default_dev, int):
            input_dev = default_dev
        if input_dev is None or int(input_dev) < 0:
            return CheckResult(
                status="RED",
                detail="no default input device",
                suggestion="在系统声音设置中设置默认麦克风。",
            )

        sd.check_input_settings(
            device=int(input_dev),
            channels=1,
            samplerate=int(cfg.audio.sample_rate),
            dtype="int16",
        )
        return CheckResult(
            status="GREEN",
            detail=f"default_input={int(input_dev)}, sample_rate={int(cfg.audio.sample_rate)}",
            suggestion="",
        )
    except Exception as exc:
        return CheckResult(
            status="RED",
            detail=f"input settings invalid ({exc})",
            suggestion="调整 [audio].sample_rate（如 16000/48000）并检查麦克风权限。",
        )


async def _check_api(cfg: AgentConfig) -> CheckResult:
    key = (cfg.api.api_key or "").strip()
    if not key or key.startswith("your-"):
        return CheckResult(
            status="RED",
            detail="api key missing",
            suggestion="在 config.toml 填写有效 [api].api_key。",
        )

    url = cfg.api.base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {key}"}
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(url, headers=headers)
        if resp.status_code == 200:
            return CheckResult(status="GREEN", detail=f"GET {url} -> 200", suggestion="")
        if resp.status_code in (401, 403):
            return CheckResult(
                status="RED",
                detail=f"GET {url} -> {resp.status_code}",
                suggestion="API key 可能无效或权限不足。",
            )
        if resp.status_code == 429:
            return CheckResult(
                status="YELLOW",
                detail=f"GET {url} -> 429",
                suggestion="触发限流，稍后重试。",
            )
        if 400 <= resp.status_code < 500:
            return CheckResult(
                status="YELLOW",
                detail=f"GET {url} -> {resp.status_code}",
                suggestion="endpoint 可达但返回客户端错误，请检查模型服务兼容性。",
            )
        return CheckResult(
            status="RED",
            detail=f"GET {url} -> {resp.status_code}",
            suggestion="服务端异常或网络不稳定，稍后重试。",
        )
    except Exception as exc:
        return CheckResult(
            status="RED",
            detail=f"request failed ({exc})",
            suggestion="检查网络/代理/base_url 是否正确。",
        )


async def run_preflight(cfg: AgentConfig, target_hwnd: Optional[int]) -> PreflightResult:
    details: dict[str, CheckResult] = {}
    try:
        details["osc"] = _check_osc(cfg)
    except Exception as exc:
        details["osc"] = CheckResult("RED", f"unexpected error ({exc})", "检查 OSC 配置。")

    try:
        details["window"] = _check_window(cfg, target_hwnd)
    except Exception as exc:
        details["window"] = CheckResult("RED", f"unexpected error ({exc})", "检查窗口选择流程。")

    try:
        details["audio"] = _check_audio(cfg)
    except Exception as exc:
        details["audio"] = CheckResult("RED", f"unexpected error ({exc})", "检查音频设备与权限。")

    try:
        details["api"] = await _check_api(cfg)
    except Exception as exc:
        details["api"] = CheckResult("RED", f"unexpected error ({exc})", "检查 API 配置。")

    notes = []
    for name in ("osc", "window", "audio", "api"):
        item = details[name]
        if item.suggestion:
            notes.append(f"{name}: {item.suggestion}")

    return PreflightResult(
        osc=details["osc"].status,
        window=details["window"].status,
        audio=details["audio"].status,
        api=details["api"].status,
        notes=notes,
        details=details,
    )


def print_preflight_report(result: PreflightResult) -> None:
    for name in ("osc", "window", "audio", "api"):
        item = result.details[name]
        line = f"[preflight] {name:<6} {item.status:<6} {item.detail}"
        if item.suggestion:
            line += f" | hint: {item.suggestion}"
        log(line)
    log(f"[preflight] summary osc={result.osc}, window={result.window}, audio={result.audio}, api={result.api}")
