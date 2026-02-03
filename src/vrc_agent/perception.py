from __future__ import annotations

import asyncio
import base64
import io
import wave
from dataclasses import dataclass

import mss
import numpy as np
import pygetwindow as gw
import sounddevice as sd
from PIL import Image

from .async_log import log
from .config import AgentConfig
from .llm_client import SiliconFlowClient
from .window_control import get_window_bbox


@dataclass
class Observation:
    scene_text: str
    heard_text: str


class Perception:
    def __init__(self, cfg: AgentConfig, llm: SiliconFlowClient, target_hwnd: int | None = None):
        self.cfg = cfg
        self.llm = llm
        self.target_hwnd = target_hwnd

    def _get_bbox(self) -> dict[str, int] | None:
        if self.target_hwnd:
            bbox = get_window_bbox(self.target_hwnd)
            if bbox is not None:
                return bbox

        keyword = self.cfg.window.title_keyword.strip()
        if not keyword:
            return None

        candidates = [w for w in gw.getAllWindows() if keyword.lower() in w.title.lower()]
        if not candidates:
            return None

        w = candidates[0]
        if w.width <= 0 or w.height <= 0:
            return None

        return {
            "left": int(w.left),
            "top": int(w.top),
            "width": int(w.width),
            "height": int(w.height),
        }

    def capture_screen_base64(self) -> tuple[str, str]:
        with mss.mss() as sct:
            bbox = self._get_bbox()
            monitor = bbox if bbox else sct.monitors[1]
            shot = sct.grab(monitor)
            img = Image.frombytes("RGB", shot.size, shot.rgb)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode("ascii"), "jpeg"

    def record_audio_base64(self) -> str:
        sr = self.cfg.audio.sample_rate
        sec = self.cfg.audio.capture_seconds
        frames = int(sr * sec)
        audio = sd.rec(frames, samplerate=sr, channels=1, dtype="int16")
        sd.wait()

        arr = np.asarray(audio).reshape(-1)
        with io.BytesIO() as bio:
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(arr.tobytes())
            return base64.b64encode(bio.getvalue()).decode("ascii")

    async def observe(self) -> Observation:
        log("[stage] observing: capture_screen...")
        image_base64, image_format = self.capture_screen_base64()
        log("[stage] observing: vision_describe...")
        try:
            scene_text = await asyncio.wait_for(
                self.llm.vision_describe(image_base64, image_format, self.cfg.prompt.vision),
                timeout=25.0,
            )
        except asyncio.TimeoutError:
            log("[warn] vision timeout (>25s), continue with empty scene.")
            scene_text = ""

        heard_text = ""
        if self.cfg.audio.enabled:
            log("[stage] observing: record_audio...")
            audio_base64 = await asyncio.to_thread(self.record_audio_base64)
            log("[stage] observing: transcribe_audio...")
            try:
                heard_text = await asyncio.wait_for(self.llm.transcribe_audio(audio_base64), timeout=30.0)
                if heard_text:
                    log(f"[asr] heard len={len(heard_text)} text={heard_text[:40]}")
                else:
                    log("[asr] heard empty")
            except asyncio.TimeoutError:
                log("[warn] asr timeout (>30s), continue without heard_text.")
                heard_text = ""

        return Observation(scene_text=scene_text, heard_text=heard_text)
