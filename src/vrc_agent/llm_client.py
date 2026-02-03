from __future__ import annotations

import asyncio
import base64
import io
import json
import re
from typing import Any

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from .async_log import log
from .config import AgentConfig


class SiliconFlowClient:
    """OpenAI-compatible client, same pattern used by MaiBot for SiliconFlow."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg
        self.client = AsyncOpenAI(
            base_url=cfg.api.base_url,
            api_key=cfg.api.api_key,
            timeout=cfg.api.timeout_sec,
            max_retries=0,
        )

    async def _call_with_retry(self, fn, *, name: str, retries: int = 3):
        last_exc: Exception | None = None
        for i in range(retries):
            try:
                return await fn()
            except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError) as exc:
                last_exc = exc
                if i == retries - 1:
                    break
                delay = 1.2 * (2**i)
                log(f"[warn] {name} failed ({type(exc).__name__}), retry in {delay:.1f}s...")
                await asyncio.sleep(delay)
        if last_exc:
            raise last_exc

    async def vision_describe(self, image_base64: str, image_format: str, prompt: str) -> str:
        image_url = f"data:image/{image_format};base64,{image_base64}"
        concise_prompt = (
            "请用简体中文做简短场景识别，优先给出："
            "1) 当前主要场景 2) 可交互对象 3) 角色/人群状态。"
            "总长度控制在120字以内，不要写长篇结构化报告。"
        )
        async def _do():
            return await self.client.chat.completions.create(
                model=self.cfg.models.vision,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{concise_prompt}\n{prompt}"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=512,
            )

        try:
            resp = await self._call_with_retry(_do, name="vision_describe", retries=3)
        except Exception as exc:
            log(f"[warn] vision_describe failed, continue with empty scene: {exc}")
            return ""
        return (resp.choices[0].message.content or "").strip()

    async def transcribe_audio(self, audio_base64: str) -> str:
        audio_bytes = base64.b64decode(audio_base64)
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=self.cfg.models.asr,
                file=("audio.wav", io.BytesIO(audio_bytes)),
                language="zh",
            )
            text = getattr(transcript, "text", "")
            return text.strip()
        except Exception as exc:
            # ASR occasionally fails with provider-side 5xx. Keep the loop alive.
            log(f"[warn] ASR failed, continue without heard_text: {exc}")
            return ""

    async def plan(self, state: dict[str, Any]) -> dict[str, Any]:
        planner_prompt = (
            f"{self.cfg.prompt.planner}\n\n"
            "Action schema:\n"
            "- move: {\"type\":\"move\",\"direction\":\"w|a|s|d\",\"seconds\":0.3}\n"
            "- toggle_crouch: {\"type\":\"toggle_crouch\"}  # key C\n"
            "- toggle_prone: {\"type\":\"toggle_prone\"}    # key Z\n"
            "- jump: {\"type\":\"jump\"}                    # key Space\n"
            "- chat_send: {\"type\":\"chat_send\",\"text\":\"你好\"}  # press Y -> paste -> Enter -> Esc\n"
            "- key_tap: {\"type\":\"key_tap\",\"key\":\"w\",\"duration\":0.15}\n"
            "- key_down/up: {\"type\":\"key_down\",\"key\":\"shift\"} / {\"type\":\"key_up\",\"key\":\"shift\"}\n"
            "- mouse_move: {\"type\":\"mouse_move\",\"dx\":20,\"dy\":-10,\"look\":true}  # rotate view\n"
            "- mouse_click: {\"type\":\"mouse_click\",\"button\":\"left\"}\n"
            "- wait: {\"type\":\"wait\",\"seconds\":0.5}\n"
            "Control rules:\n"
            "- Prefer OSC-friendly actions: move, jump, mouse_move(look), chat_send.\n"
            "- Use move for WASD navigation.\n"
            "- Use jump for jumping.\n"
            "- Use toggle_crouch/toggle_prone only when truly needed.\n"
            "- If you need to speak in game chat, prefer chat_send instead of raw key sequence.\n"
            "- Language rule: all natural-language text in `speak` and `chat_send.text` must be Simplified Chinese.\n"
            "- You will receive both short_term_memory and long_term_memory in state. Reuse relevant memory to keep behavior consistent.\n"
            "- Do not repeat the same action sequence every tick. Alternate movement direction, view angle, and interaction target.\n"
            "- Prefer concise action list (2-5 actions). Only send chat when there is clear social context.\n"
            "Only output one JSON object."
        )

        async def _do():
            return await self.client.chat.completions.create(
                model=self.cfg.models.planner,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=4096,
            )

        try:
            resp = await self._call_with_retry(_do, name="planner", retries=3)
        except Exception as exc:
            log(f"[warn] planner failed, fallback to no-op plan: {exc}")
            return {"speak": "", "actions": [], "next_focus": ""}
        text = (resp.choices[0].message.content or "{}").strip()
        return self._parse_json(text)

    async def plan_intent(self, state: dict[str, Any]) -> dict[str, Any]:
        # Low-frequency, low-token planner: intent/mood first; optional sparse actions.
        planner_prompt = (
            "You are a low-frequency intent controller for a VRChat agent.\n"
            "Return one strict JSON object with keys:\n"
            "{"
            "\"intent\": string,"
            "\"activity_level\": number(0-1),"
            "\"curiosity\": number(0-1),"
            "\"allow_move\": boolean,"
            "\"speak\": string,"
            "\"actions\": array"
            "}\n"
            "Rules:\n"
            "- Keep output concise.\n"
            "- Prefer socially natural behavior: when nearby players are visible, provide a short `speak` in Chinese.\n"
            "- In social scenes, usually include one `chat_send` action with concise text (about 10-40 Chinese chars).\n"
            "- If short_term_memory already shows a very recent chat_send, you may skip chat this turn to avoid spam.\n"
            "- In non-social scenes, actions can be empty.\n"
            "- If chat is needed, use chat_send action.\n"
            "- Language for speak/chat text: Simplified Chinese.\n"
            "- Do NOT output any extra text outside JSON."
        )

        async def _do():
            return await self.client.chat.completions.create(
                model=self.cfg.models.planner,
                messages=[
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=4096,
            )

        try:
            resp = await self._call_with_retry(_do, name="planner.intent", retries=3)
        except Exception as exc:
            log(f"[warn] planner.intent failed, fallback to keep-alive intent: {exc}")
            return {
                "intent": "observe",
                "activity_level": 0.35,
                "curiosity": 0.55,
                "allow_move": True,
                "speak": "",
                "actions": [],
            }
        text = (resp.choices[0].message.content or "{}").strip()
        data = self._parse_json(text)
        # minimal schema normalization
        if "intent" not in data:
            data["intent"] = "observe"
        if "activity_level" not in data:
            data["activity_level"] = 0.35
        if "curiosity" not in data:
            data["curiosity"] = 0.55
        if "allow_move" not in data:
            data["allow_move"] = True
        if "speak" not in data:
            data["speak"] = ""
        if "actions" not in data or not isinstance(data["actions"], list):
            data["actions"] = []
        return data

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # fallback: extract first JSON block
            m = re.search(r"\{[\s\S]*\}", raw)
            if not m:
                return {"speak": "", "actions": [], "next_focus": ""}
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {"speak": "", "actions": [], "next_focus": ""}
