from __future__ import annotations

import asyncio
import random
import time
from collections import deque
from datetime import datetime
from typing import Any, Optional
from difflib import SequenceMatcher

from .actuator import Actuator, Speaker
from .async_log import log
from .config import AgentConfig
from .llm_client import SiliconFlowClient
from .memory import MemoryStore, build_memory_item
from .perception import Observation, Perception


class AgentRuntime:
    def __init__(self, cfg: AgentConfig, target_hwnd: int | None = None, target_title: str = ""):
        self.cfg = cfg
        self.target_hwnd = target_hwnd
        self.target_title = target_title
        self.llm = SiliconFlowClient(cfg)
        self.perception = Perception(cfg, self.llm, target_hwnd=target_hwnd)
        self.actuator = Actuator(
            chat_mode=self.cfg.chat.mode,
            chat_open_key=self.cfg.chat.open_key,
            osc_host=self.cfg.chat.osc_host,
            osc_port=self.cfg.chat.osc_port,
        )
        self.speaker = Speaker()
        self.memory = deque(maxlen=8)
        self.recent_action_sigs = deque(maxlen=6)
        self.tick_id = 0
        self.last_observation: Optional[Observation] = None
        self.observe_task: Optional[asyncio.Task[Observation]] = None
        self.idle_task: Optional[asyncio.Task[None]] = None
        self.act_lock = asyncio.Lock()
        self._closed = False
        self.intent_state: dict[str, Any] = {
            "intent": "observe",
            "activity_level": 0.35,
            "curiosity": 0.55,
            "allow_move": True,
            "updated_at": 0.0,
        }
        self.intent_ttl_sec = max(1.0, float(self.cfg.runtime.intent_ttl_sec))
        self._last_llm_scene = ""
        self._last_heard = ""
        self._last_idle_sig = ""
        self._last_idle_at = 0.0
        self._last_idle_dx = 0
        self._last_manual_say_at = 0.0
        self._last_auto_chat_at = 0.0
        self._last_replied_heard = ""
        self._last_heard_reply_at = 0.0
        self._heard_latch_text = ""
        self._heard_latch_until = 0.0
        self.memory_store = MemoryStore(
            file_path=self.cfg.memory.file_path,
            max_records=self.cfg.memory.max_records,
        )
        if self.target_hwnd:
            log(f"[window] locked target: hwnd={self.target_hwnd} title={self.target_title!r}")
        self.idle_task = asyncio.create_task(self._idle_loop())

    async def tick(self) -> dict[str, Any]:
        self.tick_id += 1
        log("[stage] observing...")
        obs = self._merge_heard_latch(await self._get_observation())
        query = f"{obs.scene_text}\n{obs.heard_text}"
        long_term_memory = []
        if self.cfg.memory.enabled:
            long_term_memory = self.memory_store.retrieve(query=query, top_k=self.cfg.memory.retrieve_top_k)
        state = self._build_intent_state_payload(obs, long_term_memory)
        plan: dict[str, Any] = {"speak": "", "actions": []}
        llm_called = self._should_call_llm(obs)
        if llm_called:
            log("[stage] planning...")
            plan = await self.llm.plan_intent(state)
            self._update_intent(plan)
            self._last_llm_scene = obs.scene_text
            self._last_heard = obs.heard_text

        speak_text = str(plan.get("speak", "")).strip()
        actions = plan.get("actions", [])
        if not isinstance(actions, list):
            actions = []
        actions = self._normalize_actions(actions)
        speak_text, actions = self._ensure_reply_on_heard(obs, speak_text, actions)
        speak_text, actions = self._ensure_speak_to_chat(speak_text, actions)
        if self._should_auto_chat(obs, actions):
            auto_text = self._build_scene_short_line(obs)
            if auto_text:
                actions = list(actions)
                actions.append({"type": "chat_send", "text": auto_text})
                if not speak_text:
                    speak_text = auto_text
                self._last_auto_chat_at = time.time()
        actions = self._stabilize_actions(actions)
        actions = self._repair_chat_actions(actions, speak_text)

        if self.cfg.runtime.tts_enabled:
            log("[stage] speaking...")
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.speaker.speak, speak_text, self.cfg.runtime.dry_run),
                    timeout=15.0,
                )
            except asyncio.TimeoutError:
                log("[warn] speaking timeout (>15s), skip this step.")
        if not self.cfg.runtime.observe_only:
            log("[stage] acting...")
            try:
                async with self.act_lock:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.actuator.execute,
                            actions,
                            self.cfg.runtime.dry_run,
                            self.target_hwnd,
                            self.cfg.window.title_keyword,
                        ),
                        timeout=30.0,
                    )
            except asyncio.TimeoutError:
                log("[warn] acting timeout (>30s), skip this step.")

        summary = {
            "scene": obs.scene_text,
            "heard": obs.heard_text,
            "speak": speak_text,
            "actions": actions,
            "llm_called": llm_called,
            "intent": self.intent_state.get("intent", "observe"),
        }
        self.memory.append(summary)
        if self.cfg.memory.enabled:
            self.memory_store.append(
                build_memory_item(
                    scene=obs.scene_text,
                    heard=obs.heard_text,
                    speak=speak_text,
                    actions=actions,
                )
            )
        return summary

    def _merge_heard_latch(self, obs: Observation) -> Observation:
        now = time.time()
        heard = (obs.heard_text or "").strip()
        if heard:
            self._heard_latch_text = heard
            self._heard_latch_until = now + 10.0
            return obs
        if self._heard_latch_text and now < self._heard_latch_until:
            return Observation(scene_text=obs.scene_text, heard_text=self._heard_latch_text)
        return obs

    async def _get_observation(self) -> Observation:
        # Bootstrap async observation task once, then keep it running in background.
        if self.observe_task is None:
            self.observe_task = asyncio.create_task(self.perception.observe())

        if self.observe_task.done():
            try:
                obs = await self.observe_task
                self.last_observation = obs
            except Exception as exc:
                log(f"[warn] observe task failed, fallback to cache: {exc}")
                if self.last_observation is None:
                    # No cache available: do a foreground fetch to recover.
                    obs = await self.perception.observe()
                    self.last_observation = obs
                else:
                    obs = self.last_observation
            finally:
                self.observe_task = asyncio.create_task(self.perception.observe())
            return obs

        if self.last_observation is not None:
            log("[stage] observing: use cached observation")
            return self.last_observation

        # First frame has no cache: wait for initial observation.
        obs = await self.observe_task
        self.last_observation = obs
        self.observe_task = asyncio.create_task(self.perception.observe())
        return obs

    def _stabilize_actions(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Keep action count bounded.
        actions = actions[:8]
        # Never replace a turn that already contains chat output.
        if any(str(a.get("type", "")) == "chat_send" for a in actions):
            return actions
        sig = self._action_signature(actions)
        repeated = self.recent_action_sigs.count(sig)
        self.recent_action_sigs.append(sig)

        # If the planner keeps emitting the same script, force a different exploration pattern.
        if repeated >= 2:
            variant = self.tick_id % 3
            if variant == 0:
                return [
                    {"type": "move", "direction": "a", "seconds": 0.25},
                    {"type": "mouse_move", "dx": -30, "dy": 0},
                    {"type": "jump"},
                    {"type": "wait", "seconds": 0.25},
                ]
            if variant == 1:
                return [
                    {"type": "move", "direction": "d", "seconds": 0.25},
                    {"type": "mouse_move", "dx": 25, "dy": -8},
                    {"type": "wait", "seconds": 0.2},
                ]
            return [
                {"type": "move", "direction": "s", "seconds": 0.2},
                {"type": "mouse_move", "dx": 0, "dy": -12},
                {"type": "mouse_click", "button": "left"},
            ]
        return actions

    @staticmethod
    def _action_signature(actions: list[dict[str, Any]]) -> str:
        parts = []
        for a in actions[:5]:
            t = str(a.get("type", ""))
            if t == "move":
                parts.append(f"move:{a.get('direction','')}")
            elif t == "mouse_move":
                dx = int(a.get("dx", 0))
                dy = int(a.get("dy", 0))
                parts.append(f"mouse:{dx//10}:{dy//10}")
            else:
                parts.append(t)
        return "|".join(parts)

    @staticmethod
    def _repair_chat_actions(actions: list[dict[str, Any]], speak_text: str) -> list[dict[str, Any]]:
        repaired: list[dict[str, Any]] = []
        for action in actions:
            if str(action.get("type", "")) != "chat_send":
                repaired.append(action)
                continue

            text = str(action.get("text", "")).strip()
            # Avoid sending meaningless short numeric strings such as "4145".
            is_mostly_digits = text.isdigit() or (
                len(text) > 0 and sum(ch.isdigit() for ch in text) / len(text) >= 0.8
            )
            if len(text) < 4 or is_mostly_digits:
                text = (speak_text or "").strip()
            if len(text) > 140:
                text = text[:140]
            if not text:
                continue
            action = dict(action)
            action["text"] = text
            repaired.append(action)
        return repaired

    @staticmethod
    def _normalize_actions(actions: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for action in actions:
            if isinstance(action, dict):
                normalized.append(action)
            else:
                log(f"[warn] drop invalid action item: {type(action).__name__}")
        return normalized

    def _build_intent_state_payload(self, obs: Observation, long_term_memory: list[dict[str, Any]]) -> dict[str, Any]:
        # Token-saving compact payload: intent control does not need full scene dump.
        short_scene = (obs.scene_text or "")[:280]
        short_heard = (obs.heard_text or "")[:90]
        short_mem = []
        for item in list(self.memory)[-2:]:
            short_mem.append(
                {
                    "speak": str(item.get("speak", ""))[:80],
                    "actions": str(item.get("actions", ""))[:80],
                }
            )
        short_ltm = []
        for item in (long_term_memory or [])[:2]:
            short_ltm.append(
                {
                    "scene": str(item.get("scene", ""))[:100],
                    "speak": str(item.get("speak", ""))[:80],
                }
            )
        return {
            "time": datetime.now().isoformat(timespec="seconds"),
            "scene": short_scene,
            "heard": short_heard,
            "intent_state": {
                "intent": self.intent_state.get("intent", "observe"),
                "activity_level": self.intent_state.get("activity_level", 0.35),
                "curiosity": self.intent_state.get("curiosity", 0.55),
                "allow_move": self.intent_state.get("allow_move", True),
            },
            "short_term_memory": short_mem,
            "long_term_memory": short_ltm,
        }

    def _should_call_llm(self, obs: Observation) -> bool:
        now = time.time()
        if (obs.heard_text or "").strip() and obs.heard_text != self._last_heard:
            return True
        last_scene = self._last_llm_scene or ""
        cur_scene = obs.scene_text or ""
        if last_scene and cur_scene:
            sim = SequenceMatcher(a=last_scene[:320], b=cur_scene[:320]).ratio()
            if sim < 0.58:
                return True
        if now - float(self.intent_state.get("updated_at", 0.0)) > self.intent_ttl_sec:
            return True
        return False

    def _update_intent(self, plan: dict[str, Any]) -> None:
        intent = str(plan.get("intent", "")).strip() or str(plan.get("next_focus", "")).strip() or "observe"
        self.intent_state["intent"] = intent[:40]
        self.intent_state["activity_level"] = max(0.0, min(1.0, float(plan.get("activity_level", 0.35) or 0.35)))
        self.intent_state["curiosity"] = max(0.0, min(1.0, float(plan.get("curiosity", 0.55) or 0.55)))
        self.intent_state["allow_move"] = bool(plan.get("allow_move", True))
        self.intent_state["updated_at"] = time.time()

    async def _idle_loop(self) -> None:
        # Instinct loop: keep "alive" micro-actions without LLM.
        while not self._closed:
            interval_min = max(0.1, float(self.cfg.runtime.idle_interval_min_sec))
            interval_max = max(interval_min, float(self.cfg.runtime.idle_interval_max_sec))
            await asyncio.sleep(random.uniform(interval_min, interval_max))
            if self.cfg.runtime.observe_only:
                continue

            force_keepalive = (time.time() - self._last_idle_at) > 2.0
            actions = self._build_idle_actions(force_keepalive=force_keepalive)
            if not actions:
                continue

            try:
                async with self.act_lock:
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            self.actuator.execute,
                            actions,
                            self.cfg.runtime.dry_run,
                            self.target_hwnd,
                            self.cfg.window.title_keyword,
                        ),
                        timeout=2.0,
                    )
                self._last_idle_at = time.time()
            except asyncio.TimeoutError:
                log("[warn] idle action timeout, skip.")
            except Exception as exc:
                log(f"[warn] idle loop action failed: {exc}")

    def _build_idle_actions(self, force_keepalive: bool = False) -> list[dict[str, Any]]:
        intent = str(self.intent_state.get("intent", "observe")).lower()
        activity = float(self.intent_state.get("activity_level", 0.35))
        curiosity = float(self.intent_state.get("curiosity", 0.55))
        allow_move = bool(self.intent_state.get("allow_move", True))
        heard = (self.last_observation.heard_text if self.last_observation else "") or ""

        actions: list[dict[str, Any]] = []

        # Human-like hesitation: sometimes "thinks" and does almost nothing.
        if not force_keepalive and random.random() < self._prob(self.cfg.runtime.idle_hesitate_idle_prob):
            return []
        if not force_keepalive and random.random() < self._prob(self.cfg.runtime.idle_hesitate_pause_prob):
            return [{"type": "wait", "seconds": round(random.uniform(0.3, 0.8), 2)}]

        # Micro pause before any intent execution.
        if random.random() < 0.25:
            actions.append({"type": "wait", "seconds": round(random.uniform(0.08, 0.28), 2)})

        jitter_min_deg = max(0.2, float(self.cfg.runtime.idle_look_jitter_min_deg))
        jitter_max_deg = max(jitter_min_deg, float(self.cfg.runtime.idle_look_jitter_max_deg))
        base_jitter_deg = random.uniform(jitter_min_deg, jitter_max_deg) * (0.8 + 0.4 * curiosity)
        base_dx = self._deg_to_dx(base_jitter_deg)
        base_dx *= random.choice([-1, 1])
        base_dy = random.randint(-4, 4) if intent not in {"observe", "listen"} else random.randint(-5, 6)
        max_dx = self._deg_to_dx(jitter_max_deg * 1.35)

        # If there is fresh heard text, bias to orienting behavior.
        if heard.strip() and random.random() < 0.45:
            dx = self._soft_cap_dx(int(base_dx * 1.5), max_dx=max_dx)
            if random.random() < self._prob(self.cfg.runtime.idle_look_overshoot_prob):
                # Overshoot then pull back to mimic human correction.
                actions.append({"type": "mouse_move", "dx": dx, "dy": 0, "look": True})
                actions.append({"type": "wait", "seconds": 0.06})
                actions.append({"type": "mouse_move", "dx": int(-dx * random.uniform(0.28, 0.42)), "dy": 0, "look": True})
            else:
                actions.append({"type": "mouse_move", "dx": dx, "dy": 0, "look": True})
        else:
            dx = self._soft_cap_dx(int(base_dx), max_dx=max_dx)
            dy = base_dy
            actions.append({"type": "mouse_move", "dx": dx, "dy": dy, "look": True})

        # Not every thought produces movement.
        move_prob = self._prob(self.cfg.runtime.idle_small_step_move_prob + activity * 0.2)
        if allow_move and random.random() < move_prob:
            if random.random() < 0.28:
                actions.append({"type": "wait", "seconds": round(random.uniform(0.25, 0.5), 2)})
            else:
                direction = random.choice(["w", "a", "s", "d"])
                actions.append({"type": "move", "direction": direction, "seconds": round(random.uniform(0.12, 0.25), 2)})

        # Keep idle payload tiny and non-invasive; avoid exact repetition.
        final_actions = actions[:3]
        sig = self._action_signature(final_actions)
        if sig and sig == self._last_idle_sig:
            final_actions = self._mutate_idle_actions(final_actions, max_dx=max_dx)
            sig = self._action_signature(final_actions)
        self._last_idle_sig = sig
        return final_actions

    @staticmethod
    def _prob(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def _deg_to_dx(deg: float) -> int:
        # Empirical conversion for current mouse/OSC mapping.
        return max(1, int(round(abs(deg) * 9)))

    def _soft_cap_dx(self, dx: int, max_dx: int) -> int:
        # Prevent abrupt turn spikes and avoid identical alternating jitter.
        capped = max(-max_dx, min(max_dx, int(dx)))
        delta_cap = max(4, max_dx // 2)
        delta = capped - self._last_idle_dx
        if delta > delta_cap:
            capped = self._last_idle_dx + delta_cap
        elif delta < -delta_cap:
            capped = self._last_idle_dx - delta_cap
        self._last_idle_dx = capped
        return capped

    def _mutate_idle_actions(self, actions: list[dict[str, Any]], max_dx: int) -> list[dict[str, Any]]:
        mutated = [dict(a) for a in actions]
        for action in mutated:
            if str(action.get("type", "")) == "mouse_move":
                jitter = random.choice([-3, -2, 2, 3])
                action["dx"] = self._soft_cap_dx(int(action.get("dx", 0)) + jitter, max_dx=max_dx)
                action["dy"] = int(action.get("dy", 0)) + random.choice([-1, 0, 1])
                return mutated
        # If there is no look action, inject a tiny one instead of repeating.
        mutated.append({"type": "mouse_move", "dx": self._soft_cap_dx(random.choice([-6, 6]), max_dx=max_dx), "dy": 0, "look": True})
        return mutated[:3]

    async def run_forever(self) -> None:
        while True:
            try:
                result = await self.tick()
                log("\n=== tick ===")
                log(f"scene: {result['scene'][:220]}")
                log(f"heard: {result['heard'][:120]}")
                log(f"speak: {result['speak']}")
                log(f"actions: {result['actions']}")
            except Exception as exc:
                log(f"[error] {exc}")

            await asyncio.sleep(max(0.2, self.cfg.runtime.loop_interval_sec))

    async def say_extra_line(self) -> None:
        # F11 quick social line without changing planner logic.
        now = time.time()
        if now - self._last_manual_say_at < 0.8:
            return
        self._last_manual_say_at = now

        log("[manual] F11 waiting for scene-based line...")
        text = ""
        while not text:
            obs = self.last_observation
            if obs is None or not ((obs.scene_text or "").strip() or (obs.heard_text or "").strip()):
                try:
                    obs = await self._get_observation()
                except Exception:
                    obs = None
            text = self._build_scene_short_line(obs) if obs is not None else ""
            if text:
                break
            await asyncio.sleep(0.25)
        if len(text) > 70:
            text = text[:70]

        if self.cfg.runtime.tts_enabled:
            try:
                await asyncio.wait_for(
                    asyncio.to_thread(self.speaker.speak, text, self.cfg.runtime.dry_run),
                    timeout=8.0,
                )
            except asyncio.TimeoutError:
                log("[warn] manual speak timeout, skip TTS.")

        if self.cfg.runtime.observe_only:
            log(f"[manual] extra speak: {text}")
            return

        actions = [{"type": "chat_send", "text": text}]
        actions = self._repair_chat_actions(actions, text)
        try:
            async with self.act_lock:
                await asyncio.wait_for(
                    asyncio.to_thread(
                        self.actuator.execute,
                        actions,
                        self.cfg.runtime.dry_run,
                        self.target_hwnd,
                        self.cfg.window.title_keyword,
                    ),
                    timeout=12.0,
                )
            log(f"[manual] extra speak sent: {text}")
        except asyncio.TimeoutError:
            log("[warn] manual chat timeout, skip.")
        except Exception as exc:
            log(f"[warn] manual chat failed: {exc}")

    def _should_auto_chat(self, obs: Observation, actions: list[dict[str, Any]]) -> bool:
        if self.cfg.runtime.observe_only:
            return False
        if any(str(a.get("type", "")) == "chat_send" for a in actions):
            return False
        now = time.time()
        if now - self._last_auto_chat_at < 14.0:
            return False
        scene = (obs.scene_text or "").lower()
        heard = (obs.heard_text or "").strip()
        social_keywords = (
            "玩家", "朋友", "聊天", "房间", "角色", "avatar", "vrchat", "social", "online", "friend"
        )
        has_social_context = any(k in scene for k in social_keywords)
        if not has_social_context and not heard:
            return False
        # Natural cadence: active intent chats more often, quiet intent less often.
        activity = float(self.intent_state.get("activity_level", 0.35))
        speak_prob = 0.35 + activity * 0.45
        return random.random() < self._prob(speak_prob)

    def _build_scene_short_line(self, obs: Observation | None) -> str:
        if obs is None:
            return ""
        heard = (obs.heard_text or "").strip()
        if heard:
            heard_short = heard.replace("\n", " ").replace("\r", " ").strip()[:22]
            return f"我听到有人在说：{heard_short}，我在这边。"

        scene = (obs.scene_text or "").replace("\r", " ").replace("\n", " ").strip()
        if not scene:
            return ""
        # Strip common markdown/noise from vision outputs.
        for token in (
            "###",
            "---",
            "**",
            "好的",
            "根据您提供的图片",
            "这是对当前游戏画面的详细描述",
            "当前游戏画面描述",
            "整体场景描述",
            "可交互物体",
            "UI状态",
            "附近角色",
        ):
            scene = scene.replace(token, " ")
        scene = " ".join(scene.split())
        short = scene[:26]
        if not short:
            return ""
        return f"我在现场，看到{short}，大家继续。"

    def _ensure_reply_on_heard(
        self, obs: Observation, speak_text: str, actions: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        heard = (obs.heard_text or "").strip()
        if not heard:
            return speak_text, actions
        if any(str(a.get("type", "")) == "chat_send" for a in actions):
            return speak_text, actions
        if heard == self._last_replied_heard and (time.time() - self._last_heard_reply_at) < 12.0:
            return speak_text, actions

        heard_short = heard.replace("\r", " ").replace("\n", " ").strip()[:30]
        reply = f"收到，我听到你说“{heard_short}”，我这边在。"
        new_actions = list(actions)
        new_actions.append({"type": "chat_send", "text": reply})
        if not speak_text:
            speak_text = reply
        self._last_replied_heard = heard
        self._last_heard_reply_at = time.time()
        log(f"[chat] heard-triggered reply: {reply[:40]}")
        return speak_text, new_actions

    @staticmethod
    def _ensure_speak_to_chat(
        speak_text: str, actions: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        if not speak_text:
            return speak_text, actions
        if any(str(a.get("type", "")) == "chat_send" for a in actions):
            return speak_text, actions
        new_actions = list(actions)
        new_actions.append({"type": "chat_send", "text": speak_text})
        return speak_text, new_actions

    async def close(self) -> None:
        self._closed = True
        if self.idle_task is not None:
            self.idle_task.cancel()
            try:
                await self.idle_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            self.idle_task = None
        if self.observe_task is None:
            return
        if not self.observe_task.done():
            self.observe_task.cancel()
            try:
                await self.observe_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        self.observe_task = None
