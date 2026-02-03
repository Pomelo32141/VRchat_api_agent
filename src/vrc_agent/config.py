from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tomllib


@dataclass
class APIConfig:
    base_url: str
    api_key: str
    timeout_sec: int = 90


@dataclass
class ModelConfig:
    vision: str
    asr: str
    planner: str


@dataclass
class WindowConfig:
    title_keyword: str = ""


@dataclass
class AudioConfig:
    enabled: bool = True
    sample_rate: int = 16000
    capture_seconds: float = 3.0


@dataclass
class RuntimeConfig:
    loop_interval_sec: float = 2.0
    dry_run: bool = True
    observe_only: bool = False
    tts_enabled: bool = True
    idle_interval_min_sec: float = 0.22
    idle_interval_max_sec: float = 0.55
    idle_hesitate_idle_prob: float = 0.16
    idle_hesitate_pause_prob: float = 0.24
    idle_look_jitter_min_deg: float = 1.0
    idle_look_jitter_max_deg: float = 3.0
    idle_look_overshoot_prob: float = 0.20
    idle_small_step_move_prob: float = 0.26
    intent_ttl_sec: float = 2.8


@dataclass
class ChatConfig:
    mode: str = "auto"  # auto | osc | hotkey
    open_key: str = "y"
    osc_host: str = "127.0.0.1"
    osc_port: int = 9000


@dataclass
class MemoryConfig:
    enabled: bool = True
    file_path: str = "data/memory.jsonl"
    max_records: int = 1000
    retrieve_top_k: int = 5


@dataclass
class PromptConfig:
    vision: str
    planner: str


@dataclass
class AgentConfig:
    api: APIConfig
    models: ModelConfig
    window: WindowConfig
    audio: AudioConfig
    runtime: RuntimeConfig
    chat: ChatConfig
    memory: MemoryConfig
    prompt: PromptConfig


def _expand_env(value: str) -> str:
    # 支持 ${ENV_VAR} 展开 / Expand ${ENV_VAR} style placeholders.
    return os.path.expandvars(value)


def load_config(path: str | Path) -> AgentConfig:
    # 加载并规范化 TOML 配置 / Load and normalize TOML config.
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    api = raw.get("api", {})
    models = raw.get("models", {})
    window = raw.get("window", {})
    audio = raw.get("audio", {})
    runtime = raw.get("runtime", {})
    chat = raw.get("chat", {})
    memory = raw.get("memory", {})
    prompt = raw.get("prompt", {})

    cfg = AgentConfig(
        api=APIConfig(
            base_url=api.get("base_url", "https://api.siliconflow.cn/v1"),
            api_key=_expand_env(api.get("api_key", "")),
            timeout_sec=int(api.get("timeout_sec", 90)),
        ),
        models=ModelConfig(
            vision=models.get("vision", "Qwen/Qwen3-VL-30B-A3B-Instruct"),
            asr=models.get("asr", "FunAudioLLM/SenseVoiceSmall"),
            planner=models.get("planner", "deepseek-ai/DeepSeek-V3.2-Exp"),
        ),
        window=WindowConfig(title_keyword=window.get("title_keyword", "")),
        audio=AudioConfig(
            enabled=bool(audio.get("enabled", True)),
            sample_rate=int(audio.get("sample_rate", 16000)),
            capture_seconds=float(audio.get("capture_seconds", 3.0)),
        ),
        runtime=RuntimeConfig(
            loop_interval_sec=float(runtime.get("loop_interval_sec", 2.0)),
            dry_run=bool(runtime.get("dry_run", True)),
            observe_only=bool(runtime.get("observe_only", False)),
            tts_enabled=bool(runtime.get("tts_enabled", True)),
            idle_interval_min_sec=float(runtime.get("idle_interval_min_sec", 0.22)),
            idle_interval_max_sec=float(runtime.get("idle_interval_max_sec", 0.55)),
            idle_hesitate_idle_prob=float(runtime.get("idle_hesitate_idle_prob", 0.16)),
            idle_hesitate_pause_prob=float(runtime.get("idle_hesitate_pause_prob", 0.24)),
            idle_look_jitter_min_deg=float(runtime.get("idle_look_jitter_min_deg", 1.0)),
            idle_look_jitter_max_deg=float(runtime.get("idle_look_jitter_max_deg", 3.0)),
            idle_look_overshoot_prob=float(runtime.get("idle_look_overshoot_prob", 0.20)),
            idle_small_step_move_prob=float(runtime.get("idle_small_step_move_prob", 0.26)),
            intent_ttl_sec=float(runtime.get("intent_ttl_sec", 2.8)),
        ),
        chat=ChatConfig(
            mode=str(chat.get("mode", "auto")).lower(),
            open_key=str(chat.get("open_key", "y")).lower(),
            osc_host=str(chat.get("osc_host", "127.0.0.1")),
            osc_port=int(chat.get("osc_port", 9000)),
        ),
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", True)),
            file_path=str(memory.get("file_path", "data/memory.jsonl")),
            max_records=int(memory.get("max_records", 1000)),
            retrieve_top_k=int(memory.get("retrieve_top_k", 5)),
        ),
        prompt=PromptConfig(
            vision=prompt.get(
                "vision",
                "Describe current game scene. Focus on interactable objects, UI status, and nearby characters.",
            ),
            planner=prompt.get(
                "planner",
                "You are controlling a game character. Return strict JSON with keys: speak (string), actions (array), next_focus (string).",
            ),
        ),
    )

    if not cfg.api.api_key:
        # 明确提示密钥缺失 / Explicit missing-key error.
        raise ValueError("Missing API key. Set [api].api_key or SILICONFLOW_API_KEY env var.")

    return cfg
