from __future__ import annotations

import argparse
import asyncio
import ctypes
import shutil
from pathlib import Path

from dotenv import load_dotenv

from vrc_agent.agent import AgentRuntime
from vrc_agent.async_log import log, start_log_worker, stop_log_worker
from vrc_agent.config import load_config
from vrc_agent.preflight import print_preflight_report, run_preflight
from vrc_agent.window_picker import pick_window_ui


def parse_args() -> argparse.Namespace:
    # 命令行参数 / CLI arguments
    p = argparse.ArgumentParser(description="VRC game agent")
    p.add_argument("--config", default="config/config.toml", help="Path to config TOML")
    p.add_argument("--dry-run", action="store_true", help="Force dry-run")
    p.add_argument("--once", action="store_true", help="Run one tick and exit")
    p.add_argument("--no-window-picker", action="store_true", help="Disable startup window picker UI")
    p.add_argument("--preset", choices=["quiet", "active"], help="Startup runtime preset")
    return p.parse_args()


def apply_runtime_preset(cfg, preset: str | None) -> None:
    # 启动预设仅覆盖 runtime 参数，不改配置文件 / Startup-only runtime override
    if preset == "quiet":
        cfg.runtime.loop_interval_sec = 2.4
        cfg.runtime.idle_interval_min_sec = 0.30
        cfg.runtime.idle_interval_max_sec = 0.70
        cfg.runtime.idle_hesitate_idle_prob = 0.28
        cfg.runtime.idle_hesitate_pause_prob = 0.34
        cfg.runtime.idle_look_jitter_min_deg = 0.8
        cfg.runtime.idle_look_jitter_max_deg = 2.0
        cfg.runtime.idle_look_overshoot_prob = 0.08
        cfg.runtime.idle_small_step_move_prob = 0.14
        cfg.runtime.intent_ttl_sec = 3.4
    elif preset == "active":
        cfg.runtime.loop_interval_sec = 1.8
        cfg.runtime.idle_interval_min_sec = 0.18
        cfg.runtime.idle_interval_max_sec = 0.45
        cfg.runtime.idle_hesitate_idle_prob = 0.10
        cfg.runtime.idle_hesitate_pause_prob = 0.18
        cfg.runtime.idle_look_jitter_min_deg = 1.2
        cfg.runtime.idle_look_jitter_max_deg = 3.4
        cfg.runtime.idle_look_overshoot_prob = 0.28
        cfg.runtime.idle_small_step_move_prob = 0.26
        cfg.runtime.intent_ttl_sec = 2.4


def ensure_project_bootstrap(config_path: Path) -> None:
    # 启动自检：若缺少 config.toml，自动从示例复制
    # Startup check: auto-create config.toml from config.example.toml when missing.
    if config_path.exists():
        return

    example_path = config_path.parent / "config.example.toml"
    if not example_path.exists():
        raise FileNotFoundError(
            f"Missing config: {config_path} and template: {example_path}. "
            "Please create config manually."
        )

    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(example_path, config_path)
    log(f"[bootstrap] created missing config from template: {config_path}")


async def _hotkey_loop(agent: AgentRuntime) -> None:
    # VK_F11 = 0x7A, VK_F12 = 0x7B
    vk_f11 = 0x7A
    vk_f12 = 0x7B
    was_f11_down = False
    was_down = False
    f11_task: asyncio.Task | None = None
    while True:
        f11_down = bool(ctypes.windll.user32.GetAsyncKeyState(vk_f11) & 0x8000)
        if f11_down and not was_f11_down:
            log("[hotkey] F11 detected, trigger extra speak...")
            if f11_task is None or f11_task.done():
                f11_task = asyncio.create_task(agent.say_extra_line())
            else:
                log("[hotkey] F11 speak already running, ignore repeated trigger.")
        was_f11_down = f11_down

        down = bool(ctypes.windll.user32.GetAsyncKeyState(vk_f12) & 0x8000)
        if down and not was_down:
            log("[hotkey] F12 detected, stopping agent...")
            if f11_task is not None and not f11_task.done():
                f11_task.cancel()
                try:
                    await f11_task
                except asyncio.CancelledError:
                    pass
            return
        was_down = down
        await asyncio.sleep(0.08)


async def _main() -> None:
    args = parse_args()
    load_dotenv()
    await start_log_worker()

    agent: AgentRuntime | None = None
    try:
        config_path = Path(args.config)
        ensure_project_bootstrap(config_path)
        cfg = load_config(config_path)
        if args.dry_run:
            cfg.runtime.dry_run = True
        apply_runtime_preset(cfg, args.preset)
        if args.preset:
            log(f"[runtime] preset={args.preset}")

        target_hwnd: int | None = None
        target_title = ""
        if not args.no_window_picker:
            choice = pick_window_ui()
            target_hwnd = choice.hwnd
            target_title = choice.title

        # Startup-only preflight checks (non-blocking): report health, never abort.
        preflight = await run_preflight(cfg, target_hwnd=target_hwnd)
        print_preflight_report(preflight)

        agent = AgentRuntime(cfg, target_hwnd=target_hwnd, target_title=target_title)
        if args.once:
            result = await agent.tick()
            log(str(result))
            return

        run_task = asyncio.create_task(agent.run_forever())
        stop_task = asyncio.create_task(_hotkey_loop(agent))
        done, pending = await asyncio.wait(
            {run_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if stop_task in done and not run_task.done():
            run_task.cancel()
            try:
                await run_task
            except asyncio.CancelledError:
                pass
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    finally:
        if agent is not None:
            await agent.close()
        await stop_log_worker()


if __name__ == "__main__":
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        print("\n[exit] interrupted by user (Ctrl+C).")
