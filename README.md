# VRChat Agent

Low-frequency intent + high-frequency instinct loop VRChat agent.  
The agent observes screen/audio, plans with LLM at controlled frequency, and executes actions via OSC/local input.

## Features

- Idle instinct loop (continuous micro-actions, no per-frame LLM dependency)
- Intent gating (LLM called on scene/heard change or TTL)
- Startup preflight checks (OSC / window / audio / API)
- Runtime presets (`quiet`, `active`)
- Optional memory retrieval (`data/memory.jsonl`)
- F11 extra speak / F12 stop hotkeys

## Requirements

- Windows 10/11
- Python 3.11+
- VRChat with OSC enabled

## Install

```powershell
python -m pip install -r requirements.txt
```

## Configure

1) Copy config template:

```powershell
Copy-Item config\config.example.toml config\config.toml
```

2) Copy env template:

```powershell
Copy-Item .env.example .env
```

3) Fill `.env`:

```dotenv
SILICONFLOW_API_KEY=your_real_key
```

## Run

```powershell
python main.py --help
python main.py --preset quiet
python main.py --preset active
```

`--preflight-only`: TODO (not implemented as standalone flag yet).  
Use `scripts/run_preflight.ps1` or `scripts/run_preflight.bat` for startup preflight checks.

## VRChat Setup (short)

- Enable OSC in VRChat settings
- Keep OSC host/port aligned with config (`127.0.0.1:9000` by default)

## Notes

- This is a WIP project.
- PRs are welcome.
- Please keep contributions incremental; no architecture rewrite-only PRs.
