@echo off
setlocal
set CONFIG=%1
if "%CONFIG%"=="" set CONFIG=config\config.toml
echo [preflight] running with config: %CONFIG%
py -3 main.py --config "%CONFIG%" --once --dry-run --no-window-picker
