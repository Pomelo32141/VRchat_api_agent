param(
    [string]$Python = "py -3",
    [string]$Config = "config/config.toml"
)

Write-Host "[preflight] running with config: $Config"
& $env:ComSpec /c "$Python main.py --config $Config --once --dry-run --no-window-picker"
