# 启动本地 Web 控制台。关闭窗口或按 Ctrl+C 可停止服务。
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "未找到虚拟环境: $Py"
    exit 1
}

Set-Location $Root
& $Py -m src.web @args
