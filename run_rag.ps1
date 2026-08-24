# 始终用项目虚拟环境运行 RAG 示例，避免系统 Python 缺包
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Py = Join-Path $Root ".venv\Scripts\python.exe"

if (-not (Test-Path $Py)) {
    Write-Host "未找到虚拟环境: $Py"
    Write-Host "请先在项目目录创建 .venv 并安装 requirements.txt。"
    exit 1
}

Set-Location $Root
& $Py (Join-Path $Root "examples\01_basic_rag.py") @args
