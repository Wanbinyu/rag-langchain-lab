@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo ERROR: .venv\Scripts\python.exe not found
  pause
  exit /b 1
)

".venv\Scripts\python.exe" -m src.web %*
endlocal
