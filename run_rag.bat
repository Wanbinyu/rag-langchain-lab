@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo  RAG demo (project venv)
echo  dir: %cd%
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv\Scripts\python.exe not found
    goto :end
)

".venv\Scripts\python.exe" "examples\01_basic_rag.py" %*
set EXITCODE=%ERRORLEVEL%

echo.
if %EXITCODE%==0 (
    echo [OK] finished
) else (
    echo [FAIL] exit code: %EXITCODE%
)

:end
echo.
echo Press any key to close...
pause >nul
endlocal
exit /b %EXITCODE%
