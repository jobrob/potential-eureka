@echo off
REM Run a HEAT board game race with play-by-play output.
REM Double-click from file explorer or run from terminal.
REM Logs are saved to scripts\logs\ with a timestamp.
REM
REM Usage:
REM   scripts\run_race.bat                                  # defaults: 2 heuristic, USA, 1 lap
REM   scripts\run_race.bat --players 4 --laps 2             # 4-player 2-lap race
REM   scripts\run_race.bat --heuristic 2 --random 2         # heuristic vs random
REM   scripts\run_race.bat --seed 42                        # reproducible run
REM   scripts\run_race.bat --help                           # full options

setlocal

set SCRIPT_DIR=%~dp0
set REPO_ROOT=%SCRIPT_DIR%..

REM Create logs directory if it doesn't exist
if not exist "%SCRIPT_DIR%logs" mkdir "%SCRIPT_DIR%logs"

REM Generate log filename with timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%
set LOG_FILE=%SCRIPT_DIR%logs\race_%TIMESTAMP%.log

echo ================================================
echo   HEAT Race Runner
echo   Log will be saved to: %LOG_FILE%
echo ================================================
echo.

set PYTHONPATH=%REPO_ROOT%\src
python "%SCRIPT_DIR%run_race.py" --log "%LOG_FILE%" %*

echo.
echo ================================================
echo   Log saved to: %LOG_FILE%
echo ================================================
echo.
pause
