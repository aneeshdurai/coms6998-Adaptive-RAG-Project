@echo off
REM Script to stop the Streamlit UI server
echo Stopping Streamlit server...
taskkill /F /IM streamlit.exe 2>nul
taskkill /F /FI "WINDOWTITLE eq *streamlit*" 2>nul
for /f "tokens=2" %%a in ('netstat -ano ^| findstr :8501 ^| findstr LISTENING') do (
    echo Killing process on port 8501 (PID: %%a)
    taskkill /F /PID %%a 2>nul
)
echo Done. If server is still running, check Task Manager.

