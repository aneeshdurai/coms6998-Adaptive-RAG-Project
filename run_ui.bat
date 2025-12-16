@echo off
REM Simple batch file to run the UI using venv Python
cd /d "%~dp0"
call venv\Scripts\python.exe -m streamlit run app.py --server.headless false
pause

