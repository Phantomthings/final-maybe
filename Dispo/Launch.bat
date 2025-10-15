@echo off
cd /d %~dp0

REM 
start "" streamlit run App.py --server.address=127.0.0.1 --server.port=8501

REM 
timeout /t 2 > nul

REM 
explorer "http://127.0.0.1:8501"
