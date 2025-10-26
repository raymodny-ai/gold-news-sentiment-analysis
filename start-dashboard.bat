@echo off
echo Starting Gold News Sentiment Analysis Dashboard...
echo.
echo Dashboard will be available at:
echo   http://localhost:8501
echo   http://127.0.0.1:8501
echo   http://0.0.0.0:8501
echo.
echo Try all these URLs in your browser if one doesn't work
echo Press Ctrl+C to stop
echo.

cd /d "F:\Financial Project\gold news"

REM Check if Streamlit is installed
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo ❌ Streamlit is not installed
    echo Please run: pip install streamlit plotly altair
    pause
    exit /b 1
)

REM Check if we're in the right directory
if not exist "dashboard\app.py" (
    echo ❌ Cannot find dashboard/app.py
    echo Please make sure you're in the correct directory
    pause
    exit /b 1
)

echo ✅ Starting dashboard...
python -m streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true --browser.serverAddress=localhost

pause
