@echo off
echo 🚀 Setting up Gold News Sentiment Analysis System for Windows
echo.

echo 📦 Installing dependencies...
pip install -r requirements-simple.txt

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to install dependencies
    echo 🔧 Try installing manually: pip install -r requirements-simple.txt
    pause
    exit /b 1
)

echo ✅ Dependencies installed successfully
echo.

echo 📁 Creating directories...
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data" mkdir data

echo ✅ Directories created
echo.

echo 🗄️ Setting up SQLite database...
python -c "from app.models.database import create_tables; create_tables(); print('Database tables created successfully!')"

if %ERRORLEVEL% NEQ 0 (
    echo ❌ Failed to create database tables
    pause
    exit /b 1
)

echo ✅ Database setup completed
echo.

echo 🎉 Setup completed successfully!
echo.

echo 📋 Available commands:
echo   python run.py simple    - Interactive setup
echo   python run.py api       - Start API server
echo   python run.py dashboard - Start dashboard
echo   python main.py          - Start API directly
echo   streamlit run dashboard/app.py - Start dashboard directly
echo.

echo 🌐 Services will be available at:
echo   API Documentation: http://localhost:8000/docs
echo   Dashboard: http://localhost:8501
echo   Health Check: http://localhost:8000/api/v1/health
echo.

echo 💡 Tips:
echo   - Use Ctrl+C to stop services
echo   - Check logs/ directory for log files
echo   - Edit app/core/config.py to change settings
echo.

pause
