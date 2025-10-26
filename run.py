#!/usr/bin/env python3
"""
Quick start script for the gold news sentiment analysis system.
"""
import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed."""
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]

    missing_packages = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✅ {name} is available")
        except ImportError:
            missing_packages.append(name)
            print(f"❌ {name} is missing")

    if missing_packages:
        print(f"\n📦 Missing packages: {', '.join(missing_packages)}")
        print("Please install them:")
        print("   pip install -r requirements.txt")
        print("   or for Windows: pip install -r requirements-windows.txt")
        return False

    print("✅ All required dependencies are available")
    return True

def start_development():
    """Start development environment."""
    print("🚀 Starting development environment...")

    # Check dependencies
    if not check_requirements():
        return

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    print("📁 Created necessary directories")

    # Start services using Docker Compose (if available)
    try:
        print("🐳 Starting Docker services...")
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("✅ Docker services started")
        print("")
        print("📋 Services available at:")
        print("   🌐 API Documentation: http://localhost:8000/docs")
        print("   📊 Dashboard: http://localhost:8501")
        print("   🔧 Celery Monitor: http://localhost:5555")
        print("   🏥 Health Check: http://localhost:8000/api/v1/health")
        print("")
        print("💡 Use 'docker-compose logs -f' to view logs")
        print("💡 Use 'docker-compose down' to stop services")

    except subprocess.CalledProcessError:
        print("❌ Failed to start Docker services")
        print("Please ensure Docker and Docker Compose are installed and running")
        print("")
        print("🔧 Manual setup:")
        print("1. Start PostgreSQL and Redis")
        print("2. Run: python main.py")
        print("3. Run: streamlit run dashboard/app.py")
        print("4. Run: celery -A app.tasks.celery_config worker --loglevel=info")

def start_api_only():
    """Start only the API service."""
    print("🚀 Starting API service only...")

    if not check_requirements():
        return

    print("🔧 Starting FastAPI server...")
    print("📚 API Documentation will be available at: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/api/v1/health")
    print("")
    print("Press Ctrl+C to stop")
    print("")

    try:
        # Run the API server
        import subprocess
        subprocess.run(["python", "main.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 API server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ API server failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def start_dashboard_only():
    """Start only the dashboard service."""
    print("🚀 Starting Dashboard only...")

    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Please run: pip install -r requirements-simple.txt")
        return

    print("📊 Starting Streamlit dashboard...")
    print("📈 Dashboard will be available at: http://localhost:8501")
    print("")
    print("Press Ctrl+C to stop")
    print("")

    try:
        # Run the dashboard
        import subprocess
        subprocess.run(["streamlit", "run", "dashboard/app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Dashboard failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def start_simple():
    """Start with simplified setup (SQLite only, no Redis/Celery)."""
    print("🚀 Starting simplified setup...")
    print("📦 This uses SQLite database (no PostgreSQL/Redis required)")
    print("")

    if not check_requirements():
        return

    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)

    print("📁 Created necessary directories")
    print("")

    # Ask user what to start
    print("What would you like to start?")
    print("1. API Server only")
    print("2. Dashboard only")
    print("3. Both (API + Dashboard)")

    choice = input("Enter your choice (1-3): ").strip()

    if choice == "1":
        start_api_only()
    elif choice == "2":
        start_dashboard_only()
    elif choice == "3":
        print("🔧 Starting both API and Dashboard...")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("📊 Dashboard: http://localhost:8501")
        print("")
        print("Note: Start these in separate terminals:")
        print("  Terminal 1: python run.py api")
        print("  Terminal 2: python run.py dashboard")
        print("")
        start_api_only()
    else:
        print("❌ Invalid choice")


def main():
    """Main function."""
    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "simple":
            start_simple()
        elif command == "dev":
            start_development()
        elif command == "api":
            start_api_only()
        elif command == "dashboard":
            print("🚀 Starting Dashboard only...")
            try:
                import streamlit
                print("✅ Streamlit is available")
            except ImportError:
                print("❌ Streamlit is not installed")
                print("Please run: pip install streamlit plotly altair")
                return

            print("📊 Starting Streamlit dashboard...")
            print("📈 Dashboard will be available at:")
            print("   http://localhost:8501")
            print("   http://127.0.0.1:8501")
            print("   http://0.0.0.0:8501")
            print("")
            print("Try all these URLs in your browser if one doesn't work")
            print("Press Ctrl+C to stop")
            print("")

            try:
                # Run the dashboard
                import subprocess
                subprocess.run(['python', '-m', 'streamlit', 'run', 'dashboard/app.py',
                               '--server.port=8501', '--server.address=0.0.0.0',
                               '--server.headless=true'], check=True)
            except KeyboardInterrupt:
                print("\n👋 Dashboard stopped by user")
            except subprocess.CalledProcessError as e:
                print(f"❌ Dashboard failed: {e}")
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
        elif command == "docker":
            print("🐳 Using Docker...")
            os.system("docker-compose up -d")
        elif command == "test":
            print("🧪 Running tests...")
            os.system("pytest")
        elif command == "test-api":
            print("Testing API server...")
            os.system("python test_api.py")
        elif command == "status":
            print("System status check...")
            os.system("python system_status.py")
        elif command == "troubleshoot":
            print("Opening troubleshooting guide...")
            print("Please see troubleshooting.md for detailed help")
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: simple, dev, api, dashboard, docker, test, test-api, status")
    else:
        print("🟡 Gold News Sentiment Analysis System")
        print("📈 Real-time sentiment analysis and gold price predictions")
        print("")
        print("🚀 Quick Start Options:")
        print("  python run.py simple    - Simple setup (SQLite, no external DB)")
        print("  python run.py dev       - Full development (Docker)")
        print("  python run.py api       - API server only")
        print("  python run.py dashboard - Dashboard only")
        print("  python run.py test      - Run unit tests")
        print("  python run.py test-api  - Test API server functionality")
        print("  python run.py status    - Check system status")
        print("  python start_dashboard.py - Start dashboard with better error handling")
        print("")
        print("📦 Installation:")
        print("  pip install -r requirements-simple.txt  # For simple setup")
        print("  pip install -r requirements.txt         # For full setup")
        print("  pip install -r requirements-windows.txt # For Windows")
        print("")
        print("📖 See README.md for detailed setup instructions")

if __name__ == "__main__":
    main()
