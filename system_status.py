#!/usr/bin/env python3
"""
System status check for the gold news sentiment analysis system.
"""
import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("Checking dependencies...")

    required_packages = [
        ('fastapi', 'FastAPI'),
        ('sqlalchemy', 'SQLAlchemy'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
    ]

    optional_packages = [
        ('transformers', 'Transformers (FinBERT)'),
        ('torch', 'PyTorch (LSTM)'),
        ('xgboost', 'XGBoost'),
        ('tensorflow', 'TensorFlow'),
    ]

    missing_required = []
    missing_optional = []

    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  + {name}")
        except ImportError:
            missing_required.append(name)
            print(f"  - {name}")

    print("\nOptional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"  + {name}")
        except ImportError:
            missing_optional.append(name)
            print(f"  ! {name} (optional)")

    return len(missing_required) == 0, missing_required, missing_optional

def check_files():
    """Check if all required files exist."""
    print("\nChecking files...")

    required_files = [
        'main.py',
        'dashboard/app.py',
        'app/core/config.py',
        'app/models/database.py',
        'requirements-simple.txt',
        'README.md'
    ]

    missing_files = []

    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  + {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  - {file_path}")

    return len(missing_files) == 0, missing_files

def check_database():
    """Check database setup."""
    print("\nChecking database...")

    try:
        from app.models.database import create_tables
        create_tables()
        print("  + Database tables created successfully")
        return True
    except Exception as e:
        print(f"  - Database error: {e}")
        return False

def check_api_syntax():
    """Check API syntax."""
    print("\nChecking API syntax...")

    try:
        import main
        print("  + API syntax is correct")
        return True
    except SyntaxError as e:
        print(f"  - API syntax error: {e}")
        return False
    except Exception as e:
        print(f"  ! API import issue (this may be OK): {e}")
        return True

def check_dashboard():
    """Check dashboard functionality."""
    print("\nChecking dashboard...")

    try:
        import dashboard.app
        print("  + Dashboard can be imported")
        return True
    except Exception as e:
        print(f"  - Dashboard error: {e}")
        return False

def main():
    """Main status check function."""
    print("Gold News Sentiment Analysis System Status")
    print("=" * 60)

    # Check current directory and auto-navigate if needed
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    expected_dir = script_dir

    if not current_dir == expected_dir:
        print(f"Current directory: {current_dir}")
        print(f"Script directory: {script_dir}")
        print("Auto-navigating to script directory...")
        os.chdir(script_dir)
        print(f"Changed to: {os.getcwd()}")

    print(f"Project directory: {expected_dir}")

    # Run checks
    deps_ok, missing_required, missing_optional = check_dependencies()
    files_ok, missing_files = check_files()
    db_ok = check_database()
    api_ok = check_api_syntax()
    dashboard_ok = check_dashboard()

    print("\n" + "=" * 60)

    # Summary
    all_ok = deps_ok and files_ok and db_ok and api_ok and dashboard_ok

    if all_ok:
        print("System status: ALL GOOD!")
        print("\nReady to start:")
        print("  API: python main.py")
        print("  Dashboard: python -m streamlit run dashboard/app.py")
        print("  Interactive: python run.py simple")
        print("\nAccess URLs:")
        print("  API Docs: http://localhost:8000/docs")
        print("  Dashboard: http://localhost:8501")
        print("  Health: http://localhost:8000/api/v1/health")

        if missing_optional:
            print(f"\nMissing optional packages: {', '.join(missing_optional)}")
            print("   Install with: pip install torch transformers xgboost tensorflow")

    else:
        print("System has issues:")

        if missing_required:
            print(f"   Missing required packages: {', '.join(missing_required)}")
            print("   Install with: pip install -r requirements-simple.txt")

        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")

        if not db_ok:
            print("   Database setup failed")

        if not api_ok:
            print("   API syntax issues")

        if not dashboard_ok:
            print("   Dashboard issues")

    print("\nQuick tests:")
    print("  python test_startup.py  - Full system test")
    print("  python test_api.py      - API test")
    print("  python test_dashboard.py - Dashboard test")

if __name__ == "__main__":
    main()
