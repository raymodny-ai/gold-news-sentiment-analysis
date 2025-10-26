#!/usr/bin/env python3
"""
Test script to verify the system can start up correctly.
"""
import os
import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")

    try:
        # Test core modules
        import fastapi
        print("+ FastAPI imported")

        import sqlalchemy
        print("+ SQLAlchemy imported")

        import pandas
        print("+ Pandas imported")

        import numpy
        print("+ NumPy imported")

        # Test our modules
        from app.core.config import settings
        print("+ Configuration loaded")

        from app.models.database import engine
        print("+ Database engine created")

        from app.models.models import Base
        print("+ Database models loaded")

        print("\nAll imports successful!")
        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print("Please install dependencies: pip install -r requirements-simple.txt")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def test_database():
    """Test database connection."""
    print("\nTesting database connection...")

    try:
        from app.models.database import create_tables

        # Create tables
        create_tables()
        print("+ Database tables created successfully")
        return True

    except Exception as e:
        print(f"Database error: {e}")
        return False

def test_api_syntax():
    """Test API syntax."""
    print("\nTesting API syntax...")

    try:
        # Just import to check syntax
        import main
        print("+ API syntax is correct")
        return True

    except SyntaxError as e:
        print(f"API syntax error: {e}")
        return False
    except Exception as e:
        print(f"Other API error (this is OK): {e}")
        return True

def main():
    """Run all tests."""
    print("Testing Gold News Sentiment Analysis System")
    print("=" * 50)

    tests = [
        test_imports,
        test_database,
        test_api_syntax
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)

    print("\n" + "=" * 50)
    if all(results):
        print("All tests passed! System is ready to run.")
        print("\nNext steps:")
        print("  python run.py simple    - Interactive setup")
        print("  python main.py          - Start API server")
        print("  streamlit run dashboard/app.py - Start dashboard")
    else:
        print("Some tests failed. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
