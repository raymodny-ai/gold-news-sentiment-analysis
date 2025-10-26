#!/usr/bin/env python3
"""
Test script to verify the Streamlit dashboard works correctly.
"""
import subprocess
import time
import requests

def test_streamlit_import():
    """Test if Streamlit can import the dashboard."""
    print("Testing Streamlit dashboard...")

    try:
        import streamlit
        print("+ Streamlit is available")

        # Test dashboard import
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        try:
            import dashboard.app
            print("+ Dashboard app can be imported")
            return True
        except Exception as e:
            print(f"Error importing dashboard: {e}")
            return False

    except ImportError:
        print("Streamlit is not installed")
        print("Please run: pip install streamlit plotly altair")
        return False

def test_streamlit_startup():
    """Test if Streamlit can start the dashboard."""
    print("\nTesting Streamlit startup...")

    try:
        # Start Streamlit
        proc = subprocess.Popen(
            ['python', '-m', 'streamlit', 'run', 'dashboard/app.py', '--server.port=8501', '--server.address=0.0.0.0'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=r'F:\Financial Project\gold news'
        )

        print("Waiting for Streamlit to start...")
        time.sleep(5)

        # Check if process is still running
        if proc.poll() is None:
            print("+ Streamlit dashboard started successfully")
            print("Dashboard should be available at:")
            print("   http://localhost:8501")
            print("   http://127.0.0.1:8501")

            # Test if we can connect
            try:
                response = requests.get("http://localhost:8501", timeout=5)
                if response.status_code == 200:
                    print("+ Dashboard is responding to HTTP requests")
                else:
                    print(f"Warning: Dashboard responded with status: {response.status_code}")
            except requests.exceptions.RequestException:
                print("Warning: Dashboard may not be fully ready yet (this is normal)")

            # Stop the process
            proc.terminate()
            proc.wait()
            print("+ Streamlit stopped")
            return True
        else:
            # Check exit code
            stdout, stderr = proc.communicate()
            print(f"Streamlit failed to start: {stderr.decode()}")
            return False

    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        return False

def main():
    """Main test function."""
    print("Streamlit Dashboard Test")
    print("=" * 50)

    # Test imports
    import_ok = test_streamlit_import()

    if not import_ok:
        print("\nDashboard import failed")
        return

    # Test startup
    startup_ok = test_streamlit_startup()

    print("\n" + "=" * 50)
    if startup_ok:
        print("Dashboard test completed successfully!")
        print("\nTo start the dashboard manually:")
        print("  cd 'F:\\Financial Project\\gold news'")
        print("  python start_dashboard.py")
        print("\nOr visit these URLs:")
        print("  http://localhost:8501")
        print("  http://127.0.0.1:8501")
        print("  http://0.0.0.0:8501")
        print("\nTry all URLs in your browser if one doesn't work")
    else:
        print("Dashboard test failed")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the correct directory:")
        print("   cd 'F:\\Financial Project\\gold news'")
        print("2. Install required packages:")
        print("   pip install streamlit plotly altair")
        print("3. Start the dashboard:")
        print("   streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
