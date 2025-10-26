#!/usr/bin/env python3
"""
Simple script to start the Streamlit dashboard.
"""
import subprocess
import sys
import os

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("Starting Gold News Sentiment Analysis Dashboard...")
    print("Dashboard will be available at:")
    print("  http://localhost:8501")
    print("  http://127.0.0.1:8501")
    print("  http://0.0.0.0:8501")
    print()
    print("Try all these URLs in your browser if one doesn't work")
    print("Press Ctrl+C to stop")
    print()

    try:
        # Change to the correct directory
        os.chdir(r'F:\Financial Project\gold news')

        # Start Streamlit
        cmd = [
            'python', '-m', 'streamlit', 'run', 'dashboard/app.py',
            '--server.port=8501',
            '--server.address=0.0.0.0',
            '--server.headless=true',
            '--server.enableCORS=false',
            '--browser.serverAddress=localhost'
        ]

        print(f"Running: {' '.join(cmd)}")
        print()

        # Run Streamlit
        subprocess.run(cmd, check=True)

    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error starting dashboard: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure you're in the correct directory")
        print("2. Install required packages: pip install streamlit plotly altair")
        print("3. Try: python -m streamlit run dashboard/app.py")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    """Main function."""
    print("=" * 60)
    print("Gold News Sentiment Analysis Dashboard")
    print("=" * 60)
    print()

    # Check if we're in the right directory
    current_dir = os.getcwd()
    project_dir = r'F:\Financial Project\gold news'

    if not current_dir.endswith('gold news'):
        print(f"Current directory: {current_dir}")
        print(f"Expected directory: {project_dir}")
        print()
        print("Please run this script from the project directory:")
        print(f"cd '{project_dir}'")
        print(f"python {os.path.basename(__file__)}")
        return

    # Check dependencies
    try:
        import streamlit
        print("✓ Streamlit is available")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("Please run: pip install streamlit plotly altair")
        return

    try:
        import plotly
        print("✓ Plotly is available")
    except ImportError:
        print("❌ Plotly is not installed")
        print("Please run: pip install plotly")
        return

    print("✓ All dependencies are available")
    print()

    # Start dashboard
    start_dashboard()

if __name__ == "__main__":
    main()
