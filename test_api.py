#!/usr/bin/env python3
"""
Test script to verify the API server works correctly.
"""
import subprocess
import time
import requests
import json

def test_api_health():
    """Test API health endpoint."""
    print("Testing API health endpoint...")

    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"+ API is healthy: {data}")
            return True
        else:
            print(f"- API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"- API health check error: {e}")
        return False

def test_api_endpoints():
    """Test various API endpoints."""
    print("\nTesting API endpoints...")

    endpoints = [
        ("/api/v1/news", "GET", {"limit": "5"}),
        ("/api/v1/sentiment", "GET", {"limit": "5"}),
        ("/api/v1/gold-prices", "GET", {"limit": "5"}),
    ]

    for endpoint, method, params in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", params=params, timeout=10)
            if response.status_code == 200:
                print(f"+ {method} {endpoint} - OK")
            else:
                print(f"! {method} {endpoint} - Status: {response.status_code}")
        except Exception as e:
            print(f"- {method} {endpoint} - Error: {e}")

def start_api_server():
    """Start the API server in background."""
    print("Starting API server...")

    try:
        # Start the server
        proc = subprocess.Popen(
            ['python', 'main.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=r'F:\Financial Project\gold news'
        )

        # Wait a bit for startup
        print("Waiting for server to start...")
        time.sleep(5)

        # Check if process is still running
        if proc.poll() is None:
            print("+ API server started successfully")
            return proc
        else:
            # Check exit code
            stdout, stderr = proc.communicate()
            print(f"- API server failed to start: {stderr.decode()}")
            return None

    except Exception as e:
        print(f"- Error starting API server: {e}")
        return None

def main():
    """Main test function."""
    print("Testing Gold News API Server")
    print("=" * 50)

    # Start API server
    server_process = start_api_server()

    if server_process is None:
        print("- Failed to start API server")
        return

    try:
        # Test API endpoints
        time.sleep(2)  # Give server more time to fully start
        test_api_health()
        test_api_endpoints()

        print("\n" + "=" * 50)
        print("API server is working correctly!")
        print("\nAvailable endpoints:")
        print("  Health Check: http://localhost:8000/api/v1/health")
        print("  API Docs: http://localhost:8000/docs")
        print("  News: http://localhost:8000/api/v1/news")
        print("  Sentiment: http://localhost:8000/api/v1/sentiment")
        print("  Predictions: http://localhost:8000/api/v1/predictions")

    finally:
        # Clean up: stop the server
        print("\nStopping API server...")
        server_process.terminate()
        server_process.wait()
        print("+ API server stopped")

if __name__ == "__main__":
    main()
