#!/usr/bin/env python3
"""
Test API with module reload
"""
import subprocess
import time
import requests

print('Starting API server with module reload...')
proc = subprocess.Popen(['python', 'main.py'], cwd=r'F:\Financial Project\gold news')

time.sleep(5)

try:
    response = requests.get('http://127.0.0.1:8000/api/v1/news?limit=5')
    print(f'News API status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Found {len(data.get("data", []))} news articles')
        print('Success!')
    else:
        print(f'Error: {response.text}')
finally:
    proc.terminate()
    proc.wait()

