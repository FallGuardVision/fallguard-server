"""
Test client for Phase 3 API.
Run phase3_api.py first (uvicorn phase3_api:app --reload), then run this
in a separate terminal.
"""

import numpy as np
import requests

url = "http://127.0.0.1:8000/predict"

dummy_window = np.random.rand(30, 132).tolist()

response = requests.post(url, json={"window": dummy_window})

print("Status code:", response.status_code)
print("Response JSON:")
print(response.json())