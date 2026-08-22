import requests

response = requests.post(
    "http://127.0.0.1:8000/optimize",
    json={"prompt": "A real prompt to check.", "goal": "To do a check."}
)

print(f" Status: {response.status_code}")
print(f"response: {response.json()}")