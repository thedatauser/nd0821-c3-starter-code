import requests


API_URL = "https://nd0821-c3-starter-code-2-qzor.onrender.com/predict"

payload = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

response = requests.post(API_URL, json=payload, timeout=60)
response.raise_for_status()
print(f"HTTP status: {response.status_code}")
print(f"Model inference: {response.json()}")
