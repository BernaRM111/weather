import requests

# URL del endpoint de predicción
url = "http://localhost:8000/predict"

# Datos de entrada para la predicción
data = {
    "precipitation": 10.9,
    "temp_max": 10.6,
    "temp_min": 2.8,
    "wind": 4.5
}

# Realizar la solicitud POST
response = requests.post(url, json=data)

# Mostrar la respuesta
if response.status_code == 200:
    result = response.json()
    predicted_weather = result["predicted_weather"]
    print(f"Predicted weather: {predicted_weather}")
else:
    print(f"Error: {response.status_code}")