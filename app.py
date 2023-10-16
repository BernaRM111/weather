from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import pandas as pd

app = FastAPI(title= 'Weather Prediction')

# Cargar el modelo entrenado
model = load('model/seattle-weather-v1.joblib')

class WeatherInput(BaseModel):
    precipitation: float
    temp_max: float
    temp_min: float
    wind: float

class WeatherOutput(BaseModel):
    predicted_weather: str

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

@app.post("/predict", response_model=WeatherOutput)
def predict_weather(data: WeatherInput):
    # Crear un DataFrame a partir de los datos de entrada
    input_data = pd.DataFrame([data.dict()])

    # Realizar la predicción usando el modelo cargado
    prediction = model.predict(input_data[['precipitation', 'temp_max', 'temp_min', 'wind']])

    # Obtener la etiqueta de la predicción
    predicted_weather = prediction[0]

    # Devolver la predicción como respuesta JSON
    return {"predicted_weather": predicted_weather}