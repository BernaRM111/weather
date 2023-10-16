import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump

# Cargar datos
weather_df = pd.read_csv("data/seattle-weather.csv")

# Preparar datos
X = weather_df[['precipitation', 'temp_max', 'temp_min', 'wind']]
y = weather_df['weather']

# Dividir datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Inicializar y entrenar el modelo
print('Training model..')
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)

# Evaluar el modelo
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))

# Guardar el modelo entrenado
print('Saving model..')
dump(model, 'model/seattle-weather-v1.joblib')
