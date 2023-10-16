import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from joblib import dump

df = pd.read_csv('data/seattle-weather.csv')

X = df.pop('weather')
y = df
#X = df.drop(columns=['date', 'weather'])
#y = df['weather']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Evaluar el modelo
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Guardar el modelo entrenado en un archivo
dump(clf, 'model/seattle-weather-v1.joblib')


df = pd.read_csv(pathlib.Path('data/seattle-weather.csv'))
y = df.pop('weather')
X = df

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.3)

print ('Training model.. ')
clf = RandomForestClassifier(n_estimators = 10,
                            max_depth=2,
                            random_state=0)
clf.fit(X_train, y_train)
print ('Saving model..')

dump(clf, pathlib.Path('model/heart-disease-v1.joblib'))





