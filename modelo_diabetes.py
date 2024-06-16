import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib  # Librería para guardar el modelo

# Lectura de datos de diabetes
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
columnas = [
    "Embarazos",
    "Glucosa",
    "Presion_sanguinea",
    "Pliegue_cutaneo",
    "Insulina",
    "IMC",
    "DPF",
    "Edad",
    "Resultado",
]
datos = pd.read_csv(url, names=columnas)

# Separación de las características y la variable objetivo
X = datos[["Embarazos", "Glucosa", "Presion_sanguinea", "IMC", "Edad"]]
y = datos["Resultado"]

# División de los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Entrenando el modelo de regresión logística
modelo = LogisticRegression(max_iter=200)
modelo.fit(X_train, y_train)

# Guardando el modelo entrenado en un archivo
joblib.dump(modelo, "modelo_diabetes.pkl")
