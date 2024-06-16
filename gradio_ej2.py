# Uso de Gradio para la interfaz de prueba de un modelo de predicción.

import gradio as gr
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Simulación de datos de ejemplo
def generar_datos():
    np.random.seed(42)
    X = pd.DataFrame(
        {
            "superficie": np.random.randint(50, 200, 100),
            "no_cuartos": np.random.randint(1, 6, 100),
            "antiguedad": np.random.randint(0, 30, 100),
        }
    )
    y = (
        X["superficie"] * 3000
        + X["no_cuartos"] * 50000
        - X["antiguedad"] * 1000
        + np.random.normal(0, 25000, 100)
    )
    return X, y


# Generación de datos
X, y = generar_datos()

# Dividiendo datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Paso 2. preprocesamiento, escalando características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenando el modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)


# Función para hacer predicciones
def predecir_precio(superficie, cuartos, antiguedad):
    input_data = pd.DataFrame(
        {
            "superficie": [superficie],
            "no_cuartos": [cuartos],
            "antiguedad": [antiguedad],
        }
    )
    input_data_transf = scaler.transform(input_data)
    # Paso 3 y 4: paso al modelo y procesamiento
    prediccion = model.predict(input_data_transf)[0]
    # Paso 5: postprocesamiento
    categoria = categorizar_precio_casa(prediccion)
    # Paso 6: resultado
    return f"Predicción del precio: ${prediccion:,.2f} ({categoria})"


# Función de postprocesamiento
def categorizar_precio_casa(precio):
    if precio < 200000:
        return "Bajo"
    elif 200000 <= precio < 500000:
        return "Medio"
    else:
        return "Alto"


# Configurando interfaz de Gradio
iface = gr.Interface(
    fn=predecir_precio,
    inputs=[
        gr.Number(label="Superficie (m²)"),
        gr.Number(label="Número de habitaciones"),
        gr.Number(label="Antigüedad (años)"),
    ],
    outputs="text",
    title="Predicción de precios de casas",
    description="Ingrese las características de la casa para predecir su precio.",
)


# Ejecutar la interfaz
iface.launch()
