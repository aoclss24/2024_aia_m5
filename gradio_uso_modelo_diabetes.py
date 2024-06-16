import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
import io
import joblib  # Para cargar el modelo

# Cargando el modelo entrenado
modelo = joblib.load("modelo_diabetes.pkl")


# Función para hacer predicciones y generar gráfico
def predecir_diabetes(embarazos, glucosa, presion, imc, edad):
    entrada = pd.DataFrame(
        [[embarazos, glucosa, presion, imc, edad]],
        columns=["Embarazos", "Glucosa", "Presion_sanguinea", "IMC", "Edad"],
    )
    prediccion = modelo.predict(entrada)
    probabilidad = modelo.predict_proba(entrada)[0][1]
    resultado = "Positivo" if prediccion[0] else "Negativo"

    # Gráfico de probabilidad
    fig, ax = plt.subplots()
    categories = ["No Diabetes", "Diabetes"]
    valores_prob = [1 - probabilidad, probabilidad]
    ax.bar(categories, valores_prob, color=["blue", "red"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probabilidad")
    ax.set_title("Distribución de Probabilidades")

    # Guardar gráfico en un buffer en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)

    # Convertir buffer a imagen PIL
    imagen_resultado = Image.open(buf)

    return resultado, round(probabilidad * 100, 2), imagen_resultado


# Creación de la interfaz de Gradio con validaciones y descripciones
inputs = [
    gr.Number(
        label="Embarazos",
        minimum=0,
        maximum=5,
        step=1,
        precision=0,
        info="Número de embarazos [0-5]",
    ),
    gr.Number(
        label="Glucosa",
        minimum=0,
        maximum=199,
        step=1,
        precision=0,
        info="Nivel de glucosa en sangre [0-199]",
    ),
    gr.Number(
        label="Presión sanguínea",
        minimum=0,
        maximum=122,
        step=1,
        precision=0,
        info="Presión sanguínea (mm Hg) [0-122]",
    ),
    gr.Number(
        label="IMC",
        minimum=0,
        maximum=67.1,
        step=0.1,
        info="Índice de masa corporal [0-67.1]",
    ),
    gr.Number(
        label="Edad",
        minimum=1,
        maximum=81,
        step=1,
        precision=0,
        info="Edad en años [1-81]",
    ),
]
outputs = [
    gr.Text(label="Resultado de predicción"),
    gr.Text(label="Probabilidad (%)"),
    gr.Image(type="pil", label="Gráfico de Probabilidad"),
]

iface = gr.Interface(
    fn=predecir_diabetes,
    inputs=inputs,
    outputs=outputs,
    title="Detección de diabetes",
)

# Desplegando la interfaz
iface.launch()
