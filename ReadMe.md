# TAREA 6

## Descripción del Proyecto

El objetivo de este proyecto es entrenar, evaluar y utilizar un modelo de Red Neuronal Convolucional (CNN) para clasificar datos del conjunto de datos de cáncer de mama proporcionado por `sklearn`. Este proyecto está estructurado de manera modular para facilitar su mantenimiento, escalabilidad y reutilización.
Este proyecto incluye varios scripts de Python para entrenar, evaluar y utilizar un modelo de aprendizaje automático.

## Scripts

- [`main.py`](./src/main.py): Este es el script principal que utiliza las funciones de los otros scripts.
- [`architecture.py`](./src/architecture.py): Define la arquitectura del modelo.
- [`evaluate.py`](./src/evaluate.py): Contiene la función [`evaluate_model`](./src/evaluate.py) para evaluar el rendimiento del modelo.
- [`model_utils.py`](./src/model_utils.py): Contiene las funciones [`load_model`](./src/model_utils.py) y [`save_model`](./src/model_utils.py) para cargar y guardar el modelo.
- [`train.py`](./src/train.py): Contiene la función [`train_model`](./src/train.py) para entrenar el modelo.
- [`visualization.py`](./src/visualization.py): Contiene funciones para visualizar los resultados.


## Descripción del Modelo de CNN

**Red Neuronal Convolucional (CNN):**

- **Arquitectura**: La arquitectura del modelo se define en el archivo `architecture.py` y consiste en una única capa convolucional seguida de capas totalmente conectadas (fully connected). Específicamente:
  - Una capa convolucional 1D (`Conv1d`) con 16 filtros y un tamaño de kernel de 3.
  - Una capa totalmente conectada que toma la salida de la capa convolucional aplanada.
  - Una capa de salida que utiliza una activación sigmoide para producir una probabilidad de clase binaria (0 o 1).

- **Propósito**: La CNN entrenada está diseñada para clasificar si un tumor de mama es maligno o benigno basado en un conjunto de características extraídas de imágenes de biopsias de mama.

### Funcionalidad

1. **Entrenamiento**: El script `train.py` se encarga de entrenar el modelo utilizando los datos de entrenamiento. Se ajustan los pesos del modelo mediante el algoritmo de optimización Adam y se mide la pérdida utilizando la Entropía Cruzada Binaria (`BCELoss`). Durante el entrenamiento, se registran las pérdidas y la exactitud del modelo.

2. **Evaluación**: El script `evaluate.py` contiene una función para evaluar el modelo en un conjunto de datos de prueba, calculando la exactitud del modelo.

3. **Visualización**: El script `visualization.py` genera gráficos de la pérdida y la exactitud del modelo a lo largo de las épocas de entrenamiento, lo que permite una mejor comprensión del proceso de entrenamiento y el desempeño del modelo.

4. **Guardado y Carga del Modelo**: El script `model_utils.py` incluye funciones para guardar el estado del modelo entrenado (`state_dict`) y cargarlo posteriormente, facilitando la reutilización del modelo sin necesidad de volver a entrenarlo desde cero.

5. **Exportación a ONNX**: El modelo también se puede exportar al formato ONNX, que es un formato estándar abierto para modelos de aprendizaje automático. Esto permite que el modelo sea utilizado en diferentes frameworks y entornos compatibles con ONNX.

### Uso

- **Aplicación**: El modelo puede ser utilizado para predecir la malignidad de tumores de mama en nuevos datos no vistos, proporcionando una herramienta valiosa para ayudar en el diagnóstico médico.
- **Reutilización**: Gracias a su modularidad y capacidad de exportación a ONNX, el modelo se puede integrar fácilmente en diferentes aplicaciones o entornos, incluyendo aplicaciones móviles, servicios web, y otros sistemas de soporte de decisiones médicas.

### Estructura de Carpetas

```plaintext
proyecto/
│
├── models/                   # Carpeta para guardar modelos entrenados
│   └── modelo_entrenado.pth
│
├── plots/                    # Carpeta para guardar gráficos generados
│   ├── accuracy_{fecha}_{hora_minuto_segundo}.png
│   └── training_loss_{fecha}_{hora_minuto_segundo}.png
│
└── src/                      # Carpeta para todos los scripts de Python
    ├── architecture.py       # Archivo para definir la arquitectura del modelo
    ├── evaluate.py           # Archivo para funciones de evaluación
    ├── main.py               # Script principal que ejecuta el programa
    ├── model_utils.py        # Funciones para guardar y cargar modelos
    ├── train.py              # Funciones de entrenamiento del modelo
    └── visualization.py      # Funciones para visualizar resultados
```
