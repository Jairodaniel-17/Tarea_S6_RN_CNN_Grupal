import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from src.architecture import ArchitectureCNN


def load_model(path):
    model = ArchitectureCNN()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


model = load_model("./models/modelo_entrenado.pth")

# Preparar los datos
data = pd.read_excel("./Base_datos_cancer_test.xlsx")
features = data.loc[:, "Mean Radius":"Worst Fractal Dimension"]
labels = data["Diagnosis"]

# Normalización de características
mean = features.mean()
std = features.std()
features_normalized = (features - mean) / std

# Convertir a tensores de PyTorch
features_tensor = torch.tensor(
    features_normalized.values, dtype=torch.float32
).unsqueeze(1)
labels_tensor = torch.tensor(
    labels.apply(lambda x: 1 if x == "M" else 0).values, dtype=torch.float32
).unsqueeze(1)

# DataLoader
dataset = TensorDataset(features_tensor, labels_tensor)
loader = DataLoader(dataset, batch_size=1, shuffle=False)


# Función para hacer predicciones y mostrar diagnósticos
def make_predictions(model, loader, labels):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, _ = data
            output = model(inputs)
            prediction = "M" if output.item() > 0.5 else "B"
            print(
                f"ID {labels.index[i]}: Real Diagnosis = {labels[i]}, Predicted Diagnosis = {prediction}"
            )
            predictions.append(prediction)
    return predictions


# Hacer predicciones
make_predictions(model, loader, labels)
