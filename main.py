import torch
import datetime
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.model_utils import save_model
from src.architecture import ArchitectureCNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.visualization import plot_loss, plot_accuracy


def main() -> None:
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

    # Definir el modelo, criterio y optimizador
    model = ArchitectureCNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenar el modelo
    losses, accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, num_epochs=100
    )

    # Graficar y guardar resultados
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_loss(losses, f"training_loss_{current_time}.png")
    plot_accuracy(accuracies, f"accuracy_{current_time}.png")

    # Evaluación del modelo
    accuracy = evaluate_model(model, test_loader)
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

    # Guardar el modelo entrenado
    save_model(model, "modelo_entrenado.pth")


if __name__ == "__main__":
    main()
