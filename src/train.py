from src.evaluate import evaluate_model


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs):
    model.train()
    loss_values = []
    accuracy_values = []
    for epoch in range(num_epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
        loss_values.append(loss.item())
        # Llamar a una función para calcular la exactitud al final de cada época
        accuracy = evaluate_model(model, test_loader)
        accuracy_values.append(accuracy)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, Accuracy: {accuracy}%")
    return loss_values, accuracy_values
