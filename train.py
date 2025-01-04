import torch
from torch.utils.data import DataLoader
from torch import optim
from model import Net
from utils import load_data

def train_model():
    """Train the neural network model."""
    # Load data
    train_data, _ = load_data()
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize model, loss function, and optimizer
    model = Net()
    criterion = torch.nn.CrossEntropyLoss()  # Use torch.nn here
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Training loop
    epochs = 5
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

    # Save the trained model
    torch.save(model.state_dict(), "saved_models/mnist_model.pth")
    print("Model saved to saved_models/mnist_model.pth")