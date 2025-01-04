import torch
from torch.utils.data import DataLoader
from model import Net
from utils import load_data, plot_image

def evaluate_model():
    """Evaluate the trained model."""
    # Load data
    _, test_data = load_data()
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # Load the trained model
    model = Net()
    model.load_state_dict(torch.load("saved_models/mnist_model.pth"))
    model.eval()  # Set the model to evaluation mode

    # Evaluate the model
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {100 * correct / total:.2f}%")

    # Plot a sample image with its predicted label
    sample_image, sample_label = test_data[0]
    output = model(sample_image.unsqueeze(0))
    _, predicted_label = torch.max(output, 1)
    plot_image(sample_image, predicted_label.item())