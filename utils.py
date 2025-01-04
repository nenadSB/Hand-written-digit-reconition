import matplotlib.pyplot as plt
from torchvision import datasets, transforms

def load_data():
    """Load the MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values
    ])

    # Download and load the training data
    train_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )

    # Download and load the test data
    test_data = datasets.MNIST(
        root="data", train=False, download=True, transform=transform
    )

    return train_data, test_data

def plot_image(image, label):
    """Plot a single image with its label."""
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.show()