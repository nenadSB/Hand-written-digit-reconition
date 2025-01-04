import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    """Neural network model for handwritten digit recognition."""
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)       # Fully connected layer 2
        self.fc3 = nn.Linear(64, 10)        # Output layer (10 classes for digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input image
        x = F.relu(self.fc1(x))  # Apply ReLU activation
        x = F.relu(self.fc2(x))  # Apply ReLU activation
        x = self.fc3(x)          # Output layer (no activation)
        return x