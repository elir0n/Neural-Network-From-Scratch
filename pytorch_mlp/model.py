import torch.nn as nn
import torch.nn.functional as F


class NeuralNetwork(nn.Module):
    """Three-layer MLP for Fashion-MNIST classification.

    Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (LogSoftmax)
    """

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x
