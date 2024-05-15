import torch
import torch.nn as nn


class ArchitectureCNN(nn.Module):
    def __init__(self):
        super(ArchitectureCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(16 * 30, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
