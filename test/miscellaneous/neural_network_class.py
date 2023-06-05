import os
import torch
from torch import nn
from torch.utils.data import DataLoader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
                torch.nn.Flatten(),
                torch.nn.Linear(2, 32),
                torch.nn.Sigmoid(),
                torch.nn.Linear(32, 2)
            )

    def forward(self, x):
        return self.model(x)