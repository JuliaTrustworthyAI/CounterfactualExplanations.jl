from torch import nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2, 32),
            nn.Sigmoid(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.model(x)