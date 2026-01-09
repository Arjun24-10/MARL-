# model.py
import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        # 1. Define the physical layers
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    # 2. ADD THIS FUNCTION - It is the "missing link" causing your error
    def forward(self, x):
        """
        This tells PyTorch how to move the 'state' through 
         the layers to get the 'Q-values'.
        """
        return self.net(x)