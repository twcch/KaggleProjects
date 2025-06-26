import torch
import torch.nn as nn

class LogisticRegressionTorch(nn.Module):
    def __init__(self, input_dim: int = 1, output_dim: int = 1):
        super().__init__()
        self.linear = nn.linear(input_dim, output_dim)
        
    def forward(self, x):
        x = self.linear.__call__(x)
        x = torch.sigmoid(x)
        
        return x


