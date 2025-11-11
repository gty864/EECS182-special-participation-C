import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int = 2, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)      # logits