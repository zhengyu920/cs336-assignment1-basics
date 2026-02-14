import torch
from torch import nn


class Softmax(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, i: int) -> torch.Tensor:
        x = x - x.max(dim=i, keepdim=True).values
        return torch.exp(x) / torch.sum(torch.exp(x), dim=i, keepdim=True)
