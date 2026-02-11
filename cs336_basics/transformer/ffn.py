import torch
from torch import nn
from cs336_basics.transformer.linear import Linear


class SiLU(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_ff: int, d_model: int, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.l1 = Linear(d_model, d_ff, device, dtype)
        self.l3 = Linear(d_model, d_ff, device, dtype)
        self.l2 = Linear(d_ff, d_model, device, dtype)
        self.silu = SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.silu(self.l1(x))*self.l3(x))
