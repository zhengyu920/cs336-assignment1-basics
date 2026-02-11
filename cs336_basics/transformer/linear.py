import torch
from cs336_basics.transformer import init_param


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w = init_param.init_param(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)
