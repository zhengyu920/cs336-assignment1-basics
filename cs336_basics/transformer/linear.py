import torch
import math


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std = math.sqrt(2.0 / (in_features + out_features))
        self.w = torch.nn.Parameter(torch.nn.init.trunc_normal_(
            tensor=torch.zeros([out_features, in_features]),
            mean=0,
            std=self.std,
            a=3 * self.std,
            b=3 * self.std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.w.T)
