import torch
from torch import nn
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5,  device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.g = nn.Parameter(nn.init.trunc_normal_(torch.zeros(d_model)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) = x.size()
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(torch.square(
            x) + self.eps, dim=-1))  # (batch_size, seq_len)
        rms = rearrange(rms, "batch_size seq_len -> batch_size seq_len 1")
        g = rearrange(self.g, "d_model -> 1 d_model")
        result = x / rms * g
        return result.to(in_dtype)
