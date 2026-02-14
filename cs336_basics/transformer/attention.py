import torch
from torch import nn
from cs336_basics.transformer.functional import Softmax
from jaxtyping import Bool, Float, Int
import math

from einops import einsum


class ScaledDotProductAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.softmax = Softmax()

    def forward(self,
                Q: Float[torch.Tensor, " ... queries d_k"],
                K: Float[torch.Tensor, " ... keys d_k"],
                V: Float[torch.Tensor, " ... values d_v"],
                mask: Bool[torch.Tensor, " ... queries keys"] | None = None
                ) -> Float[torch.Tensor, " ... queries d_v"]:
        d_k = Q.shape[-1]
        qk = einsum(
            Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
        qk = qk/math.sqrt(d_k)
        if (mask is not None):
            qk = qk.masked_fill(~mask, -torch.inf)
        sm = self.softmax(qk, -1)
        return einsum(sm, V, "... queries keys, ... keys d_v -> ... queries d_v")
