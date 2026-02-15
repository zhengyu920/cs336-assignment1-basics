import torch
from torch import nn
from cs336_basics.transformer.functional import Softmax
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.init_param import init_param
from jaxtyping import Bool, Float, Int
import math

from einops import einsum


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert (d_model % num_heads == 0)
        self.d_k = int(d_model / num_heads)
        self.q_proj_weight = init_param(
            self.d_k * self.num_heads, self.d_model)
        self.k_proj_weight = init_param(
            self.d_k * self.num_heads, self.d_model)
        self.v_proj_weight = init_param(
            self.d_k * self.num_heads, self.d_model)
        self.o_proj_weight = init_param(
            self.d_k * self.num_heads, self.d_model)
        self.attention = ScaledDotProductAttention()

    def forward(self, x: Float[torch.Tensor, "... seq d_model"]) -> Float[torch.Tensor, " ... sequence_length d_out"]:
        Q = einsum(x, self.q_proj_weight,
                   "... seq d_model_in, d_model_out d_model_in -> ... seq d_model_out")
        K = einsum(x, self.k_proj_weight,
                   "... seq d_model_in, d_model_out d_model_in -> ... seq d_model_out")
        V = einsum(x, self.v_proj_weight,
                   "... seq d_model_in, d_model_out d_model_in -> ... seq d_model_out")
        seq = x.shape[-2]
        mask = torch.tril(torch.ones(seq, seq)).bool()
        attentions = [self.attention(Q[..., i * self.d_k: (i+1)*self.d_k], K[..., i * self.d_k: (i+1)*self.d_k], V[..., i * self.d_k: (i+1)*self.d_k], mask)
                      for i in range(self.num_heads)]
        return einsum(torch.concat(attentions, dim=-1), self.o_proj_weight, "... seq d_model_in, d_model_out d_model_in -> ... seq d_model_out")
