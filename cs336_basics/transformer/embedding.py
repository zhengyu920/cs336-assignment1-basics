import torch
from torch import nn
from cs336_basics.transformer import init_param
import math
from einops import rearrange, einsum


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        self.embeddings = init_param.init_param(num_embeddings, embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        rotations = []
        for i in range(max_seq_len):
            r_i = []
            for k in range(1, int(d_k/2) + 1):
                theta_i_k = i / math.pow(theta, (2*k-2)/d_k)
                r_i_k = torch.Tensor([
                    [math.cos(theta_i_k), -math.sin(theta_i_k)],
                    [math.sin(theta_i_k), math.cos(theta_i_k)]])
                r_i.append(r_i_k)
            rotations.append(torch.block_diag(*r_i))
        self.register_buffer('rope',
                             torch.stack(rotations),
                             persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        rotations = self.rope[token_positions, :, :]
        rotations = rearrange(
            rotations, "... seq_len d_k_1 d_k_2 -> ... seq_len d_k_2 d_k_1")
        return einsum(x, rotations, "... seq_len d_k_in, ... seq_len d_k_in d_k_out -> ... seq_len d_k_out")
