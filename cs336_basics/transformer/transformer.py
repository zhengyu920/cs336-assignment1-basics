import torch
from torch import nn
import jaxtyping
from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.norm import RMSNorm
from cs336_basics.transformer.ffn import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, d_ff: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_ff, d_model)
        self.muti_head_att = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta)

    def forward(self, x, token_positions):
        y = self.ln1(x)
        y = y + self.muti_head_att(y, token_positions)
        y = self.ln2(y)
        y = y + self.ffn(y)
        return y
