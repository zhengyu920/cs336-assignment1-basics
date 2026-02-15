import torch
from torch import nn
from jaxtyping import Float, Int
from cs336_basics.transformer.attention import MultiHeadSelfAttention
from cs336_basics.transformer.norm import RMSNorm
from cs336_basics.transformer.ffn import SwiGLU
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.functional import Softmax


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, d_ff: int) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_ff, d_model)
        self.muti_head_att = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta)

    def forward(self, x, token_positions):
        y = x + self.muti_head_att(self.ln1(x), token_positions)
        y += self.ffn(self.ln2(y))
        return y


class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float) -> None:
        super().__init__()
        self.embed = Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, context_length, rope_theta, d_ff) for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model)
        self.output_embed = Linear(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(self, x: Int[torch.Tensor, "batch_size sequence_length"]) -> Float[torch.Tensor, "batch_size sequence_length vocab_size"]:
        temp = self.embed(x)
        seq_len = x.shape[-1]
        token_positions = torch.arange(seq_len)
        for t in self.transformer_blocks:
            temp = t(temp, token_positions)
        temp = self.ln_final(temp)
        temp = self.output_embed(temp)
        return temp
