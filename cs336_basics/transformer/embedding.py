import torch
from torch import nn
import math


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype
        std = math.sqrt(2 / (num_embeddings + embedding_dim))
        eb_init = nn.init.trunc_normal_(tensor=torch.zeros([num_embeddings, embedding_dim]),
                                        mean=0,
                                        std=std,
                                        a=-3*std,
                                        b=3*std)
        self.embeddings = nn.Parameter(eb_init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
