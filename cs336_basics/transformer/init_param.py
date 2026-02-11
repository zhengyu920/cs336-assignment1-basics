import torch
import math


def init_param(in_features: int, out_features: int) -> torch.nn.Parameter:
    std = math.sqrt(2.0 / (in_features + out_features))
    return torch.nn.Parameter(torch.nn.init.trunc_normal_(
        tensor=torch.zeros([out_features, in_features]),
        mean=0,
        std=std,
        a=-3 * std,
        b=3 * std))
