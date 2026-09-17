import torch.nn as nn
import torch
import math

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theat = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        positions = torch.arange(
            max_seq_len,
            device=self.device,
            dtype = torch.float32,
        )

        pair_indices = torch.arange(
            d_k//2,
            device= self.device,
            dtype = torch.float32,
        )

        frequencies = self.theat ** (-2 * pair_indices / d_k)
        angles = positions[:,None] * frequencies[None,:]
        cos_values = torch.cos(angles)
        sin_values = torch.sin(angles)

        self.register_buffer("cos_cached",cos_values,persistent=False)
        self.register_buffer("sin_cached",sin_values,persistent=False)

        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        even = x[...,0::2]
        odd = x[...,1::2]

        even_rotated = even * cos - odd*sin
        odd_rotated = even * sin + odd *cos

        result = torch.stack([even_rotated,odd_rotated],dim=-1)
        result = result.flatten(start_dim = -2,end_dim = -1)
        return result