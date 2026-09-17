import einops
import torch
from .softmax import softmax
def scaled_dot_product_attention(keys:torch.tensor,queries:torch.tensor,values:torch.tensor,mask = None):
    d_k = keys.shape[-1]
    scores = einops.einsum(queries,keys,"... n d_k,... m d_k -> ... n m")
    scores = scores / (d_k ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(~mask,float("-inf"))
    softmax_values = softmax(scores,-1)
    attention = softmax_values @ values
    return attention