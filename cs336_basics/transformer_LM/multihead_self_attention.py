import einops
import torch.nn as nn
import torch
from .rope import RoPE
from .scaled_dot_product_attention import scaled_dot_product_attention
from .linear import Linear

class causal_multihead_self_attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float |None = None, max_seq_len: int |None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads ==0

        self.head_dim = self.d_model//self.num_heads

        self.ql = Linear(d_model,d_model)
        self.kl = Linear(d_model,d_model)
        self.vl = Linear(d_model,d_model)
        self.output_layer = Linear(d_model,d_model)
        if theta is not None and max_seq_len is not None:
            self.rope = RoPE(
                theta=theta,
                d_k = self.head_dim,
                max_seq_len=max_seq_len,
            )
        else :self.rope = None
        
    def forward(self,in_features:torch.Tensor,token_positions:torch.Tensor | None =None):
        Q = self.ql(in_features)
        K = self.kl(in_features)
        V = self.vl(in_features)
        Q = einops.rearrange(
            Q,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads = self.num_heads,
        )
        K = einops.rearrange(
            K,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads = self.num_heads,
        )
        V = einops.rearrange(
            V,
            "... seq (heads head_dim) -> ... heads seq head_dim",
            heads =self.num_heads,
        )

        seq_len = in_features.shape[-2]

    
        if self.rope is not None:
            if token_positions is None:
                    token_positions = torch.arange(
                        in_features.shape[-2],
                        device=in_features.device,
                    )

            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)

        mask = torch.tril(
            torch.ones(
                seq_len,seq_len,
                dtype = torch.bool,
                device= in_features.device,
            )
        )

        attention = scaled_dot_product_attention(K,Q,V,mask)
        attention = einops.rearrange(
            attention,
            "... heads seq head_dim -> ... seq (heads head_dim)"
        )

        return self.output_layer(attention)

