from .multihead_self_attention import causal_multihead_self_attention
from .rmsnorm import Rmsnorm
from .positionwise_feedforward import SwiGLU
import torch.nn as nn
import torch

class transformer_block(nn.Module):
    def __init__(self, d_model:int,num_heads:int,d_ff:int,theta:float,max_seq_len:int):
        super().__init__()

        self.norm1 = Rmsnorm(d_model=d_model)
        self.norm2 = Rmsnorm(d_model=d_model)
        self.SwiGLU = SwiGLU(d_model=d_model,d_ff=d_ff)
        self.multihead_layer = causal_multihead_self_attention(d_model=d_model,num_heads=num_heads,theta=theta,max_seq_len=max_seq_len)


    def forward(self,x:torch.Tensor):
        norm1_process = self.norm1(x)
        multihead_process = self.multihead_layer(norm1_process)
        halfway_result = x+multihead_process
        norm2_process = self.norm2(halfway_result)
        feedforward_process = self.SwiGLU(norm2_process)
        result = halfway_result+feedforward_process
        return result

