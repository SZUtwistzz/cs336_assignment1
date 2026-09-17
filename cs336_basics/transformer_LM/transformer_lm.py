from .transformer_block import transformer_block
from .embedding import Embedding
from .rmsnorm import Rmsnorm
from .linear import Linear
from .softmax import softmax
import torch.nn as nn
import torch

class transformer_LM(nn.Module):
    def __init__(self, 
                 vocab_size:int,
                 context_length:int,
                 num_layers,
                 d_model:int,
                 num_heads:int,
                 d_ff:int,
                 theta:float,
                ):
        super().__init__()

        self.blocks = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                theta=theta,
                max_seq_len=context_length,
            )
            for _ in range (num_layers)
        ])

        self.context_length = context_length
        
        self.embedding = Embedding(num_embeddings=vocab_size,embedding_dim=d_model)
        self.norm = Rmsnorm(d_model=d_model)
        self.linear = Linear(in_features=d_model,out_features=vocab_size)


        
    def forward(self,x:torch.Tensor):
        embedded_tokens = self.embedding(x)
        transformer_process =embedded_tokens

        for block in self.blocks:
            transformer_process = block(transformer_process)

        norm_process = self.norm(transformer_process)
        logits = self.linear(norm_process)

        return logits
