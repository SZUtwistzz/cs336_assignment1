import einops
import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(self, in_features:int,out_features:int,device = None,dtype = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.dtype = dtype

        #empty用来提前给weights留位置
        self.weight = torch.empty(self.out_features,self.in_features,device=self.device,dtype=self.dtype)
        self.weight = nn.Parameter(self.weight)

        result_of_std = math.sqrt(2/(self.in_features+self.out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std= result_of_std,
            a = -3* result_of_std,
            b=3* result_of_std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = einops.einsum(
            x, self.weight,
            "... d_in,d_out d_in -> ... d_out",
        )
        return result
    
