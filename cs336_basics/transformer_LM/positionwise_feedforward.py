import torch.nn as nn
import torch
from .linear import Linear
import math
class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff=None):
        super().__init__()
        self.d_model = d_model
        if d_ff is not None:
            self.d_ff = d_ff
        else:
            self.d_ff = math.ceil((8/3)*self.d_model/64)*64

        self.l1=Linear(self.d_model,self.d_ff)
        self.l2=Linear(self.d_ff,self.d_model)
        self.l3=Linear(self.d_model,self.d_ff)

    def forward(self,in_feartures:torch.tensor):
        result_of_l1  =self.l1(in_feartures)
        result = result_of_l1*torch.sigmoid(result_of_l1)*self.l3(in_feartures)
        result = self.l2(result)

        return result
