import torch.nn as nn
import torch

class Rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device = self.device,
                dtype = self.dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #计算RMS(a)
        in_dtype = x.dtype
        x_float = x.to(torch.float32)

        mear_square = torch.mean(x_float**2,dim = -1,keepdim=True)
        rms = torch.sqrt(mear_square+self.eps)
        rmsnorm = x_float/rms
        result = rmsnorm *self.weight
        return result.to(in_dtype)