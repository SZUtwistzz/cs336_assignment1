import torch
import torch.nn as nn
import math
def gradient_clipping(params:nn.Parameter,M):
    if params is None:
        return

    eps = 1e-6
    Enorm2 = 0
    list_of_params = list(params)
    for param in list_of_params:
        gradient = param.grad
        if gradient is None:
            continue
        Enorm2+=torch.sum(gradient**2)
    if Enorm2!=0:
        Enorm = torch.sqrt(Enorm2)
    else: return
    
    factor = M/(Enorm+eps)
    if Enorm>=M:
        for param in list_of_params:
            if param.grad is not None:
                param.grad*=factor
    
