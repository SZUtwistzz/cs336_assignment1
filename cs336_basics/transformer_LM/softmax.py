import torch

def softmax(tensor:torch.Tensor,dim:int):
    max_values = tensor.amax(dim,keepdim = True)
    shifted = tensor - max_values
    exp_values = torch.exp(shifted)
    sum =  torch.sum(exp_values,dim,keepdim=True)
    result = exp_values/sum
    return result
