import torch

def SiLU(in_features:torch.tensor):
    result = in_features/(1+torch.exp(-1*in_features))
    return result