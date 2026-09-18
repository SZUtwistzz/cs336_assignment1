import os
import typing

import torch.nn as nn
import torch

def save_checkpoint(model:nn.Module, 
                    optimizer:torch.optim.Optimizer, 
                    iteration,
                    out:str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    infodict = {}
    infodict["iteration"] = iteration
    infodict["model"] = model.state_dict()
    infodict["optimizer"] =optimizer.state_dict()
    torch.save(infodict,out)

def load_checkpoint(src, model, optimizer):
    # Loading on CPU also permits restoring a CUDA checkpoint without that GPU.
    infodict = torch.load(src, map_location="cpu", weights_only=True)
    model.load_state_dict(infodict["model"])
    optimizer.load_state_dict(infodict["optimizer"])

    return infodict["iteration"]
    
