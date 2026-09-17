import numpy as np
import torch

def data_loader(x:np.ndarray,batch_size:int,context_length:int,device="cuda" if torch.cuda.is_available() else "cpu"):
    inputs = []
    targets = []

    starts = np.random.randint(0,len(x)-context_length,batch_size)
    for start in starts:
        targeted_start = start+1
        inputs.append(x[start:start+context_length])
        targets.append(x[targeted_start:targeted_start+context_length])

    inputs = np.stack(inputs,axis=0)
    targets = np.stack(targets,axis=0)
    inputs_to_tensor = torch.tensor(inputs,dtype = torch.long,device = device)
    targets_to_tensor = torch.tensor(targets,dtype = torch.long,device=device)

    return inputs_to_tensor,targets_to_tensor