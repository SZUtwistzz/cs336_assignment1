from typing import Callable, Optional
import math
import torch

class adamw(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-8):
        if lr<0:
            raise ValueError(f"Invalid learning rate: {lr}")

        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay,"eps":eps}
        super().__init__(params,defaults)

    def step(self,closure:Optional[Callable]=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t",1)
                m = state.get("m",torch.zeros_like(p))
                v = state.get("v",torch.zeros_like(p))
                grad = p.grad.data
                lr_t = lr*math.sqrt(1-betas[1]**t)/(1-betas[0]**t)
                p.data = (1-lr*weight_decay)*p.data

                m = betas[0]*m+(1-betas[0])*grad
                v= betas[1]*v+(1-betas[1])*(grad**2)
                state["m"]=m
                state["v"]= v
                state["t"] = t + 1

                with torch.no_grad():
                    p.data -= lr_t*m/(torch.sqrt(v)+eps)

        return loss