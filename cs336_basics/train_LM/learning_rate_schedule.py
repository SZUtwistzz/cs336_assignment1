import torch
import math
def learning_rate_schedule(t:int,a_max,a_min,Tw,Tc):
    lr = 0
    if t<Tw:
        lr = (t/Tw)*a_max
    elif t>=Tw and t<=Tc:
        lr = a_min + 0.5*(1+math.cos(math.pi*(t-Tw)/(Tc-Tw)))*(a_max - a_min)
    else:
        lr = a_min
    return lr
