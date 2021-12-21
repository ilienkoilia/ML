import torch
import numpy as np
import pandas as pd
def pi(u):
    return torch.exp(u)/(torch.sum(torch.exp(u)))

def grad_example10(u):
    u = u.clone()
    u.requires_grad_(True)

    f = -torch.sum(pi(u) * torch.log(pi(u)))
    f.backward()

    return u.grad

print(grad_example10(torch.tensor([-4.,3.,2.])))