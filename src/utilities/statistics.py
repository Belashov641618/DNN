import torch

from typing import Union

def autocorrelation(a:torch.Tensor, dims:Union[int,tuple[int,...]]=()):
    count = len(a.size())
    if isinstance(dims, int): dims = (dims, )

    expanded_a = a.expand(*[-1]*count, *[a.size(dim) for dim in dims])

    paddings  = [0 for i in range(2*len(a.size()))]
    paddings_ = [0 for i in range(2*len(a.size()))]
    for dim in dims:
        paddings[2*dim] = (a.size(dim) + 1)//2
        paddings[2*dim + 1] = paddings[2*dim]
        paddings_[2*dim] = -paddings[2*dim]
        paddings_[2*dim + 1] = -paddings[2*dim + 1]

