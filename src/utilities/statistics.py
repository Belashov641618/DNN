import torch
import numpy
import math

from typing import Union

def autocorrelation(a:torch.Tensor, dims:Union[int,tuple[int,...]], mean_dim:int):
    if isinstance(dims, int): dims = (dims, )

    a = a - torch.mean(a, dim=mean_dim, keepdim=True)
    a = a / torch.sqrt(torch.sum(a**2, dim=mean_dim, keepdim=True)/(a.size(mean_dim) - 1))

    paddings  = [0 for i in range(2*len(a.size()))]
    paddings_ = [0 for i in range(2*len(a.size()))]
    multiplier = 1.0
    for dim in dims:
        paddings[2*dim]      = (a.size(dim) + 1)//2
        paddings[2*dim + 1]  = paddings[2*dim]
        paddings_[2*dim]     = -paddings[2*dim]
        paddings_[2*dim + 1] = -paddings[2*dim + 1]
        multiplier *= a.size(dim)

    a = torch.nn.functional.pad(a, paddings)
    spectrum    = torch.fft.fftshift(torch.fft.fftn(a, dim=dims))
    convolution = torch.fft.ifftshift(torch.fft.ifftn(spectrum*spectrum.conj(), dim=dims)).abs()
    convolution = torch.nn.functional.pad(convolution, paddings_)

    result = torch.sum(convolution, dim=mean_dim) / (a.size(mean_dim) - 1)

    return result

def distribution(a:torch.Tensor, N:int=100):
    values = torch.linspace(a.min(), a.max(), N+1)[1:]
    results = torch.zeros(N, dtype=torch.float32, device=a.device)
    for i, value in enumerate(values):
        results[i] = torch.sum(a <= value)
    results /= a.numel()
    return results
