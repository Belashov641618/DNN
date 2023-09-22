import torch

class Accuracy:

    _bits : int
    tensor_float : torch.dtype
    tensor_complex : torch.dtype

    def __init__(self):
        self._bits = 32
        self.tensor_float = torch.float32
        self.tensor_complex = torch.complex64

    def set(self, bits:int):
        if bits == 16:
            self._tensor_float = torch.float16
            self._tensor_complex = torch.complex32
        elif bits == 32:
            self._tensor_float = torch.float32
            self._tensor_complex = torch.complex64
        elif bits == 64:
            self._tensor_float = torch.float64
            self._tensor_complex = torch.complex128
        else:
            raise ValueError('bits may be 16, 32 or 64')
        self._bits = bits

    def get(self):
        return self._bits


