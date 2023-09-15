import torch
from typing import Union, Iterable
from copy import deepcopy
from utilities.DecimalPrefixes import nm

from .AbstractMaskLayer import AbstractMaskLayer

class HeightMaskLayer(AbstractMaskLayer):

    @property
    def heights(self):
        class Selector:
            _self:HeightMaskLayer
            def __init__(self, _self:HeightMaskLayer):
                self._self = _self
            def __call__(self, to_cpu:bool=True):
                return self.prepared(to_cpu)
            def raw(self, to_cpu:bool=True):
                Heights = (self._self._max_height*self._self._normalize()).clone().detach()
                if to_cpu:
                    return Heights.cpu()
                return Heights
            def prepared(self, to_cpu:bool=True):
                Heights = (self._self._max_height*self._self._scale()).clone().detach()
                if to_cpu:
                    return Heights.cpu()
                return Heights
        return Selector(self)

    def _scale(self):
        if self._smooth_matrix is None:
            return super(HeightMaskLayer, self)._scale()
        else:
            return torch.nn.functional.conv2d(super(HeightMaskLayer, self)._scale().expand(1, 1, -1, -1), self._smooth_matrix.expand(1, 1, -1, -1), padding='same').squeeze() / torch.sum(self._smooth_matrix)
    def _mask(self):
        return torch.exp(2.0j*torch.pi*(self._mask_reflection/(self._mask_reflection-self._space_reflection))*self._scale())

    _smooth_matrix : Union[torch.Tensor, None]
    @property
    def smooth_matrix(self):
        class Selector:
            _self:HeightMaskLayer
            def __init__(self, _self:HeightMaskLayer):
                self._self = _self
            def get(self, to_cpu:bool=True):
                if to_cpu:
                    return self._self._smooth_matrix.clone().detach().cpu()
                else:
                    return self._self._smooth_matrix.clone().detach()
            def set(self, matrix:Union[torch.Tensor, None]):
                if isinstance(matrix, torch.Tensor):
                    device = 'cpu'
                    if hasattr(self._self, '_parameters'):
                        device = self._self._parameters.device
                    matrix = matrix.to(device)
                    matrix = matrix.to(self._self._accuracy.tensor_float)
                    self._self.register_buffer('_smooth_matrix', matrix)
                else:
                    self._self._smooth_matrix = None
        return Selector(self)

    _max_height : float
    @property
    def max_height(self):
        class Selector:
            _self:HeightMaskLayer
            def __init__(self, _self:HeightMaskLayer):
                self._self = _self
            def get(self):
                return deepcopy(self._self._max_height)
        return Selector(self)
    def _recalc_max_height(self):
        self._max_height = torch.max(self._wavelength / torch.real(self._mask_reflection - self._space_reflection)).item()

    _wavelength : torch.Tensor
    @property
    def wavelength(self):
        return self._wavelength
    @wavelength.setter
    def wavelength(self, length:Union[float,Iterable,torch.Tensor]):
        self._wavelength = (length.to(self._accuracy.tensor_float).requires_grad_(False) if torch.is_tensor(length) else torch.tensor([length] if type(length) is float else length, requires_grad=False, dtype=self._accuracy.tensor_float))
        self._delayed.add(self._recalc_max_height)

    _space_reflection : torch.Tensor
    @property
    def space_reflection(self):
        return self._space_reflection
    @space_reflection.setter
    def space_reflection(self, reflection:Union[float, Iterable, torch.Tensor]):
        self._space_reflection = (reflection.to(self._accuracy.tensor_complex).requires_grad_(False) if torch.is_tensor(reflection) else torch.tensor([reflection] if type(reflection) is float  else reflection, requires_grad=False, dtype=self._accuracy.tensor_complex))
        if self._space_reflection.size() != self._wavelength.size():
            if self._space_reflection.size(0) == 1:
                self._space_reflection = self._space_reflection.repeat(self._wavelength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        self._delayed.add(self._recalc_max_height)

    _mask_reflection : torch.Tensor
    @property
    def mask_reflection(self):
        return self._mask_reflection
    @mask_reflection.setter
    def mask_reflection(self, reflection:Union[float, Iterable, torch.Tensor]):
        self._mask_reflection = (reflection.to(self._accuracy.tensor_complex).requires_grad_(False) if torch.is_tensor(reflection) else torch.tensor([reflection] if type(reflection) is float  else reflection, requires_grad=False, dtype=self._accuracy.tensor_complex))
        if self._mask_reflection.size() != self._wavelength.size():
            if self._mask_reflection.size(0) == 1:
                self._mask_reflection = self._mask_reflection.repeat(self._wavelength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        self._delayed.add(self._recalc_max_height)

    def __init__(self,  pixels:int=20,
                        up_scaling:int=8,
                        wavelength:Union[float,Iterable,torch.Tensor]=532*nm,
                        space_reflection:Union[float,Iterable,torch.Tensor]=1.0,
                        mask_reflection:Union[float,Iterable,torch.Tensor]=1.5):
        super(HeightMaskLayer, self).__init__(pixels=pixels, up_scaling=up_scaling)
        self.wavelength = wavelength
        self.space_reflection = space_reflection
        self.mask_reflection = mask_reflection
        self._recalc_max_height()