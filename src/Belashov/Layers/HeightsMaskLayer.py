import torch
from typing import Union, Iterable
from copy import deepcopy
from src.AdditionalUtilities.DecimalPrefixes import mm, um, nm

from src.Belashov.Layers.AbstractLayer import AbstractLayer


class HeightsMaskLayer(AbstractLayer):
    """
    Описание класса
    """

    _SmoothMatrixBuffer : Union[Iterable,None,torch.Tensor]
    _SmoothMatrixSum    : float
    def SetSmoothMatrix(self, matrix:Union[Iterable,None,torch.Tensor]):
        if matrix is None:
            self._SmoothMatrixBuffer = None
            self._SmoothMatrixSum = 0
        else:
            if not torch.is_tensor(matrix):
                matrix = torch.tensor(matrix)
            if hasattr(self, '_SmoothMatrixBuffer'):
                device = (self._Parameters.device if self._SmoothMatrixBuffer is None else self._SmoothMatrixBuffer.device)
                self.register_buffer('_SmoothMatrixBuffer', matrix.to(device).to(self._tensor_float_type))
            else:
                self.register_buffer('_SmoothMatrixBuffer', matrix.to(self._tensor_float_type))
        if self._SmoothMatrixBuffer is not None:
            self._SmoothMatrixSum = torch.sum(self._SmoothMatrixBuffer).item()
    def GetSmoothMatrix(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._SmoothMatrixBuffer).requires_grad_(False).cpu()
        return deepcopy(self._SmoothMatrixBuffer).requires_grad_(False)

    _PropagationBuffer : None
    _PropagationShiftBuffer : None
    def _recalc_PropagationBuffer(self):
        if hasattr(self, '_PropagationBuffer'):
            device = self._PropagationBuffer.device
            self.register_buffer('_PropagationBuffer', (2j*torch.pi*(self._MaskReflection + self._SpaceReflection)/self._WaveLength).to(self._tensor_complex_type).to(device))
        else:
            self.register_buffer('_PropagationBuffer', (2j*torch.pi*(self._MaskReflection + self._SpaceReflection)/self._WaveLength).to(self._tensor_complex_type).to(torch.complex64))
    def _recalc_PropagationShiftBuffer(self):
        if hasattr(self, '_PropagationShiftBuffer'):
            device = self._PropagationShiftBuffer.device
            self.register_buffer('_PropagationShiftBuffer', torch.exp(2j*torch.pi*self._MaximumHeight*self._SpaceReflection/self._WaveLength).to(self._tensor_complex_type).to(device))
        else:
            self.register_buffer('_PropagationShiftBuffer', torch.exp(2j*torch.pi*self._MaximumHeight*self._SpaceReflection/self._WaveLength).to(self._tensor_complex_type))
    def GetPropagationBuffer(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._PropagationBuffer).requires_grad_(False).cpu()
        return deepcopy(self._PropagationBuffer).requires_grad_(False)
    def PropagationShiftBuffer(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._PropagationShiftBuffer).requires_grad_(False).cpu()
        return deepcopy(self._PropagationShiftBuffer).requires_grad_(False)

    _MaximumHeight: float
    def _recalc_MaximumHeight(self):
        self._MaximumHeight = torch.max(self._WaveLength / torch.real(self._MaskReflection - self._SpaceReflection)).item()
    def GetMaximumHeight(self):
        return deepcopy(self._MaximumHeight)

    _Parameters : torch.nn.Parameter
    def _recalc_Parameters(self):
        with torch.no_grad():
            if hasattr(self, '_Parameters'):
                device = self._Parameters.device
                ParametersCopy = self._Parameters.clone()
                if ParametersCopy.size(0) < self._PixelsCount:
                    Parameters = torch.normal(mean=0, std=10, size=(self._PixelsCount, self._PixelsCount), dtype=self._tensor_float_type)
                    nx1, nx2 = int(self._PixelsCount/2 - ParametersCopy.size()[0]/2), int(self._PixelsCount/2 + ParametersCopy.size()[0]/2)
                    ny1, ny2 = int(self._PixelsCount/2 - ParametersCopy.size()[1]/2), int(self._PixelsCount/2 + ParametersCopy.size()[1]/2)
                    if nx2 - nx1 < ParametersCopy.size()[0]: nx2 += 1
                    if ny2 - ny1 < ParametersCopy.size()[1]: ny2 += 1
                    Parameters[nx1:nx2, ny1:ny2] = ParametersCopy
                else:
                    nx1, nx2 = int(ParametersCopy.size()[0]/2 - self._PixelsCount/2), int(ParametersCopy.size()[0]/2 + self._PixelsCount/2) + 1
                    ny1, ny2 = int(ParametersCopy.size()[1]/2 - self._PixelsCount/2), int(ParametersCopy.size()[1]/2 + self._PixelsCount/2) + 1
                    if nx2 - nx1 > self._PixelsCount: nx2 -= 1
                    if ny2 - ny1 > self._PixelsCount: ny2 -= 1
                    Parameters = ParametersCopy[nx1:nx2, ny1:ny2]
                self._Parameters = torch.nn.Parameter(Parameters.to(device))
            else:
                self._Parameters = torch.nn.Parameter(torch.normal(mean=0, std=10, size=(self._PixelsCount, self._PixelsCount), dtype=self._tensor_float_type))
    def SetParameters(self, parameters:Union[Iterable,torch.Tensor]):
        if not torch.is_tensor(parameters):
            parameters = torch.tensor(parameters)
        if parameters.size() != self._Parameters.size(): raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': parameters must have size [PixelsCount, PixelsCount]!'))
        parameters.to(self._Parameters.device).to(self._tensor_float_type)
        self._Parameters = torch.nn.Parameter(parameters)
    def GetParameters(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._Parameters).requires_grad_(False).cpu()
        return deepcopy(self._Parameters).requires_grad_(False)


    _WaveLength : torch.Tensor
    @property
    def WaveLength(self):
        return self._WaveLength
    @WaveLength.setter
    def WaveLength(self, wave_length:Union[float,Iterable,torch.Tensor]):
        self._WaveLength = (wave_length.to(self._tensor_float_type).requires_grad_(False) if torch.is_tensor(wave_length) else torch.tensor([wave_length] if type(wave_length) is float else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._add_DelayedFunction(self._recalc_MaximumHeight,           1.0)
        self._add_DelayedFunction(self._recalc_PropagationShiftBuffer,  0.5)
        self._add_DelayedFunction(self._recalc_PropagationBuffer,       0.0)

    _SpaceReflection : torch.Tensor
    @property
    def SpaceReflection(self):
        return self._SpaceReflection
    @SpaceReflection.setter
    def SpaceReflection(self, space_reflection:Union[float,Iterable,torch.Tensor]):
        self._SpaceReflection = (space_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(space_reflection) else torch.tensor([space_reflection] if type(space_reflection) is float else space_reflection, requires_grad=False, dtype=self._tensor_float_type))
        if self._SpaceReflection.size() != self._WaveLength.size():
            if self._SpaceReflection.size(0) == 1:
                self._SpaceReflection = self._SpaceReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        self._add_DelayedFunction(self._recalc_MaximumHeight,           1.0)
        self._add_DelayedFunction(self._recalc_PropagationShiftBuffer,  0.5)
        self._add_DelayedFunction(self._recalc_PropagationBuffer,       0.0)

    _MaskReflection : torch.Tensor
    @property
    def MaskReflection(self):
        return self._MaskReflection
    @MaskReflection.setter
    def MaskReflection(self, mask_reflection:Union[float,Iterable,torch.Tensor]):
        self._MaskReflection = (mask_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(mask_reflection) else torch.tensor([mask_reflection] if type(mask_reflection) is float else mask_reflection, requires_grad=False, dtype=self._tensor_float_type))
        if self._MaskReflection.size() != self._WaveLength.size():
            if self._MaskReflection.size(0) == 1:
                self._MaskReflection = self._MaskReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': mask_reflection size must be one or equal wave_length size!'))
        self._add_DelayedFunction(self._recalc_MaximumHeight,           1.0)
        self._add_DelayedFunction(self._recalc_PropagationBuffer,       0.0)

    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        self._add_DelayedFunction(self._recalc_Parameters,              2.0)

    _UpScaling : int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling


    def _heights(self):
        return self._MaximumHeight * torch.sigmoid(self._Parameters)
    def _prepared_heights(self):
        if self._SmoothMatrixBuffer is None:
            return torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0)
        else:
            return torch.nn.functional.conv2d(torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0).expand(1, 1, -1, -1), self._SmoothMatrixBuffer.expand(1, 1, -1, -1), padding='same').squeeze() / self._SmoothMatrixSum
    def _propagators(self):
        return self._PropagationShiftBuffer.expand(self._UpScaling*self._PixelsCount, self._UpScaling*self._PixelsCount, -1).movedim(2,0) \
               * \
               torch.exp(self._PropagationBuffer.expand(self._UpScaling*self._PixelsCount, self._UpScaling*self._PixelsCount, -1).movedim(2,0) * self._prepared_heights())

    def GetOriginalHeights(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._heights()).requires_grad_(False).cpu()
        return deepcopy(self._heights()).requires_grad_(False)
    def GetPreparedHeights(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._prepared_heights()).requires_grad_(False).cpu()
        return deepcopy(self._prepared_heights()).requires_grad_(False)
    def GetPropagators(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._propagators()).requires_grad_(False).cpu()
        return deepcopy(self._propagators()).requires_grad_(False)


    def __init__(self, wave_length:Union[float,Iterable,torch.Tensor]=600*nm, space_reflection:Union[float,Iterable,torch.Tensor]=1.0, mask_reflection:Union[float,Iterable,torch.Tensor]=1.5, pixels_count:int=50, up_scaling:int=20, smoothing_matrix:Union[None,Iterable,torch.Tensor]=None):
        super(HeightsMaskLayer, self).__init__()

        self._WaveLength        = (wave_length.to(self._tensor_float_type).requires_grad_(False)        if torch.is_tensor(wave_length)         else torch.tensor([wave_length]         if type(wave_length) is float       else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._SpaceReflection   = (space_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(space_reflection)    else torch.tensor([space_reflection]    if type(space_reflection) is float  else space_reflection, requires_grad=False, dtype=self._tensor_float_type))
        self._MaskReflection    = (mask_reflection.to(self._tensor_complex_type).requires_grad_(False)  if torch.is_tensor(mask_reflection)     else torch.tensor([mask_reflection]     if type(mask_reflection) is float   else mask_reflection, requires_grad=False, dtype=self._tensor_float_type))
        self._PixelsCount       = int(pixels_count)
        self._UpScaling         = int(up_scaling)

        if self._SpaceReflection.size() != self._WaveLength.size():
            if self._SpaceReflection.size(0) == 1:
                self._SpaceReflection = self._SpaceReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        if self._MaskReflection.size() != self._WaveLength.size():
            if self._MaskReflection.size(0) == 1:
                self._MaskReflection = self._MaskReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': mask_reflection size must be one or equal wave_length size!'))

        self.SetSmoothMatrix(smoothing_matrix)

        self._recalc_Parameters()
        self._recalc_MaximumHeight()
        self._recalc_PropagationShiftBuffer()
        self._recalc_PropagationBuffer()

    def forward(self, field:torch.Tensor):
        super(HeightsMaskLayer, self).forward(field)
        return field * self._propagators()