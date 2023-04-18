import torch
from typing import Union, Iterable
from copy import deepcopy
from src.AdditionalUtilities.DecimalPrefixes import mm, um, nm

from src.Belashov.Layers.AbstractLayer import AbstractLayer

import inspect

class FourierPropagationLayer(AbstractLayer):
    """
    Описание класса
    """

    _PropagationBuffer : None
    def _recalc_PropagationBuffer(self):
        device = torch.device('cpu')
        if hasattr(self, '_PropagationBuffer'):
            device = self._PropagationBuffer.device

        fx = torch.fft.fftshift(torch.fft.fftfreq(self._PixelsCount*self._UpScaling + 2*self._BorderPixelsCount, d=self._PlaneLength / (self._PixelsCount*self._UpScaling), device=device))
        fy = torch.fft.fftshift(torch.fft.fftfreq(self._PixelsCount*self._UpScaling + 2*self._BorderPixelsCount, d=self._PlaneLength / (self._PixelsCount*self._UpScaling), device=device))
        fxx, fyy = torch.meshgrid(fx, fy, indexing='ij')
        wave_length = self._WaveLength.expand(1, 1, -1).movedim(2, 0).to(device)
        space_reflection = self._SpaceReflection.expand(1, 1, -1).movedim(2, 0).to(device)
        Kz = ((2 * torch.pi) * torch.sqrt(0j + (1.0 / (wave_length * space_reflection)) ** 2 - fxx ** 2 - fyy ** 2)).to(dtype=self._tensor_complex_type)

        PropagationBuffer: torch.Tensor
        if isinstance(self._DiffractionLength, float):
            PropagationBuffer = torch.exp(1j * Kz * self._DiffractionLength)
        else:
            PropagationBuffer = Kz

        self.register_buffer('_PropagationBuffer', PropagationBuffer)
    def GetPropagationBuffer(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._PropagationBuffer).requires_grad_(False).cpu()
        return deepcopy(self._PropagationBuffer).requires_grad_(False)

    _BorderPixelsCount : int
    def _recalc_BorderPixelsCount(self):
        self._BorderPixelsCount = int(self._BorderLength * self._PixelsCount * self._UpScaling / self._PlaneLength)
    def GetBorderPixelsCount(self):
        return deepcopy(self._BorderPixelsCount)


    _WaveLength : torch.Tensor
    @property
    def WaveLength(self):
        return self._WaveLength
    @WaveLength.setter
    def WaveLength(self, wave_length:Union[float,Iterable,torch.Tensor]):
        self._WaveLength = (wave_length.to(self._tensor_float_type).requires_grad_(False) if torch.is_tensor(wave_length) else torch.tensor([wave_length] if type(wave_length) is float else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _SpaceReflection : torch.Tensor
    @property
    def SpaceReflection(self):
        return self._WaveLength
    @SpaceReflection.setter
    def SpaceReflection(self, space_reflection:Union[float,Iterable,torch.Tensor]):
        self._SpaceReflection = (space_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(space_reflection) else torch.tensor([space_reflection] if type(space_reflection) is float  else space_reflection, requires_grad=False, dtype=self._tensor_float_type))
        if self._SpaceReflection.size() != self._WaveLength.size():
            if self._SpaceReflection.size(0) == 1:
                self._SpaceReflection = self._SpaceReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _PlaneLength : float
    @property
    def PlaneLength(self):
        return self._PlaneLength
    @PlaneLength.setter
    def PlaneLength(self, plane_length:float):
        self._PlaneLength = plane_length
        self._add_DelayedFunction(self._recalc_BorderPixelsCount, 0.0)
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        self._add_DelayedFunction(self._recalc_BorderPixelsCount, 0.0)
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _DiffractionLength : Union[float, torch.nn.Parameter]
    @property
    def DiffractionLength(self):
        return self._DiffractionLength
    @DiffractionLength.setter
    def DiffractionLength(self, diffraction_length:float):
        if isinstance(self._DiffractionLength, float):
            self._DiffractionLength = diffraction_length
            self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)
        else:
            self._DiffractionLength = torch.nn.Parameter(torch.tensor([diffraction_length], dtype=self._tensor_float_type))
    def DiffractionLengthAsParameter(self, mode:bool=True):
        if mode:
            if isinstance(self._DiffractionLength, float):
                old_value = self._DiffractionLength
                del self._DiffractionLength
                self._DiffractionLength = torch.nn.Parameter(torch.tensor([old_value], dtype=self._tensor_float_type))
                self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)
        else:
            if not isinstance(self._DiffractionLength, float):
                old_value = float(torch.abs(self._DiffractionLength.data[0].detach()))
                del self._DiffractionLength
                setattr(self, '_DiffractionLength', old_value)
                self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _UpScaling : int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling
        self._add_DelayedFunction(self._recalc_BorderPixelsCount, 0.0)
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _BorderLength : float
    @property
    def BorderLength(self):
        return self._BorderLength
    @BorderLength.setter
    def BorderLength(self, border_length:float):
        self._BorderLength = border_length
        self._add_DelayedFunction(self._recalc_BorderPixelsCount, 0.0)
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)


    _TrainMode : bool
    def train(self, mode:bool=True):
        self._TrainMode = mode
        if not mode:
            self._evaluate()
        return super(FourierPropagationLayer, self).train(mode)
    def eval(self):
        self._evaluate()
        return super(FourierPropagationLayer, self).eval()
    def _evaluate(self):
        self._TrainMode = False


    def __init__(self, wave_length:Union[float,Iterable,torch.Tensor]=600*nm, space_reflection:Union[float,Iterable,torch.Tensor]=1.0, plane_length:float=1.0*mm, pixels_count:int=21, diffraction_length:float=20.0*mm, up_scaling:int=20, border_length:float=0.0, register_diffraction_length_as_parameter:bool=False):
        super(FourierPropagationLayer, self).__init__()

        self._WaveLength        = (wave_length.to(self._tensor_float_type).requires_grad_(False)        if torch.is_tensor(wave_length)         else torch.tensor([wave_length]         if type(wave_length) is float       else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._SpaceReflection   = (space_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(space_reflection)    else torch.tensor([space_reflection]    if type(space_reflection) is float  else space_reflection, requires_grad=False, dtype=self._tensor_float_type))
        self._PlaneLength       = float(plane_length)
        self._PixelsCount       = int(pixels_count)
        self._UpScaling         = int(up_scaling)
        self._BorderLength      = float(border_length)

        if register_diffraction_length_as_parameter:
            self._DiffractionLength = torch.nn.Parameter(torch.tensor([diffraction_length], dtype=self._tensor_float_type))
        else:
            self._DiffractionLength = float(diffraction_length)

        if self._SpaceReflection.size() != self._WaveLength.size():
            if self._SpaceReflection.size(0) == 1:
                self._SpaceReflection = self._SpaceReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))

        self._TrainMode = False

        self._recalc_BorderPixelsCount()
        self._recalc_PropagationBuffer()

    def forward(self, field:torch.Tensor):
        super(FourierPropagationLayer, self).forward(field)

        field = torch.nn.functional.pad(field, (+self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount))
        field = torch.fft.fftshift(torch.fft.fft2(field))

        if isinstance(self._DiffractionLength, float):
            field = torch.fft.ifft2(torch.fft.ifftshift(field * self._PropagationBuffer))
        else:
            field = torch.fft.ifft2(torch.fft.ifftshift(field * torch.exp(1j * self._PropagationBuffer * torch.abs(self._DiffractionLength))))

        field = torch.nn.functional.pad(field, (-self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount))
        return field


def auto_test(pixel_emission_test=True, colorized_pixel_emission_test=True):
    from itertools import product
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from src.AdditionalUtilities.Formaters import Format
    from src.AdditionalUtilities.TitledFigure import Titles
    from src.AdditionalUtilities.UniversalTestsAndOther import DrawPropagationLayerSinglePixelEmission

    if pixel_emission_test:
        print('<FourierPropagationLayer> - Pixel emission test: ', end='')
        try:
            Layer = FourierPropagationLayer(wave_length=500*nm, plane_length=21*30*um, diffraction_length=8*mm)

            BorderLengthList    = torch.arange(3)*Layer.PlaneLength
            UpScalingList       = 2**(torch.arange(5)+1) - 1

            Cols = len(UpScalingList)
            Rows = len(BorderLengthList)

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Распространение излучения от пикселя при различных параметрах', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for (col, up_scaling), (row, border_length) in product(enumerate(UpScalingList), enumerate(BorderLengthList)):
                Layer.BorderLength = border_length.item()
                Layer.UpScaling = int(up_scaling.item())
                axis = Fig.add_axes((col+1, row+1))
                axis.set_title('Множитель разрешения: ' + str(Layer.UpScaling) + '\nВычислительный пиксель: ' + Format.Engineering(Layer.PlaneLength/(Layer.PixelsCount*Layer.UpScaling), 'm') + '\nРазмер паддинга: ' + Format.Engineering(Layer.BorderLength,'m'), **Format.Text('Default', {'fontsize':10}))
                DrawPropagationLayerSinglePixelEmission(axis, Layer, length_limits=(Layer.DiffractionLength*0.0, Layer.DiffractionLength), use_fast_recurrent_method=False)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if colorized_pixel_emission_test:
        print('<FourierPropagationLayer> - Colorized pixel emission test: ', end='')
        try:
            Layer = FourierPropagationLayer(wave_length=torch.linspace(400*nm, 650*nm, 20), plane_length=21*30*um, diffraction_length=8*mm)

            BorderLengthList    = torch.arange(3)*Layer.PlaneLength
            UpScalingList       = 2**(torch.arange(4)+1) - 1

            Cols = len(UpScalingList)
            Rows = len(BorderLengthList)

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Распространение излучения от пикселя при различных параметрах', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for (col, up_scaling), (row, border_length) in product(enumerate(UpScalingList), enumerate(BorderLengthList)):
                Layer.BorderLength = border_length.item()
                Layer.UpScaling = int(up_scaling.item())
                axis = Fig.add_axes((col+1, row+1))
                axis.set_title('Множитель разрешения: ' + str(Layer.UpScaling) + '\nВычислительный пиксель: ' + Format.Engineering(Layer.PlaneLength/(Layer.PixelsCount*Layer.UpScaling), 'm') + '\nРазмер паддинга: ' + Format.Engineering(Layer.BorderLength,'m'), **Format.Text('Default', {'fontsize':10}))
                DrawPropagationLayerSinglePixelEmission(axis, Layer, length_limits=(Layer.DiffractionLength*0.0, Layer.DiffractionLength), use_fast_recurrent_method=False)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    plt.show()
if __name__ == '__main__':
    auto_test()