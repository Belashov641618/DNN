import matplotlib.pyplot as plt
import torch
from typing import Union, Iterable
from copy import deepcopy
from src.utilities.DecimalPrefixes import mm, um, nm

from src.modules.layers.AbstractLayer import AbstractLayer


class KirchhoffPropagationLayer(AbstractLayer):
    """
    Описание класса
    """

    _PropagationBuffer : None
    def _recalc_PropagationBuffer(self):
        mesh = torch.linspace(0, self._PlaneLength, self._PixelsCount*self._UpScaling)
        x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        R = torch.sqrt(self._DiffractionLength**2 + x_grid**2 + y_grid**2)
        wave_length = self._WaveLength.expand(1, 1, -1).movedim(2, 0)
        space_reflection = self._SpaceReflection.expand(1, 1, -1).movedim(2, 0)
        # propagator = (wave_length * space_reflection + 2j * torch.pi / R) * torch.exp(0.5j * wave_length * space_reflection * R / torch.pi) / R
        propagator = (self._DiffractionLength / R**2)+(1.0/(2*torch.pi*R) + 1.0/(1j*wave_length*space_reflection))*torch.exp(2j*torch.pi*R/(wave_length*space_reflection))*(self._PlaneLength/(self._PixelsCount*self._UpScaling))**2
        PropagationBuffer = torch.nn.functional.pad(propagator, mode='reflect', pad=(self._PixelsCount*self._UpScaling-1, 0, self._PixelsCount*self._UpScaling-1, 0)).unsqueeze(1).to(self._tensor_complex_type)

        if hasattr(self, '_PropagationBuffer'):
            device = self._PropagationBuffer.device
            self.register_buffer('_PropagationBuffer', PropagationBuffer.to(device))
        else:
            self.register_buffer('_PropagationBuffer', PropagationBuffer)
    def GetPropagationBuffer(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._PropagationBuffer).requires_grad_(False).cpu()
        return deepcopy(self._PropagationBuffer).requires_grad_(False)


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
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _DiffractionLength : float
    @property
    def DiffractionLength(self):
        return self._DiffractionLength
    @DiffractionLength.setter
    def DiffractionLength(self, diffraction_length:float):
        self._DiffractionLength = diffraction_length
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)

    _UpScaling : int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling
        self._add_DelayedFunction(self._recalc_PropagationBuffer, 1.0)


    def __init__(self, wave_length:Union[float,Iterable,torch.Tensor]=600*nm, space_reflection:Union[float,Iterable,torch.Tensor]=1.0, plane_length:float=1.0*mm, pixels_count:int=21, diffraction_length:float=20.0*mm, up_scaling:int=20):
        super(KirchhoffPropagationLayer, self).__init__()

        self._WaveLength        = (wave_length.to(self._tensor_float_type).requires_grad_(False)        if torch.is_tensor(wave_length)         else torch.tensor([wave_length]         if type(wave_length) is float       else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._SpaceReflection   = (space_reflection.to(self._tensor_complex_type).requires_grad_(False) if torch.is_tensor(space_reflection)    else torch.tensor([space_reflection]    if type(space_reflection) is float  else space_reflection, requires_grad=False, dtype=self._tensor_float_type))
        self._PlaneLength       = float(plane_length)
        self._PixelsCount       = int(pixels_count)
        self._DiffractionLength = float(diffraction_length)
        self._UpScaling         = int(up_scaling)

        if self._SpaceReflection.size() != self._WaveLength.size():
            if self._SpaceReflection.size(0) == 1:
                self._SpaceReflection = self._SpaceReflection.repeat(self._WaveLength.size(0))
            else:
                raise ValueError("\033[31m\033[1m{}".format(self._get_name() + ': space_reflection size must be one or equal wave_length size!'))

        self._recalc_PropagationBuffer()

    def forward(self, field:torch.Tensor):
        super(KirchhoffPropagationLayer, self).forward(field)
        field = torch.nn.functional.conv2d(field, self._PropagationBuffer, padding=self._PixelsCount*self._UpScaling-1, groups=self._PropagationBuffer.size(0))
        return field



def auto_test(pixel_emission_test=True, colorized_pixel_emission_test=True):
    from itertools import product
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from src.utilities.Formaters import Format
    from src.utilities.TitledFigure import Titles
    from src.utilities.UniversalTestsAndOther import DrawPropagationLayerSinglePixelEmission


    if pixel_emission_test:
        print('<FourierPropagationLayer> - Pixel emission test: ', end='')
        try:
            Layer = KirchhoffPropagationLayer(wave_length=500*nm, plane_length=21*30*um, diffraction_length=8*mm)

            UpScalingList = (torch.arange(4)+1)**2
            UpScalingList = torch.tensor([10])

            Samples = len(UpScalingList)
            Rows = int(Samples**0.5)
            Cols = int(Samples/Rows) + (Samples%Rows!=0)

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Распространение излучения от пикселя при различных параметрах', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for up_scaling, (col, row) in zip(UpScalingList, product(range(Cols), range(Rows))):
                Layer.UpScaling = int(up_scaling.item())
                axis = Fig.add_axes((col+1, row+1))
                axis.set_title('Множитель разрешения: ' + str(Layer.UpScaling) + '\nВычислительный пиксель: ' + Format.Engineering(Layer.PlaneLength/(Layer.PixelsCount*Layer.UpScaling), 'm'), **Format.Text('Default', {'fontsize':10}))
                DrawPropagationLayerSinglePixelEmission(axis, Layer, length_limits=(Layer.DiffractionLength*0.0, Layer.DiffractionLength), use_fast_recurrent_method=False)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if colorized_pixel_emission_test:
        print('<FourierPropagationLayer> - Colorized pixel emission test: ', end='')
        try:
            Layer = KirchhoffPropagationLayer(wave_length=torch.linspace(400*nm, 650*nm, 20), plane_length=21*30*um, diffraction_length=8*mm)

            UpScalingList = (torch.arange(4) + 1) ** 2
            UpScalingList = torch.tensor([10])

            Samples = len(UpScalingList)
            Rows = int(Samples ** 0.5)
            Cols = int(Samples / Rows) + (Samples % Rows != 0)

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Распространение излучения от пикселя при различных параметрах', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for up_scaling, (col, row) in zip(UpScalingList, product(range(Cols), range(Rows))):
                Layer.UpScaling = int(up_scaling.item())
                axis = Fig.add_axes((col+1, row+1))
                axis.set_title('Множитель разрешения: ' + str(Layer.UpScaling) + '\nВычислительный пиксель: ' + Format.Engineering(Layer.PlaneLength/(Layer.PixelsCount*Layer.UpScaling), 'm'), **Format.Text('Default', {'fontsize':10}))
                DrawPropagationLayerSinglePixelEmission(axis, Layer, length_limits=(Layer.DiffractionLength*0.0, Layer.DiffractionLength), use_fast_recurrent_method=True)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    plt.show()
if __name__ == '__main__':
    auto_test(colorized_pixel_emission_test=True)