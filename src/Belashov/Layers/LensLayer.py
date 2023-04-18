import torch
from typing import Union, Iterable
from copy import deepcopy
from src.AdditionalUtilities.DecimalPrefixes import mm, um, nm, cm

from src.Belashov.Layers.AbstractLayer import AbstractLayer


class LensLayer(AbstractLayer):
    """
    Описание класса
    """

    _PhaseBuffer : None
    def _recalc_PhaseBuffer(self):
        wave_length = self._WaveLength.expand(1, 1, -1).movedim(2, 0)
        mesh = torch.linspace(-self._PlaneLength/2, +self._PlaneLength/2, int(self._PixelsCount*self._UpScaling))
        x_grid, y_grid = torch.meshgrid(mesh, mesh, indexing='ij')
        PhaseBuffer = torch.exp(-1j*torch.pi*(x_grid**2 + y_grid**2)/(wave_length*self._FocusLength)).to(dtype=self._tensor_complex_type)

        if hasattr(self, '_PhaseBuffer'):
            device = self._PhaseBuffer.device
            self.register_buffer('_PhaseBuffer', PhaseBuffer.to(device))
        else:
            self.register_buffer('_PhaseBuffer', PhaseBuffer)
    def GetPhaseBuffer(self, to_cpu:bool=True):
        if to_cpu:
            return deepcopy(self._PhaseBuffer).requires_grad_(False).cpu()
        return deepcopy(self._PhaseBuffer).requires_grad_(False)


    _FocusLength : float
    @property
    def FocusLength(self):
        return self._FocusLength
    @FocusLength.setter
    def FocusLength(self, focus_length:float):
        self._FocusLength = focus_length
        self._add_DelayedFunction(self._recalc_PhaseBuffer, 1.0)

    _WaveLength : torch.Tensor
    @property
    def WaveLength(self):
        return self._WaveLength
    @WaveLength.setter
    def WaveLength(self, wave_length:Union[float,torch.Tensor,Iterable]):
        self._WaveLength = (wave_length.to(self._tensor_float_type).requires_grad_(False) if torch.is_tensor(wave_length) else torch.tensor([wave_length] if type(wave_length) is float else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._add_DelayedFunction(self._recalc_PhaseBuffer, 1.0)

    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        self._add_DelayedFunction(self._recalc_PhaseBuffer, 1.0)

    _PlaneLength: float
    @property
    def PlaneLength(self):
        return self._PlaneLength
    @PlaneLength.setter
    def PlaneLength(self, plane_length:float):
        self._PlaneLength = plane_length
        self._add_DelayedFunction(self._recalc_PhaseBuffer, 1.0)

    _UpScaling: int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling
        self._add_DelayedFunction(self._recalc_PhaseBuffer, 1.0)


    def __init__(self, focus_length:float=10*mm, wave_length:Union[float,torch.Tensor,Iterable]=600*nm, plane_length:float=1.0*mm, pixels_count:int=21, up_scaling:int=20):
        super(LensLayer, self).__init__()

        self._FocusLength       = float(focus_length)
        self._WaveLength        = (wave_length.to(self._tensor_float_type).requires_grad_(False)        if torch.is_tensor(wave_length)         else torch.tensor([wave_length]         if type(wave_length) is float       else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        self._PlaneLength       = float(plane_length)
        self._PixelsCount       = int(pixels_count)
        self._UpScaling         = int(up_scaling)

        self._recalc_PhaseBuffer()

    def forward(self, field:torch.Tensor):
        super(LensLayer, self).forward(field)
        return field * self._PhaseBuffer


def auto_test(fourier_propagation_test=True, kirchhoff_propagation_test=True):
    from itertools import product
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from src.AdditionalUtilities.Formaters import Format
    from src.AdditionalUtilities.TitledFigure import Titles
    from src.Belashov.Layers.FourierPropagationLayer import FourierPropagationLayer
    from src.Belashov.Layers.KirchhoffPropagationLayer import KirchhoffPropagationLayer
    from src.AdditionalUtilities.UniversalTestsAndOther import DrawLensSystemImages


    if fourier_propagation_test:
        print('<LensLayer> - Fourier propagation lens test: ', end='')
        try:
            PixelsCount = 40
            PlaneLength = PixelsCount*30*um
            WaveLength = 500*nm
            UpScaling = 20
            FocusLength = 10*cm

            PropagationModule = FourierPropagationLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, up_scaling=UpScaling, border_length=4*PlaneLength)
            LensModule = LensLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, up_scaling=UpScaling, focus_length=FocusLength)

            BeforeAndAfterLengthsList = [
                (1.0*FocusLength, 1.0*FocusLength),
                (2.0*FocusLength, 2.0*FocusLength),
                (3.0*FocusLength, 1.5*FocusLength),
                (1.5*FocusLength, 3.0*FocusLength)
            ]

            Cols = len(BeforeAndAfterLengthsList)
            Rows = 2

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Начальные и конечные поля систем с линзами\nРаспространение методом Фурье', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for col, (length_before_lens, length_after_lens) in enumerate(BeforeAndAfterLengthsList):
                input_axis = Fig.add_axes((col+1, 1))
                input_axis.set_title('Система: ' + Format.Scientific(length_before_lens/FocusLength, 'f', 1) + '-' + Format.Scientific(length_after_lens/FocusLength, 'f', 1) + '\nВычислительный пиксель: ' + Format.Engineering(PlaneLength/(PixelsCount*UpScaling), 'm') + '\nВходное поле:', **Format.Text('Default', {'fontsize':10}))

                output_axis = Fig.add_axes((col+1, 2))
                output_axis.set_title('Выходное поле:', **Format.Text('Default', {'fontsize':10}))

                DrawLensSystemImages(input_axis, output_axis, PropagationModule, LensModule, FocusLength, length_before_lens=length_before_lens, length_after_lens=length_after_lens)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    if kirchhoff_propagation_test:
        print('<LensLayer> - Kirchhoff propagation lens test: ', end='')
        try:
            PixelsCount = 40
            PlaneLength = PixelsCount*30*um
            WaveLength = 500*nm
            UpScaling = 20
            FocusLength = 10*cm

            PropagationModule = KirchhoffPropagationLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, up_scaling=UpScaling)
            LensModule = LensLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, up_scaling=UpScaling, focus_length=FocusLength)

            BeforeAndAfterLengthsList = [
                (1.0*FocusLength, 1.0*FocusLength),
                (2.0*FocusLength, 2.0*FocusLength)
            ]

            Cols = len(BeforeAndAfterLengthsList)
            Rows = 2

            fig = plt.figure(figsize=(12*Cols/Rows, 12))
            fig.suptitle('Начальные и конечные поля систем с линзами\nРаспространение методом Кирхгофа(Фраунгоффера)', **Format.Text('BigHeader'))
            Fig = Titles(fig, (Cols, Rows))

            for col, (length_before_lens, length_after_lens) in enumerate(BeforeAndAfterLengthsList):
                input_axis = Fig.add_axes((col+1, 1))
                input_axis.set_title('Система: ' + Format.Scientific(length_before_lens/FocusLength, 'f', 1) + '-' + Format.Scientific(length_after_lens/FocusLength, 'f', 1) + '\nВычислительный пиксель: ' + Format.Engineering(PlaneLength/(PixelsCount*UpScaling), 'm') + '\nВходное поле:', **Format.Text('Default', {'fontsize':10}))

                output_axis = Fig.add_axes((col+1, 2))
                output_axis.set_title('Выходное поле:', **Format.Text('Default', {'fontsize':10}))

                DrawLensSystemImages(input_axis, output_axis, PropagationModule, LensModule, FocusLength, length_before_lens=length_before_lens, length_after_lens=length_after_lens)

            print('Pass! Check result in plot')
        except Exception as e:
            print('Failed! (' + str(e) + ')')
            return

    plt.show()
if __name__ == '__main__':
    auto_test(kirchhoff_propagation_test=False)