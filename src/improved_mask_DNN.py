import torch
import numpy
from typing import List
if __name__ == '__main__':
    from src.AdditionalUtilities.DecimalPrefixes import nm, um, mm
    from src.AdditionalUtilities.Formaters import Format
    from src.AdditionalUtilities.TitledFigure import Titles
else:
    from .AdditionalUtilities.DecimalPrefixes import    nm, um, mm
    from .AdditionalUtilities.Formaters import Format
    from .AdditionalUtilities.TitledFigure import Titles


class HeightMaskLayer(torch.nn.Module):
    """
    Описание класса
    """

    _PixelsCount : int
    _MaximumHeight : float

    _UpScaling : int
    _UpScaledPixelsCount : int

    _SmoothMatrix : None
    _SmoothMatrixSum = None

    _PropagationBuffer : None
    _PropagationShiftBuffer : None

    _Parameters : torch.nn.Parameter

    def __init__(self, wave_length=600*nm, space_reflection=1.0, mask_reflection=1.5, pixels_count=50, up_scaling=20, smoothing_matrix=None, parameters=None):
        super(HeightMaskLayer, self).__init__()

        #Setting all class fields:
        _Lambda = torch.tensor(requires_grad=False, dtype=torch.float32,    data=wave_length)

        _SpaceComplexReflection = torch.tensor(requires_grad=False, dtype=torch.complex64,  data=space_reflection)
        _MaskComplexReflection  = torch.tensor(requires_grad=False, dtype=torch.complex64,  data=mask_reflection)

        self._MaximumHeight = torch.max(_Lambda / torch.real(_MaskComplexReflection - _SpaceComplexReflection)).item()

        self._PixelsCount           = int(pixels_count)
        self._UpScaling             = int(up_scaling)
        self._UpScaledPixelsCount   = self._PixelsCount * self._UpScaling

        if smoothing_matrix is not None:
            if isinstance(smoothing_matrix, type(torch.tensor)):
                self.register_buffer('_SmoothMatrix', smoothing_matrix.to(torch.float32))
            else:
                self.register_buffer('_SmoothMatrix', torch.tensor(smoothing_matrix).to(torch.float32))
            self._SmoothMatrixSum = torch.sum(self._SmoothMatrix).item()


        #Registring parameters
        self._Parameters = torch.nn.Parameter(torch.rand(pixels_count, pixels_count))
        if parameters is not None: self._Parameters = torch.nn.Parameter(parameters.to(torch.float32))


        #Checking if everything ok and fixing some issues
        if True:
            if _Lambda.size() == 1:
                if _SpaceComplexReflection.size() != 1:
                    raise Exception('Space reflection should be single if wave_length is single')
                if _MaskComplexReflection.size() != 1:
                    raise Exception('Mask reflection should be single if wave_length is single')
            else:
                if _SpaceComplexReflection.size() != _Lambda.size():
                    if _SpaceComplexReflection.size() == 1:
                        _SpaceComplexReflection = _SpaceComplexReflection.repeat(_Lambda.size().item())
                    else:
                        raise Exception('Space reflection should be same size as wave length or single value')
                if _MaskComplexReflection.size() != _Lambda.size():
                    if _MaskComplexReflection.size() == 1:
                        _MaskComplexReflection = _MaskComplexReflection.repeat(_Lambda.size().item())
                    else:
                        raise Exception('Mask reflection should be same size as wave length or single value')
            if self._Parameters.size() != torch.Size((pixels_count, pixels_count)):
                raise Exception('parameters must have size (pixels_count, pixels_count) or equals None for auto creating')


        #Registering buffers
        self.register_buffer('_PropagationBuffer',      (2j*torch.pi*(_MaskComplexReflection + _SpaceComplexReflection)/_Lambda).to(torch.complex64))
        self.register_buffer('_PropagationShiftBuffer', torch.exp(2j*torch.pi*self._MaximumHeight*_SpaceComplexReflection/_Lambda).to(torch.complex64))

    def _heights(self):
        return self._MaximumHeight * torch.sigmoid(self._Parameters)
    def _prepared_heights(self):
        if self._SmoothMatrixSum is None:
            return torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0)
        else:
            return torch.nn.functional.conv2d(
                torch.repeat_interleave(torch.repeat_interleave(self._heights(), self._UpScaling, dim=1), self._UpScaling, dim=0).expand(1, 1, -1, -1),
                self._SmoothMatrix.expand(1, 1, -1, -1),
                padding='same').squeeze() / self._SmoothMatrixSum
    def _propagator(self):
        return self._PropagationShiftBuffer.expand(self._UpScaledPixelsCount, self._UpScaledPixelsCount, -1).movedim(2,0) \
               * \
               torch.exp(self._PropagationBuffer.expand(self._UpScaledPixelsCount, self._UpScaledPixelsCount, -1).movedim(2,0) * self._prepared_heights())

    def GetOriginalHeights(self):
        return self._heights().clone().detach()
    def GetPreparedHeights(self):
        return self._prepared_heights().clone().detach()
    def GetMaximumHeight(self):
        return self._MaximumHeight
    def GetSmoothMatrix(self):
        return self._SmoothMatrix.clone().detach() / self._SmoothMatrixSum

    def forward(self, field):
        return field * self._propagator()

class PaddedDiffractionLayer(torch.nn.Module):

    _BorderPixelsCount : int

    _PropagationArguments : None
    def _init_PropagationArguments(self, wave_length, up_scaled_pixels_count, up_scaled_pixel_length, diffraction_length, space_reflection, border_pixels_count):
        # Этот код скопирован из DiffractiveLayer, но затухающие моды учитываются и добавлен учёт границы
        wave_length = numpy.array(wave_length).reshape((-1, 1, 1))
        fx = numpy.fft.fftshift(numpy.fft.fftfreq(up_scaled_pixels_count + 2*border_pixels_count, d=up_scaled_pixel_length))
        fy = numpy.fft.fftshift(numpy.fft.fftfreq(up_scaled_pixels_count + 2*border_pixels_count, d=up_scaled_pixel_length))
        fxx, fyy = numpy.meshgrid(fx, fy)

        Kz = torch.tensor((2 * numpy.pi) * numpy.sqrt(0j + (1.0/(wave_length*space_reflection))**2 - fxx**2 - fyy**2), dtype=torch.complex64)
        self.register_buffer('_PropagationArguments', torch.exp(1j * Kz * diffraction_length))

    def __init__(self, wave_length=600*nm, space_reflection=1.0, plane_length=1.5*mm, pixels_count=50, diffraction_length=1.0*mm, up_scaling=20, border_length=0.0):
        super(PaddedDiffractionLayer, self).__init__()

        self._BorderPixelsCount = int(border_length * pixels_count * up_scaling / plane_length)

        self._init_PropagationArguments(wave_length, pixels_count*up_scaling, plane_length/(pixels_count*up_scaling), diffraction_length, space_reflection, self._BorderPixelsCount)

    def forward(self, field):
        field_padded                = torch.nn.functional.pad(field, (+self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount, +self._BorderPixelsCount))
        field_angular_spectrum      = torch.fft.fftshift(torch.fft.fft2(field_padded))
        field_propagated_padded     = torch.fft.ifft2(torch.fft.ifftshift(field_angular_spectrum * self._PropagationArguments))
        field_propagated_un_padded  = torch.nn.functional.pad(field_propagated_padded, (-self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount, -self._BorderPixelsCount))
        return field_propagated_un_padded
        
class ImprovedD2NN(torch.nn.Module):
    def __init__(self):
        super(ImprovedD2NN, self).__init__()


def test_HeightMaskLayer(show=False):
    from itertools import product
    import matplotlib.pyplot as plt

    print('<HeightMaskLayer> - Height smoothing test: ', end='')
    try:
        PixelsCount = 5
        UpScaling = 50
        Samples = 4

        SmoothingMatrix = numpy.zeros((Samples, int(UpScaling / 2), int(UpScaling / 2)))

        SmoothingParameters = numpy.linspace(3.0, 0.5, Samples)
        MaxPixelsShift = UpScaling / 4.0
        CentralPixel = UpScaling / 4.0
        for s, i, j in product(range(Samples), range(int(UpScaling / 2)), range(int(UpScaling / 2))):
            weight = MaxPixelsShift - numpy.sqrt((CentralPixel - i) ** 2 + (CentralPixel - j) ** 2)
            if weight >= 0:
                SmoothingMatrix[s][i][j] = weight ** SmoothingParameters[s]

        fig = plt.figure(figsize=(3.0 * Samples, 3.0 * 4))
        fig.suptitle('Сравнение сглаженных и не сглаженных масок при различных фильтрах', **Format.Text('BigHeader'))
        Fig = Titles(fig, (Samples, 4), topspace=0.05, wspace=0.05)

        for s in range(Samples):
            Mask = HeightMaskLayer(pixels_count=PixelsCount, up_scaling=UpScaling, smoothing_matrix=SmoothingMatrix[s])

            Heights = Mask.GetOriginalHeights().numpy()
            SmoothedHeights = Mask.GetPreparedHeights().numpy()
            SmoothingFilter = Mask.GetSmoothMatrix().numpy()

            hmin = numpy.min(Heights)
            hmax = numpy.max(Heights)

            axes1 = Fig.add_axes((s+1, 1))
            axes1.set_title('Сглаживающая маска', **Format.Text('Default', {'fontweight': 'bold'}))
            axes1.xaxis.set_tick_params(labelsize=6)
            axes1.yaxis.set_tick_params(labelsize=6)
            axes1.set_xlabel('Up Scaled Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes1.set_ylabel('Up Scaled Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes1.imshow(SmoothingFilter, cmap='gray', origin='lower', extent=[-UpScaling / 4, +UpScaling / 4, -UpScaling / 4, +UpScaling / 4])
            cbar = fig.colorbar(image, ax=axes1, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes2 = Fig.add_axes((s + 1, 2))
            axes2.set_title('Оригинальные высоты', **Format.Text('Default', {'fontweight': 'bold'}))
            axes2.xaxis.set_tick_params(labelsize=6)
            axes2.yaxis.set_tick_params(labelsize=6)
            axes2.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes2.set_ylabel('Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes2.imshow(Heights, cmap='viridis', origin='lower', extent=[-PixelsCount / 2, +PixelsCount / 2, -PixelsCount / 2, +PixelsCount / 2], vmin=hmin, vmax=hmax)
            cbar = fig.colorbar(image, ax=axes2, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes3 = Fig.add_axes((s + 1, 3))
            axes3.set_title('Сглаженные высоты', **Format.Text('Default', {'fontweight': 'bold'}))
            axes3.xaxis.set_tick_params(labelsize=6)
            axes3.yaxis.set_tick_params(labelsize=6)
            axes3.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes3.set_ylabel('Ny', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            image = axes3.imshow(SmoothedHeights, cmap='viridis', origin='lower', extent=[-PixelsCount / 2, +PixelsCount / 2, -PixelsCount / 2, +PixelsCount / 2], vmin=hmin, vmax=hmax)
            cbar = fig.colorbar(image, ax=axes3, location='right', pad=0.1, shrink=0.7)
            cbar.ax.tick_params(labelsize=6)

            axes4 = Fig.add_axes((s + 1, 4))
            axes4.set_title('Сравнение профилей', **Format.Text('Default', {'fontweight': 'bold'}))
            axes4.xaxis.set_tick_params(labelsize=6)
            axes4.yaxis.set_tick_params(labelsize=6)
            axes4.set_xlabel('Nx', **Format.Text('Caption', {'fontweight': 'bold'}))
            axes4.set_ylabel('Height', **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes4.plot(numpy.repeat(numpy.repeat(Heights, UpScaling, axis=1), UpScaling, axis=0)[int(UpScaling*PixelsCount/2)], color='maroon', linestyle='--')
            axes4.plot(SmoothedHeights[int(UpScaling * PixelsCount / 2)], color='indigo', linestyle='--')

        print('Pass! Check result in plot')
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    print('<HeightMaskLayer> - Transmitting test: ', end='')
    try:
        Toller = 1.0E-5
        Channels = 5
        Tests = 5

        WaveLength = numpy.linspace(100*nm, 1000*nm, Channels)
        SpaceReflection = numpy.linspace(2.0, 1.0, Channels) + 1j*numpy.linspace(0.5, 1.0, Channels)
        MaskReflection  = numpy.linspace(2.5, 1.5, Channels) + 1j*numpy.linspace(0.2, 1.0, Channels)
        PixelsCount = 20
        UpScaling = 10

        Mask = HeightMaskLayer(WaveLength, SpaceReflection, MaskReflection, PixelsCount, UpScaling)

        MaskInputField = torch.rand(Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling) * torch.exp(1j * torch.pi * torch.rand(Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling))
        MaskOutputField = Mask.forward(MaskInputField).clone().detach().numpy()

        Heights = Mask.GetPreparedHeights().numpy()
        MaximumHeight = Mask.GetMaximumHeight()

        RealInputField = MaskInputField.clone().detach().numpy()
        RealOutputField = numpy.zeros((Tests, Channels, PixelsCount*UpScaling, PixelsCount*UpScaling), dtype=complex)
        for s, k, i, j in product(range(Tests), range(Channels), range(PixelsCount*UpScaling), range(PixelsCount*UpScaling)):
            RealOutputField[s][k][i][j] = RealInputField[s][k][i][j] * numpy.exp(2j*numpy.pi*(Heights[i][j]*MaskReflection[k] + (MaximumHeight-Heights[i][j])*SpaceReflection[k])/WaveLength[k])

        Deviations          = numpy.abs(MaskOutputField - RealOutputField)
        MaximumDeviation    = numpy.max(Deviations)
        AverageDeviation    = numpy.mean(Deviations)

        if MaximumDeviation >= Toller:
            raise Exception('Maximum absolute value deviation greater then Toller. Maximum Deviation = ' + str(MaximumDeviation) + ' |  Average Deviation = ' + str(AverageDeviation) + ' |  Toller = ' + Format.Scientific(Toller))

        print('Pass! Maximum Deviation = ' + Format.Scientific(MaximumDeviation) + ' |  Average Deviation = ' + Format.Scientific(AverageDeviation) + ' |  Toller = ' + Format.Scientific(Toller))
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    if show:
        plt.show()
def test_PaddedDiffractionLayer(show=False):
    import matplotlib.pyplot as plt
    from torchvision.transforms import functional
    from itertools import product

    print('<PaddedDiffractionLayer> - Field propagation test: ', end='')
    try:
        WaveLength = 600*nm
        PixelsCount = 9
        PlaneLength = PixelsCount * 50*um
        MaxLength = 10.0*mm

        UpScaling = numpy.array([1, 3, 15])
        BorderLength = numpy.linspace(0, 2*PlaneLength, len(UpScaling))

        N = len(UpScaling)

        fig = plt.figure(figsize=(N*4.0, N*4.0))
        fig.suptitle('Распрастронение поля при различных параметрах вычисления', **Format.Text('BigHeader'))
        Fig = Titles(fig, (N, N), hspace=0.1)
        for (nx, US), (ny, BL) in product(enumerate(UpScaling), enumerate(BorderLength)):
            ls = numpy.linspace(0, MaxLength, 100)
            cut = numpy.zeros((100, PixelsCount*US))
            field = torch.nn.functional.pad(torch.ones((US, US), dtype=torch.complex64), (int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2), int((PixelsCount-1)*US/2)))
            for i, l in enumerate(ls):
                Propagator = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=l, up_scaling=US, border_length=BL)
                cut[i] = (numpy.abs(Propagator.forward(field).clone().detach().numpy())**1)[0][int(PixelsCount*US/2)]

            unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
            unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

            axes = Fig.add_axes((nx+1, ny+1))
            axes.set_title('Множитель разрешения: ' + str(US) + '\nРасширение на: ' + Format.Engineering(BL, 'm'), **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
            axes.xaxis.set_tick_params(labelsize=6)
            axes.yaxis.set_tick_params(labelsize=6)
            axes.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
            axes.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes.imshow(cut, cmap='viridis', origin='lower', extent=[-PlaneLength*multX/2, +PlaneLength*multX/2, 0, MaxLength*multZ], aspect='auto')

            axes.autoscale(enable=False)

            axes.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
            for k in range(1, 8+1):
                tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                xy0 = (0, 0)
                y1 = MaxLength * multZ
                x1 = y1 * tan * multX / multZ
                axes.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                axes.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

        print('Pass! Check result in plot')
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    print('<PaddedDiffractionLayer> - Field propagation scaling test: ', end='')
    try:
        MaxLength = 10.0*mm
        UpScaling = 15

        Ratios = [1.0, 2.0, 0.5, 4.0, 0.25]

        WaveLength_  = 600*nm
        PixelSize_   = 50*um

        N = len(Ratios)

        normed_cuts = []

        fig = plt.figure(figsize=(N * 4.0, 3 * 4.0))
        fig.suptitle('Распрастронение поля при пропорционально изменяющихся длинах волн и размерах пикселя', **Format.Text('BigHeader'))
        Fig = Titles(fig, (N, 3), hspace=0.1)
        for (c, ratio) in enumerate(Ratios):
            WaveLength = WaveLength_*ratio
            PixelSize = PixelSize_*ratio

            PixelsCount = int(8 / ratio)

            PlaneLength = PixelsCount * PixelSize
            BorderLength = PlaneLength

            ls = numpy.linspace(0, MaxLength, 100)
            cut = numpy.zeros((100, PixelsCount * UpScaling))

            field = torch.zeros(PixelsCount*UpScaling, PixelsCount*UpScaling)
            min_index = int(PixelsCount*UpScaling/2 - UpScaling/2)
            max_index = int(min_index + UpScaling)
            field[min_index:max_index, min_index:max_index] = torch.ones(UpScaling, UpScaling)
            for i, l in enumerate(ls):
                Propagator = PaddedDiffractionLayer(wave_length=WaveLength, plane_length=PlaneLength, pixels_count=PixelsCount, diffraction_length=l, up_scaling=UpScaling, border_length=BorderLength)
                cut[i] = (numpy.abs(Propagator.forward(field).clone().detach().numpy()) ** 1)[0][int(PixelsCount * UpScaling / 2)]

            unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
            unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

            axes1 = Fig.add_axes((c+1, 1))
            axes1.set_title('Длинна волны: ' + Format.Engineering(WaveLength) + '\nРазмер пикселя: ' + Format.Engineering(PixelSize) + '\nМножитель: ' + str(ratio), **Format.Text('Default', {'fontweight': 'bold'}), pad=21)
            axes1.xaxis.set_tick_params(labelsize=6)
            axes1.yaxis.set_tick_params(labelsize=6)
            axes1.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
            axes1.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes1.imshow(cut, cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
            axes1.autoscale(enable=False)
            axes1.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
            for k in range(1, 8 + 1):
                tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                xy0 = (0, 0)
                y1 = MaxLength * multZ
                x1 = y1 * tan * multX / multZ
                axes1.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                axes1.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

            axes2 = Fig.add_axes((c+1, 2))
            axes2.set_title('Нормированное на множитель', **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
            axes2.xaxis.set_tick_params(labelsize=6)
            axes2.yaxis.set_tick_params(labelsize=6)
            axes2.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
            axes2.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes2.imshow(cut/ratio, cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
            axes2.autoscale(enable=False)
            axes2.axline((0, 0), (0, MaxLength * multZ), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
            for k in range(1, 8 + 1):
                tan = numpy.sqrt(1.0 / (1.0 - (WaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
                xy0 = (0, 0)
                y1 = MaxLength * multZ
                x1 = y1 * tan * multX / multZ
                axes2.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
                axes2.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

            normed_cut = torch.tensor(cut/ratio)
            if c != 0:
                normed_cut = functional.resize(normed_cut.expand(1,1,-1,-1), normed_cuts[0].shape).squeeze()
            normed_cuts.append(normed_cut.numpy())

        for c, cut in enumerate(normed_cuts):
            PixelSize = PixelSize_ * Ratios[c]
            PixelsCount = int(8 / Ratios[c])
            PlaneLength = PixelsCount * PixelSize

            unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
            unitZ, multZ = Format.Engineering_Separated(MaxLength, 'm')

            axes = Fig.add_axes((c+1, 3))
            axes.set_title('Разница нормированных срезов', **Format.Text('Default', {'fontweight': 'bold'}), pad=10)
            axes.xaxis.set_tick_params(labelsize=6)
            axes.yaxis.set_tick_params(labelsize=6)
            axes.set_xlabel('X, ' + unitX, **Format.Text('Caption', {'fontweight': 'bold'}))
            axes.set_ylabel('Z, ' + unitZ, **Format.Text('Caption', {'fontweight': 'bold', 'rotation': 90}))
            axes.imshow(numpy.abs(cut-normed_cuts[0]), cmap='viridis', origin='lower', extent=[-PlaneLength * multX / 2, +PlaneLength * multX / 2, 0, MaxLength * multZ], aspect='auto', vmin=0, vmax=2)
            if c == 0:
                axes.text(0, MaxLength * multZ/2, 'Reference', **Format.Text('BigHeader'))

        print('Pass! Check result in plot')
    except Exception as e:
        print('Failed! (' + str(e) + ')')
        return

    if show:
        plt.show()
def test_ImprovedD2NN(show=False):
    import matplotlib.pyplot as plt



    if show:
        plt.show()
def test_show():
    import matplotlib.pyplot as plt
    plt.show()
if __name__ == '__main__':
    test_HeightMaskLayer()
    test_PaddedDiffractionLayer()
    test_ImprovedD2NN()
    test_show()



