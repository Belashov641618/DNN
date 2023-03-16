import torch
import numpy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from functools import partial

if __name__ == '__main__':
    from src.AdditionalUtilities.DecimalPrefixes import nm, um, mm
    from src.AdditionalUtilities.Formaters import Format
    from src.AdditionalUtilities.TitledFigure import Titles
else:
    from .DecimalPrefixes import nm, um, mm
    from .Formaters import Format
    from .TitledFigure import Titles


class Colorizer:
    @staticmethod
    def _colorizer_gaussian(amplitude, average, width_on_half_amplitude, x):
        if torch.is_tensor(x):
            return amplitude * torch.exp(-(2.77*(x-average)**2) / (width_on_half_amplitude**2))
        else:
            return amplitude * numpy.exp(-(2.77*(x-average)**2) / (width_on_half_amplitude**2))

    def __init__(self,
                 R_AM=0.6, R_AV=600*nm, R_HW=120*nm,
                 r_AM=0.3, r_AV=430*nm, r_HW=40*nm,
                 G_AM=0.6, G_AV=550*nm, G_HW=100*nm,
                 B_AM=1.0, B_AV=460*nm, B_HW=60*nm,
                 I_AM=1.0, I_AV=550*nm, I_HW=250*nm):
        self.R_parameters = (R_AM, R_AV, R_HW)
        self.r_parameters = (r_AM, r_AV, r_HW)
        self.G_parameters = (G_AM, G_AV, G_HW)
        self.B_parameters = (B_AM, B_AV, B_HW)
        self.I_parameters = (I_AM, I_AV, I_HW)

    def Colorize(self, intensity : type(torch.tensor), lambdas : type(torch.tensor), norm=(0.0, 1.0)):
        intensity = intensity.movedim(1,3)
        colored_intensity = torch.zeros([3] + list(intensity.size()))

        R_filter = ((self._colorizer_gaussian(*self.R_parameters, lambdas) + self._colorizer_gaussian(*self.r_parameters, lambdas))* self._colorizer_gaussian(*self.I_parameters, lambdas)).expand_as(intensity)
        G_filter = (self._colorizer_gaussian(*self.G_parameters, lambdas) * self._colorizer_gaussian(*self.I_parameters, lambdas)).expand_as(intensity)
        B_filter = (self._colorizer_gaussian(*self.B_parameters, lambdas) * self._colorizer_gaussian(*self.I_parameters, lambdas)).expand_as(intensity)

        colored_intensity[0] = R_filter * intensity
        colored_intensity[1] = G_filter * intensity
        colored_intensity[2] = B_filter * intensity

        colored_intensity = colored_intensity.movedim(0,4).movedim(3,1)
        colored_intensity = torch.mean(colored_intensity, dim=1)

        maximum = torch.max(colored_intensity)
        minimum = torch.min(colored_intensity)

        colored_intensity = (colored_intensity - minimum) / (maximum - minimum)
        colored_intensity = colored_intensity * (norm[1] - norm[0]) + norm[0]

        return colored_intensity

    def GetFilters(self, lambdas : type(torch.tensor)):
        R_filter = (self._colorizer_gaussian(*self.R_parameters, lambdas) + self._colorizer_gaussian(*self.r_parameters, lambdas)) * self._colorizer_gaussian(*self.I_parameters, lambdas)
        G_filter = self._colorizer_gaussian(*self.G_parameters, lambdas) * self._colorizer_gaussian(*self.I_parameters, lambdas)
        B_filter = self._colorizer_gaussian(*self.B_parameters, lambdas) * self._colorizer_gaussian(*self.I_parameters, lambdas)
        return R_filter, G_filter, B_filter

    def SetupFilters(self, lambdas = torch.linspace(300*nm, 800*nm, 100), intensity_steps=20, norm_lines=False):
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Настройка раскрашивания спектров', **Format.Text('BigHeader'))
        Fig = Titles(fig, (20,24))

        axes = Fig.add_axes((1,1), (9,9))
        unitLam, multLam = Format.Engineering_Separated(torch.max(lambdas).item(), 'm')
        axes.set_title('Цвет в зависимости от\nотносительной интенсивности и длины волны', **Format.Text('Header'), pad=21)
        axes.xaxis.set_tick_params(labelsize=10)
        axes.yaxis.set_tick_params(labelsize=10)
        axes.set_xlabel('Relative Intensity', **Format.Text('Default', {'fontweight': 'bold'}))
        axes.set_ylabel('Wave Length, ' + unitLam, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
        image = axes.imshow(torch.zeros(len(lambdas), intensity_steps, 3), origin='lower', extent=[0, 1.0, torch.min(lambdas).item()*multLam, torch.max(lambdas).item()*multLam], aspect='auto')
        def update_image():
            temp = torch.zeros(1, len(lambdas), len(lambdas), intensity_steps)
            for n in range(len(lambdas)):
                temp[0][n][n] = torch.arange(intensity_steps)
            data = self.Colorize(temp, lambdas)[0].numpy()
            if norm_lines:
                for n in range(len(lambdas)):
                    maximum = numpy.max(data[n])
                    minimum = numpy.min(data[n])
                    data[n] = (data[n] - minimum) / (maximum - minimum)
            image.set_data(data)
        update_image()

        axes = Fig.add_axes((12,1), (20,9))
        unitLam, multLam = Format.Engineering_Separated(torch.max(lambdas).item(), 'm')
        axes.set_title('Функции спектральной чувствительности', **Format.Text('Header'), pad=21)
        axes.xaxis.set_tick_params(labelsize=10)
        axes.yaxis.set_tick_params(labelsize=10)
        axes.set_xlabel('Wave Length, ' + unitLam, **Format.Text('Default', {'fontweight': 'bold'}))
        axes.set_ylabel('Relative Sensitivity', **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
        axes.grid(True)
        FilterR, = axes.plot(lambdas*multLam, 1.0*torch.arange(len(lambdas)) / len(lambdas), linestyle='--', color='red')
        FilterG, = axes.plot(lambdas*multLam, 1.0*torch.arange(len(lambdas)) / len(lambdas), linestyle='--', color='green')
        FilterB, = axes.plot(lambdas*multLam, 1.0*torch.arange(len(lambdas)) / len(lambdas), linestyle='--', color='blue')
        def update_filters():
            FilterRdata, FilterGdata, FilterBdata = self.GetFilters(lambdas)
            FilterR.set_data(lambdas*multLam, FilterRdata)
            FilterG.set_data(lambdas*multLam, FilterGdata)
            FilterB.set_data(lambdas*multLam, FilterBdata)
        update_filters()

        lambda_limits   = (torch.min(lambdas).item()*multLam, torch.max(lambdas).item()*multLam)
        width_limits    = (torch.min(lambdas).item()*multLam/50, torch.max(lambdas).item()*multLam/5)
        lambda_step     = (torch.max(lambdas)-torch.min(lambdas)).item()*multLam/1000

        Sliders = []
        def setAverage(letter, x):
            self.__setattr__(letter + '_parameters', (self.__getattribute__(letter + '_parameters')[0], x / multLam, self.__getattribute__(letter + '_parameters')[2]))
            update_filters()
            update_image()
        def setWidth(letter, x):
            self.__setattr__(letter + '_parameters', (self.__getattribute__(letter + '_parameters')[0], self.__getattribute__(letter + '_parameters')[1], x / multLam))
            update_filters()
            update_image()
        def setAmplitude(letter, x):
            self.__setattr__(letter + '_parameters', (x, self.__getattribute__(letter + '_parameters')[1], self.__getattribute__(letter + '_parameters')[2]))
            update_filters()
            update_image()
        for s, (name, parameter) in enumerate([('Red', self.R_parameters), ('Green', self.G_parameters), ('Blue', self.B_parameters), ('red', self.r_parameters)]):
            AverageSlider   = Slider(Fig.add_axes((3,11+4*s+0), (20,11+4*s+0)), name+' Average '    + unitLam, *lambda_limits,  valinit=parameter[1]*multLam,   valstep=lambda_step)
            HalfWidthSlider = Slider(Fig.add_axes((3,11+4*s+1), (20,11+4*s+1)), name+' Half Width ' + unitLam, *width_limits,   valinit=parameter[2]*multLam,   valstep=lambda_step)
            AmplitudeSlider = Slider(Fig.add_axes((3,11+4*s+2), (20,11+4*s+2)), name+' Amplitude '  + unitLam, 0, 1.0,          valinit=parameter[0],           valstep=1.0/1000)

            AverageSlider.on_changed(partial(setAverage, name[0]))
            HalfWidthSlider.on_changed(partial(setWidth, name[0]))
            AmplitudeSlider.on_changed(partial(setAmplitude, name[0]))

            Sliders.append(AverageSlider)
            Sliders.append(HalfWidthSlider)
            Sliders.append(AmplitudeSlider)

        plt.show()




def test_Colorizer():
    from src.improved_mask_DNN import PaddedDiffractionLayer

    colorizer = Colorizer()
    samples = 60
    Lambdas = torch.linspace(350*nm, 700*nm, samples)
    Fields = torch.zeros(1, samples, samples, samples)
    for num, lam in enumerate(Lambdas):
        Fields[0][num][num] = torch.arange(samples)
    ColoredField = colorizer.Colorize(Fields, Lambdas)[0].numpy()

    unit, mult = Format.Engineering_Separated(torch.max(Lambdas).item(), 'm')

    fig = plt.figure(figsize=(12,12))
    fig.suptitle('Раскрашивание спектральных картинок', **Format.Text('BigHeader'))
    Fig = Titles(fig, (2,2), topspace=0.1, leftspace=0.05, rightspace=0.05)

    axes1 = Fig.add_axes((1,1))
    axes1.set_title('Цвет в зависимости от\nотносительной интенсивности и длины волны', **Format.Text('Header'), pad=21)
    axes1.xaxis.set_tick_params(labelsize=10)
    axes1.yaxis.set_tick_params(labelsize=10)
    axes1.set_xlabel('Relative Intensity', **Format.Text('Default', {'fontweight': 'bold'}))
    axes1.set_ylabel('Wave Length, ' + unit, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
    axes1.imshow(ColoredField, origin='lower', extent=[0, 1.0, torch.min(Lambdas).item()*mult, torch.max(Lambdas).item()*mult], aspect='auto')

    R_filter, G_filter, B_filter = colorizer.GetFilters(Lambdas)
    R_filter = R_filter.numpy()
    G_filter = G_filter.numpy()
    B_filter = B_filter.numpy()

    axes2 = Fig.add_axes((2, 1))
    axes2.set_title('Спектральные фильтры', **Format.Text('Header'), pad=21)
    axes2.xaxis.set_tick_params(labelsize=10)
    axes2.yaxis.set_tick_params(labelsize=10)
    axes2.set_xlabel('Wave Length, ' + unit, **Format.Text('Default', {'fontweight': 'bold'}))
    axes2.set_ylabel('Relative Sensitivity', **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
    axes2.grid(True)

    axes2.plot((Lambdas*mult).numpy(), R_filter, linestyle='--', color='red')
    axes2.plot((Lambdas*mult).numpy(), G_filter, linestyle='--', color='green')
    axes2.plot((Lambdas*mult).numpy(), B_filter, linestyle='--', color='blue')

    Fields = torch.zeros(1, samples, samples, samples)
    for num, lam in enumerate(Lambdas):
        temp = torch.zeros(samples, samples)
        temp[num][:] = torch.ones(samples)
        temp = torch.swapdims(temp, 0, 1)
        temp[num][:] = torch.ones(samples)
        temp = torch.swapdims(temp, 0, 1)
        temp[num][num] = 2
        Fields[0][num] = temp
    ColoredField = colorizer.Colorize(Fields, Lambdas)[0].numpy()

    axes3 = Fig.add_axes((1, 2))
    axes3.set_title('Смешивание волн с различными длиннами', **Format.Text('Header'), pad=21)
    axes3.xaxis.set_tick_params(labelsize=10)
    axes3.yaxis.set_tick_params(labelsize=10)
    axes3.set_xlabel('Wave Length, ' + unit, **Format.Text('Default', {'fontweight': 'bold'}))
    axes3.set_ylabel('Wave Length, ' + unit, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
    axes3.imshow(ColoredField, origin='lower', extent=[torch.min(Lambdas).item() * mult, torch.max(Lambdas).item() * mult, torch.min(Lambdas).item() * mult, torch.max(Lambdas).item() * mult], aspect='auto')

    PixelSize = 50*um
    PixelsCount = 15
    UpScaling = 31
    DiffractionLength = 20*mm
    Steps = 200
    Length = torch.linspace(0, DiffractionLength, Steps)
    InputField = torch.zeros(samples, PixelsCount*UpScaling, PixelsCount*UpScaling, dtype=torch.complex64)
    MinIndex = int(PixelsCount*UpScaling/2 - UpScaling/2)
    MaxIndex = int(MinIndex + UpScaling)
    InputField[:, MinIndex:MaxIndex, MinIndex:MaxIndex] = torch.ones(samples, UpScaling, UpScaling)

    # Cut = torch.zeros(samples, PixelsCount*UpScaling, Steps)
    # for n, l in enumerate(Length):
    #     Propagator = PaddedDiffractionLayer(Lambdas, plane_length=PixelsCount * PixelSize, pixels_count=PixelsCount, diffraction_length=l, up_scaling=UpScaling, border_length=PixelsCount * PixelSize)
    #     Cut[:, :, n] = torch.abs(Propagator.forward(InputField).clone().detach()[:,:,int(PixelsCount*UpScaling/2)])**1
    # ColoredCut = colorizer.Colorize(Cut.expand(1, -1, -1, -1), Lambdas)[0].numpy()

    Cut = torch.zeros((samples, PixelsCount*UpScaling, Steps), dtype=torch.float32)
    Propagator = PaddedDiffractionLayer(Lambdas, plane_length=PixelsCount * PixelSize, pixels_count=PixelsCount, diffraction_length=Length[1]-Length[0], up_scaling=UpScaling, border_length=PixelsCount * PixelSize/4).cuda()
    PreviousField = InputField.cuda()
    for n in range(Steps):
        Cut[:, :, n] = (torch.abs(PreviousField[:,:,int(PixelsCount*UpScaling/2)])**1).cpu()
        PreviousField = Propagator(PreviousField)
    ColoredCut = colorizer.Colorize(Cut.expand(1, -1, -1, -1), Lambdas)[0].numpy()


    unitX, multX = Format.Engineering_Separated(DiffractionLength, 'm')
    unitZ, multZ = Format.Engineering_Separated(PixelSize*PixelsCount, 'm')

    axes4 = Fig.add_axes((2, 2))
    axes4.set_title('Распространение', **Format.Text('Header'), pad=21)
    axes4.xaxis.set_tick_params(labelsize=10)
    axes4.yaxis.set_tick_params(labelsize=10)
    axes4.set_xlabel('Z, ' + unitZ, **Format.Text('Default', {'fontweight': 'bold'}))
    axes4.set_ylabel('X, ' + unitX, **Format.Text('Default', {'fontweight': 'bold', 'rotation': 90}))
    axes4.imshow(ColoredCut, origin='lower', extent=[0, DiffractionLength * multX, -PixelsCount*PixelSize*multZ/2, +PixelsCount*PixelSize*multZ/2], aspect='auto')

    plt.show()
    return
if __name__ == '__main__':
    test_Colorizer()
    # colorizer = Colorizer()
    # colorizer.SetupFilters(lambdas=torch.linspace(490*nm, 500*nm, 100))