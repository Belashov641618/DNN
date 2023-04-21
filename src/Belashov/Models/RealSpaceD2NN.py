import torch
from copy import deepcopy
from typing import Union, List, Iterable, Any
import pickle

from src.AdditionalUtilities.DecimalPrefixes import cm, mm, um, nm
from src.AdditionalUtilities.Formaters import Format

from src.Belashov.Layers.FourierPropagationLayer import FourierPropagationLayer
from src.Belashov.Layers.LensLayer import LensLayer
from src.Belashov.Layers.HeightsMaskLayer import HeightsMaskLayer
from src.Belashov.Layers.DetectorsLayer import DetectorsLayer
from src.Belashov.Layers.AmplificationLayer import AmplificationLayer


class RealSpaceD2NN(torch.nn.Module):

    _DefaultPath        : str = 'src/Belashov/Models/SavedModels/'
    _DefaultFilename    : str = 'RealSpaceD2NN.data'
    @staticmethod
    def load(file_name:str=None):
        if file_name is None:
            file_name = RealSpaceD2NN._DefaultPath + RealSpaceD2NN._DefaultFilename
        try:
            file = open(file_name, 'rb')
            self = pickle.load(file)
            file.close()
            return self
        except Exception as e:
            print(e)
    def save(self, file_name:str=None):
        if file_name is None:
            file_name = RealSpaceD2NN._DefaultPath + RealSpaceD2NN._DefaultFilename
        try:
            file = open(file_name, 'wb')
            pickle.dump(self, file)
            file.close()
        except Exception as e:
            print(e)


    _accuracy_bits          : int
    _tensor_float_type      : torch.dtype
    _tensor_complex_type    : torch.dtype
    def AccuracySet(self, bits:int):
        if bits == 32:
            self._tensor_float_type     = torch.float32
            self._tensor_complex_type   = torch.complex64
        elif bits == 64:
            self._tensor_float_type     = torch.float64
            self._tensor_complex_type   = torch.complex128
    def AccuracyGet(self):
        return self._accuracy_bits


    _AmplificationModule            : torch.nn.Module
    _DetectorsModule                : torch.nn.Module
    _HeightsMaskModuleList          : torch.nn.ModuleList
    _HeightsMaskPropagationModule   : torch.nn.Module

    def FinalizeChanges(self):
        modules_to_finalize_list = [self._DetectorsModule, self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in modules_to_finalize_list:
            if hasattr(module, 'FinalizeChanges'):
                getattr(module, 'FinalizeChanges')()


    _WaveLength : torch.Tensor
    @property
    def WaveLength(self):
        return self._WaveLength
    @WaveLength.setter
    def WaveLength(self, wave_length:Union[float,Iterable,torch.Tensor]):
        self._WaveLength = (wave_length.to(self._tensor_float_type).requires_grad_(False) if torch.is_tensor(wave_length) else torch.tensor([wave_length] if type(wave_length) is float else wave_length, requires_grad=False, dtype=self._tensor_float_type))
        synchronize_modules = [self._DetectorsModule, self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'WaveLength'):
                setattr(module, 'WaveLength', self._WaveLength)

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
        synchronize_modules = [self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'SpaceReflection'):
                setattr(module, 'SpaceReflection', self._SpaceReflection)

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
        synchronize_modules = list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'MaskReflection'):
                setattr(module, 'MaskReflection', self._MaskReflection)

    _PlaneLength : float
    @property
    def PlaneLength(self):
        return self._PlaneLength
    @PlaneLength.setter
    def PlaneLength(self, plane_length:float):
        self._PlaneLength = plane_length
        synchronize_modules = [self._DetectorsModule, self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'PlaneLength'):
                setattr(module, 'PlaneLength', self._PlaneLength)

    _PixelsCount : int
    @property
    def PixelsCount(self):
        return self._PixelsCount
    @PixelsCount.setter
    def PixelsCount(self, pixels_count:int):
        self._PixelsCount = pixels_count
        synchronize_modules = [self._DetectorsModule, self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'PixelsCount'):
                setattr(module, 'PixelsCount', self._PixelsCount)

    _UpScaling : int
    @property
    def UpScaling(self):
        return self._UpScaling
    @UpScaling.setter
    def UpScaling(self, up_scaling:int):
        self._UpScaling = up_scaling
        synchronize_modules = [self._DetectorsModule, self._HeightsMaskPropagationModule] + list(self._HeightsMaskModuleList)
        for module in synchronize_modules:
            if hasattr(module, 'UpScaling'):
                setattr(module, 'UpScaling', self._UpScaling)

    _DiffractionLength : float
    @property
    def DiffractionLength(self):
        if self._DiffractionLengthAsParameter:
            self._DiffractionLength = float(torch.abs(self._HeightsMaskPropagationModule.DiffractionLength.data[0].detach()))
        return self._DiffractionLength
    @DiffractionLength.setter
    def DiffractionLength(self, diffraction_length:float):
        self._DiffractionLength = diffraction_length
        synchronize_modules = [self._HeightsMaskPropagationModule]
        for module in synchronize_modules:
            if hasattr(module, 'DiffractionLength'):
                setattr(module, 'DiffractionLength', self._DiffractionLength)

    _MaskBorderLength : float
    @property
    def MaskBorderLength(self):
        return self._MaskBorderLength
    @MaskBorderLength.setter
    def MaskBorderLength(self, mask_propagation_border_length:float):
        self._MaskBorderLength = mask_propagation_border_length
        synchronize_modules = [self._HeightsMaskPropagationModule]
        for module in synchronize_modules:
            if hasattr(module, 'BorderLength'):
                setattr(module, 'BorderLength', self._MaskBorderLength)



    _DetectorsEnable : bool
    def EnableDetectors(self, state:bool=True):
        self._DetectorsEnable = state
    def DisableDetectors(self, state:bool=True):
        self._DetectorsEnable = not state

    _DiffractionLengthAsParameter : bool
    def DiffractionLengthAsParameter(self, mode:bool=True):
        if hasattr(self._HeightsMaskPropagationModule, 'DiffractionLengthAsParameter'):
            getattr(self._HeightsMaskPropagationModule, 'DiffractionLengthAsParameter')(mode)
        else:
            raise AttributeError("\033[31m\033[1m{}".format('RealSpaceD2NN: HeightsMaskPropagationModule must have attribute "DiffractionLengthAsParameter"!'))
        self._DiffractionLengthAsParameter = mode


    def __init__(self, masks_count:int=5, wave_length:Union[float,Iterable,torch.Tensor]=600*nm, space_reflection:Union[float,Iterable,torch.Tensor]=1.0, mask_reflection:Union[float, Iterable, torch.Tensor]=1.5, plane_length:float=1.0*mm, pixels_count:int=21, up_scaling:int=20, mask_interspacing:float=20.0*mm, mask_propagation_border_length:float=0.0, smoothing_matrix:Union[None,Iterable,torch.Tensor]=None, detectors_masks:Union[str,Iterable,torch.Tensor]='Square'):
        """
        :param masks_count:                     Количество высотных масок в модели.
        :param wave_length:                     Длинна волны излучения (можно использовать набор длинн волн)
        :param space_reflection:                Коэффициент преломления среды (если использовать единое значение, то оно продублируется для всех длинн волн)
        :param mask_reflection:                 Коэффициент преломления высотных масок (если использовать единое значение, то оно продублируется для всех длинн волн)
        :param plane_length:                    Размер высотной макси в метрах
        :param pixels_count:                    Количество пикселей в высотной маске (пиксели принтера)
        :param up_scaling:                      Множитель разрешения (количество вычислительных пикселей = up_scaling*pixels_count)
        :param mask_interspacing:               Расстояние между высотными масками
        :param mask_propagation_border_length:  Размер падинга при расчёте дифракции между слоями в метрах
        :param lens_propagation_border_length:  Размер падинга при расчёте дифракции вокруг линз в метрах
        :param smoothing_matrix:                Ядро сглаживания высотных масок (если None, то сглаживания не будет)
        :param focus_length:                    Фокусное растояние линз
        :param detectors_masks:                 Маски детекторов (можно указывать с помощью строки для автоматической генерации, например 'Square')
        """
        super(RealSpaceD2NN, self).__init__()

        self._accuracy_bits = 32
        self.AccuracySet(self._accuracy_bits)

        self._AmplificationModule           = AmplificationLayer()
        self._DetectorsModule               = DetectorsLayer(detectors_masks=detectors_masks)
        self._HeightsMaskModuleList         = torch.nn.ModuleList([HeightsMaskLayer(smoothing_matrix=smoothing_matrix) for i in range(masks_count)])
        self._HeightsMaskPropagationModule  = FourierPropagationLayer()

        self.WaveLength         = wave_length
        self.SpaceReflection    = space_reflection
        self.MaskReflection     = mask_reflection
        self.PlaneLength        = plane_length
        self.PixelsCount        = pixels_count
        self.UpScaling          = up_scaling
        self.DiffractionLength  = mask_interspacing
        self.MaskBorderLength   = mask_propagation_border_length

        self._DetectorsEnable   = True

        self._DiffractionLengthAsParameter = False

    def forward(self, field:torch.Tensor, record_history:bool=False):
        field = self._AmplificationModule(field)
        if record_history:
            length = 0

            history = [('Амплитуда перед маской №' + str(1) + ': ' + Format.Engineering(length, 'm', 1), (torch.abs(field) ** 1).cpu())]

            for i, _MaskLayer in enumerate(self._HeightsMaskModuleList[:-1]):
                field = _MaskLayer(field)
                field = self._HeightsMaskPropagationModule(field)
                length += self._DiffractionLength
                history.append(('Амплитуда перед маской №' + str(i+2) + ': ' + Format.Engineering(length, 'm', 1), (torch.abs(field) ** 1).cpu()))
            field = self._HeightsMaskModuleList[-1](field)

            history.append(('Амплитуда перед детекторами' + ': ' + Format.Engineering(length, 'm', 1), (torch.abs(field) ** 1).cpu()))

            if self._DetectorsEnable:
                labels = self._DetectorsModule(field)
            else:
                labels = field
            return labels, history
        else:

            for _MaskLayer in self._HeightsMaskModuleList[:-1]:
                field = _MaskLayer(field)
                field = self._HeightsMaskPropagationModule(field)
            field = self._HeightsMaskModuleList[-1](field)

            if self._DetectorsEnable:
                labels = self._DetectorsModule(field)
            else:
                labels = field
            return labels