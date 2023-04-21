import torch
from torch.utils.data import DataLoader as TorchDataLoader
import torchvision
from typing import Union, Iterable, List, Any, Tuple, Dict
from itertools import product
import inspect
from functools import partial
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src.AdditionalUtilities.Colorize import Colorizer
from src.AdditionalUtilities.Formaters import Format
from src.AdditionalUtilities.DecimalPrefixes import nm, um, mm, cm
from src.AdditionalUtilities.TitledFigure import Titles

from src.Belashov.Layers.DetectorsLayer import DetectorsLayer

def StringToDataSetRedirector(data_set_name:str, train:bool=True, transformation:Any=None, input_pixels:int=None):

    data_sets_root = 'data/'

    data_sets_dict = {
        'MNIST' :           (torchvision.datasets.MNIST,        ['mnist', 'Mnist']),
        'FashionMNIST' :    (torchvision.datasets.FashionMNIST, ['FMNIST', 'Fmnist', 'fashion_mnist']),
        'CIFAR10' :         (torchvision.datasets.CIFAR10,      ['cifar10', 'Cifar10']),
    }

    if (transformation is None) and (input_pixels is not None):
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(input_pixels, input_pixels)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])

    if data_set_name in data_sets_dict.keys():
        return data_sets_dict[data_set_name][0](train=train, download=True, transform=transformation, root=data_sets_root)
    else:
        for key, (data_set, other_names) in data_sets_dict.items():
            if data_set_name in other_names:
                return data_set(train=train, download=True, transform=transformation, root=data_sets_root)

    raise ValueError("\033[31m\033[1m{}".format('There is no data set with name:' + data_set_name + '!'))


def GenerateSingleUnscaledSampleMNIST(only_image=False):
    transformation = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomRotation((-90, -90)),
        torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
    loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    image, labels = next(iter(loader))
    if only_image:
        return image
    return image, labels


def ConvertModelAttributesToString(Model:torch.nn.Module, ExcludeAttributes:Union[List,Tuple,Iterable]=None, AttributeToNameAndFormat:Dict=None, CombineDicts:bool=True):
    DefaultAttributeToNameAndFormat = {
        'WaveLength':           ('Длинна волны',                            partial(Format.Engineering, unit='m',   precision=1)),
        'SpaceReflection':      ('Коэффициент преломления среды',           partial(Format.Scientific,  unit='',    precision=1)),
        'MaskReflection':       ('Коэффициент преломления масок',           partial(Format.Scientific,  unit='',    precision=1)),
        'PlaneLength':          ('Размер масок',                            partial(Format.Engineering, unit='m',   precision=1)),
        'PixelsCount':          ('Количество пикселей масок',               partial(Format.Engineering, unit='шт',  precision=1)),
        'UpScaling':            ('Множитель разрешения',                    partial(Format.Scientific,  unit='',    precision=1)),
        'DiffractionLength':    ('Расстояние между масками',                partial(Format.Engineering, unit='m',   precision=1)),
        'MaskBorderLength':     ('Паддинг при диффракции между масками',    partial(Format.Engineering, unit='m',   precision=1)),
        'LensBorderLength':     ('Паддинг при диффракции вокруг линз',      partial(Format.Engineering, unit='m',   precision=1)),
        'FocusLength':          ('Фокусное расстояние линз',                partial(Format.Engineering, unit='m',   precision=1))
    }

    if CombineDicts and (AttributeToNameAndFormat is not None):
        DefaultAttributeToNameAndFormat.update(AttributeToNameAndFormat)
    elif AttributeToNameAndFormat is not None:
        DefaultAttributeToNameAndFormat = AttributeToNameAndFormat
    AttributeToNameAndFormat = DefaultAttributeToNameAndFormat

    if ExcludeAttributes is not None:
        for key in ExcludeAttributes: AttributeToNameAndFormat.pop(key)

    String = 'Параметры модели: '
    for attribute_name, (name, format_function) in AttributeToNameAndFormat.items():
        if hasattr(Model, attribute_name):
            value = getattr(Model, attribute_name)
            if torch.is_tensor(value):
                value = value.item()
            String += '(' + name + ' : ' + format_function(value) + '), '
    String = String[:-2]
    return String
def ConvertModelToString(Model:torch.nn.Module, _string:str='', _numeration:List=None, show_variables:bool=True):
    if hasattr(Model, 'FinalizeChanges'):
        getattr(Model, 'FinalizeChanges')()

    attributes_list = [attribute_name for attribute_name in dir(Model) if not (attribute_name.startswith('__') or (callable(getattr(Model, attribute_name)) and not isinstance(getattr(Model, attribute_name), (torch.nn.Module, torch.nn.ModuleList))) or (isinstance(getattr(type(Model), attribute_name), property) if hasattr(type(Model), attribute_name) else False))]

    unsolved_list = []

    modules_lists_list  = []
    modules_list        = []
    variables_list      = []
    buffers_list        = []
    parameters_list     = []

    for attribute_name in attributes_list:
        attribute = getattr(Model, attribute_name)
        if isinstance(attribute, torch.nn.ModuleList):
            modules_lists_list.append(attribute_name)
        elif isinstance(attribute, torch.nn.Module):
            modules_list.append(attribute_name)
        elif isinstance(attribute, (int, float, complex)):
            variables_list.append(attribute_name)
        elif isinstance(attribute, torch.Tensor):
            if attribute_name in dict(Model.named_buffers()).keys():
                buffers_list.append(attribute_name)
            elif attribute_name in dict(Model.named_parameters()).keys():
                parameters_list.append(attribute_name)
            else:
                variables_list.append(attribute_name)
        else:
            unsolved_list.append(attribute_name)

    FirstCall = False
    if _numeration is None:
        _numeration = [0]
        _string += 'Model ' + type(Model).__name__ + ' attributes info:\n'
        FirstCall = True
    def numeration_to_string(numeration):
        numeration_string = '\t' + str(numeration[0]) + '.'
        for num in numeration[1:]:
            numeration_string = '\t' + numeration_string + str(num) + '.'
        return numeration_string


    if len(parameters_list) > 0:
        _numeration[-1] += 1
        _string += numeration_to_string(_numeration) + ' Parameters:\n'
        _numeration.append(1)
        for name in parameters_list:
            if not hasattr(Model, name):
                pass
            parameter = getattr(Model, name)
            _string += numeration_to_string(_numeration) + ' ' + name + ' - '
            _string += 'size:' + str(list(parameter.size())) + ', '
            _string += 'requires_grad:' + str(parameter.requires_grad) + ', '
            _string += 'memory:' + Format.Memory(parameter.numel() * parameter.element_size() if not (torch.sum(torch.isnan(parameter)).item() > 0) else 0)
            _string += '\n'
            _numeration[-1] += 1
        _numeration = _numeration[:-1]

    if len(buffers_list) > 0:
        _numeration[-1] += 1
        _string += numeration_to_string(_numeration) + ' Buffers:\n'
        _numeration.append(1)
        for name in buffers_list:
            if not hasattr(Model, name):
                pass
            buffer = getattr(Model, name)
            _string += numeration_to_string(_numeration) + ' ' + name + ' - '
            _string += 'size:' + str(list(buffer.size())) + ', '
            _string += 'requires_grad:' + str(buffer.requires_grad) + ', '
            _string += 'memory:' + Format.Memory(buffer.numel() * buffer.element_size() if not (torch.sum(torch.isnan(buffer)).item() > 0) else 0)
            _string += '\n'
            _numeration[-1] += 1
        _numeration = _numeration[:-1]

    if len(variables_list) > 0 and show_variables:
        _numeration[-1] += 1
        _string += numeration_to_string(_numeration) + ' Variables:\n'
        _numeration.append(1)
        for name in variables_list:
            if not hasattr(Model, name):
                pass
            variable = getattr(Model, name)
            if torch.is_tensor(variable):
                if variable.size(0) == 1:
                    variable = variable.item()
                else:
                    variable = list(variable)
            _string += numeration_to_string(_numeration) + ' ' + name + ' - '
            _string += str(variable)
            _string += '\n'
            _numeration[-1] += 1
        _numeration = _numeration[:-1]

    if len(modules_list) + len(modules_lists_list) > 0:
        _numeration[-1] += 1
        _string += numeration_to_string(_numeration) + ' Modules:\n'
        _numeration.append(1)
        for name in modules_lists_list:
            if not hasattr(Model, name):
                pass
            _modules_list = getattr(Model, name)
            _string += numeration_to_string(_numeration) + ' ' + name + ':\n'
            _numeration.append(1)
            for i, module in enumerate(_modules_list):
                _string += numeration_to_string(_numeration) + ' ' + str(module)[:-2] + ' №' + str(i + 1) + ':\n'
                _numeration.append(1)
                _string, _numeration = ConvertModelToString(module, _string, _numeration, show_variables)
                _numeration = _numeration[:-1]
                _numeration[-1] += 1
            _numeration = _numeration[:-1]
            _numeration[-1] += 1

        for name in modules_list:
            if not hasattr(Model, name):
                pass
            module = getattr(Model, name)
            _string += numeration_to_string(_numeration) + ' ' + name + ':\n'
            _numeration.append(0)
            _string, _numeration = ConvertModelToString(module, _string, _numeration, show_variables)
            _numeration = _numeration[:-1]
            _numeration[-1] += 1
        _numeration = _numeration[:-1]

    if FirstCall:
        _numeration[-1] += 1
        _string += numeration_to_string(_numeration) + ' Characteristics:\n'
        _numeration.append(1)

        total_memory = sum(p.numel() * p.element_size() for p in Model.parameters())
        total_memory += sum(b.numel() * b.element_size() for b in Model.buffers())
        _string += numeration_to_string(_numeration) + ' Total Memory Usage: ' + Format.Memory(total_memory) + '\n'
        _numeration[-1] += 1

        if torch.cuda.is_available():
            device = torch.device('cuda')
            Model.cpu()

            allocated_memory0   = torch.cuda.memory_allocated(device=device)
            Model.to(device)
            allocated_memory1   = torch.cuda.memory_allocated(device=device)
            Model.cpu()
            _string += numeration_to_string(_numeration) + ' Real Total Memory Usage: ' + Format.Memory(allocated_memory1 - allocated_memory0) + '\n'
            _numeration[-1] += 1

            reserved_memory      = torch.cuda.memory_reserved(device=device)
            _string += numeration_to_string(_numeration) + ' Total Reserved Memory: ' + Format.Memory(reserved_memory) + '\n'
            _numeration[-1] += 1

        _numeration = _numeration[:-1]

    if FirstCall:
        return _string
    return _string, _numeration


def CalculateMaximumBatchSize(Model:torch.nn.Module, input_size:Union[Tuple,List,int]=None):
    if not torch.cuda.is_available():
        raise SystemError("\033[31m\033[1m{}".format('CalculateMaximumBatchSize: Torch cuda is not avaliable!'))

    device = torch.device('cuda')

    if input_size is None:
        UpScaling = 0
        if hasattr(Model, 'UpScaling'):
            UpScaling = getattr(Model, 'UpScaling')
        else: raise AttributeError("\033[31m\033[1m{}".format('CalculateMaximumBatchSize: Model must have attribute "UpScaling" or define "input_size" in function call!'))
        PixelsCount = 0
        if hasattr(Model, 'PixelsCount'):
            PixelsCount = getattr(Model, 'PixelsCount')
        else: raise AttributeError("\033[31m\033[1m{}".format('CalculateMaximumBatchSize: Model must have attribute "PixelsCount" or define "input_size" in function call!'))
        input_size = [PixelsCount*UpScaling, PixelsCount*UpScaling]
    if isinstance(input_size, int):
        input_size = [input_size]
    if type(input_size) is Tuple:
        input_size = list(input_size)

    if len(input_size) == 2:
        input_size = [1] + input_size

    torch.cuda.empty_cache()

    Model.to(device)
    Model.train()

    batch_size:int = 1
    while True:
        try:
            Model(torch.rand([batch_size] + input_size).to(device))
            batch_size = int(batch_size*2)
        except torch.cuda.OutOfMemoryError:
            break
    delta:int = int(batch_size/2)
    while delta != 0:
        try:
            for i in range(5):
                output = Model(torch.rand([batch_size] + input_size).to(device))
                output_ = deepcopy(output.clone().detach())
            batch_size += delta
            delta = int(delta/2)
        except torch.cuda.OutOfMemoryError:
            batch_size -= delta
    while True:
        try:
            for i in range(10):
                Model(torch.rand([batch_size] + input_size).to(device))
            break
        except torch.cuda.OutOfMemoryError:
            batch_size -= 1
    torch.cuda.empty_cache()

    Model.eval()
    Model.cpu()

    return batch_size


def CalculatePropagationLayerEmission(Layer:torch.nn.Module, initial_field:torch.tensor, length_limits:Iterable=None, length_steps:int=100, relative_cut_position:float=0.5, use_fast_recurrent_method:bool=False):
    """
    :param Layer: Слой, считающий распространение света.
    :param initial_field: Начальное распределение поля.
    :param length_limits: Минимальное и максимальное расстояние расчёта.
    :param length_steps: Количество шагов расчёта.
    :param relative_cut_position: Относительное расположение разреза от 0 до 1.
    :param use_fast_recurrent_method: Разрешить ускорение вычислений путём итеративного расчёта поля.
    :return: Тензор [N][length_steps], срез распространяющегося поля плоскостью, перпендикулярной направлению распространения.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Layer.to(device)
    initial_field.to(device)

    for attribute_name in ['PixelsCount', 'UpScaling', 'DiffractionLength', 'WaveLength']:
        if not hasattr(Layer, attribute_name): raise AttributeError("\033[31m\033[1m{}".format('CalculatePropagationLayerEmission: Layer must have attribute ' + attribute_name + '!'))

    if length_limits is None:
        length_limits = (0, getattr(Layer, 'DiffractionLength'))

    WaveLength  = getattr(Layer, 'WaveLength')
    InputPixels = int(getattr(Layer, 'PixelsCount') * getattr(Layer, 'UpScaling'))
    CutPixel    = int((InputPixels-1)*relative_cut_position)
    DiffractionLength = getattr(Layer, 'DiffractionLength')

    if not torch.is_tensor(WaveLength): raise TypeError("\033[31m\033[1m{}".format('CalculatePropagationLayerEmission: Layer attribute WaveLength must be a tensor!'))
    if initial_field.size() != torch.Size((1, WaveLength.size(0), InputPixels, InputPixels)): raise ValueError("\033[31m\033[1m{}".format('CalculatePropagationLayerEmission: initial_field must have size [1, WaveLength.size(0), PixelsCount*UpScaling, PixelsCount*UpScaling]!'))

    StackedFieldsIntensity = torch.zeros((length_steps, WaveLength.size(0), InputPixels), dtype=torch.float32, requires_grad=False)
    with torch.no_grad():
        if use_fast_recurrent_method:
            if length_limits[0] != 0:
                setattr(Layer, 'DiffractionLength', length_limits[0])
                field = Layer(initial_field)
            else:
                field = initial_field
            StackedFieldsIntensity[0] = (torch.abs(field[0,:,CutPixel,:])**1).cpu()
            dl = (length_limits[1] - length_limits[0]) / (length_steps - 1)
            setattr(Layer, 'DiffractionLength', dl)
            for i in range(1, length_steps):
                field = Layer(field)
                # print('NaNs:', torch.sum(torch.isnan(field)), 'Mean:', torch.mean(torch.abs(field)), 'For buffer:', field.size())
                StackedFieldsIntensity[i] = (torch.abs(field[0,:,CutPixel,:])**1).cpu()
        else:
            for i, length in enumerate(torch.linspace(length_limits[0], length_limits[1], length_steps)):
                setattr(Layer, 'DiffractionLength', length.item())
                field = Layer(initial_field)
                # print('NaNs:', torch.sum(torch.isnan(field)), 'Mean:', torch.mean(torch.abs(field)), 'For buffer:', field.size())
                StackedFieldsIntensity[i] = (torch.abs(field[0,:,CutPixel,:])**1).cpu()
        StackedFieldsIntensity = StackedFieldsIntensity.movedim(0, 2)
        if WaveLength.size(0) > 1:
            colorizer = Colorizer()
            StackedFieldsIntensity = colorizer.Colorize(StackedFieldsIntensity.expand(1,-1,-1,-1)**2, WaveLength).squeeze()
        else:
            StackedFieldsIntensity = StackedFieldsIntensity[0]

    setattr(Layer, 'DiffractionLength', DiffractionLength)
    return StackedFieldsIntensity
def CalculateLensSystemImage(PropagationLayer:torch.nn.Module, LensLayer:torch.nn.Module, focus_length:float=10*cm, lens_spacing_in_focus_lengths=1.0, initial_field:torch.tensor=None, length_before_lens:float=None, length_after_lens:float=None):
    if (length_before_lens is not None) and (length_after_lens is not None):
        lens_spacing_in_focus_lengths = None
    else:
        length_before_lens  = focus_length*lens_spacing_in_focus_lengths
        length_after_lens   = focus_length*lens_spacing_in_focus_lengths

    PropagationLayerAttributes  = ['DiffractionLength', 'WaveLength', 'PixelsCount', 'UpScaling', 'PlaneLength']
    LensLayerAttributes         = ['WaveLength', 'PixelsCount', 'UpScaling', 'PlaneLength']

    for attribute_name in PropagationLayerAttributes:
        if not hasattr(PropagationLayer,    attribute_name): raise AttributeError("\033[31m\033[1m{}".format('CalculateLensSystemImage: PropagationLayer must have attribute ' + attribute_name + '!'))
    for attribute_name in LensLayerAttributes:
        if not hasattr(LensLayer,           attribute_name): raise AttributeError("\033[31m\033[1m{}".format('CalculateLensSystemImage: LensLayer must have attribute ' + attribute_name + '!'))

    PixelsCount = getattr(PropagationLayer, 'PixelsCount')
    UpScaling   = getattr(PropagationLayer, 'UpScaling')
    CalculatingPixelsCount = int(PixelsCount*UpScaling)
    WaveLength = getattr(PropagationLayer, 'WaveLength')

    if initial_field is None:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(CalculatingPixelsCount, CalculatingPixelsCount)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        initial_field, labels = next(iter(loader))
        if WaveLength.size(0) > 1:
            initial_field = initial_field.expand(-1, WaveLength.size(0), -1, -1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    PropagationLayer.to(device)
    LensLayer.to(device)
    initial_field = initial_field.to(device)

    with torch.no_grad():
        field = initial_field
        setattr(PropagationLayer, 'DiffractionLength', length_before_lens)
        field = PropagationLayer(field)
        field = LensLayer(field)
        setattr(PropagationLayer, 'DiffractionLength', length_after_lens)
        field = PropagationLayer(field)

        input_field     = torch.abs(initial_field)**1
        output_field    = torch.abs(field)**1

        if WaveLength.size(0) > 1:
            colorizer = Colorizer()
            input_field     = colorizer.Colorize(input_field, WaveLength).squeeze()
            output_field    = colorizer.Colorize(output_field, WaveLength).squeeze()
        else:
            input_field     = input_field.squeeze()
            output_field    = output_field.squeeze()

    return input_field, output_field


def DrawPropagationLayerSinglePixelEmission(axis:Axes, Layer:torch.nn.Module, length_limits:Iterable=None, length_steps:int=100, draw_lines:int=8, use_fast_recurrent_method:bool=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for attribute_name in ['PixelsCount', 'UpScaling', 'DiffractionLength', 'WaveLength', 'PlaneLength']:
        if not hasattr(Layer, attribute_name): raise AttributeError("\033[31m\033[1m{}".format('DrawPropagationLayerSinglePixelEmission: Layer must have attribute ' + attribute_name + '!'))

    if length_limits is None:
        length_limits = (0, getattr(Layer, 'DiffractionLength'))

    WaveLength  = getattr(Layer, 'WaveLength')
    PlaneLength = getattr(Layer, 'PlaneLength')
    InputPixels = int(getattr(Layer, 'PixelsCount') * getattr(Layer, 'UpScaling'))
    PixelsCount = int(getattr(Layer, 'PixelsCount'))
    UpScaling   = int(getattr(Layer, 'UpScaling'))

    if not torch.is_tensor(WaveLength): raise TypeError("\033[31m\033[1m{}".format('DrawPropagationLayerSinglePixelEmission: Layer attribute WaveLength must be a tensor!'))

    InputField = torch.zeros((1,WaveLength.size(0),InputPixels,InputPixels), dtype=torch.complex64, requires_grad=False).to(device)
    Center = int(InputPixels/2)
    Nx1 = int(InputPixels/2 + UpScaling/2)
    Ny1 = int(InputPixels/2 + UpScaling/2)
    Nx0 = Nx1 - UpScaling
    Ny0 = Ny1 - UpScaling
    InputField[:,:,Nx0:Nx1,Ny0:Ny1] = torch.ones((1, WaveLength.size(0), UpScaling, UpScaling))

    StackedFieldsIntensity = CalculatePropagationLayerEmission(Layer, InputField, length_limits, length_steps, 0.5, use_fast_recurrent_method)

    unitX, multX = Format.Engineering_Separated(PlaneLength, 'm')
    unitY, multY = Format.Engineering_Separated(length_limits[1], 'm')
    extent = [-PlaneLength*multX/2, +PlaneLength*multX/2, length_limits[0]*multY, length_limits[1]*multY]

    image = axis.imshow(StackedFieldsIntensity.swapaxes(0,1), origin='lower', extent=extent, aspect='auto')
    axis.xaxis.set_tick_params(labelsize=8)
    axis.yaxis.set_tick_params(labelsize=8)
    axis.set_xlabel('X, ' + unitX, Format.Text('Caption'))
    axis.set_ylabel('Z, ' + unitY, Format.Text('Caption', {'rotation':90}))

    if draw_lines:
        axis.autoscale(enable=False)
        MeanWaveLength = torch.mean(WaveLength)
        axis.axline((0, 0), (0, length_limits[1]*multY), linestyle='--', linewidth=1.5, color='maroon', alpha=1.0)
        for k in range(1, draw_lines+1):
            tan = torch.sqrt(1.0 / (1.0 - (MeanWaveLength * (k + 0.5) * PixelsCount / PlaneLength) ** 2) - 1.0)
            xy0 = (0, 0)
            y1 = length_limits[1]*multY
            x1 = y1*tan*multX/multY
            axis.axline(xy0, (0 + x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)
            axis.axline(xy0, (0 - x1, y1), linestyle='--', linewidth=(1.5 / (k + 0.5)) ** 0.6, color='maroon', alpha=(1.0 / (k + 0.5)) ** 0.6)

    return image
def DrawLensSystemImages(input_field_axis:Axes, output_field_axis:Axes, PropagationLayer:torch.nn.Module, LensLayer:torch.nn.Module, focus_length:float=10*cm, lens_spacing_in_focus_lengths=1.0, initial_field:torch.tensor=None, length_before_lens:float=None, length_after_lens:float=None):
    if (length_before_lens is not None) and (length_after_lens is not None):
        lens_spacing_in_focus_lengths = None
    else:
        length_before_lens  = focus_length*lens_spacing_in_focus_lengths
        length_after_lens   = focus_length*lens_spacing_in_focus_lengths

    PropagationLayerAttributes = ['DiffractionLength', 'WaveLength', 'PixelsCount', 'UpScaling', 'PlaneLength']
    LensLayerAttributes = ['WaveLength', 'PixelsCount', 'UpScaling', 'PlaneLength']

    for attribute_name in PropagationLayerAttributes:
        if not hasattr(PropagationLayer, attribute_name): raise AttributeError("\033[31m\033[1m{}".format('CalculateLensSystemImage: PropagationLayer must have attribute ' + attribute_name + '!'))
    for attribute_name in LensLayerAttributes:
        if not hasattr(LensLayer, attribute_name): raise AttributeError("\033[31m\033[1m{}".format('CalculateLensSystemImage: LensLayer must have attribute ' + attribute_name + '!'))

    PlaneLength = getattr(PropagationLayer, 'PlaneLength')
    PixelsCount = getattr(PropagationLayer, 'PixelsCount')
    UpScaling   = getattr(PropagationLayer, 'UpScaling')
    CalculatingPixelsCount = int(PixelsCount*UpScaling)
    WaveLength = getattr(PropagationLayer, 'WaveLength')

    if initial_field is None:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(CalculatingPixelsCount, CalculatingPixelsCount)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        initial_field, labels = next(iter(loader))
        if WaveLength.size(0) > 1:
            initial_field = initial_field.expand(-1, WaveLength.size(0), -1, -1)

    input_field, output_field = CalculateLensSystemImage(PropagationLayer, LensLayer, focus_length=focus_length, initial_field=initial_field, length_before_lens=length_before_lens, length_after_lens=length_after_lens)
    input_field     = input_field.cpu()
    output_field    = output_field.cpu()

    unit, mult = Format.Engineering_Separated(PlaneLength, 'm')
    extent = [-PlaneLength*mult/2, +PlaneLength*mult/2, -PlaneLength*mult/2, +PlaneLength*mult/2]

    input_image = input_field_axis.imshow(input_field.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto')
    input_field_axis.xaxis.set_tick_params(labelsize=8)
    input_field_axis.yaxis.set_tick_params(labelsize=8)
    input_field_axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
    input_field_axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

    output_image = output_field_axis.imshow(output_field.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto')
    output_field_axis.xaxis.set_tick_params(labelsize=8)
    output_field_axis.yaxis.set_tick_params(labelsize=8)
    output_field_axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
    output_field_axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

    return input_image, output_image
def DrawThroughModelPropagation(axes:Union[List,Tuple,Iterable], Model:torch.nn.Module, initial_field:torch.tensor=None, input_pixels:int=None):
    if initial_field is None:
        if input_pixels is None:
            PixelsCount = 0
            UpScaling = 0
            if hasattr(Model, 'PixelsCount'):   PixelsCount = getattr(Model, 'PixelsCount')
            else: raise AttributeError("\033[31m\033[1m{}".format('DrawThroughModelPropagation: Model must have attribute "PixelsCount" or define input_pixels in function call!'))
            if hasattr(Model, 'UpScaling'):     UpScaling = getattr(Model, 'UpScaling')
            else: raise AttributeError("\033[31m\033[1m{}".format('DrawThroughModelPropagation: Model must have attribute "UpScaling" or define input_pixels in function call!'))
            input_pixels = UpScaling*PixelsCount
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.Resize(size=(input_pixels, input_pixels)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        initial_field, labels = next(iter(loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model.to(device)
    initial_field = initial_field.to(device)

    output_field, history = None, None
    if hasattr(Model, 'DisableDetectors'):
        getattr(Model, 'DisableDetectors')()
    with torch.no_grad():
        if 'record_history' in inspect.signature(Model.forward).parameters:
            output_field, history = Model.forward(initial_field, record_history=True)
        else:
            output_field, history = Model.forward(initial_field)

    if len(history) > len(axes):
        print("\033[31m\033[1m{}".format('DrawThroughModelPropagation: len(history) > len(axes), so contracting history to axes length (' + str(len(axes)) + ')!'))
        history = history[:len(axes)]
    if len(history) < len(axes):
        print("\033[31m\033[1m{}".format('DrawThroughModelPropagation: len(history) < len(axes), so contracting axes to history length (' + str(len(history)) + ')!'))
        axes = axes[:len(history)]

    if hasattr(Model, 'PlaneLength'):
        PlaneLength = getattr(Model, 'PlaneLength')
        unit, mult = Format.Engineering_Separated(PlaneLength, 'm')
        extent = [-PlaneLength * mult / 2, +PlaneLength * mult / 2, -PlaneLength * mult / 2, +PlaneLength * mult / 2]
    else:
        unit = ''
        mult = 1.0
        extent = [-0.5, +0.5, -0.5, +0.5]

    images = []
    for i, (data, axis) in enumerate(zip(history, axes)):
        if isinstance(data[0], str):
            name, image = data
        else:
            name = 'Срез №' + str(i+1)
            image = data
        if image.size(0) != 1:
            raise ValueError("\033[31m\033[1m{}".format('DrawThroughModelPropagation: this method does not support batched input!'))
        if image.size(1) > 2:
            if hasattr(Model, 'WaveLength'):
                colorizer = Colorizer()
                image = colorizer.Colorize(image, getattr(Model, 'WaveLength'))
            else:
                image = torch.sum(image, dim=1)
        image = image.squeeze()

        image_ = axis.imshow(image.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto')
        axis.set_title(name, **Format.Text('Default', {'fontsize':9}))
        axis.xaxis.set_tick_params(labelsize=8)
        axis.yaxis.set_tick_params(labelsize=8)
        axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
        axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

        images.append(image_)
    return images


def PlotDiffractionSystemOneParameterDistribution(ModuleList:Union[torch.nn.ModuleList, List, torch.nn.Module], ParameterName:str, ParameterValues:Union[List,Iterable], basic_input_field:torch.Tensor=None, RescaleModule:torch.nn.Module=None, show=True):
    """
    Внимание, данная функция будет варьировать аттрибут, только если он есть внутри слоя
    :param ModuleList: Лист, содержащий все слои сети.
    :param RescaleModule: Модуль, который масштабирует basic_input_field к необходимому размеру.
    :param ParameterName: Название параметра, который нужно варьировать.
    :param ParameterValues: Лист со значениями варьированного параметра.
    :param basic_input_field: Базовое входное поле, которое сьедает модель.
    :param show: Вызывать ли функцию plt.show()
    :return:
    """

    Model:Union[torch.nn.Module, Any] = None
    if isinstance(ModuleList, torch.nn.Module):
        Model = ModuleList
        ModuleList = None


    if basic_input_field is None:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        basic_input_field, labels = next(iter(loader))

    if RescaleModule is None:
        class NoRescale(torch.nn.Module):
            def __init__(self):
                super(NoRescale, self).__init__()
                setattr(self, ParameterName, ParameterValues[0])
            def forward(self, field_:torch.Tensor):
                self.__getattr__(ParameterName)
                return field_
        RescaleModule = NoRescale()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if ModuleList is None:
        Model.to(device)
    else:
        for module in ModuleList:
            module.to(device)
    RescaleModule.to(device)
    basic_input_field = basic_input_field.to(device)


    Samples = len(ParameterValues) + 1
    Rows = int(Samples ** 0.5)
    Cols = int(Samples / Rows) + (Samples % Rows != 0)

    fig = plt.figure(figsize=(12 * Cols / Rows, 12))
    fig.suptitle('Распространение излучения от пикселя при различных параметрах', **Format.Text('BigHeader'))
    Fig = Titles(fig, (Cols, Rows))

    PlaneLength = None
    if ModuleList is None:
        if hasattr(Model, 'PlaneLength'):
            PlaneLength = getattr(Model, 'PlaneLength')
    else:
        for module in ModuleList:
            if hasattr(module, 'PlaneLength'):
                PlaneLength = getattr(module, 'PlaneLength')
                break
    if PlaneLength is None:
        unit = ''
        extent = [-0.5, +0.5, -0.5, +0.5]
    else:
        unit, mult = Format.Engineering_Separated(PlaneLength, 'm')
        extent = [-PlaneLength * mult / 2, +PlaneLength * mult / 2, -PlaneLength * mult / 2, +PlaneLength * mult / 2]

    with torch.no_grad():
        ParameterValues = [None] + list(ParameterValues)
        for parameter, (row, col) in zip(ParameterValues, product(range(Rows), range(Cols))):
            axis = Fig.add_axes((col+1, row+1))
            if parameter is None:
                field = RescaleModule(basic_input_field)
                field = (torch.abs(field)**1).squeeze().cpu()
                axis.imshow(field.swapaxes(0,1), origin='lower', extent=extent, aspect='auto')
                axis.set_title('Исходное изображение', **Format.Text('Default'))
                axis.xaxis.set_tick_params(labelsize=8)
                axis.yaxis.set_tick_params(labelsize=8)
                axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
                axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))
            else:
                if ModuleList is None:
                    if hasattr(Model, ParameterName):
                        setattr(Model, ParameterName, parameter)
                else:
                    for module in ModuleList:
                        if hasattr(module, ParameterName):
                            setattr(module, ParameterName, parameter)
                if hasattr(RescaleModule, ParameterName):
                    setattr(RescaleModule, ParameterName, parameter)
                field = RescaleModule(basic_input_field)
                if ModuleList is None:
                    field = Model(field)
                else:
                    for module in ModuleList:
                        field = module(field)
                field = (torch.abs(field)**1).squeeze().cpu()
                axis.imshow(field.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto')
                axis.set_title(ParameterName + ' : ' + Format.Scientific(parameter), **Format.Text('Default'))
                axis.xaxis.set_tick_params(labelsize=8)
                axis.yaxis.set_tick_params(labelsize=8)
                axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
                axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

    if show:
        plt.show()

    return
def PlotDiffractionModelOneParameterDistribution(Model:torch.nn.Module, ParameterName:str, ParameterValues:Union[List,Iterable], ParameterFormat=None, basic_input_field:torch.Tensor=None, RescaleModule:torch.nn.Module=None, ForceModelParametersZero:bool=False, show=True):
    ParameterValues = list(ParameterValues)

    if ParameterFormat is None:
        ParameterFormat = lambda x: str(x)

    if not hasattr(Model, ParameterName):
        raise AttributeError("\033[31m\033[1m{}".format('PlotDiffractionModelOneParameterDistribution: Model must have attribute "' + ParameterName + '"!'))

    ParameterValueRestore = getattr(Model, ParameterName)

    if basic_input_field is None:
        transformation = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation((-90, -90)),
            torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
        ])
        dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
        loader = TorchDataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
        basic_input_field, labels = next(iter(loader))

    if RescaleModule is None:
        class NoRescale(torch.nn.Module):
            def __init__(self):
                super(NoRescale, self).__init__()
                setattr(self, ParameterName, ParameterValues[0])
            def forward(self, field_:torch.Tensor):
                self.__getattr__(ParameterName)
                return field_
        RescaleModule = NoRescale()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Model.to(device)
    RescaleModule.to(device)
    basic_input_field = basic_input_field.to(device)

    history_length = None
    if hasattr(Model, 'DisableDetectors'):
        getattr(Model, 'DisableDetectors')(True)
    elif hasattr(Model, 'EnableDetectors'):
        getattr(Model, 'EnableDetectors')(False)
    with torch.no_grad():
        field = RescaleModule(basic_input_field)
        if 'record_history' in inspect.signature(Model.forward).parameters:
            output_field, history = Model.forward(field, record_history=True)
        else:
            output_field, history = Model.forward(field)
        history_length = len(history)

    Cols = history_length
    Rows = len(ParameterValues)

    fig = plt.figure(figsize=(12 * Cols / Rows, 12))
    fig.suptitle('Распространение излучения через сеть при вариации параметра (' + ParameterName + ').' + (' Веса сети установлены в ноль' if ForceModelParametersZero else ''), **Format.Text('BigHeader'))
    Fig = Titles(fig, (Cols, Rows))
    Fig.add_bottom_annotation(Format.SmartWrappedText(ConvertModelAttributesToString(Model), 250, ','), **Format.Text('Header'))
    for row, parameter in enumerate(ParameterValues):
        Fig.add_row_annotation(row+1, text=ParameterName + ': ' + ParameterFormat(parameter), **Format.Text('Default', {'fontweight':'bold'}))

    for row, parameter in enumerate(ParameterValues):
        axes = [Fig.add_axes((col+1, row+1)) for col in range(Cols)]

        if hasattr(RescaleModule, ParameterName):
            setattr(RescaleModule, ParameterName, parameter)
        setattr(Model, ParameterName, parameter)

        old_params = []
        if ForceModelParametersZero:
            for param in Model.parameters():
                old_params.append(param.data.clone())
                param.data.zero_()

        DrawThroughModelPropagation(axes, Model, RescaleModule(basic_input_field))

        if ForceModelParametersZero:
            for param, old_param in zip(Model.parameters(), old_params):
                param.data = old_param

    setattr(Model, ParameterName, ParameterValueRestore)
    if hasattr(RescaleModule, ParameterName):
        setattr(RescaleModule, ParameterName, ParameterValueRestore)

    if hasattr(Model, 'DisableDetectors'):
        getattr(Model, 'DisableDetectors')(False)
    elif hasattr(Model, 'EnableDetectors'):
        getattr(Model, 'EnableDetectors')(True)

    if show:
        plt.show()
def PlotTroughModelPropagationSamples(Model:torch.nn.Module, data_set_name:str='MNIST', input_pixels:int=None, images:torch.Tensor=None, labels:torch.Tensor=None, samples:int=4, show=True):
    if images is None:
        if data_set_name is None:
            raise ValueError("\033[31m\033[1m{}".format('PlotTroughModelPropagationSamples: define "data_set_name" or "initial_images" in function call!'))
        if input_pixels is None:
            up_scaling = 0
            if hasattr(Model, 'UpScaling'):
                up_scaling = getattr(Model, 'UpScaling')
            else : raise AttributeError("\033[31m\033[1m{}".format('PlotTroughModelPropagationSamples: Model must have attribute "UpScaling" or define input_pixels in function call!'))
            pixels_count = 0
            if hasattr(Model, 'PixelsCount'):
                pixels_count = getattr(Model, 'PixelsCount')
            else : raise AttributeError("\033[31m\033[1m{}".format('PlotTroughModelPropagationSamples: Model must have attribute "UpScaling" or define input_pixels in function call!'))
            input_pixels = up_scaling*pixels_count

        loader = TorchDataLoader(StringToDataSetRedirector(data_set_name, train=False, input_pixels=input_pixels), batch_size=samples, shuffle=True, num_workers=0)
        images, labels = next(iter(loader))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    Model.to(device)

    output = None
    history = None
    history_length = None
    with torch.no_grad():
        if 'record_history' in inspect.signature(Model.forward).parameters:
            output, history = Model.forward(images, record_history=True)
        else:
            output, history = Model.forward(images)
        history_length = len(history)

    output = output.cpu()
    Model.cpu()

    Cols = history_length
    Rows = samples

    detectors_module = None
    for module in Model.modules():
        if isinstance(module, DetectorsLayer):
            Cols += 1
            detectors_module = module
            break

    fig = plt.figure(figsize=(12*Cols/Rows, 12))
    fig.suptitle('Распространение излучения через сеть', **Format.Text('BigHeader'))
    Fig = Titles(fig, (Cols, Rows), leftspace=0.05, rightspace=0.05)
    Fig.add_bottom_annotation(Format.SmartWrappedText(ConvertModelAttributesToString(Model), 180, ','), **Format.Text('Header'))

    if hasattr(Model, 'PlaneLength'):
        PlaneLength = getattr(Model, 'PlaneLength')
        unit, mult = Format.Engineering_Separated(PlaneLength, 'm')
        extent = [-PlaneLength * mult / 2, +PlaneLength * mult / 2, -PlaneLength * mult / 2, +PlaneLength * mult / 2]
    else:
        unit = ''
        mult = 1.0
        extent = [-0.5, +0.5, -0.5, +0.5]

    plot_images = []
    for col, history_col in enumerate(history):
        name, images = None, None
        if isinstance(history_col[0], str):
            name, images = history_col
        else:
            name = 'Срез №' + str(col + 1)
            images = history_col

        if (images.size(1) != 1) and hasattr(Model, 'WaveLength'):
            colorizer = Colorizer()
            images = colorizer.Colorize(images, getattr(Model, 'WaveLength'))
        else:
            images = torch.sum(images, dim=1)

        plot_images.append([])
        for row, image in enumerate(images):
            axis = Fig.add_axes((col+1, row+1))
            image_ = axis.imshow(image.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto')
            axis.set_title(name, **Format.Text('Default', {'fontsize': 9}))
            axis.xaxis.set_tick_params(labelsize=8)
            axis.yaxis.set_tick_params(labelsize=8)
            axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
            axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

            plot_images[col].append(image_)

    if detectors_module is not None:
        detectors_masks = torch.mean(detectors_module.GetDetectorsMasksBuffer(), dim=0)
        for row in range(Rows):
            axis = Fig.add_axes((Cols, row+1))
            axis.imshow(detectors_masks.swapaxes(0, 1), origin='lower', extent=extent, aspect='auto', cmap='gray')
            axis.set_title('Маски детекторов', **Format.Text('Default', {'fontsize': 9}))
            axis.xaxis.set_tick_params(labelsize=8)
            axis.yaxis.set_tick_params(labelsize=8)
            axis.set_xlabel('X, ' + unit, Format.Text('Caption'))
            axis.set_ylabel('Y, ' + unit, Format.Text('Caption', {'rotation': 90}))

    if show:
        plt.show()

    return plot_images




