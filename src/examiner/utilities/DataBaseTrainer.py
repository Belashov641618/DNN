import torch
import numpy
from typing import Union, List, Tuple, Any, Dict
import inspect
from torch.utils.data import DataLoader
from src.utilities.CycleTimePredictor import CycleTimePredictor
from src.utilities.Formaters import Format

from src.utilities.UniversalTestsAndOther import CalculateMaximumBatchSize, StringToDataSetRedirector
from src.utilities.DelayedFunctions import DelayedFunctions

class DataBaseTrainer:

    _DelayedFunctions : DelayedFunctions

    _dataset : str
    @property
    def dataset(self):
        return self._dataset
    @dataset.setter
    def dataset(self, name:str):
        self._dataset = name
        self._DelayedFunctions.add(self._reset_loaders, 0.)
        self._DelayedFunctions.add(self._reset_accuracy, 0.)

    _batches : int
    @property
    def batches(self):
        return self._batches
    @batches.setter
    def batches(self, amount:Union[int, None]):
        if amount is None:
            if hasattr(self, '_Model') and self._Model is not None:
                self._batches = CalculateMaximumBatchSize(self._Model)
            else : raise Exception('Автоматическое вычисление размера одного пакета данных невозможно без предварительной установки модели!')
        else:
            self._batches = amount
        self._DelayedFunctions.add(self._reset_loaders, 0.)

    _pixels : int
    @property
    def pixels(self):
        return self._pixels
    @pixels.setter
    def pixels(self, amount:int):
        self._pixels = amount
        self._DelayedFunctions.add(self._reset_loaders, 0.)
        self._DelayedFunctions.add(self._reset_accuracy, 0.)

    _Model : Union[torch.nn.Module,torch.nn.Sequential]
    @property
    def model(self):
        return self._Model
    @model.setter
    def model(self, network:Union[torch.nn.Module,torch.nn.Sequential,List,Tuple]):
        if isinstance(network, (List, Tuple)):
            self._Model = torch.nn.Sequential(*network)
        else:
            self._Model = network
        if hasattr(network, 'PixelsCount') and hasattr(network, 'UpScaling'):
            self.pixels = int(getattr(network, 'PixelsCount') * getattr(network, 'UpScaling'))
        self._DelayedFunctions.add(self._reset_accuracy, 0.)

    _TrainLoader : DataLoader
    _TestLoader  : DataLoader
    def _reset_loaders(self):
        self._TestLoader    = DataLoader(StringToDataSetRedirector(self._dataset, train=False, input_pixels=self._pixels), batch_size=self._batches, shuffle=True, num_workers=0)
        self._TrainLoader   = DataLoader(StringToDataSetRedirector(self._dataset, train=True, input_pixels=self._pixels), batch_size=self._batches, shuffle=True, num_workers=0)

    _Optimizer       : Any
    _OptimizerType   : Any
    _OptimizerKwargs : Dict
    @property
    def optimizer_type(self):
        if hasattr(self, '_OptimizerType'):
            return self._OptimizerType
        else:
            return None
    @optimizer_type.setter
    def optimizer_type(self, torch_optimizer_type:Any):
        self._OptimizerType = torch_optimizer_type
        self._construct_optimizer()
    @property
    def optimizer_kwargs(self):
        if hasattr(self, '_OptimizerKwargs'):
            return self._OptimizerKwargs
        else:
            return None
    @optimizer_kwargs.setter
    def optimizer_kwargs(self, torch_optimizer_parameters:Dict):
        self._OptimizerKwargs = torch_optimizer_parameters
        self._construct_optimizer()
    def optimizer(self, optimizer_type, **kwargs):
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = kwargs
    def _construct_optimizer(self):
        if not hasattr(self, '_OptimizerKwargs'):               self._OptimizerKwargs = {}
        if not hasattr(self, '_OptimizerType'):                 self._OptimizerType = torch.optim.Adam
        if not hasattr(self, '_Model') or self._Model is None:  self._DelayedFunctions.add(self._construct_optimizer)
        else:                                                   self._Optimizer = self._OptimizerType(self._Model.parameters(), **self._OptimizerKwargs)

    _LossFunction : Any
    @property
    def loss(self):
        return self._LossFunction
    @loss.setter
    def loss(self, loss_function:Any):
        self._LossFunction = loss_function

    _epochs : int
    @property
    def epochs(self):
        return self._epochs
    @epochs.setter
    def epochs(self, amount:int):
        self._epochs = amount

    _device_name : str
    _device : torch.device
    @property
    def device(self):
        return self._device_name
    @device.setter
    def device(self, name:str):
        self._device_name = name
        self._device = torch.device(name)

    _accuracy : Union[float, None]
    @property
    def accuracy(self):
        if self._accuracy is None: self._accuracy_test()
        return self._accuracy
    def _reset_accuracy(self):
        self._accuracy = None

    def __init__(self, Model:Union[torch.nn.Module,torch.nn.Sequential,List,Tuple]=None, Dataset:str='MNIST', LossFunction:Any=None, OptimizerType:Any=None, OptimizerKwargs:Dict=None):
        self._DelayedFunctions = DelayedFunctions()

        self.model = Model
        self.dataset = Dataset
        if LossFunction is None:        LossFunction = torch.nn.CrossEntropyLoss()
        self.loss = LossFunction
        if OptimizerType is None:       OptimizerType = torch.optim.Adadelta
        if OptimizerKwargs is None:     OptimizerKwargs = {}
        self.optimizer(OptimizerType, **OptimizerKwargs)

    def _accuracy_test(self):
        if not hasattr(self, '_device') or self._device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._DelayedFunctions.launch()

        self._Model.to(self._device)
        if inspect.isclass(self._LossFunction) and hasattr(self._LossFunction, 'to'):
            self._LossFunction.to(self._device)

        self._Model.eval()

        Total: int = 0
        Correct: int = 0
        Percent: float = 0
        percent_function = lambda: 'Правильных ответов: ' + str(Correct) + ' из ' + str(Total) + ' - ' + str(round(Percent, 1)) + '%'

        with torch.no_grad():
            for (images, labels) in CycleTimePredictor(self._TestLoader, [percent_function]):
                images = images.to(self._device)
                labels = labels.to(self._device)

                output = self._Model(images)

                Correct += torch.sum(torch.eq(torch.argmax(output, dim=1), labels)).item()
                Total += len(labels)

                Percent = 100 * Correct / Total

        self._accuracy = Percent
    def _epoch_step(self, images, labels):
        images = images.to(self._device)
        labels = labels.to(self._device)

        output = self._Model(images)
        if isinstance(self._LossFunction, torch.nn.CrossEntropyLoss):
            loss = self._LossFunction(output, labels)
        else:
            labels = torch.eye(output.size(1), output.size(1), device=self._device, dtype=output.dtype, requires_grad=True)[labels]
            loss = self._LossFunction(output, labels)

        torch.autograd.detect_anomaly()

        self._Optimizer.zero_grad()
        loss.backward()
        self._Optimizer.step()

        return loss
    def _epoch(self, echo:bool=True):
        loss_buffer_first_iter = True
        loss_buffer_size = 20
        loss_buffer = numpy.zeros(loss_buffer_size)
        loss_buffer_index = 0
        loss_average = 0
        loss_string_function = lambda: 'Средний по ' + str(loss_buffer_size) + ' тестам лосс: ' + Format.Scientific(loss_average) + '± ' + Format.Scientific(numpy.std(loss_buffer)/numpy.sqrt(loss_buffer_size-1))

        CycleStringFunctions = [loss_string_function]

        gradient_string_function = lambda: 'Средний модуль градиента: ' + Format.Scientific((torch.mean(torch.abs(torch.cat([param.grad.view(-1) for param in self._Model.parameters() if param.grad is not None]))).item() if len([p.grad.view(-1) for p in self._Model.parameters() if p.grad is not None]) > 0 else 0.0),'')
        CycleStringFunctions.append(gradient_string_function)

        if hasattr(self._Model, '_DiffractionLengthAsParameter') and hasattr(self._Model, 'DiffractionLength'):
            if getattr(self._Model, '_DiffractionLengthAsParameter'):
                diffraction_length_function = lambda: 'Расстояние между масками: ' + Format.Engineering(getattr(self._Model, 'DiffractionLength') ,'m')
                CycleStringFunctions.append(diffraction_length_function)

        self._Model.train()
        for images, labels in (self._TrainLoader if not echo else CycleTimePredictor(self._TrainLoader, CycleStringFunctions)):
            loss = self._epoch_step(images, labels)
            if loss_buffer_first_iter:
                loss_buffer.fill(loss.item())
                loss_buffer_first_iter = False
            else:
                loss_buffer[loss_buffer_index] = loss.item()
                loss_buffer_index = (loss_buffer_index+1)%loss_buffer_size
            loss_average = numpy.mean(loss_buffer)
        self._Model.eval()

    def train(self, epochs:int=None, echo:bool=True):
        self._accuracy = None
        if not hasattr(self, '_device') or self._device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if epochs is not None:
            self.epochs = epochs
        elif not hasattr(self, '_epochs') or self._epochs is None:
            self.epochs = 1

        self._DelayedFunctions.launch()

        self._Model.to(self._device)
        if inspect.isclass(self._LossFunction) and hasattr(self._LossFunction, 'to'):
            self._LossFunction.to(self._device)

        if echo:
            print('Точность сети на начало тренировки: ' + str(round(self.accuracy, 1)) + '%')
        for epoch in range(self.epochs):
            self._epoch(echo=echo)
            if echo:
                print('Точность сети после эпохи обучения №' + str(epoch+1) + ' - ' + str(round(self.accuracy, 1)) + '%')

    @property
    def name(self):
        class NamesContainer:
            _self: DataBaseTrainer
            @staticmethod
            def _name(obj: Any):
                if hasattr(obj, '__class__'):
                    return obj.__class__.__name__
                elif hasattr(obj, '__name__'):
                    return obj.__name__
                else:
                    return str(obj)
            def __init__(self, original: DataBaseTrainer):
                self._self = original
            @property
            def model(self):
                return self._name(self._self._Model)
            @property
            def dataset(self):
                return self._self._dataset
            @property
            def loss(self):
                return self._name(self._self._LossFunction)
            @property
            def optimizer(self):
                return self._name(self._self._Optimizer)

        return NamesContainer(self)
    @property
    def info(self):
        self._DelayedFunctions.launch()
        return ' | '.join([
            'DataSet: '         + self.name.dataset,
            'LossFunction: '    + self.name.loss,
            'Optimizer: '       + self.name.optimizer,
            'Batches: '         + str(self.batches),
            'Epochs: '          + str(self.epochs)
        ])