import torch
import numpy
from typing import Union, List, Iterable, Tuple, Any
from torch.utils.data import DataLoader
import torchvision
from src.AdditionalUtilities.CycleTimePredictor import CycleTimePredictor
from src.AdditionalUtilities.Formaters import Format

from src.AdditionalUtilities.UniversalTestsAndOther import CalculateMaximumBatchSize

class Trainer:

    _Model          : Union[torch.nn.Module,torch.nn.Sequential]
    _TrainLoader    : DataLoader
    _TestLoader     : DataLoader
    _LossFunction   : Any
    _Optimizer      : Any

    def __init__(self, Model:Union[torch.nn.Module,torch.nn.Sequential,List,Tuple], DataSet:str='MNIST', batch_size:int=None, InputSize:int=None, TrainLoader:DataLoader=None, TestLoader:DataLoader=None, LossFunction:Any=None, Optimizer:Any=None):
        if isinstance(Model, List) or isinstance(Model, Tuple):
            Model = torch.nn.Sequential(*Model)

        if DataSet is not None:
            if InputSize is None:
                if hasattr(Model, 'PixelsCount') and hasattr(Model, 'UpScaling'):
                    InputSize = int(getattr(Model, 'PixelsCount')*getattr(Model, 'UpScaling'))
                else: raise AttributeError("\033[31m\033[1m{}".format('Model does not have attributes "PixelsCount" and "UpScaling", so there is no ability to calculate InputSize. Define InputSize or this attributes in Model'))
            if batch_size is None:
                # batch_size = CalculateMaximumBatchSize(Model, [InputSize, InputSize])
                batch_size = CalculateMaximumBatchSize(Model, [InputSize, InputSize])
                print("\033[32m\033[1m{}\033[0m".format('Automatically calculated "batch_size": ' + str(batch_size) + ''))
            if DataSet == 'MNIST':
                transformation = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.RandomRotation((-90, -90)),
                    torchvision.transforms.Resize(size=(InputSize, InputSize)),
                    torchvision.transforms.ConvertImageDtype(dtype=torch.complex64)
                ])
                training = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
                TrainLoader = DataLoader(training, batch_size=batch_size, shuffle=True, num_workers=0)
                testing = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transformation)
                TestLoader = DataLoader(testing, batch_size=batch_size, shuffle=False, num_workers=0)
            else:
                raise ValueError("\033[31m\033[1m{}".format('There is no data set with name:' + DataSet + '!'))

        self._TestLoader = TestLoader
        self._TrainLoader = TrainLoader
        self._Model = Model

        if LossFunction is None:
            LossFunction = torch.nn.MSELoss()
        if Optimizer is None:
            # Optimizer = torch.optim.SGD(self._Model.parameters(), lr=1000, momentum=0.1)
            Optimizer = torch.optim.Adam(self._Model.parameters(), lr=0.005)

        self._LossFunction   = LossFunction
        self._Optimizer      = Optimizer


    def CalculateAccuracy(self, device:Any=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._Model.to(device)
        self._LossFunction.to(device)

        Total:int = 0
        Correct:int = 0
        Percent:float = 0
        percent_function = lambda: 'Правильных ответов: ' + str(Correct) + ' из ' + str(Total) + ' - ' + str(round(Percent,1)) + '%'

        with torch.no_grad():
            for (images, labels) in CycleTimePredictor(self._TestLoader, [percent_function]):
                images = images.to(device)
                labels = labels.to(device)

                output = self._Model(images)

                Correct += torch.sum(torch.eq(torch.argmax(output, dim=1), labels)).item()
                Total += len(labels)

                Percent = 100*Correct/Total

        return Percent

    def TrainEpoch(self, device:Any=None, loss_buffer_size=20, show_average_absolute_gradient:bool=True):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._Model.to(device)
        self._LossFunction.to(device)

        loss_buffer_first_iter = True
        loss_buffer = numpy.zeros(loss_buffer_size)
        loss_buffer_index = 0
        loss_average = 0
        loss_string_function = lambda: 'Средний по ' + str(loss_buffer_size) + ' тестам лосс: ' + Format.Scientific(loss_average) + '± ' + Format.Scientific(numpy.std(loss_buffer)/numpy.sqrt(loss_buffer_size-1))

        CycleStringFunctions = [loss_string_function]

        if show_average_absolute_gradient:
            gradient_string_function = lambda: 'Средний модуль градиента: ' + Format.Scientific((torch.mean(torch.abs(torch.cat([param.grad.view(-1) for param in self._Model.parameters() if param.grad is not None]))).item() if len([p.grad.view(-1) for p in self._Model.parameters() if p.grad is not None]) > 0 else 0.0),'')
            CycleStringFunctions.append(gradient_string_function)

        if hasattr(self._Model, '_DiffractionLengthAsParameter') and hasattr(self._Model, 'DiffractionLength'):
            if getattr(self._Model, '_DiffractionLengthAsParameter'):
                diffraction_length_function = lambda: 'Расстояние между масками: ' + Format.Engineering(getattr(self._Model, 'DiffractionLength') ,'m')
                CycleStringFunctions.append(diffraction_length_function)

        self._Model.train()
        for images, labels in CycleTimePredictor(self._TrainLoader, CycleStringFunctions):
            images = images.to(device)
            labels = labels.to(device)

            output = self._Model(images)

            if isinstance(self._LossFunction, torch.nn.CrossEntropyLoss):
                loss = self._LossFunction(output, labels)
            else:
                labels = torch.eye(output.size(1), output.size(1), device=device, dtype=output.dtype, requires_grad=True)[labels]
                loss = self._LossFunction(output, labels)

            # torch.autograd.detect_anomaly()

            self._Optimizer.zero_grad()
            loss.backward()
            self._Optimizer.step()

            if loss_buffer_first_iter:
                loss_buffer.fill(loss.item())
                loss_buffer_first_iter = False
            else:
                loss_buffer[loss_buffer_index] = loss.item()
                loss_buffer_index = (loss_buffer_index+1)%loss_buffer_size
            loss_average = numpy.mean(loss_buffer)
        self._Model.eval()
        self.CalculateAccuracy()

    def Train(self, epochs:int, device:Any=None, loss_buffer_size:int=20, show_average_absolute_gradient:bool=True):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._Model.to(device)
        self._LossFunction.to(device)

        self._Model.eval()
        self.CalculateAccuracy()

        for epoch in range(epochs):
            self.TrainEpoch(device, loss_buffer_size=loss_buffer_size, show_average_absolute_gradient=show_average_absolute_gradient)