import torch
import numpy
from copy import deepcopy
from typing import Any, Dict, Union
import inspect

from torch.utils.data import DataLoader

from src.utilities.Formaters import Format
from src.utilities.CycleTimePredictor import CycleTimePredictor
from src.utilities.DelayedFunctions import DelayedFunctions
from src.modules.models.AbstractModel import AbstractModel
from src.utilities.UniversalTestsAndOther import StringToDataSetRedirector


class Trainer:
    _delayed : DelayedFunctions

    _dataset : str
    _train_loader : DataLoader
    _test_loader : DataLoader
    def _reset_loaders(self):
        self._train_loader  = DataLoader(StringToDataSetRedirector(self._dataset, train=True,  input_pixels=self._pixels), batch_size=self._batches, shuffle=True, num_workers=0)
        self._test_loader   = DataLoader(StringToDataSetRedirector(self._dataset, train=False, input_pixels=self._pixels), batch_size=self._batches, shuffle=True, num_workers=0)
    @property
    def dataset(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def MNIST(self):
                self._self._dataset = 'MNIST'
                self._self._delayed.add(self._self._reset_loaders, 0.)
        return Selector(self)

    _pixels : int
    @property
    def pixels(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def __call__(self, amount:int):
                self._self._pixels = amount
                self._self._delayed.add(self._self._reset_loaders, 0.)
        return Selector(self)

    _model : AbstractModel
    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, network:AbstractModel):
        self._model = deepcopy(network)
        if hasattr(network, '_pixels') and hasattr(network, '_up_scaling'):
            self.pixels(int(getattr(network, '_pixels') * getattr(network, '_up_scaling')))

        # TODO разобраться, почему finalize не выполняет те-же функции, что и вызов нейронки
        self._model(torch.zeros(1,1,self._pixels, self._pixels))
        # self._model.finalize()

        self._reset_accuracy()
        self._delayed.add(self._reset_optimizer)

    _sub_batches : int
    _batches : int
    @property
    def batches(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def get(self):
                return deepcopy(self._self._batches)
            def __call__(self, amount:int):
                self._self._batches = amount
                self._self._sub_batches = 1
                self._self._delayed.add(self._self._reset_loaders, 0.)
        return Selector(self)

    _epochs : int
    @property
    def epochs(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def __call__(self, amount:int):
                self._self._epochs = amount
        return Selector(self)

    _optimizer_type : type(torch.optim.Optimizer)
    _optimizer_kwargs : Dict
    _optimizer : torch.optim.Optimizer
    def _reset_optimizer(self):
        if not hasattr(self, '_model') or self._model is None:
            self._delayed.add(self._reset_optimizer)
        else:
            self._optimizer = self._optimizer_type(self._model.parameters(), **self._optimizer_kwargs)
    @property
    def optimizer(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def Adam(self, lr:float=...):
                self._self._optimizer_type = torch.optim.Adam
                self._self._optimizer_kwargs = {'lr':lr}
                self._self._delayed.add(self._self._reset_optimizer)
            def SGD(self, lr:float=..., momentum:float=...):
                self._self._optimizer_type = torch.optim.SGD
                self._self._optimizer_kwargs = {
                    'lr':lr,
                    'momentum':momentum
                }
                self._self._delayed.add(self._self._reset_optimizer)
        return Selector(self)

    _loss_function : Any
    @property
    def loss_function(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def MSE(self):
                self._self._loss_function = torch.nn.MSELoss()
            def CrossEntropy(self):
                self._self._loss_function = torch.nn.CrossEntropyLoss()
        return Selector(self)

    _device : torch.device
    @property
    def device(self):
        class Selector:
            _self : Trainer
            def __init__(self, _self:Trainer):
                self._self = _self
            def cpu(self):
                self._self._device = torch.device('cpu')
            def cuda(self):
                self._self._device = torch.device('cuda')
            def default(self):
                self._self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return Selector(self)

    _accuracy : Union[float, None]
    @property
    def accuracy(self):
        if not hasattr(self, '_accuracy') or self._accuracy is None: self._accuracy_test()
        return self._accuracy
    def _reset_accuracy(self):
        self._accuracy = None

    def _accuracy_test(self):
        if not hasattr(self, '_device') or self._device is None:
            self.device.default()

        self._delayed.launch()

        self._model.to(self._device)
        if inspect.isclass(self._loss_function) and hasattr(self._loss_function, 'to'):
            self._loss_function.to(self._device)

        self._model.eval()

        Total: int = 0
        Correct: int = 0
        Percent: float = 0
        percent_function = lambda: 'Правильных ответов: ' + str(Correct) + ' из ' + str(Total) + ' - ' + str(round(Percent, 1)) + '%'

        with torch.no_grad():
            try:
                for (images, labels) in CycleTimePredictor(self._test_loader, [percent_function]):
                    for images_, labels_ in zip(torch.split(images, int(self._batches/self._sub_batches), dim=0), torch.split(labels, int(self._batches/self._sub_batches), dim=0)):
                        images_ = images_.to(self._device)
                        labels_ = labels_.to(self._device)

                        output = self._model(images_)

                        Correct += torch.sum(torch.eq(torch.argmax(output, dim=1), labels_)).item()
                        Total += len(labels_)

                        Percent = 100 * Correct / Total
            except torch.cuda.OutOfMemoryError:
                print('')
                print("torch.cuda.OutOfMemoryError happened - expanding sub_batches form " + str(self._sub_batches) + " to " + str(self._sub_batches + 1))
                self._sub_batches += 1
                self._accuracy_test()

        self._accuracy = Percent

    def _epoch_step(self, images, labels):
        try:
            loss = None
            for images_, labels_ in zip(torch.split(images, int(self._batches / self._sub_batches), dim=0), torch.split(labels, int(self._batches / self._sub_batches), dim=0)):
                images_ = images_.to(self._device)
                labels_ = labels_.to(self._device)

                output = self._model(images_)
                if isinstance(self._loss_function, torch.nn.CrossEntropyLoss):
                    loss = self._loss_function(output, labels_)
                else:
                    labels_ = torch.eye(output.size(1), output.size(1), device=self._device, dtype=output.dtype, requires_grad=True)[labels_]
                    loss = self._loss_function(output, labels_)

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

            return loss.item()
        except torch.cuda.OutOfMemoryError:
            print('')
            print("torch.cuda.OutOfMemoryError happened - expanding sub_batches form " + str(self._sub_batches) + " to " + str(self._sub_batches+1))
            self._sub_batches += 1
            self._epoch_step(images, labels)
    def _epoch(self, echo:bool=True):
        loss_buffer_first_iter = True
        loss_buffer_size = 20
        loss_buffer = numpy.zeros(loss_buffer_size)
        loss_buffer_index = 0
        loss_average = 0
        loss_string_function = lambda: 'Средний по ' + str(loss_buffer_size) + ' тестам лосс: ' + Format.Scientific(loss_average) + '± ' + Format.Scientific(numpy.std(loss_buffer)/numpy.sqrt(loss_buffer_size-1))
        CycleStringFunctions = [loss_string_function]

        gradient_string_function = lambda: 'Средний модуль градиента: ' + Format.Scientific((torch.mean(torch.abs(torch.cat([param.grad.view(-1) for param in self._model.parameters() if param.grad is not None]))).item() if len([p.grad.view(-1) for p in self._model.parameters() if p.grad is not None]) > 0 else 0.0),'')
        CycleStringFunctions.append(gradient_string_function)

        parameters1 = [param.clone().detach() for param in self._model.parameters()]
        parameters2 = [param.clone().detach() for param in self._model.parameters()]
        parameters_deviation_string_function1 = lambda: 'Среднее изменение параметров: ' + Format.Scientific(sum(torch.mean(torch.abs(p2-p1)).item() for p1, p2 in zip(parameters1, parameters2)) / len(parameters2), '', 6)
        CycleStringFunctions.append(parameters_deviation_string_function1)
        parameters_deviation_string_function2 = lambda: 'Максимальное изменение параметров: ' + Format.Scientific(max(torch.max(torch.abs(p2-p1)).item() for p1, p2 in zip(parameters1, parameters2)), '', 6)
        CycleStringFunctions.append(parameters_deviation_string_function2)

        self._model.train()
        with torch.autograd.set_grad_enabled(True):
            for images, labels in (self._train_loader if not echo else CycleTimePredictor(self._train_loader, CycleStringFunctions)):
                loss = self._epoch_step(images, labels)
                if loss_buffer_first_iter:
                    loss_buffer.fill(loss)
                    loss_buffer_first_iter = False
                else:
                    loss_buffer[loss_buffer_index] = loss
                    loss_buffer_index = (loss_buffer_index+1)%loss_buffer_size
                loss_average = numpy.mean(loss_buffer)
                parameters1 = [param.clone().detach() for param in parameters2]
                parameters2 = [param.clone().detach() for param in self._model.parameters()]

        self._model.eval()
        self._reset_accuracy()
    def train(self, epochs:int=None, echo:bool=True):
        self._accuracy = None
        if not hasattr(self, '_device') or self._device is None:
            self.device.default()
        if not hasattr(self, '_epochs'):
            self.epochs(1)
        if epochs is not None:
            self.epochs(epochs)

        self._delayed.launch()

        self._model.to(self._device)
        if inspect.isclass(self._loss_function) and hasattr(self._loss_function, 'to'):
            self._loss_function.to(self._device)

        if echo:
            print('Точность сети на начало тренировки: ' + str(round(self.accuracy, 1)) + '%')
        for epoch in range(self._epochs):
            self._epoch(echo=echo)
            if echo:
                print('Точность сети после эпохи обучения №' + str(epoch+1) + ' - ' + str(round(self.accuracy, 1)) + '%')

    def __init__(self, model:AbstractModel=None):
        self._delayed = DelayedFunctions()

        self.dataset.MNIST()
        self.model = model
        self.batches(64)
        self.epochs(1)
        self.optimizer.Adam(lr=0.01)
        self.loss_function.CrossEntropy()
        self.device.default()
        self._reset_accuracy()



if __name__ == "__main__":
    from src.modules.models.FourierSpaceD2NN import FourierSpaceD2NN
    from src.modules.models.RealSpaceD2NN import RealSpaceD2NN

    model = FourierSpaceD2NN()
    model.up_scaling(6)
    model.pixels(20)
    model.plane_length(20*50.0E-6)
    model.space(5.0E-3)
    model.border(5.0E-3)
    model.focus(10.0E-3)
    model.focus_border(10.0E-3)

    trainer = Trainer(model)
    trainer.batches(64)
    trainer.loss_function.CrossEntropy()
    trainer.optimizer.Adam(lr=0.01)

    trainer.train()