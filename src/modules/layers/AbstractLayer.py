import torch
from typing import List, Callable

class AbstractLayer(torch.nn.Module):
    """
    Abstract Layer
    Абстрактный класс для создания других слоёв диффракционной сети. Функционал этого класса
    содержит в себе две системы.\n
    #1. Система отложенных вызовов:
    \t-С помощью метода _add_DelayedFunction(function, priority) можно добавлять функции которые будут запущены при следующем вызове forward(...). Приоритет определяет последовательность вызова функций - чем выше приоритет, тем раньше вызовится функция.\n
    \t-Если убрать автоматический вызов отложенных функций с помощью AutoFinalizeChanges(False), то отложенные функции не будут вызываться в forward, но их выполнение можно начильно вызвать методом FinalizeChanges().\n

    #2. Система установки точности:
    \t-При создании тензоров к дочерних классах следует указывать тип данных _tensor_float_type или _tensor_complex_type, тогда, можно будет изменять точность переменных.\n
    \t-Для установки битовой точности вызовите метод AccuracySet(bits), для того, чтобы узнать - AccuracyGet().\n
    """

    _DelayedFunctions : List
    def _launch_DelayedFunctions(self):
        if hasattr(self, '_DelayedFunctions'):
            self._DelayedFunctions.sort(key=lambda element: element[1])
            for (function, priority) in self._DelayedFunctions:
                function()
            self._DelayedFunctions.clear()
    def _add_DelayedFunction(self, function:Callable, priority:float=0.0):
        if not hasattr(self, '_DelayedFunctions'):
            self._DelayedFunctions = [[function, priority]]
        elif [function, priority] not in self._DelayedFunctions:
            self._DelayedFunctions.append([function, priority])
    def FinalizeChanges(self):
        self._launch_DelayedFunctions()

    _AutoFinalizeChanges : bool
    def AutoFinalizingChanges(self, mode:bool):
        self._AutoFinalizeChanges = mode

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

    def __init__(self):
        super(AbstractLayer, self).__init__()

        self._AutoFinalizeChanges = True

        self._accuracy_bits = 32
        self.AccuracySet(self._accuracy_bits)

    def forward(self, field:torch.Tensor):
        if self._AutoFinalizeChanges: self._launch_DelayedFunctions()