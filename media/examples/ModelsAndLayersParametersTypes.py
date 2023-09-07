import torch
import numpy
from typing import Union, List, Tuple, Any


class AnyModelOrLayerFloat(torch.nn.Module):

    # Плавующая точка
    _float_type : float
    @property
    def float_type(self):
        class SubClass:
            _self : AnyModelOrLayerFloat # Ссылка на экземпляр оригинального класса
            def __init__(self, _self:AnyModelOrLayerFloat):
                self._self = _self
            def __call__(self, value:float):
                self._self._float_type = value
                # Не забудьте изменить зависимые данные вручную или с помощью
                # функционала отложенных функций.
        return SubClass(self)

    def __init__(self):
        super(AnyModelOrLayerFloat, self).__init__()

    def forward(self, x):
        return


class AnyModelOrLayerInt(torch.nn.Module):

    # Целое число
    _int_type : int
    @property
    def int_type(self):
        class SubClass:
            _self : AnyModelOrLayerInt # Ссылка на экземпляр оригинального класса
            def __init__(self, _self:AnyModelOrLayerInt):
                self._self = _self
            def __call__(self, value:int):
                self._self._int_type = value
                # Не забудьте изменить зависимые данные вручную или с помощью
                # функционала отложенных функций.
        return SubClass(self)

    def __init__(self):
        super(AnyModelOrLayerInt, self).__init__()

    def forward(self, x):
        return


class AnyModelOrLayerRange(torch.nn.Module):

    # Диапазон
    _range_type : Union[Tuple, List, numpy.ndarray, torch.Tensor]
    @property
    def range_type(self):
        class SubClass:
            _self : AnyModelOrLayerRange # Ссылка на экземпляр оригинального класса
            def __init__(self, _self:AnyModelOrLayerRange):
                self._self = _self
            def __call__(self, value:Tuple[float, float, int]):
                if isinstance(self._self._range_type, list):
                    self._self._range_type = list(numpy.linspace(value[0], value[1], value[2]))
                elif isinstance(self._self._range_type, numpy.ndarray):
                    self._self._range_type = numpy.linspace(value[0], value[1], value[2])
                elif isinstance(self._self._range_type, torch.Tensor):
                    self._self._range_type = torch.linspace(value[0], value[1], value[2])
                elif isinstance(self._self._range_type, tuple):
                    self._self._range_type = value
                else:
                    # Если тип хранения диапазона иной
                    return
                # Не забудьте изменить зависимые данные вручную или с помощью
                # функционала отложенных функций.
        return SubClass(self)

    def __init__(self):
        super(AnyModelOrLayerRange, self).__init__()

    def forward(self, x):
        return


class AnyModelOrLayerKit(torch.nn.Module):

    # Набор
    _kit_type : Union[Tuple, List, numpy.ndarray, torch.Tensor]
    @property
    def kit_type(self):
        class SubClass:
            _self : AnyModelOrLayerKit # Ссылка на экземпляр оригинального класса
            def __init__(self, _self:AnyModelOrLayerKit):
                self._self = _self
            def __call__(self, value:Union[List, torch.Tensor, numpy.ndarray]):
                if isinstance(self._self._kit_type, list):
                    self._self._kit_type = list(value)
                elif isinstance(self._self._kit_type, numpy.ndarray):
                    self._self._kit_type = numpy.array(value)
                elif isinstance(self._self._kit_type, torch.Tensor):
                    self._self._kit_type = torch.tensor(value)
                else:
                    # Если тип хранения набора иной
                    return
                # Не забудьте изменить зависимые данные вручную или с помощью
                # функционала отложенных функций.
        return SubClass(self)

    def __init__(self):
        super(AnyModelOrLayerKit, self).__init__()

    def forward(self, x):
        return


class AnyModelOrLayerVariant(torch.nn.Module):

    # Варианты
    _variant_type : Any
    @property
    def variant_type(self):
        class SubClass:
            _self : AnyModelOrLayerVariant # Ссылка на экземпляр оригинального класса
            def __init__(self, _self:AnyModelOrLayerVariant):
                self._self = _self
            @property
            def variant1(self):
                class VariantClass:
                    _self : AnyModelOrLayerVariant
                    def __init__(self, _self:AnyModelOrLayerVariant):
                        self._self = _self
                    def __call__(self, parameter1:float, parameter2:float, parameter3:float):
                        # Вставьте код реализующий вариант № 1.
                        # Параметры каждого варианта могут должны быть реализованы также, как
                        # параметры слоёв и моделей.
                        # Не забудьте изменить зависимые данные вручную или с помощью
                        # функционала отложенных функций.
                        return
                return VariantClass(self._self)
            @property
            def variant2(self):
                class VariantClass:
                    _self : AnyModelOrLayerVariant
                    def __init__(self, _self:AnyModelOrLayerVariant):
                        self._self = _self
                    def __call__(self, parameter1:float, parameter2:float, parameter3:float):
                        # Код реализующий вариант № 2
                        # Не забудьте изменить зависимые данные вручную или с помощью
                        # Функционала отложенных функций
                        return
                return VariantClass(self._self)
        return SubClass(self)

    def __init__(self):
        super(AnyModelOrLayerVariant, self).__init__()

    def forward(self, x):
        return