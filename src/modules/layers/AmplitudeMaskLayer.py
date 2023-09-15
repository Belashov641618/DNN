import torch

from .AbstractMaskLayer import AbstractMaskLayer

#Класс давольно бесполезный, стоит позже пересмотреть и удалить этот класс, потому что AbstractMaskLayer делает абсолютно тоже самое.
class AmplitudeMaskLayer(AbstractMaskLayer):

    def __init__(self, pixels:int=20, up_scaling:int=8):
        super(AmplitudeMaskLayer, self).__init__(pixels=pixels, up_scaling=up_scaling)