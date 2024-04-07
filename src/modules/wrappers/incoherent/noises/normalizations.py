import torch
import math

from tqdm import tqdm

if __name__ == '__main__':
    from generators import Generator
    from src.utilities.statistics import distribution
else:
    from .generators import Generator
    from .....utilities.statistics import distribution

class Normalizer:
    _generator: Generator
    _steps:int

    def _limited(self):
        sample = self._generator.sample()
        return (sample - sample.min()) / (sample.max() - sample.min())
    def _distribution(self):
        return distribution(self._limited(), self._steps)

    @staticmethod
    def _compare_function(a:torch.Tensor, b:torch.Tensor):
        return torch.mean(torch.abs(a-b)**2)
    _parameters : torch.nn.Parameter
    def _function(self, x:torch.Tensor):
        raise NotImplementedError
    def difference(self):
        return self._compare_function(self._function(torch.linspace(0, 1, self._steps, device=self._generator.device)), self._distribution())
    def optimize(self, steps:int=1000):
        optimizer = torch.optim.Adam((self._parameters, ), lr=1e-1)
        for step in tqdm(range(steps)):
            loss = self.difference()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def __init__(self, generator:Generator, steps:int=None):
        if steps is None: steps = math.sqrt(self._generator.numel())
        self._generator = generator
        self._steps = steps
    def sample(self):
        return self._function(self._limited())
    def function(self):
        array = torch.linspace(0, 1, self._steps, device=self._generator.device)
        return array, self._function(array)

class GaussianNormalizer(Normalizer):
    def _function(self, x:torch.Tensor):
        return 0.5*(torch.erf((x-self._parameters[1])/(self._parameters[0]*1.41421356)) + 1)*self._parameters[2] + self._parameters[3]
    def __init__(self, generator:Generator, steps:int=None):
        super().__init__(generator, steps)
        self._parameters = torch.nn.Parameter(torch.rand(4, device=self._generator.device, dtype=self._generator.dtype))


def normalize(generator:Generator, steps:int=None) -> Normalizer:
    """

    :rtype: object
    """
    normalizers = [
        GaussianNormalizer
    ]
    losses = []
    for i, Norm in enumerate(normalizers):
        Norm = Norm(generator, 20)
        Norm.optimize(20)
        losses.append(Norm.difference())
    index = losses.index(min(losses))
    Norm = normalizers[index](generator, steps)
    Norm.optimize(100)
    return Norm
