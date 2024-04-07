
class Generator:
    @property
    def size(self):
        raise NotImplementedError
    @property
    def device(self):
        raise NotImplementedError
    @property
    def dtype(self):
        raise NotImplementedError
    def numel(self):
        raise NotImplementedError
    def sample(self):
        raise NotImplementedError