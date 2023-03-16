import matplotlib.pyplot as plt

class Titles:
    def __init__(self, figure, titles, wspace=-1, hspace=-1, topspace=0.1, bottomspace=0.05, leftspace=0.05, rightspace=0.05):
        self.figure = figure

        self.titles = titles

        if wspace == -1:
            wspace = (1.0 / titles[0])/5
        if hspace == -1:
            hspace = (1.0 / titles[1])/5

        self.wspace = wspace
        self.hspace = hspace
        self.topspace = topspace
        self.bottomspace = bottomspace
        self.leftspace = leftspace
        self.rightspace = rightspace

        self.wpg = (1.0 - wspace * (titles[0] - 1) - leftspace - rightspace) / titles[0]
        self.hpg = (1.0 - hspace * (titles[1] - 1) - topspace - bottomspace) / titles[1]
    def _repos(self, pos1, pos2):
        left    = (min(pos1[0], pos2[0]) - 1) * (self.wpg + self.wspace) + self.leftspace
        bottom  = 1.0 - (max(pos1[1], pos2[1]) - 1) * (self.hpg + self.hspace) - self.topspace - self.hpg
        width   = (abs(pos1[0]-pos2[0]))*(self.wpg+self.wspace) + self.wpg
        height  = (abs(pos1[1]-pos2[1]))*(self.hpg+self.hspace) + self.hpg

        return [left, bottom, width, height]

    def add_axes(self, pos1, pos2=(-1,-1), projection=None):
        if pos2 == (-1,-1):
            pos2 = pos1
        return self.figure.add_axes(self._repos(pos1, pos2), projection=projection)

