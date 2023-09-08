import matplotlib.pyplot as plt
from matplotlib.axes import Axes

class Titles:
    def __init__(self, figure:plt.Figure, titles, wspace=-1, hspace=-1, topspace=0.05, bottomspace=0.01, leftspace=0.005, rightspace=0.005):
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

        self._recalc()
    def _recalc(self):
        self.wpg = (1.0 - self.wspace * (self.titles[0] - 1) - self.leftspace - self.rightspace) / self.titles[0]
        self.hpg = (1.0 - self.hspace * (self.titles[1] - 1) - self.topspace - self.bottomspace) / self.titles[1]
    def _repos(self, pos1, pos2):
        left    = (min(pos1[0], pos2[0]) - 1) * (self.wpg + self.wspace) + self.leftspace
        bottom  = 1.0 - (max(pos1[1], pos2[1]) - 1) * (self.hpg + self.hspace) - self.topspace - self.hpg
        width   = (abs(pos1[0]-pos2[0]))*(self.wpg+self.wspace) + self.wpg
        height  = (abs(pos1[1]-pos2[1]))*(self.hpg+self.hspace) + self.hpg

        return [left, bottom, width, height]

    def add_axes(self, pos1, pos2=(-1,-1), projection=None) -> Axes:
        if pos2 == (-1,-1):
            pos2 = pos1
        return self.figure.add_axes(self._repos(pos1, pos2), projection=projection)

    def add_top_annotation(self, text:str, **text_kwargs):
        self.topspace += 0.1
        self._recalc()
        self.figure.text(0.5, 0.95, text, text_kwargs)
    def add_bottom_annotation(self, text:str, **text_kwargs):
        text_kwargs.update({'verticalalignment':'top'})
        self.bottomspace += 0.1
        self._recalc()
        self.figure.text(0.5, 0.05, text, text_kwargs)

    _ColumnAnnotationEnable = False
    def add_column_annotation(self, col_from:int, col_to:int=None, text:str='',  **text_kwargs):
        text_kwargs.update({'rotation': 0})

        if not self._ColumnAnnotationEnable:
            self.bottomspace += 0.1
            self._recalc()
            self._ColumnAnnotationEnable = True
        if col_to is None: col_to = col_from

        left, bottom, width, height = self._repos((col_from,0), (col_to,0))
        self.figure.text(left + width/2, 0.05, text, text_kwargs)

    _RowAnnotationEnable = False
    def add_row_annotation(self, row_from:int, row_to:int=None, text:str='',  **text_kwargs):
        text_kwargs.update({'rotation': 90})
        text_kwargs.update({'horizontalalignment':'right'})

        if not self._RowAnnotationEnable:
            self.leftspace += 0.05
            self._recalc()
            self._RowAnnotationEnable = True
        if row_to is None: row_to = row_from

        left, bottom, width, height = self._repos((0, row_from), (0, row_to))
        self.figure.text(0.025, bottom+height/2, text, text_kwargs)