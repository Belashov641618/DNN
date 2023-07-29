import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Text, Figure, Axes

from typing import Dict, Union, Tuple, List
from copy import deepcopy

DefaultFont = {
    'alpha' : 1.0,
    'backgroundcolor' : (0, 0, 0, 0),
    'c' : (0,0,0,1),
    'horizontalalignment' : 'center',
    'verticalalignment' : 'center',
    'multialignment' : 'center',
    'font' : 'Times New Roman',
    'fontsize': 12,
    'fontweight': 'normal',
    'rotation' : 0,
    'fontstyle' : 'normal'
}
class FontClass:
    _FontParameters : Dict
    def __init__(self, Parameters:Dict):
        self._FontParameters = DefaultFont.copy()
        if Parameters is not None:
            self.Update(Parameters)

    def Pass(self):
        return self._FontParameters.copy()
    def SpecialPass(self):
        temp = self.Pass()
        temp.pop('alpha')
        temp.pop('backgroundcolor')
        temp.pop('c')
        temp.pop('horizontalalignment')
        temp.pop('verticalalignment')
        return temp
    def __call__(self):
        return self.Pass()

    def Update(self, Parameters:Dict):
        self._FontParameters.update(Parameters)

    @property
    def Rotation(self):
        return self._FontParameters['rotation']
    @Rotation.setter
    def Rotation(self, rotation:float):
        self._FontParameters['rotation'] = rotation

    @property
    def FontStyle(self):
        return self._FontParameters['fontstyle']
    @FontStyle.setter
    def FontStyle(self, style:str):
        self._FontParameters['fontstyle'] = style

    @property
    def FontWeight(self):
        return self._FontParameters['fontweight']
    @FontWeight.setter
    def FontWeight(self, weight: str):
        self._FontParameters['fontweight'] = weight

    @property
    def FontSize(self):
        return self._FontParameters['fontsize']
    @FontSize.setter
    def FontSize(self, size:float):
        self._FontParameters['fontsize'] = size

    @property
    def Font(self):
        return self._FontParameters['font']
    @Font.setter
    def Font(self, font:str):
        self._FontParameters['font'] = font

    @property
    def VerticalAlignment(self):
        return self._FontParameters['verticalalignment']
    @VerticalAlignment.setter
    def VerticalAlignment(self, alignment:str):
        self._FontParameters['verticalalignment'] = alignment

    @property
    def HorizontalAlignment(self):
        return self._FontParameters['horizontalalignment']
    @HorizontalAlignment.setter
    def HorizontalAlignment(self, alignment:str):
        self._FontParameters['horizontalalignment'] = alignment

    @property
    def MultiAlignment(self):
        return self._FontParameters['multialignment']
    @MultiAlignment.setter
    def MultiAlignment(self, alignment:str):
        self._FontParameters['multialignment'] = alignment

    @property
    def Color(self):
        return self._FontParameters['c']
    @Color.setter
    def Color(self, color:str):
        self._FontParameters['c'] = color

    @property
    def BackGroundColor(self):
        return self._FontParameters['backgroundcolor']
    @BackGroundColor.setter
    def BackGroundColor(self, color:str):
        self._FontParameters['backgroundcolor'] = color

    @property
    def Alpha(self):
        return self._FontParameters['alpha']
    @Alpha.setter
    def Alpha(self, alpha:float):
        self._FontParameters['alpha'] = alpha

DefaultBigHeader    = FontClass({
                'fontsize'      : 24,
                'fontweight'    : 'bold',
                'fontstyle'     : 'normal'
            })
DefaultHeader       = FontClass({
                'fontsize'      : 20,
                'fontweight'    : 'bold',
                'fontstyle'     : 'normal'
            })
DefaultDefault      = FontClass({
                'fontsize'      : 16,
                'fontweight'    : 'normal',
                'fontstyle'     : 'normal'
            })
DefaultCaption      = FontClass({
                'fontsize'      : 12,
                'fontweight'    : 'normal',
                'fontstyle'     : 'normal'
            })
DefaultSmallCaption = FontClass({
                'fontsize'      : 8,
                'fontweight'    : 'normal',
                'fontstyle'     : 'italic'
            })
class FontLibraryClass:
    class FontsClass:
        BigHeader           : FontClass
        Header              : FontClass
        Default             : FontClass
        Caption             : FontClass
        SmallCaption        : FontClass

        PlotTitle               : FontClass
        DescriptionTop          : FontClass
        DescriptionBottom       : FontClass
        DescriptionLeft         : FontClass
        DescriptionRight        : FontClass
        ColumnDescriptionTop    : FontClass
        ColumnDescriptionBottom : FontClass
        RowDescriptionLeft      : FontClass
        RowDescriptionRight     : FontClass
        GraphTitle              : FontClass
        GraphDescription        : FontClass
        AxisX                   : FontClass
        AxisY                   : FontClass

        Fonts                : List[FontClass]
        def __init__(self):
            self.BigHeader      = deepcopy(DefaultBigHeader)
            self.Header         = deepcopy(DefaultHeader)
            self.Default        = deepcopy(DefaultDefault)
            self.Caption        = deepcopy(DefaultCaption)
            self.SmallCaption   = deepcopy(DefaultSmallCaption)

            self.PlotTitle                                  = deepcopy(self.BigHeader)
            self.PlotTitle.VerticalAlignment                = 'top'

            self.DescriptionTop                             = deepcopy(self.Default)
            self.DescriptionTop.VerticalAlignment           = 'top'

            self.DescriptionBottom                          = deepcopy(self.Default)
            self.DescriptionBottom.VerticalAlignment        = 'top'

            self.DescriptionLeft                            = deepcopy(self.Default)
            self.DescriptionLeft.VerticalAlignment          = 'center'
            self.DescriptionLeft.HorizontalAlignment        = 'left'
            self.DescriptionLeft.Rotation                   = 90

            self.DescriptionRight                           = deepcopy(self.Default)
            self.DescriptionRight.VerticalAlignment         = 'center'
            self.DescriptionRight.HorizontalAlignment       = 'right'
            self.DescriptionRight.Rotation                  = 90

            self.ColumnDescriptionTop                       = deepcopy(self.Header)
            self.ColumnDescriptionTop.VerticalAlignment     = 'bottom'

            self.ColumnDescriptionBottom                    = deepcopy(self.Header)
            self.ColumnDescriptionBottom.VerticalAlignment  = 'top'

            self.RowDescriptionLeft                         = deepcopy(self.Header)
            self.RowDescriptionLeft.VerticalAlignment       = 'center'
            self.RowDescriptionLeft.HorizontalAlignment     = 'right'
            self.RowDescriptionLeft.Rotation                = 90

            self.RowDescriptionRight                        = deepcopy(self.Header)
            self.RowDescriptionRight.VerticalAlignment      = 'center'
            self.RowDescriptionRight.HorizontalAlignment    = 'left'
            self.RowDescriptionRight.Rotation               = 90

            self.GraphTitle                                 = deepcopy(self.Header)
            self.GraphTitle.VerticalAlignment               = 'bottom'

            self.GraphDescription                           = deepcopy(self.Caption)
            self.GraphDescription.VerticalAlignment         = 'top'

            self.AxisX                                      = deepcopy(self.Caption)
            self.AxisX.VerticalAlignment                    = 'top'

            self.AxisY                                      = deepcopy(self.Caption)
            self.AxisY.VerticalAlignment                    = 'bottom'
            self.AxisY.HorizontalAlignment                  = 'center'
            self.AxisY.Rotation                             = 90

            self.Fonts = [self.BigHeader, self.Header, self.Default, self.Caption, self.SmallCaption, self.PlotTitle, self.DescriptionTop, self.DescriptionBottom, self.DescriptionLeft, self.DescriptionRight, self.ColumnDescriptionTop, self.ColumnDescriptionBottom, self.RowDescriptionLeft, self.RowDescriptionRight, self.GraphTitle, self.GraphDescription, self.AxisX, self.AxisY]
    Fonts : FontsClass

    def __init__(self):
        self.Fonts = self.FontsClass()

    def MultiplyFontSize(self, multiplier:float):
        for font in self.Fonts.Fonts:
            font.FontSize = font.FontSize * multiplier
    def SynchronizeFont(self, font:str):
        for _font in self.Fonts.Fonts:
            _font.Font = font
    def SynchronizeColor(self, color:Union[Tuple[float,float,float],Tuple[float,float,float,float],str]):
        for font in self.Fonts.Fonts:
            font.Color = color

def CheckTextSpell(text:str):
    return text

class WrappedText:
    @staticmethod
    def WrapText(text:str, columns:int, separator:str=' '):
        titles = text.split(separator)
        text = ''
        line = ''
        title = titles[0]
        while len(title) > columns:
            text += title[:columns - 1] + '-' + '\n'
            title = title[columns - 1:]
        line += title
        for title in titles[1:]:
            if len(line) + len(separator) + len(title) < columns:
                line += separator + title
            else:
                text += line + '\n'
                line = ''
                while len(title) > columns:
                    text += title[:columns - 1] + '-' + '\n'
                    title = title[columns - 1:]
                line += title
        if line != '':
            text += line
        if text[-1] == '\n':
            text = text[:-1]
        return text
    @staticmethod
    def CalculateTextSize(text:str, font:FontClass, dpi:float=100, figure:Figure=None):
        close = False
        if figure is None:
            figure:Figure = plt.figure()
            close = True
        figure.set_dpi(dpi)
        frame:Text = figure.text(0,0, text, **font.Pass())
        bbox = frame.get_window_extent(dpi=dpi)
        width   = bbox.width / dpi
        height  = bbox.height / dpi
        if close:
            plt.close(figure)
        return width, height

    _original   : str
    _font       : FontClass
    _dpi        : float
    _available_columns  : List[int]
    _available_width    : List[float]
    _available_height   : List[float]
    def _append_new_variant(self, text:str, columns:int, figure:Figure=None):
        width, height = WrappedText.CalculateTextSize(text, self._font, self._dpi, figure)
        self._available_columns.append(columns)
        self._available_width.append(width)
        self._available_height.append(height)
    def __init__(self, text:str, font:FontClass, dpi:float=100, spell:bool=True):
        if spell: text = CheckTextSpell(text)

        self._original = text
        self._font = font
        self._dpi = dpi

        min_letters = 5
        min_columns = min([(len(word) if len(word) >= min_letters else min_letters) for word in text.split(' ')])
        max_columns = (len(text)+1 if len(text) >= min_letters else min_letters+1)

        self._available_columns = []
        self._available_width   = []
        self._available_height  = []

        last_text = None

        figure = plt.figure()
        for columns in reversed(range(min_columns, max_columns+1)):
            current_text = WrappedText.WrapText(text, columns, ' ')
            if current_text != last_text:
                last_text = current_text
                self._append_new_variant(last_text, columns, figure)
        plt.close(figure)

        columns = self._available_columns[0]
        self._selected_text = WrappedText.WrapText(self._original, columns, ' ')
        self._selected_width = self._available_width[0]
        self._selected_height = self._available_height[0]

    _selected_text      : str
    _selected_width     : float
    _selected_height    : float
    def AdjustWidth(self, width:float):
        best_n = None
        best_width_variant = None
        best_n_ = 0
        best_width_variant_ = self._available_width[0]
        for n, width_variant in enumerate(self._available_width):
            if abs(width_variant - width) < abs(best_width_variant_ - width):
                best_width_variant_ = width_variant
                best_n_ = n
            if width_variant <= width and (best_width_variant is None or (width - best_width_variant) > (width - width_variant)):
                best_width_variant = width_variant
                best_n = n
        if best_width_variant is None:
            best_width_variant = best_width_variant_
            best_n = best_n_

        columns = self._available_columns[best_n]
        self._selected_text = WrappedText.WrapText(self._original, columns, ' ')
        self._selected_width = self._available_width[best_n]
        self._selected_height = self._available_height[best_n]
    def AdjustHeight(self, height:float):
        best_n = None
        best_height_variant = None
        best_n_ = 0
        best_height_variant_ = self._available_height[0]
        for n, height_variant in enumerate(self._available_height):
            if abs(height_variant - height) < abs(best_height_variant_ - height):
                best_height_variant_ = height_variant
                best_n_ = n
            if height_variant <= height and (best_height_variant is None or (height - best_height_variant) > (height - height_variant)):
                best_height_variant = height_variant
                best_n = n
        if best_height_variant is None:
            best_height_variant = best_height_variant_
            best_n = best_n_

        columns = self._available_columns[best_n]
        self._selected_text = WrappedText.WrapText(self._original, columns, ' ')
        self._selected_width = self._available_width[best_n]
        self._selected_height = self._available_height[best_n]

    def __call__(self):
        return self._selected_text
    def text(self):
        return self._selected_text
    def width(self):
        return self._selected_width
    def height(self):
        return self._selected_height
    def font(self):
        return self._font.Pass()

    def ExamineAvailableVariants(self):
        for columns, width, height in zip(self._available_columns, self._available_width, self._available_height):
            text = WrappedText.WrapText(self._original, columns, ' ')
            print('Columns:', columns, ', Width:', width, ', Height:', height, 'Text:')
            print(text)
            print('')

class TiledPlot:
    # Changeable members
    FontLibrary : FontLibraryClass

    # Private members
    _Figure         : Figure
    _FigureAxes     : Axes
    _MaxWidth       : Union[None, float]
    _MaxHeight      : Union[None, float]
    _Columns        : int
    _Rows           : int
    _AxesList       : List[Axes]
    _AxesCordsList  : List[Tuple[Tuple[int, int], Tuple[int, int]]]
    _SelectedIndex  : Union[None, int]

    # Plot texts
    _Title              : Union[None, WrappedText]
    _TopDescription     : Union[None, WrappedText]
    _BottomDescription  : Union[None, WrappedText]
    _LeftDescription    : Union[None, WrappedText]
    _RightDescription   : Union[None, WrappedText]
    def _init_plot_texts(self):
        self._Title             = None
        self._TopDescription    = None
        self._BottomDescription = None
        self._LeftDescription   = None
        self._RightDescription  = None

    # Columns and Rows texts
    _ColumnTopDescriptionList           : List[Union[WrappedText]]
    _ColumnTopDescriptionCordsList      : List[Tuple[int, int]]
    _ColumnBottomDescriptionList        : List[Union[WrappedText]]
    _ColumnBottomDescriptionCordsList   : List[Tuple[int, int]]
    _RowLeftDescriptionList             : List[Union[WrappedText]]
    _RowLeftDescriptionCordsList        : List[Tuple[int, int]]
    _RowRightDescriptionList            : List[Union[WrappedText]]
    _RowRightDescriptionCordsList       : List[Tuple[int, int]]
    def _init_rows_and_columns_texts(self):
        self._ColumnTopDescriptionList              = []
        self._ColumnBottomDescriptionList           = []
        self._RowLeftDescriptionList                = []
        self._RowRightDescriptionList               = []
        self._ColumnTopDescriptionCordsList       = []
        self._ColumnBottomDescriptionCordsList    = []
        self._RowLeftDescriptionCordsList            = []
        self._RowRightDescriptionCordsList           = []

    # Graphs texts
    _GraphTitleList         : List[Union[None, WrappedText]]
    _GraphXAxisList         : List[Union[None, WrappedText]]
    _GraphYAxisList         : List[Union[None, WrappedText]]
    _GraphDescriptionList   : List[Union[None, WrappedText]]
    def _init_graphs_texts(self):
        self._GraphTitleList        = []
        self._GraphXAxisList        = []
        self._GraphYAxisList        = []
        self._GraphDescriptionList  = []

    # Paddings
    _TitlePad                   : float
    _TopDescriptionPad          : float
    _BottomDescriptionPad       : float
    _LeftDescriptionPad         : float
    _RightDescriptionPad        : float
    _ColumnTopDescriptionPad    : float
    _ColumnBottomDescriptionPad : float
    _RowLeftDescriptionPad      : float
    _RowRightDescriptionPad     : float
    _GraphDescriptionPad        : float
    _GraphVerticalPad           : float
    _GraphHorizontalPad         : float
    def _init_paddings(self):
        CharacteristicLength = ((self._MaxWidth if self._MaxWidth is not None else self._MaxHeight) + self._MaxHeight if self._MaxHeight is not None else self._MaxWidth) / 2
        self._TitlePad                      = CharacteristicLength * 0.0050
        self._TopDescriptionPad             = CharacteristicLength * 0.0050
        self._BottomDescriptionPad          = CharacteristicLength * 0.0050
        self._LeftDescriptionPad            = CharacteristicLength * 0.0050
        self._RightDescriptionPad           = CharacteristicLength * 0.0050
        self._ColumnTopDescriptionPad       = CharacteristicLength * 0.0025
        self._ColumnBottomDescriptionPad    = CharacteristicLength * 0.0025
        self._RowLeftDescriptionPad         = CharacteristicLength * 0.0025
        self._RowRightDescriptionPad        = CharacteristicLength * 0.0025
        self._GraphDescriptionPad           = CharacteristicLength * 0.0010
        self._GraphVerticalPad              = CharacteristicLength * 0.0020
        self._GraphHorizontalPad            = CharacteristicLength * 0.0020

    # Axes paddings
    _AxesTopPad     : float
    _AxesBottomPad  : float
    _AxesLeftPad    : float
    _AxesRightPad   : float
    def _calculate_axes_paddings(self):
        self._AxesTopPad    = 0
        self._AxesBottomPad = 0
        self._AxesLeftPad   = 0
        self._AxesRightPad  = 0

        size = ((self._MaxWidth if self._MaxWidth is not None else self._MaxHeight) + (self._MaxHeight if self._MaxHeight is not None else self._MaxWidth)) / 2
        x0      = size/4
        y0      = size/4
        width   = size/2
        height  = size/2
        x1 = x0 + width
        y1 = y0 + width
        for axes, title, x_axis, y_axis in zip(self._AxesList, self._GraphTitleList, self._GraphXAxisList, self._GraphYAxisList):
            axes.set_position((self._relative_x(x0), self._relative_y(y0), self._relative_width(width), self._relative_height(height)))
            if title is not None:
                title.AdjustWidth(width)
                axes.set_title(title.text(), **title.font())
            if x_axis is not None:
                x_axis.AdjustWidth(width)
                axes.set_xlabel(x_axis.text(), **x_axis.font())
            if y_axis is not None:
                y_axis.AdjustHeight(height)
                axes.set_ylabel (y_axis.text(), **y_axis.font())

            real_x0     = axes.get_tightbbox().x0 / self._Figure.get_dpi()
            real_y0     = axes.get_tightbbox().y0 / self._Figure.get_dpi()
            real_x1     = axes.get_tightbbox().x1 / self._Figure.get_dpi()
            real_y1     = axes.get_tightbbox().y1 / self._Figure.get_dpi()

            top_pad     = real_y1 - y1 - (title.height()    if title    is not None else 0)
            bottom_pad  = y0 - real_y0 - (x_axis.height()   if x_axis   is not None else 0)
            left_pad    = x0 - real_x0 - (y_axis.width()    if y_axis   is not None else 0)
            right_pad   = real_x1 - x1

            self._AxesTopPad    = max(self._AxesTopPad,     top_pad)
            self._AxesBottomPad = max(self._AxesBottomPad,  bottom_pad)
            self._AxesLeftPad   = max(self._AxesLeftPad,    left_pad)
            self._AxesRightPad  = max(self._AxesRightPad,   right_pad)

    # Area highlighters and their options
    _HighlightFillColor                 : Union[Tuple[float, float, float], Tuple[float, float, float, float], str]
    _HighlightBorderColor               : Union[Tuple[float, float, float], Tuple[float, float, float, float], str]
    _HighlightTitle                     : bool
    _HighlightTopDescription            : bool
    _HighlightBottomDescription         : bool
    _HighlightLeftDescription           : bool
    _HighlightRightDescription          : bool
    _HighlightColumnTopDescription      : bool
    _HighlightColumnBottomDescription   : bool
    _HighlightRowLeftDescription        : bool
    _HighlightRowRightDescription       : bool
    _HighlightGraphDescription          : bool
    def _init_highlighters(self):
        self._HighlightFillColor                = (0.2, 0.2, 0.7, 0.5)
        self._HighlightBorderColor              = (0.1, 0.1, 0.4, 0.7)
        self._HighlightTitle                    = False
        self._HighlightTopDescription           = False
        self._HighlightBottomDescription        = False
        self._HighlightLeftDescription          = False
        self._HighlightRightDescription         = False
        self._HighlightColumnTopDescription     = False
        self._HighlightColumnBottomDescription  = False
        self._HighlightRowLeftDescription       = False
        self._HighlightRowRightDescription      = False
        self._HighlightGraphDescription         = False
    def _highlight(self, x:float, y:float, width:float, height:float):
        self._FigureAxes.add_patch(Rectangle((x, y), width, height, edgecolor=self._HighlightBorderColor, facecolor=self._HighlightFillColor))

    # Additional parameters
    _AxesWidthToHeight : float

    # Just initialization and some special methods
    def __init__(self, MaxWidth:float=None, MaxHeight:float=None, DPI:float=100):
        self.FontLibrary = FontLibraryClass()
        if (MaxWidth is None) and (MaxHeight is None):
            figure = plt.figure()
            figure_manager = figure.canvas.manager
            window = figure_manager.window
            MaxWidth  = 0.9 * window.winfo_screenwidth()  / DPI
            MaxHeight = 0.9 * window.winfo_screenheight() / DPI
            plt.close(figure)
        size = ((MaxWidth if MaxWidth is not None else MaxHeight) + (MaxHeight if MaxHeight is not None else MaxWidth)) / 2
        self._Figure = plt.figure(dpi=DPI, figsize=(size, size))
        self._FigureAxes = self._Figure.add_axes((0,0,1,1))
        self._MaxWidth = MaxWidth
        self._MaxHeight = MaxHeight
        self._Columns = 0
        self._Rows = 0
        self._AxesList = []
        self._AxesCordsList = []
        self._SelectedIndex = None
        self._AxesWidthToHeight = 1.0
        self._init_plot_texts()
        self._init_rows_and_columns_texts()
        self._init_graphs_texts()
        self._init_paddings()
        self._init_highlighters()
    def show    (self, finalize:bool=True):
        if finalize: self.finalize()
        plt.show()
    def finalize(self):
        self._find_best_construction()
    def examine_virtual_figure_construction(self):
        self._calculate_axes_paddings()
        padding = 0.1
        points = 300

        min_size = 0
        max_size = max(self._MaxWidth, self._MaxHeight)
        length = max_size - min_size
        min_size = min_size - length * padding
        max_size = max_size + length * padding

        import numpy
        size_array = numpy.linspace(min_size, max_size, points)
        width_array = numpy.zeros(points)
        height_array = numpy.zeros(points)
        for n, size in enumerate(size_array):
            width_array[n], height_array[n], *_ = self._virtual_construct_figure(size)

        figure:Figure = plt.figure(figsize=(12,12))
        axes:Axes = figure.add_axes((0.1,0.1,0.8,0.8))
        axes.grid(True)
        axes.set_title('Зависимость ширины от параметра size', **self.FontLibrary.Fonts.GraphTitle())
        axes.set_xlabel('size, Дюймы', **self.FontLibrary.Fonts.AxisX())
        axes.set_ylabel('Ширина, Дюймы', **self.FontLibrary.Fonts.AxisY())
        axes.plot(size_array, width_array)
        axes.axhline(self._MaxWidth, linestyle='--')
        axes.axvline(0, linestyle='--')
        axes.axvline(self._MaxWidth, linestyle='--')

        figure:Figure = plt.figure(figsize=(12, 12))
        axes:Axes = figure.add_axes((0.1, 0.1, 0.8, 0.8))
        axes.grid(True)
        axes.set_title('Зависимость высоты от параметра size', **self.FontLibrary.Fonts.GraphTitle())
        axes.set_xlabel('size, Дюймы', **self.FontLibrary.Fonts.AxisX())
        axes.set_ylabel('Высота, Дюймы', **self.FontLibrary.Fonts.AxisY())
        axes.plot(size_array, height_array)
        axes.axhline(self._MaxHeight, linestyle='--')
        axes.axvline(0, linestyle='--')
        axes.axvline(self._MaxHeight, linestyle='--')

        plt.show()

    # Figure construction
    def _relative_x             (self, x:float):
        return x / self._Figure.get_figwidth()
    def _relative_y             (self, y:float):
        return y / self._Figure.get_figheight()
    def _relative_width         (self, width:float):
        return abs(self._relative_x(width) - self._relative_x(0))
    def _relative_height        (self, height:float):
        return abs(self._relative_y(height) - self._relative_y(0))
    def _relative_coordinates   (self, x:float, y:float):
        return self._relative_x(x), self._relative_y(y)

    def _virtual_construct_figure   (self, size:float):
        # Size of space where it`s able to draw
        AxesW = size * self._AxesWidthToHeight
        AxesH = size

        GraphTitleW = AxesW
        GraphTitleH = 0
        GraphXAxisW = AxesW
        GraphXAxisH = 0
        GraphYAxisW = 0
        GraphYAxisH = AxesH
        GraphDescriptionW = AxesW
        GraphDescriptionH = 0
        for ((c0, r0), (c1, r1)), title, x_axis, y_axis, description in zip(self._AxesCordsList, self._GraphTitleList, self._GraphXAxisList, self._GraphYAxisList, self._GraphDescriptionList):
            width   = (c1 - c0 + 1) * AxesW
            height  = (r1 - r0 + 1) * AxesH
            if title is not None:
                title.AdjustWidth(width)
                GraphTitleH = max(GraphTitleH, title.height())
            if x_axis is not None:
                x_axis.AdjustWidth(width)
                GraphXAxisH = max(GraphXAxisH, x_axis.height())
            if y_axis is not None:
                y_axis.AdjustHeight(height)
                GraphYAxisW = max(GraphYAxisW, y_axis.width())
            if description is not None:
                description.AdjustWidth(width)
                GraphDescriptionH = max(GraphDescriptionH, description.height())

        # Size of single graph with it`s title and labels
        GraphW = AxesW + GraphYAxisW + self._AxesLeftPad + self._AxesRightPad
        GraphH = AxesH + GraphTitleH + GraphXAxisH + self._AxesTopPad + self._AxesBottomPad

        # Size of single graph tile, graph with it`s title, labels and description
        GraphTileW = GraphW
        GraphTileH = GraphH + GraphDescriptionH + self._GraphDescriptionPad

        # Size of graph space, of all tiles
        GraphSpaceW = GraphTileW * self._Columns + self._GraphHorizontalPad * (self._Columns - 1)
        GraphSpaceH = GraphTileH * self._Rows + self._GraphVerticalPad * (self._Rows - 1)

        ColumnTopDescriptionW = AxesW
        ColumnTopDescriptionH = 0
        for (c0, c1), description in zip(self._ColumnTopDescriptionCordsList, self._ColumnTopDescriptionList):
            width = (c1 - c0 + 1) * AxesW
            description.AdjustWidth(width)
            ColumnTopDescriptionH = max(ColumnTopDescriptionH, description.height())

        ColumnBottomDescriptionW = AxesW
        ColumnBottomDescriptionH = 0
        for (c0, c1), description in zip(self._ColumnBottomDescriptionCordsList, self._ColumnBottomDescriptionList):
            width = (c1 - c0 + 1) * AxesW
            description.AdjustWidth(width)
            ColumnBottomDescriptionH = max(ColumnBottomDescriptionH, description.height())

        RowLeftDescriptionW = 0
        RowLeftDescriptionH = AxesH
        for (r0, r1), description in zip(self._RowLeftDescriptionCordsList, self._RowLeftDescriptionList):
            height = (r1 - r0 + 1) * AxesH
            description.AdjustHeight(height)
            RowLeftDescriptionW = max(RowLeftDescriptionW, description.width())

        RowRightDescriptionW = 0
        RowRightDescriptionH = AxesH
        for (r0, r1), description in zip(self._RowRightDescriptionCordsList, self._RowRightDescriptionList):
            height = (r1 - r0 + 1) * AxesH
            description.AdjustHeight(height)
            RowRightDescriptionW = max(RowRightDescriptionW, description.width())

        # Expanded graph space with descriptions through columns and rows
        ExpandedGraphSpaceW = GraphSpaceW + RowLeftDescriptionW + RowRightDescriptionW + self._RowLeftDescriptionPad + self._RowRightDescriptionPad
        ExpandedGraphSpaceH = GraphSpaceH + ColumnTopDescriptionH + ColumnBottomDescriptionH + self._ColumnTopDescriptionPad + self._ColumnBottomDescriptionPad

        LeftDescriptionW = 0.0
        LeftDescriptionH = ExpandedGraphSpaceH
        if self._LeftDescription is not None:
            self._LeftDescription.AdjustHeight(LeftDescriptionH)
            LeftDescriptionW = max(LeftDescriptionW, self._LeftDescription.width())

        RightDescriptionW = 0.0
        RightDescriptionH = ExpandedGraphSpaceH
        if self._RightDescription is not None:
            self._RightDescription.AdjustHeight(RightDescriptionH)
            RightDescriptionW = max(RightDescriptionW, self._RightDescription.width())

        # Central space that includes expanded graph space and left and right descriptions
        CentralSpaceW = ExpandedGraphSpaceW + LeftDescriptionW + RightDescriptionW + self._LeftDescriptionPad + self._RightDescriptionPad
        CentralSpaceH = ExpandedGraphSpaceH

        TopDescriptionW = CentralSpaceW
        TopDescriptionH = 0.0
        if self._TopDescription is not None:
            self._TopDescription.AdjustWidth(TopDescriptionW)
            TopDescriptionH = max(TopDescriptionH, self._TopDescription.height())

        BottomDescriptionW = CentralSpaceW
        BottomDescriptionH = 0.0
        if self._BottomDescription is not None:
            self._BottomDescription.AdjustWidth(BottomDescriptionW)
            BottomDescriptionH = max(BottomDescriptionH, self._BottomDescription.height())

        TitleW = CentralSpaceW
        TitleH = 0
        if self._Title is not None:
            self._Title.AdjustWidth(TitleW)
            TitleH = self._Title.height()

        # Whole plot size
        PlotW = CentralSpaceW
        PlotH = CentralSpaceH + TopDescriptionH + BottomDescriptionH + TitleH + self._TopDescriptionPad + self._BottomDescriptionPad + self._TitlePad

        parameters = (
            AxesW,
            AxesH,
            GraphTitleW,
            GraphTitleH,
            GraphXAxisW,
            GraphXAxisH,
            GraphYAxisW,
            GraphYAxisH,
            GraphDescriptionW,
            GraphDescriptionH,
            GraphW,
            GraphH,
            GraphTileW,
            GraphTileH,
            GraphSpaceW,
            GraphSpaceH,
            ColumnTopDescriptionW,
            ColumnTopDescriptionH,
            ColumnBottomDescriptionW,
            ColumnBottomDescriptionH,
            RowLeftDescriptionW,
            RowLeftDescriptionH,
            RowRightDescriptionW,
            RowRightDescriptionH,
            ExpandedGraphSpaceW,
            ExpandedGraphSpaceH,
            LeftDescriptionW,
            LeftDescriptionH,
            RightDescriptionW,
            RightDescriptionH,
            CentralSpaceW,
            CentralSpaceH,
            TopDescriptionW,
            TopDescriptionH,
            BottomDescriptionW,
            BottomDescriptionH,
            TitleW,
            TitleH
        )

        return PlotW, PlotH, parameters
    def _adjust_height              (self):
        tolerance = 1.0E-6
        max_divisions = 10
        min_size = 0
        max_size = self._MaxHeight

        def parameter(size):
            width, height, _ = self._virtual_construct_figure(size)
            return self._MaxHeight > height

        size1 = min_size
        parameter1 = parameter(size1)
        size2 = max_size
        parameter2 = parameter(size2)

        if parameter1 == parameter2:
            n = 1
            shift = (max_size - min_size) / 2
            step = 1
            stop = False
            for i in range(max_divisions):
                for pi in range(n):
                    size0 = min_size + shift + n*step
                    parameter0 = parameter(size0)
                    if parameter0 != parameter1:
                        parameter1 = parameter0
                        size1 = size0
                        stop = True
                        break
                if stop:
                    break
                step /= 2
                shift /= 2
                n *= 2
            if not stop: raise ValueError('Can`t find best size for such configuration :(')

        while abs(size1 - size2) > tolerance:
            size0 = (size1 + size2) / 2
            parameter0 = parameter(size0)
            if parameter0 == parameter1:    size1 = size0
            elif parameter0 == parameter2:  size2 = size0

        return self._virtual_construct_figure((size1 + size2) / 2)
    def _adjust_width               (self):
        tolerance = 1.0E-6
        max_divisions = 10
        min_size = 0
        max_size = self._MaxWidth

        def parameter(size):
            width, height, _ = self._virtual_construct_figure(size)
            return self._MaxWidth > width

        size1 = min_size
        parameter1 = parameter(size1)
        size2 = max_size
        parameter2 = parameter(size2)

        if parameter1 == parameter2:
            n = 1
            shift = (max_size - min_size) / 2
            step = 1
            stop = False
            for i in range(max_divisions):
                for pi in range(n):
                    size0 = min_size + shift + n * step
                    parameter0 = parameter(size0)
                    if parameter0 != parameter1:
                        parameter1 = parameter0
                        size1 = size0
                        stop = True
                        break
                if stop:
                    break
                step /= 2
                shift /= 2
                n *= 2
            if not stop: raise ValueError('Can`t find best size for such configuration :(')

        while abs(size1 - size2) > tolerance:
            size0 = (size1 + size2) / 2
            parameter0 = parameter(size0)
            if parameter0 == parameter1:
                size1 = size0
            elif parameter0 == parameter2:
                size2 = size0

        return self._virtual_construct_figure((size1 + size2) / 2)
    def _find_best_construction     (self):
        self._calculate_axes_paddings()
        if (self._MaxWidth is not None) and (self._MaxHeight is None):
            PlotW, PlotH, parameters = self._adjust_width()
            self._construct_figure(PlotW, PlotH, parameters)
        elif (self._MaxWidth is None) and (self._MaxHeight is not None):
            PlotW, PlotH, parameters = self._adjust_height()
            self._construct_figure(PlotW, PlotH, parameters)
        elif (self._MaxWidth is not None) and (self._MaxHeight is not None):
            try:
                PlotW, PlotH, parameters = self._adjust_height()
                if PlotW <= self._MaxWidth:
                    self._construct_figure(PlotW, PlotH, parameters)
                else: raise ValueError
            except ValueError:
                try:
                    PlotW, PlotH, parameters = self._adjust_width()
                    if PlotH <= self._MaxHeight:
                        self._construct_figure(PlotW, PlotH, parameters)
                    else:
                        raise ValueError
                except ValueError: raise Exception('Cant find best size for both configurations :(')
        else: raise Exception('Somehow max width and max height are None')
    def _construct_figure           (self, PlotW:float, PlotH:float, parameters:Tuple):
        (
            AxesW,
            AxesH,
            GraphTitleW,
            GraphTitleH,
            GraphXAxisW,
            GraphXAxisH,
            GraphYAxisW,
            GraphYAxisH,
            GraphDescriptionW,
            GraphDescriptionH,
            GraphW,
            GraphH,
            GraphTileW,
            GraphTileH,
            GraphSpaceW,
            GraphSpaceH,
            ColumnTopDescriptionW,
            ColumnTopDescriptionH,
            ColumnBottomDescriptionW,
            ColumnBottomDescriptionH,
            RowLeftDescriptionW,
            RowLeftDescriptionH,
            RowRightDescriptionW,
            RowRightDescriptionH,
            ExpandedGraphSpaceW,
            ExpandedGraphSpaceH,
            LeftDescriptionW,
            LeftDescriptionH,
            RightDescriptionW,
            RightDescriptionH,
            CentralSpaceW,
            CentralSpaceH,
            TopDescriptionW,
            TopDescriptionH,
            BottomDescriptionW,
            BottomDescriptionH,
            TitleW,
            TitleH
        ) = parameters

        figure = self._Figure
        x = self._relative_x
        y = self._relative_y
        width = self._relative_width
        height = self._relative_height

        figure.set_figwidth(PlotW)
        figure.set_figheight(PlotH)

        if self._Title              is not None: figure.text(x(PlotW/2),    y(PlotH),                                                                   self._Title.text(),             **self._Title.font())
        if self._TopDescription     is not None: figure.text(x(PlotW/2),    y(PlotH-TitleH-self._TitlePad),                                             self._TopDescription.text(),    **self._TopDescription.font())
        if self._BottomDescription  is not None: figure.text(x(PlotW/2),    y(BottomDescriptionH),                                                      self._BottomDescription.text(), **self._BottomDescription.font())
        if self._LeftDescription    is not None: figure.text(x(0),          y(BottomDescriptionH + self._BottomDescriptionPad + LeftDescriptionH/2),    self._LeftDescription.text(),   **self._LeftDescription.font())
        if self._RightDescription   is not None: figure.text(x(PlotW),      y(BottomDescriptionH + self._BottomDescriptionPad + RightDescriptionH/2),   self._RightDescription.text(),  **self._RightDescription.font())

        # if self._HighlightTitle:                self._highlight(x(0), y(PlotH-TitleH), width(TitleW), height(TitleH))
        # if self._HighlightTopDescription:       self._highlight(x(0), y(PlotH-TitleH-self._TitlePad), width(TopDescriptionW), height(TopDescriptionH))
        # if self._HighlightBottomDescription:    self._highlight(x(0), y(0), width(BottomDescriptionW), height(BottomDescriptionH))
        # if self._HighlightLeftDescription:      self._highlight(x(0), y(BottomDescriptionH + self._BottomDescriptionPad), width(LeftDescriptionW), height(LeftDescriptionH))
        # if self._HighlightRightDescription:     self._highlight(x(0), y(BottomDescriptionH + self._BottomDescriptionPad), width(RightDescriptionW), height(LeftDescriptionH))

        col_to_axes_x = lambda col: x(LeftDescriptionW + self._LeftDescriptionPad + RowLeftDescriptionW + GraphYAxisW + self._AxesLeftPad + col*(GraphTileW + self._GraphHorizontalPad))
        row_to_axes_y = lambda row: y(BottomDescriptionH + self._BottomDescriptionPad + ColumnBottomDescriptionH + self._ColumnBottomDescriptionPad + GraphDescriptionH + self._GraphDescriptionPad + GraphXAxisH + self._AxesBottomPad + (self._Rows - row - 1)*(GraphTileH + self._GraphVerticalPad))

        left_description_x = x(LeftDescriptionW + self._LeftDescriptionPad + RowLeftDescriptionW)
        for (r0, r1), description in zip(self._RowLeftDescriptionCordsList, self._RowLeftDescriptionList):
            figure.text(left_description_x, (row_to_axes_y(r0) + row_to_axes_y(r1) + y(AxesH))/2, description.text(), **description.font())
            # if self._HighlightRowLeftDescription: self._highlight(left_description_x-x(RowLeftDescriptionW), row_to_axes_y(r0), width(RowLeftDescriptionW), row_to_axes_y(r1)+y(AxesH)-row_to_axes_y(r0))

        right_description_x = x(PlotW - RightDescriptionW - self._RightDescriptionPad - RowRightDescriptionW)
        for (r0, r1), description in zip(self._RowRightDescriptionCordsList, self._RowRightDescriptionList):
            figure.text(right_description_x, (row_to_axes_y(r0) + row_to_axes_y(r1) + y(AxesH))/2, description.text(), **description.font())
            # if self._HighlightRowRightDescription: self._highlight(right_description_x, row_to_axes_y(r0), width(RowRightDescriptionW), row_to_axes_y(r1)+y(AxesH)-row_to_axes_y(r0))

        top_description_y = y(PlotH - TitleH - self._TitlePad - TopDescriptionH - self._TopDescriptionPad - ColumnTopDescriptionH)
        for (c0, c1), description in zip(self._ColumnTopDescriptionCordsList, self._ColumnTopDescriptionList):
            figure.text((col_to_axes_x(c0) + col_to_axes_x(c1) + x(AxesW))/2, top_description_y, description.text(), **description.font())

        bottom_description_y = y(BottomDescriptionH + self._BottomDescriptionPad + ColumnBottomDescriptionH)
        for (c0, c1), description in zip(self._ColumnBottomDescriptionCordsList, self._ColumnBottomDescriptionList):
            figure.text((col_to_axes_x(c0) + col_to_axes_x(c1) + x(AxesW))/2, bottom_description_y, description.text(), **description.font())

        for axes, ((c0, r0), (c1, r1)), title, x_axis, y_axis, description in zip(self._AxesList, self._AxesCordsList, self._GraphTitleList, self._GraphXAxisList, self._GraphYAxisList, self._GraphDescriptionList):
            axes_x0 = col_to_axes_x(c0)
            axes_y0 = row_to_axes_y(r1)
            axes_x1 = col_to_axes_x(c1)
            axes_y1 = row_to_axes_y(r0)
            axes_width = axes_x1 - axes_x0 + width(AxesW)
            axes_height = axes_y1 - axes_y0 + height(AxesH)
            axes.set_position((axes_x0, axes_y0, axes_width, axes_height))
            if title is not None:   axes.set_title(title.text(), **title.font())
            if x_axis is not None:  axes.set_xlabel(x_axis.text(), **x_axis.font())
            if y_axis is not None:  axes.set_ylabel(y_axis.text(), **y_axis.font())
            if description is not None:
                figure.text(axes_x0 + axes_width/2, axes_y0 - y(self._AxesBottomPad + GraphXAxisH + self._GraphDescriptionPad), description.text(), **description.font())

    # Axes manipulations
    @staticmethod
    def _correct_positions(position1:Union[int,List[int],Tuple[int,int],None], position2:Union[int,List[int],Tuple[int,int],None]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        if position1 is None and position2 is None:
            position1:Tuple[int, int] = (0, 0)
            position2:Tuple[int, int] = (0, 0)
        elif isinstance(position1, int) and isinstance(position2, int):
            position1:Tuple[int, int] = (position1, position2)
            position2:Tuple[int, int] = (position1[0], position1[1])
        elif isinstance(position1, (List, Tuple)) and position2 is None:
            position1:Tuple[int, int] = (position1[0], position1[1])
            position2:Tuple[int, int] = (position1[0], position1[1])
        elif isinstance(position1, (List, Tuple)) and isinstance(position2, (List, Tuple)):
            position1:Tuple[int, int] = (position1[0], position1[1])
            position2:Tuple[int, int] = (position2[0], position2[1])
        else:
            raise AttributeError('Wrong Positioning!')
        position1, position2 = (min(position1[0], position2[0]), min(position1[1], position2[1])), (max(position1[0], position2[0]), max(position1[1], position2[1]))
        return position1, position2
    def _add_axes(self, position1:Union[int,List[int],Tuple[int,int]]=None, position2:Union[int,List[int],Tuple[int,int]]=None, **kwargs) -> Axes:
        position1, position2 = TiledPlot._correct_positions(position1, position2)

        self._AxesList.append(self._Figure.add_axes((0, 0, 1, 1), *kwargs))
        self._AxesCordsList.append((position1, position2))
        self._SelectedIndex = len(self._AxesList) - 1
        self._AxesList[self._SelectedIndex].xaxis.set_tick_params(labelsize=self.FontLibrary.Fonts.SmallCaption.FontSize)
        self._AxesList[self._SelectedIndex].yaxis.set_tick_params(labelsize=self.FontLibrary.Fonts.SmallCaption.FontSize)

        if self._Columns    < position2[0]+1: self._Columns   = position2[0]+1
        if self._Rows       < position2[1]+1: self._Rows      = position2[1]+1

        self._GraphTitleList.append(None)
        self._GraphXAxisList.append(None)
        self._GraphYAxisList.append(None)
        self._GraphDescriptionList.append(None)

        return self._selected_axes
    def _select_axes(self, axes:Axes=None, index:int=None, cords:Tuple[int, int]=None):
        if axes is not None:
            index = self._AxesList.index(axes)
            if index is None:
                raise Exception('Can`t find such axes!')
            self._SelectedIndex = index
        elif index is not None:
            if index >= len(self._AxesList):
                raise Exception('Index too large or below zero! Choose index from ' + str(0) + ' to ' + str(len(self._AxesList) - 1) + '!')
            self._SelectedIndex = index
        elif cords is not None:
            for (index, (position1, position2)) in enumerate(self._AxesCordsList):
                if isinstance(position1, int): position1 = [position1, position1+1]
                if isinstance(position2, int): position2 = [position2, position2+1]
                if (cords[0] >= position1[0]) and (cords[1] >= position1[1]) and (cords[0] < position2[0]) and (cords[1] < position2[1]):
                    self._SelectedIndex = index
                    return
            LimitsX = [self._AxesCordsList[0][0][0], self._AxesCordsList[0][0][0]]
            LimitsY = [self._AxesCordsList[0][0][1], self._AxesCordsList[0][0][1]]
            for index, (position1, position2) in enumerate(self._AxesCordsList):
                if isinstance(position1, int): position1 = [position1, position1+1]
                if isinstance(position2, int): position2 = [position2, position2+1]
                if position1[0] < LimitsX[0]: LimitsX[0] = position1[0]
                if position2[0] < LimitsX[0]: LimitsX[0] = position2[0]
                if position1[0] > LimitsX[1]: LimitsX[1] = position1[0]
                if position2[0] > LimitsX[1]: LimitsX[1] = position2[0]
                if position1[1] < LimitsY[0]: LimitsY[0] = position1[1]
                if position2[1] < LimitsY[0]: LimitsY[0] = position2[1]
                if position1[1] > LimitsY[1]: LimitsY[1] = position1[1]
                if position2[1] > LimitsY[1]: LimitsY[1] = position2[1]
            raise Exception('Can`t find axes with such position, pass position {[' + str(LimitsX[0]) + ';' + str(LimitsX[1]) + '), [' + str(LimitsY[0]) + ';' + str(LimitsY[1]) + ')}')
        else:
            raise AttributeError('To select axes you have to pass "axes" or "index" or "cords" attributes!')
    @property
    def _selected_axes(self) -> Axes:
        return self._AxesList[self._SelectedIndex]
    @_selected_axes.setter
    def _selected_axes(self, parameter:Union[Axes, int, Tuple[int, int]]):
        if   isinstance(parameter, Axes):   self._select_axes(axes=parameter)
        elif isinstance(parameter, int):    self._select_axes(index=parameter)
        elif isinstance(parameter, Tuple):  self._select_axes(cords=parameter)

    # Plot text manipulations
    def _title               (self, text:str):
        self._Title = WrappedText(text, font=self.FontLibrary.Fonts.PlotTitle, dpi=self._Figure.get_dpi())
    def _top_description     (self, text:str):
        self._TopDescription    = WrappedText(text, font=self.FontLibrary.Fonts.DescriptionTop, dpi=self._Figure.get_dpi())
    def _bottom_description  (self, text:str):
        self._BottomDescription = WrappedText(text, font=self.FontLibrary.Fonts.DescriptionBottom, dpi=self._Figure.get_dpi())
    def _left_description    (self, text:str):
        self._LeftDescription   = WrappedText(text, font=self.FontLibrary.Fonts.DescriptionLeft, dpi=self._Figure.get_dpi())
    def _right_description   (self, text:str):
        self._RightDescription  = WrappedText(text, font=self.FontLibrary.Fonts.DescriptionRight, dpi=self._Figure.get_dpi())

    # Columns text manipulations
    def _correct_columns(self, column1:Union[int,None], column2:Union[int,None]):
        if column2 is None:
            if column1 is None:
                if self._SelectedIndex is None:
                    raise AttributeError('Provide column1 or column2 or both or select an axes!')
                column1 = self._AxesCordsList[self._SelectedIndex][0][0]
                column2 = self._AxesCordsList[self._SelectedIndex][1][0] - 1
            else:
                column2 = column1
        elif column1 is None:
            column1 = column2
        column1, column2 = min(column1, column2), max(column1, column2)
        return column1, column2
    @staticmethod
    def _check_columns_intersection(column1:int, column2:int, columns_list:List[Tuple[int,int]]):
        for (col1, col2) in columns_list:
            if (col1 <= column1 <= col2) or (col1 <= column2 <= col2):
                return False
        return True
    def _column_top_description      (self, text:str, column1:int=None, column2:int=None):
        column1, column2 = self._correct_columns(column1, column2)
        if not TiledPlot._check_columns_intersection(column1, column2, self._ColumnTopDescriptionCordsList):
            raise AttributeError('Wrong column1, column2 were passed and description intersect with other description!')
        self._ColumnTopDescriptionList.append(WrappedText(text, font=self.FontLibrary.Fonts.ColumnDescriptionTop, dpi=self._Figure.get_dpi()))
        self._ColumnTopDescriptionCordsList.append((column1, column2))
    def _column_bottom_description   (self, text:str, column1:int=None, column2:int=None):
        column1, column2 = self._correct_columns(column1, column2)
        if not TiledPlot._check_columns_intersection(column1, column2, self._ColumnBottomDescriptionCordsList):
            raise AttributeError('Wrong column1, column2 were passed and description intersect with other description!')
        self._ColumnBottomDescriptionList.append(WrappedText(text, font=self.FontLibrary.Fonts.ColumnDescriptionBottom, dpi=self._Figure.get_dpi()))
        self._ColumnBottomDescriptionCordsList.append((column1, column2))

    # Rows text manipulations
    def _correct_rows(self, row1:Union[int, None], row2:Union[int, None]):
        if row2 is None:
            if row1 is None:
                if self._SelectedIndex is None:
                    raise AttributeError('Provide column1 or column2 or both or select an axes!')
                row1 = self._AxesCordsList[self._SelectedIndex][0][1]
                row2 = self._AxesCordsList[self._SelectedIndex][1][1] - 1
            else:
                row2 = row1
        elif row1 is None:
            row1 = row2
        row1, row2 = min(row1, row2), max(row1, row2)
        return row1, row2
    @staticmethod
    def _check_rows_intersection(row1:int, row2:int, rows_list:List[Tuple[int, int]]):
        for (col1, col2) in rows_list:
            if (col1 <= row1 <= col2) or (col1 <= row2 <= col2):
                return False
        return True
    def _row_left_description    (self, text:str, row1:int=None, row2:int=None):
        row1, row2 = self._correct_columns(row1, row2)
        if not TiledPlot._check_rows_intersection(row1, row2, self._RowLeftDescriptionCordsList):
            raise AttributeError('Wrong row1, row2 were passed and description intersect with other description!')
        self._RowLeftDescriptionList.append(WrappedText(text, font=self.FontLibrary.Fonts.RowDescriptionLeft, dpi=self._Figure.get_dpi()))
        self._RowLeftDescriptionCordsList.append((row1, row2))
    def _row_right_description   (self, text:str, row1:int=None, row2:int=None):
        row1, row2 = self._correct_columns(row1, row2)
        if not TiledPlot._check_rows_intersection(row1, row2, self._RowRightDescriptionCordsList):
            raise AttributeError('Wrong row1, row2 were passed and description intersect with other description!')
        self._RowRightDescriptionList.append(WrappedText(text, font=self.FontLibrary.Fonts.RowDescriptionRight, dpi=self._Figure.get_dpi()))
        self._RowRightDescriptionCordsList.append((row1, row2))

    # Graph text manipulations
    def _reselect_axes(self, axes:Axes=None, index:int=None, cords:Tuple[int,int]=None):
        if (axes is not None) or (index is not None) or (cords is not None):
            self._select_axes(axes=axes, index=index, cords=cords)
        elif self._SelectedIndex is None:
            raise AttributeError('Provide selected_axes or selected_index or selected_cords or select an axes firstly!')
    def _graph_title         (self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int, int]=None):
        self._reselect_axes(axes=selected_axes, index=selected_index, cords=selected_cords)
        self._GraphTitleList[self._SelectedIndex] = WrappedText(text, font=self.FontLibrary.Fonts.GraphTitle, dpi=self._Figure.get_dpi())
    def _graph_axis_x        (self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int, int]=None):
        self._reselect_axes(axes=selected_axes, index=selected_index, cords=selected_cords)
        self._GraphXAxisList[self._SelectedIndex] = WrappedText(text, font=self.FontLibrary.Fonts.AxisX, dpi=self._Figure.get_dpi())
    def _graph_axis_y        (self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int, int]=None):
        self._reselect_axes(axes=selected_axes, index=selected_index, cords=selected_cords)
        self._GraphYAxisList[self._SelectedIndex] = WrappedText(text, font=self.FontLibrary.Fonts.AxisY, dpi=self._Figure.get_dpi())
    def _graph_description   (self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int, int]=None):
        self._reselect_axes(axes=selected_axes, index=selected_index, cords=selected_cords)
        self._GraphDescriptionList[self._SelectedIndex] = WrappedText(text, font=self.FontLibrary.Fonts.GraphDescription, dpi=self._Figure.get_dpi())

    # Padding manipulations
    def _title_pad                       (self, height:float):
        self._TitlePad = height
    def _top_description_pad             (self, height:float):
        self._TopDescriptionPad = height
    def _bottom_description_pad          (self, height:float):
        self._BottomDescriptionPad = height
    def _left_description_pad            (self, width:float):
        self._LeftDescriptionPad = width
    def _right_description_pad           (self, width:float):
        self._RightDescriptionPad = width
    def _column_top_description_pad      (self, height:float):
        self._ColumnTopDescriptionPad = height
    def _column_bottom_description_pad   (self, height:float):
        self._ColumnBottomDescriptionPad = height
    def _row_left_description_pad        (self, width:float):
        self._RowLeftDescriptionPad = width
    def _row_right_description_pad       (self, width:float):
        self._RowRightDescriptionPad = width
    def _graph_description_pad           (self, height:float):
        self._GraphDescriptionPad = height
    def _graph_vertical_pad              (self, height:float):
        self._GraphVerticalPad = height
    def _graph_horizontal_pad            (self, width:float):
        self._GraphHorizontalPad = width

    # Highlight manipulations
    def highlight_all(self):
        self._HighlightTitle                    = True
        self._HighlightTopDescription           = True
        self._HighlightBottomDescription        = True
        self._HighlightLeftDescription          = True
        self._HighlightRightDescription         = True
        self._HighlightColumnTopDescription     = True
        self._HighlightColumnBottomDescription  = True
        self._HighlightRowLeftDescription       = True
        self._HighlightRowRightDescription      = True
        self._HighlightGraphDescription         = True

    # Convenient access
    def width_to_height(self, value:float):
        self._AxesWidthToHeight = value
    def title(self, text:str):
        self._title(text)
    @property
    def description(self):
        class Accesser:
            _plot : TiledPlot
            def __init__(self, plot:TiledPlot):
                self._plot = plot
            def top(self, text:str):
                self._plot._top_description(text)
            def bottom(self, text:str):
                self._plot._bottom_description(text)
            def left(self, text:str):
                self._plot._left_description(text)
            def right(self, text:str):
                self._plot._right_description(text)
            @property
            def column(self):
                class ColumnAccesser:
                    _plot : TiledPlot
                    def __init__(self, plot:TiledPlot):
                        self._plot = plot
                    def top(self, text:str, column1:int=None, column2:int=None):
                        self._plot._column_top_description(text, column1, column2)
                    def bottom(self, text:str, column1:int=None, column2:int=None):
                        self._plot._column_bottom_description(text, column1, column2)
                return  ColumnAccesser(self._plot)
            @property
            def row(self):
                class RowAccesser:
                    _plot : TiledPlot
                    def __init__(self, plot:TiledPlot):
                        self._plot = plot
                    def left(self, text:str, row1:int=None, row2:int=None):
                        self._plot._row_left_description(text, row1, row2)
                    def right(self, text:str, row1:int=None, row2:int=None):
                        self._plot._row_right_description(text, row1, row2)
                return RowAccesser(self._plot)
            def graph(self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int,int]=None):
                self._plot._graph_description(text, selected_axes, selected_index, selected_cords)
        return Accesser(self)
    @property
    def graph(self):
        class Accesser:
            _plot : TiledPlot
            def __init__(self, plot:TiledPlot):
                self._plot = plot
            def title(self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int,int]=None):
                self._plot._graph_title(text, selected_axes, selected_index, selected_cords)
            def description(self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int,int]=None):
                self._plot._graph_description(text, selected_axes, selected_index, selected_cords)
            @property
            def label(self):
                class LabelAccesser:
                    _plot : TiledPlot
                    def __init__(self, plot:TiledPlot):
                        self._plot = plot
                    def x(self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int,int]=None):
                        self._plot._graph_axis_x(text, selected_axes, selected_index, selected_cords)
                    def y(self, text:str, selected_axes:Axes=None, selected_index:int=None, selected_cords:Tuple[int,int]=None):
                        self._plot._graph_axis_y(text, selected_axes, selected_index, selected_cords)
                return LabelAccesser(self._plot)
            def width_to_height(self, value:float):
                self._plot.width_to_height(value)
        return Accesser(self)
    @property
    def pad(self):
        class Accesser:
            _plot : TiledPlot
            def __init__(self, plot:TiledPlot):
                self._plot = plot
            def title(self, height:float):
                self._plot._title_pad(height)
            @property
            def description(self):
                class DescriptionAccesser:
                    _plot : TiledPlot
                    def __init__(self, plot:TiledPlot):
                        self._plot = plot
                    def top(self, height:float):
                        self._plot._top_description_pad(height)
                    def bottom(self, height:float):
                        self._plot._bottom_description_pad(height)
                    def left(self, width:float):
                        self._plot._left_description_pad(width)
                    def right(self, width:float):
                        self._plot._right_description_pad(width)
                    @property
                    def column(self):
                        class ColumnAccesser:
                            _plot : TiledPlot
                            def __init__(self, plot:TiledPlot):
                                self._plot = plot
                            def top(self, height:float):
                                self._plot._column_top_description_pad(height)
                            def bottom(self, height:float):
                                self._plot._column_bottom_description_pad(height)
                        return ColumnAccesser(self._plot)
                    @property
                    def row(self):
                        class RowAccesser:
                            _plot : TiledPlot
                            def __init__(self, plot:TiledPlot):
                                self._plot = plot
                            def left(self, width:float):
                                self._plot._row_left_description_pad(width)
                            def right(self, width:float):
                                self._plot._row_right_description_pad(width)
                        return RowAccesser(self._plot)
                    def graph(self, height:float):
                        self._plot._graph_description_pad(height)
                return DescriptionAccesser(self._plot)
            @property
            def graph(self):
                class GraphAccesser:
                    _plot : TiledPlot
                    def __init__(self, plot:TiledPlot):
                        self._plot = plot
                    def vertical(self, height:float):
                        self._plot._graph_vertical_pad(height)
                    def horizontal(self, width:float):
                        self._plot._graph_horizontal_pad(width)
                    def description(self, height:float):
                        self._plot._graph_description_pad(height)
                return GraphAccesser(self._plot)
        return Accesser(self)
    @property
    def axes(self):
        class Accesser:
            _plot : TiledPlot
            def __init__(self, plot:TiledPlot):
                self._plot = plot
            def add(self, position1:Union[int,List[int],Tuple[int,int]]=None, position2:Union[int,List[int],Tuple[int,int]]=None, **kwargs) -> Axes:
                return self._plot._add_axes(position1, position2, **kwargs)
            def select(self, axes:Axes=None, index:int=None, cords:Tuple[int,int]=None):
                return self._plot._select_axes(axes, index, cords)
            @property
            def selected(self) -> Axes:
                return self._plot._selected_axes
            @selected.setter
            def selected(self, parameter:Union[Axes,int,Tuple[int,int]]):
                self._plot._selected_axes = parameter
            def width_to_height(self, value:float):
                self._plot.width_to_height(value)
        return Accesser(self)

def TiledPlotExample1():
    more_text1 = ". Оно, как и любое описание, может быть сколь угодно длинным и обработчик автоматически перенесёт слова так, чтобы оно вмещалось на картинку."
    more_text2 = ". Оно, как и любое описание, может быть сколь угодно длинным и обработчик автоматически перенесёт слова так, чтобы оно вмещалось на картинку. В общем это верно для любой надписи, но это написано здесь только чтобы удлинить текст."

    Plot = TiledPlot()
    Plot.FontLibrary.MultiplyFontSize(0.7)
    # Plot.highlight_all()

    Plot._title('Название массива графиков')
    Plot._top_description('Верхнее описание массива графиков' + more_text2)
    Plot._bottom_description('Нижнее описание массива графиков' + more_text2)
    Plot._left_description('Левое описание массива графиков' + more_text2)
    Plot._right_description('Правое описание массива графиков' + more_text2)

    Plot._add_axes((0, 0))
    Plot._selected_axes.grid(True)
    Plot._graph_title('Название графика')
    Plot._graph_axis_x('Ось X')
    Plot._graph_axis_y('Ось Y')
    Plot._graph_description('Описание Графика' + more_text1)

    Plot._add_axes((1, 0))
    Plot._selected_axes.grid(True)
    Plot._graph_title('Название графика')
    Plot._graph_axis_x('Ось X')
    Plot._graph_axis_y('Ось Y')
    Plot._graph_description('Описание Графика' + more_text1)

    Plot._add_axes((0, 1))
    Plot._selected_axes.grid(True)
    Plot._graph_title('Название графика')
    Plot._graph_axis_x('Ось X')
    Plot._graph_axis_y('Ось Y')
    Plot._graph_description('Описание Графика' + more_text1)

    Plot._add_axes((1, 1))
    Plot._selected_axes.grid(True)
    Plot._graph_title('Название графика')
    Plot._graph_axis_x('Ось X')
    Plot._graph_axis_y('Ось Y')
    Plot._graph_description('Описание Графика' + more_text1)

    Plot._add_axes((2, 0), (3, 1))
    Plot._selected_axes.grid(True)
    Plot._graph_title('Название графика')
    Plot._graph_axis_x('Ось X')
    Plot._graph_axis_y('Ось Y')
    Plot._graph_description('Описание Графика' + more_text2)


    # Plot.add_axes((2, 2))
    # Plot.selected_axes.grid(True)
    # Plot.graph_title('Название графика')
    # Plot.graph_axis_x('Ось X')
    # Plot.graph_axis_y('Ось Y')
    # Plot.graph_description('Описание Графика' + more_text1)
    #
    # Plot.add_axes((2, 3))
    # Plot.selected_axes.grid(True)
    # Plot.graph_title('Название графика')
    # Plot.graph_axis_x('Ось X')
    # Plot.graph_axis_y('Ось Y')
    # Plot.graph_description('Описание Графика' + more_text1)
    #
    # Plot.add_axes((3, 2))
    # Plot.selected_axes.grid(True)
    # Plot.graph_title('Название графика')
    # Plot.graph_axis_x('Ось X')
    # Plot.graph_axis_y('Ось Y')
    # Plot.graph_description('Описание Графика' + more_text1)
    #
    # Plot.add_axes((3, 3))
    # Plot.selected_axes.grid(True)
    # Plot.graph_title('Название графика')
    # Plot.graph_axis_x('Ось X')
    # Plot.graph_axis_y('Ось Y')
    # Plot.graph_description('Описание Графика' + more_text1)
    #
    # Plot.add_axes((0, 2), (1, 3))
    # Plot.selected_axes.grid(True)
    # Plot.graph_title('Название графика')
    # Plot.graph_axis_x('Ось X')
    # Plot.graph_axis_y('Ось Y')
    # Plot.graph_description('Описание Графика' + more_text2)

    Plot._row_left_description('Первая строчка слева' + more_text1, 0)
    Plot._row_left_description('Вторая строчка слева' + more_text1, 1)
    # Plot.row_left_description('Третья и четвёргтая строчка слева' + more_text2, 2, 3)

    Plot._row_right_description('Первая и вторая строчка справа' + more_text2, 0, 1)
    # Plot.row_right_description('Третья строчка справа' + more_text1, 2)
    # Plot.row_right_description('Четвёртая строчка справа' + more_text1, 3)

    Plot._column_top_description('Первая колонка сверху' + more_text1, 0)
    Plot._column_top_description('Вторая колонка сверху' + more_text1, 1)
    Plot._column_top_description('Третья и четвёртая колонка сверху' + more_text2, 2, 3)

    Plot._column_bottom_description('Первая и вторая колонка снизу' + more_text2, 0, 1)
    Plot._column_bottom_description('Третья колонка снизу' + more_text1, 2)
    Plot._column_bottom_description('Четвёртая колонка снизу' + more_text1, 3)

    # Plot.examine_virtual_figure_construction()
    Plot.finalize()
    Plot.show()

def TiledPlotExample2():
    more_text1 = ". Оно, как и любое описание, может быть сколь угодно длинным и обработчик автоматически перенесёт слова так, чтобы оно вмещалось на картинку."
    more_text2 = ". Оно, как и любое описание, может быть сколь угодно длинным и обработчик автоматически перенесёт слова так, чтобы оно вмещалось на картинку. В общем это верно для любой надписи, но это написано здесь только чтобы удлинить текст."

    Plot = TiledPlot()
    Plot.FontLibrary.MultiplyFontSize(0.5)
    Plot.FontLibrary.SynchronizeFont('Comic Sans MS')
    Plot.FontLibrary.SynchronizeColor('black')

    Plot.width_to_height(4/3)

    Plot.title('Название массива графиков')
    Plot.description.top('Верхнее описание массива графиков' + more_text2)
    Plot.description.bottom('Нижнее описание массива графиков' + more_text2)
    Plot.description.left('Левое описание массива графиков' + more_text2)
    Plot.description.right('Правое описание массива графиков' + more_text2)

    Plot.axes.add((0, 0))
    Plot.axes.selected.grid(True)
    Plot.graph.title('Название графика')
    Plot.graph.label.x('Ось X')
    Plot.graph.label.y('Ось Y')
    Plot.graph.description('Описание Графика' + more_text1)

    Plot.axes.add((1, 0))
    Plot.axes.selected.grid(True)
    Plot.graph.title('Название графика')
    Plot.graph.label.x('Ось X')
    Plot.graph.label.y('Ось Y')
    Plot.graph.description('Описание Графика' + more_text1)

    Plot.axes.add((0, 1))
    Plot.axes.selected.grid(True)
    Plot.graph.title('Название графика')
    Plot.graph.label.x('Ось X')
    Plot.graph.label.y('Ось Y')
    Plot.graph.description('Описание Графика' + more_text1)

    Plot.axes.add((1, 1))
    Plot.axes.selected.grid(True)
    Plot.graph.title('Название графика')
    Plot.graph.label.x('Ось X')
    Plot.graph.label.y('Ось Y')
    Plot.graph.description('Описание Графика' + more_text1)

    Plot.axes.add((2, 0), (3, 1))
    Plot.axes.selected.grid(True)
    Plot.graph.title('Название графика')
    Plot.graph.label.x('Ось X')
    Plot.graph.label.y('Ось Y')
    Plot.graph.description('Описание Графика' + more_text2)

    Plot.description.row.left('Первая строчка слева' + more_text1, 0)
    Plot.description.row.left('Вторая строчка слева' + more_text1, 1)

    Plot.description.row.right('Первая и вторая строчка справа' + more_text2, 0, 1)

    Plot.description.column.top('Первая колонка сверху' + more_text1, 0)
    Plot.description.column.top('Вторая колонка сверху' + more_text1, 1)
    Plot.description.column.top('Третья и четвёртая колонка сверху' + more_text2, 2, 3)

    Plot.description.column.bottom('Первая и вторая колонка снизу' + more_text2, 0, 1)
    Plot.description.column.bottom('Третья колонка снизу' + more_text1, 2)
    Plot.description.column.bottom('Четвёртая колонка снизу' + more_text1, 3)

    Plot.show()

if __name__ == '__main__':
    # TiledPlotExample1()
    TiledPlotExample2()