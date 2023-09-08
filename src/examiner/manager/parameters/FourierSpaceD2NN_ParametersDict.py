from src.utilities.Formaters import Format
from functools import partial

ParametersDict = {
    "DiffractionLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 0.1,
        "ValueStep": 0.0001,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    },
    "FocusLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 0.05,
        "ValueStep": 5e-05,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    },
    "LensBorderLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 0.0,
        "ValueStep": 0.0,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    },
    "MaskBorderLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 0.0,
        "ValueStep": 0.0,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    },
    "MaskReflection": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 7.5,
        "ValueStep": 0.0075,
        "Format": partial(Format.Scientific, unit='', precision=2)
    },
    "PixelsCount": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 105,
        "ValueStep": 1,
        "Format": lambda x: str(int(x))
    },
    "PlaneLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 0.005,
        "ValueStep": 5e-06,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    },
    "SpaceReflection": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 5.0,
        "ValueStep": 0.005,
        "Format": partial(Format.Scientific, unit='', precision=2)
    },
    "UpScaling": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 100,
        "ValueStep": 1,
        "Format": lambda x: str(int(x))
    },
    "WaveLength": {
        "DataType": "int",
        "MinValue": 0,
        "MaxValue": 3.000000106112566e-06,
        "ValueStep": 3.0000001061125657e-09,
        "Format": partial(Format.Engineering, unit='m', precision=1)
    }
}