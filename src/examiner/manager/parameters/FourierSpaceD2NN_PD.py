from src.utilities.Formaters import Format

ParametersDict = {
    "accuracy": {
        "bits16": {},
        "bits32": {},
        "bits64": {}
    },
    "amplification": {
        "disable": {},
        "enable": {}
    },
    "border": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        }
    },
    "detectors": {
        "__call__": {
            "amount": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "detectors_type": {
        "polar": {
            "borders": {
                "type": "float",
                "default": 0.05,
                "max": 0.1,
                "min": 0.025
            },
            "space": {
                "type": "float",
                "default": 0.2,
                "max": 0.4,
                "min": 0.1
            },
            "power": {
                "type": "float",
                "default": 0.5,
                "max": 1.0,
                "min": 0.25
            }
        },
        "square": {
            "borders": {
                "type": "float",
                "default": 0.05,
                "max": 0.1,
                "min": 0.025
            },
            "space": {
                "type": "float",
                "default": 0.2,
                "max": 0.4,
                "min": 0.1
            }
        }
    },
    "focus": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        }
    },
    "focus_border": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        }
    },
    "layers": {
        "__call__": {
            "amount": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "mask_reflection": {
        "__call__": {
            "reflection": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        },
        "range": {
            "reflection0": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "reflection1": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "N": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "normalization": {
        "integral": {},
        "maximum": {},
        "softmax": {}
    },
    "parameters_normalization": {
        "sigmoid": {},
        "sinus": {
            "period": {
                "type": "float",
                "default": 100.0,
                "max": 200.0,
                "min": 50.0
            }
        }
    },
    "pixels": {
        "__call__": {
            "amount": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "plane_length": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        }
    },
    "space": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        }
    },
    "space_reflection": {
        "__call__": {
            "reflection": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        },
        "range": {
            "reflection0": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "reflection1": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "N": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "up_scaling": {
        "__call__": {
            "amount": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    },
    "wavelength": {
        "__call__": {
            "length": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            }
        },
        "range": {
            "length0": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "length1": {
                "type": "float",
                "default": 1.0,
                "max": 2.0,
                "min": 0.5
            },
            "N": {
                "type": "int",
                "default": 1,
                "max": 2,
                "min": 0
            }
        }
    }
}