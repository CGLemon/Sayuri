import torch
import torch.nn as nn
import packaging
import packaging.version
import math

DEFAULT_ACTIVATION = "relu"

def activation_func(activation, inplace=False):
    if activation == "identity":
        return nn.Identity()
    if activation == "relu":
        return nn.ReLU(inplace=inplace)
    if activation == "elu":
        return nn.ELU(inplace=inplace)
    if activation == "selu":
        return nn.SELU(inplace=inplace)
    if activation == "gelu":
        return nn.GELU(inplace=inplace)
    if activation == "mish":
        return nn.Mish(inplace=inplace)
    if activation == "swish":
        return nn.SiLU(inplace=inplace)
    if activation == "hardswish":
        if packaging.version.parse(torch.__version__) > packaging.version.parse("1.6.0"):
            return nn.Hardswish(inplace=inplace)
        else:
            return nn.Hardswish()
    raise Exception("The {} is invalid activation function.".format(activation))

def compute_gain(activation):
    if activation == "identity":
        gain = 1.0
    elif activation == "relu":
        gain = math.sqrt(2.0)
    elif activation == "elu":
        gain = math.sqrt(1.55052)
    elif activation == "selu":
        gain = 3/4
    elif activation == "gelu":
        gain = math.sqrt(2.351718)
    elif activation == "mish":
        gain = math.sqrt(2.210277)
    elif activation == "swish":
        gain = math.sqrt(2.0) # TODO:
    elif activation == "hardswish":
        gain = math.sqrt(2.0)
    else:
        raise Exception("The {} is invalid activation function for computing gain.".format(activation))
    return gain
