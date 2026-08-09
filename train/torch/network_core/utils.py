import torch
import torch.nn as nn
import torch.nn.functional as F
import packaging
import packaging.version
import math
import struct

CRAZY_NEGATIVE_VALUE = -5000.0
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

def dwconv_to_text(in_channels, out_channels, kernel_size):
    return "DepthwiseConvolution {iC} {oC} {KS}\n".format(
               iC=in_channels,
               oC=out_channels,
               KS=kernel_size)

def conv_to_text(in_channels, out_channels, kernel_size):
    return "Convolution {iC} {oC} {KS}\n".format(
               iC=in_channels,
               oC=out_channels,
               KS=kernel_size)

def fullyconnect_to_text(in_size, out_size):
    return "FullyConnect {iS} {oS}\n".format(iS=in_size, oS=out_size)

def bn_to_text(channels):
    return "BatchNorm {C}\n".format(C=channels)

def float_to_bin(num, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.pack(fmt, num)

def bin_to_float(bnum, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.unpack(fmt, bnum)[0]

def str_to_bin(st):
    return bytearray(st, "utf-8")

def ffffffff_nan():
    return b'\xff\xff\xff\xff'

def tensor_to_list(t: torch.Tensor):
    return t.detach().cpu().numpy().ravel()

def tensor_to_bin(t: torch.Tensor):
    return b''.join([float_to_bin(w, False) for w in tensor_to_list(t)]) + ffffffff_nan()

def tensor_to_text(t: torch.Tensor, use_bin):
    if use_bin:
        return tensor_to_bin(t)
    return " ".join([str(w) for w in tensor_to_list(t)]) + "\n"

def write_network_to_file(network, filename, use_bin):
    wrap = str_to_bin if use_bin else (lambda s: s)
    mode = "wb" if use_bin else "w"
    float_type = "float32bin" if use_bin else "float32"

    def write_stack(f, stack):
        f.write(wrap("get stack\n"))
        for s in stack:
            f.write(wrap("{}\n".format(s)))
        f.write(wrap("end stack\n"))

    def write_struct(f, layers_collector):
        f.write(wrap("get struct\n"))
        for layer in layers_collector:
            f.write(wrap(layer.shape_to_text()))
        f.write(wrap("end struct\n"))

    def write_params(f, layers_collector):
        f.write(wrap("get parameters\n"))
        for layer in layers_collector:
            f.write(layer.tensors_to_text(use_bin))
        f.write(wrap("end parameters\n"))

    with open(filename, mode) as f:
        f.write(wrap("get main\n"))

        f.write(wrap("get info\n"))
        f.write(wrap("NNType {}\n".format(network.nntype)))
        f.write(wrap("Version {}\n".format(network.version)))
        f.write(wrap("FloatType {}\n".format(float_type)))
        f.write(wrap("InputChannels {}\n".format(network.input_channels)))
        f.write(wrap("ResidualChannels {}\n".format(network.residual_channels)))
        f.write(wrap("ResidualBlocks {}\n".format(len(network.stack))))
        f.write(wrap("PolicyHeadChannels {}\n".format(network.policy_head_channels)))
        f.write(wrap("ValueHeadChannels {}\n".format(network.value_head_channels)))
        f.write(wrap("ValueMisc {}\n".format(network.value_misc)))
        f.write(wrap("PolicyHeadType {}\n".format(network.policy_head_type["Type"])))
        f.write(wrap("ActivationFunction {}\n".format(network.activation)))
        f.write(wrap("end info\n"))

        write_stack(f, network.get_stack_name(network.stack))
        write_struct(f, network.layers_collector)
        write_params(f, network.layers_collector)

        f.write(wrap("end main"))

def network_info_to_text(network):
    info = str()
    info += "NN Type: {type}\n".format(type=network.nntype)
    info += "NN size [x,y]: [{xsize}, {ysize}]\n".format(xsize=network.xsize, ysize=network.ysize)
    info += "Input channels: {channels}\n".format(channels=network.input_channels)
    info += "Residual channels: {channels}\n".format(channels=network.residual_channels)
    info += "Residual tower: size -> {s} [\n".format(s=len(network.stack))
    for s in network.get_stack_name(network.stack):
        info += "  {}\n".format(s)
    info += "]\n"
    info += "Policy head channels: {polhead}\n".format(polhead=network.policy_head_channels)
    info += "Value head channels: {valhead}\n".format(valhead=network.value_head_channels)
    info += "Value misc size: {valuemisc}\n".format(valuemisc=network.value_misc)
    info += "Policy Head Type: {polheadtype}\n".format(polheadtype=network.policy_head_type["Type"])
    info += "Default activation: {act}\n".format(act=network.activation)
    return info

def make_soft_porb(prob, policy_mask, eps=1e-7, t=4):
    soft_prob = (prob + eps) * policy_mask
    soft_prob = torch.pow(soft_prob, 1/t)
    soft_prob /= torch.sum(soft_prob, dim=1, keepdim=True)
    return soft_prob

def cross_entropy(pred, target, weight=1.):
    loss_sum = -torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def huber_loss(x, y, delta, weight=1.):
    absdiff = torch.abs(x - y)
    loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
    loss_sum = torch.sum(loss, dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def mse_loss(pred, target, weight=1.):
    loss_sum = torch.mean(torch.square(pred - target), dim=1)
    return torch.mean(weight * loss_sum, dim=0)

def mse_loss_spat(pred, target, mask_sum_hw, weight=1.):
    loss_sum = torch.sum(torch.square(pred - target), dim=1) / mask_sum_hw
    return torch.mean(weight * loss_sum, dim=0)

def square_huber_loss(pred, x, y, delta, eps, weight=1.):
    sqerror = torch.square(x - y) + eps
    loss = huber_loss(pred, sqerror, delta=delta, weight=weight)
    return loss
