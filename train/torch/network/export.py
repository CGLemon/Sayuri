import torch
import struct

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

def bnorm_to_text(channels):
    return "BatchNorm {C}\n".format(C=channels)

def rmsnorm_to_text(channels):
    return "RMSNorm {C}\n".format(C=channels)

def rope_to_text(head_dim, num_heads):
    return "RoPE {HD} {NH}\n".format(HD=head_dim, NH=num_heads)

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
