'''
    This code is discarded.
'''

import argparse

from network import Network, FullyConnect, Convolve, ConvBlock
from config import gather_config

import torch

def swa(cfg, input_weights, outs_name, recompute):
    avarage_list = []
    out_net = Network(cfg)

    inputs_size = 0
    for filename in input_weights:
        avarage_list.append(Network(cfg));
        avarage_list[inputs_size].load_pt(filename)
        inputs_size += 1

    lsize = len(out_net.layer_collector)
    for i in range(lsize):
        out_layer = out_net.layer_collector[i]

        # set zeros
        if isinstance(out_layer, FullyConnect):
            out_layer.linear.weight.data *= 0
            out_layer.linear.bias.data *= 0
        elif isinstance(out_layer, Convolve):
            out_layer.conv.weight.data *= 0
            out_layer.conv.bias.data *= 0
        elif isinstance(out_layer, ConvBlock):
            out_layer.conv.weight.data *= 0
            out_layer.bn.running_mean.data *= 0
            out_layer.bn.running_var.data *= 0
            if out_layer.bn.gamma is not None:
                out_layer.bn.gamma.data *= 0
            out_layer.bn.beta.data *= 0

        # average
        for j in range(inputs_size):
            input_layer = avarage_list[j].layer_collector[i]

            if isinstance(out_layer, FullyConnect):
                out_layer.linear.weight.data += (input_layer.linear.weight.data / inputs_size)
                out_layer.linear.bias.data += (input_layer.linear.bias.data / inputs_size)
            elif isinstance(out_layer, Convolve):
                out_layer.conv.weight.data += (input_layer.conv.weight.data / inputs_size)
                out_layer.conv.bias.data += (input_layer.conv.bias.data / inputs_size)
            elif isinstance(out_layer, ConvBlock):
                out_layer.conv.weight.data += (input_layer.conv.weight.data / inputs_size)
                out_layer.bn.running_mean.data += (input_layer.bn.running_mean.data / inputs_size)
                out_layer.bn.running_var.data += (input_layer.bn.running_var.data / inputs_size)
                if out_layer.bn.gamma is not None:
                    out_layer.bn.gamma.data += (input_layer.bn.gamma.data / inputs_size)
                out_layer.bn.beta.data += (input_layer.bn.beta.data / inputs_size)

    out_net.transfer_to_bin(args.network + ".bin.txt")
    # out_net.transfer_to_text(args.network + ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)
    parser.add_argument("-n", "--network", metavar="<string>",
                        help="The output network name prefix.", type=str)
    parser.add_argument("-i", "--inputs", nargs="+",
                        help="The input networks list.")
    args = parser.parse_args()


    if args.json == None:
        print("Please give the setting json file.")
    elif args.network == None:
        print("Plase give the network name.")
    elif args.inputs == None:
        print("")
    else:
        swa(gather_config(args.json), args.inputs, args.network, False)
