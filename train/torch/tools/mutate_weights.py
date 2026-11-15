#!/usr/bin/env python3
# Mutate one parameter of a .pt checkpoint and export it as the .txt weights.
# It is the negative control of compare_raw_nn.py: the mutated .txt must not
# match the original .pt.
# Run it from train/torch as a module: python3 -m tools.mutate_weights
import argparse

import torch

from common.config import Config
from network import Network
from common.status_dict import StatusDict

def load_network(pt):
    status_dict = StatusDict()
    status_dict.load(pt)
    cfg = Config(status_dict.fancy_get(StatusDict.JSON_KEY), is_file=False)
    net = Network(cfg)
    status_dict.load_module(StatusDict.MODEL_KEY, net)
    net.eval()
    return net

def main(args):
    net = load_network(args.pt)
    params = dict(net.named_parameters())

    if args.list:
        for name, p in params.items():
            print("{:60s} {}".format(name, tuple(p.shape)))
        return

    if args.param not in params:
        raise Exception("{} is not a parameter, see --list".format(args.param))
    if args.output is None:
        raise Exception("should provide the output path")
    if args.scale is None and args.flip_dim is None:
        raise Exception("should provide --scale or --flip-dim")

    p = params[args.param]
    with torch.no_grad():
        if args.scale is not None:
            p.mul_(args.scale)
        if args.flip_dim is not None:
            p.copy_(p.flip(args.flip_dim))
    net.transfer_to_text(args.output)
    print("Mutated {} {}, saved: {}".format(args.param, tuple(p.shape), args.output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", metavar="<string>", required=True,
                        help="The .pt weights from gen_random_weights.py.", type=str)
    parser.add_argument("--list", default=False,
                        help="List the parameter names and shapes.", action="store_true")
    parser.add_argument("--param", metavar="<string>",
                        help="The parameter name to mutate.", type=str)
    parser.add_argument("--scale", metavar="<float>",
                        help="Multiply the parameter by this value.", type=float)
    parser.add_argument("--flip-dim", metavar="<integer>",
                        help="Reverse the parameter along this dimension.", type=int)
    parser.add_argument("-o", "--output", metavar="<string>",
                        help="The mutated .txt weights.", type=str)
    main(parser.parse_args())
