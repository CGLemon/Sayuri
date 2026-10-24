#!/usr/bin/env python3
# Generate a randomly perturbed network from a setting json, then save it as
#   - <out>.pt  : the StatusDict checkpoint for pysayuri (-c <out>.pt)
#   - <out>.txt : the text weights for the C++ engine (-w <out>.txt)
import argparse
import json
import os
import sys

import torch

TORCH_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TORCH_DIR)

from config import Config
from network import Network
from status_dict import StatusDict

# pysayuri always builds the network with this board size.
PYSAYURI_BOARD_SIZE = 19

def load_setting(filename):
    with open(filename, "r") as f:
        setting = json.load(f)

    nn_setting = setting.get("NeuralNetwork", None)
    if nn_setting is None:
        raise Exception("the setting json should have the \"NeuralNetwork\" section")
    if nn_setting.get("MaxBoardSize", 19) != PYSAYURI_BOARD_SIZE:
        raise Exception("the MaxBoardSize should be {} because pysayuri always uses it".format(
                            PYSAYURI_BOARD_SIZE))

    # Only the network is used, but the Config still requires these keys.
    train_setting = setting.setdefault("Train", dict())
    train_setting.setdefault("UseGPU", False)
    train_setting.setdefault("TrainDirectory", "train-dir")
    train_setting.setdefault("StorePath", "workspace")
    return setting

def perturb_network(net, noise, weight_scale):
    # The default init keeps the normalization gammas = 1 and the biases = 0,
    # which would hide the loading bugs. Perturb every parameter and the BN
    # statistics.
    with torch.no_grad():
        for _, p in net.named_parameters():
            if p.dim() > 1:
                p.mul_(weight_scale)
            p.add_(torch.randn_like(p) * noise)
        for name, b in net.named_buffers():
            if name.endswith("running_mean"):
                b.normal_(0.0, 0.1)
            elif name.endswith("running_var"):
                b.uniform_(0.5, 1.5)

def main(args):
    setting = load_setting(args.json)

    torch.manual_seed(args.seed)
    cfg = Config(json.dumps(setting, indent=4), is_file=False)
    net = Network(cfg)
    net.eval()
    perturb_network(net, args.noise, args.weight_scale)

    status_dict = StatusDict()
    status_dict.fancy_set(StatusDict.JSON_KEY, cfg.json_str)
    status_dict.set_module(StatusDict.MODEL_KEY, net)
    status_dict.set_module(StatusDict.SWA_KEY, net)
    status_dict.fancy_set(StatusDict.STEPS_KEY, 0)
    status_dict.save(args.output + ".pt")
    net.transfer_to_text(args.output + ".txt")

    print(net.get_simple_info())
    print("Saved: {out}.pt, {out}.txt".format(out=args.output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", metavar="<string>", required=True,
                        help="The setting json file of the network.", type=str)
    parser.add_argument("-o", "--output", metavar="<string>", required=True,
                        help="The output path prefix.", type=str)
    parser.add_argument("--seed", metavar="<integer>", default=1234,
                        help="The random seed.", type=int)
    parser.add_argument("--noise", metavar="<float>", default=0.1,
                        help="The stddev of the noise added to every parameter.", type=float)
    parser.add_argument("--weight-scale", metavar="<float>", default=0.5,
                        help="Scale the weight matrices to keep the outputs unsaturated.", type=float)
    main(parser.parse_args())
