import argparse

from nnprocess import NNProcess
from config import gather_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="The json file name", type=str)
    parser.add_argument("-n", "--network", help="", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)
    net = NNProcess(cfg)
    net.load_ckpt(args.network + ".ckpt")
    net.transfer2text(args.network + ".txt")
