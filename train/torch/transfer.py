import argparse

from nnprocess import NNProcess
from config import gather_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The json file name.", type=str)
    parser.add_argument("-n", "--network", metavar="<string>",
                        help="The network which we want to transfer.", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)
    net = NNProcess(cfg)
    net.load_pt(args.network + ".pt")
    net.transfer2text(args.network + ".txt")
