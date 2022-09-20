import argparse

from network import Network
from config import gather_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)

    parser.add_argument("-n", "--network", metavar="<string>",
                        help="The network name which we want to transfer.", type=str)

    parser.add_argument("-b", "--binary", default=False,
                        help="Save the weights as binary.", action="store_true")

    args = parser.parse_args()

    if args.json == None:
        print("Please give the setting json file.")
    elif args.network == None:
        print("Plase give the network name.")
    else:
        cfg = gather_config(args.json)

        net = Network(cfg)
        net.load_pt(args.network + ".pt")
        if args.binary:
            net.transfer_to_bin(args.network + ".bin.txt")
        else:
            net.transfer_to_text(args.network + ".txt")

