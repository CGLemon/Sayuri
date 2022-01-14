import argparse

from train import *
from config import *

def main(args, cfg):
    pipe = TrainingPipe(cfg)

    if args.input != None:
        pipe.load_pt(args.input + ".pt")

    if args.dummy != True:
        pipe.prepare_data()
        pipe.fit()

    if args.output != None:
        pipe.save_pt(args.output + ".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dummy", help="Genrate a dummy network.", action="store_true")
    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The json file name.", type=str)
    parser.add_argument("-o", "--output", metavar="<string>",
                        help="The ouput weights prefix name.", type=str)
    parser.add_argument("-i", "--input", metavar="<string>",
                        help="The intput weights prefix name.", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)

    if args.json != None:
        main(args, cfg)
