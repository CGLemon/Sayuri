import argparse

from train import *
from config import *

def cover(args, cfg):
    if args.verbose >= 1:
        cfg.misc_verbose = True
    if args.verbose >= 2:
        cfg.debug_verbose = True

def main(args, cfg):
    pipe = TrainingPipe(cfg)

    if args.input != None:
        pipe.load_pt(args.input + ".pt")

    if args.dummy != True:
        pipe.fit()

    if args.output != None:
        pipe.save_pt(args.output + ".pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dummy", help="Genrating the dummy network.", action="store_true")
    parser.add_argument("-j", "--json", help="The json file name", type=str)
    parser.add_argument("-v", "--verbose", help="", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument("-o", "--output", help="the ouput weights prefix name ", type=str)
    parser.add_argument("-i", "--input", help="the intput weights prefix name ", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)
    cover(args, cfg)
    
    if cfg.misc_verbose:
        dump_dependent_version()

    if args.json != None:
        main(args, cfg)
