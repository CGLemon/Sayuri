import argparse

from train import TrainingPipe
from config import Config

def main(cfg):
    pipe = TrainingPipe(cfg)
    pipe.fit_and_store()

def overwrite(cfg, args):
    if not args.workspace is None:
        cfg.store_path = args.workspace
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)
    parser.add_argument("-w", "--workspace", metavar="<string>",
                        help="Overwrite the store path.", type=str)
    args = parser.parse_args()

    if args.json == None:
        print("Please give the setting json file.")
    else:
        cfg = Config(args.json)
        cfg = overwrite(cfg, args)
        main(cfg)
