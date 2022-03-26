import argparse

from train import TrainingPipe
from config import gather_config
from loader import Loader

def main(args, cfg):
    pipe = TrainingPipe(cfg)

    if args.input != None:
        pipe.load_pt("{}-s{}.pt".format(args.input, args.steps))

    if args.dummy != True:
        pipe.fit_and_store(args.output, args.steps, args.log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dummy", default=False,
                        help="Genrate a dummy network.", action="store_true")

    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)

    parser.add_argument("-o", "--output", metavar="<string>",
                        help="The ouput weights prefix path/name.", type=str)

    parser.add_argument("-s", "--steps", metavar="<string>", default=0,
                        help="The initialization steps.", type=int)

    parser.add_argument("-i", "--input", metavar="<string>",
                        help="The intput weights prefix path/name.", type=str)

    parser.add_argument("-l", "--log-file", metavar="<string>", default="log.txt",
                        help="The log file name.", type=str)

    args = parser.parse_args()
    cfg = gather_config(args.json)

    if args.json == None:
        print("Please give the setting json file.")
    elif args.output == None:
        print("Plase give the ouput weights prefix path/name.")
    else:
        main(args, cfg)
