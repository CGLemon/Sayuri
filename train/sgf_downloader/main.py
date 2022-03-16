import argparse
from request_process import RequestProcess

def check(args):
    success = True
    if args.games == None:
        print('Please add: --games <int>')
        success = False
    if args.games == None:
        print('Please add: --directory <string>')
        success = False
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--games', metavar='<int>',
                        help='Number of downloading games.', type=int)
    parser.add_argument('-d', '--directory', metavar='<string>',
                        help='The saving directory.', type=str)
    parser.add_argument('-u', '--url', metavar='<string>',
                        help='Try to download the sgfs file from this url.', type=str)

    args = parser.parse_args()

    if check(args):
        rp = RequestProcess(args.directory, args.games, args.url)
