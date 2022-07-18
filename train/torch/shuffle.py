import numpy as np
import argparse, glob, os, random, time, io, gzip
import multiprocessing as mp

DATA_LINES = 45

def gather_data(filename):
    data_list = list()
    stream = None

    if filename.find(".gz") >= 0:
        with gzip.open(filename, 'rt') as f:
            stream = io.StringIO(f.read())
    else:
        with open(filename, 'r') as f:
            stream = io.StringIO(f.read())

    running = True
    while running:
        data = str()
        for cnt in range(DATA_LINES):
            line = stream.readline()
            if len(line) == 0:
                running = False
                assert(cnt == 0)
                break
            data += "{}".format(line)
        if running:
            data_list.append(data)
    return data_list


def write_data(filename, data_list):
    random.shuffle(data_list)

    if filename.find(".gz") >= 0:
        with gzip.open(filename, 'wt') as f:
            for data in data_list:
                f.write(data)
    else:
        with open(filename, 'w') as f:
            for data in data_list:
                f.write(data)

def get_new_filename(new_dir, filename):
    path_list = os.path.split(filename)
    postfix = path_list[-1]

    out_name = os.path.join(new_dir, postfix)

    if out_name.find(".gz") < 0:
        out_name += ".gz"

    return out_name

def do_shuffle(file_que, output_dir):
    while not file_que.empty():
        filename = file_que.get()
        print("Parse the {}...".format(filename))

        data_list = gather_data(filename)

        print("Data number {}".format(len(data_list)))

        random.shuffle(data_list)
        new_filename = get_new_filename(output_dir, filename)
        write_data(new_filename, data_list)

        print("Parsed done")

def gather_recursive_files(root):
    l = list()
    for name in glob.glob(os.path.join(root, '*')):
        if os.path.isdir(name):
            l.extend(gather_recursive_files(name))
        else:
            l.append(name)
    return l

def shuffle_process(num_cores, target_dir, output_dir):
    file_list = gather_recursive_files(target_dir)

    file_que = mp.Queue(len(file_list))
    for i in file_list:
        file_que.put(i)

    print("Use {} processes".format(num_cores))

    processes = []
    for _ in range(num_cores):
        p = mp.Process(target=do_shuffle, args=(file_que, output_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def valid(args):
    if args.input_dir is None or args.output_dir is None:
       return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-cores", metavar="<int>",
                        default=1,
                        help="", type=int)
    parser.add_argument("-i", "--input-dir", metavar="<string>",
                        help="", type=str)
    parser.add_argument("-o", "--output-dir", metavar="<string>",
                        help="", type=str)
    args = parser.parse_args()

    if valid(args):
        shuffle_process(args.num_cores, args.input_dir, args.output_dir)
       
