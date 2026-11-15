import argparse, random, glob, os, re

INT_REGEX = "([+-]?\\d+)"

def gather_files(root, sort_key_fn=None):
    def gather_recursive_files(root):
        l = list()
        for name in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(name):
                l.extend(gather_recursive_files(name))
            else:
                l.append(name)
        return l
    files = gather_recursive_files(root)
    if sort_key_fn is None:
        random.shuffle(files)
    else:
        files.sort(key=sort_key_fn, reverse=False)
    return files

def single_match(line, fmt, as_type=None):
    val = re.search(fmt, line).group(1)
    if as_type:
        val = as_type(val)
    return val

def get_theory_computation(blocks, channels):
    return blocks * channels * channels

def read_queries_file(filename):
    with open(filename, "r") as f:
        buf = f.read().strip()
    return [ line.strip() for line in buf.split("\n") ][-1]

def get_network_size(network_name):
    if network_name == "random":
        return network_name
    blocks = single_match(network_name, "-b{}x".format(INT_REGEX), int)
    channels = single_match(network_name, "xc{}-".format(INT_REGEX), int)
    return "b{}xc{}".format(blocks, channels)

def compute_line_computation(line):
    reuslt = line.split()
    played_games = int(reuslt[0])
    network_name = reuslt[1]
    network_queries = int(reuslt[2])
    baseline_b20_c256 = get_theory_computation(20, 256)

    if network_name == "random":
        b20_c256_eval_queries = 0
    else:
        blocks = single_match(network_name, "-b{}x".format(INT_REGEX), int)
        channels = single_match(network_name, "xc{}-".format(INT_REGEX), int)
        # steps = single_match(network_name, "-s{}-".format(INT_REGEX), int)
        # chunks = single_match(network_name, "-c{}-".format(INT_REGEX), int)
        # window = single_match(network_name, "-w{}-".format(INT_REGEX), int)
        b20_c256_eval_queries = (get_theory_computation(blocks, channels) / baseline_b20_c256) * network_queries
    return network_name, played_games, b20_c256_eval_queries

def plot_img(accum_node):
    import matplotlib.pyplot as plt

    BASE_LINEWIDTH = 3.5
    BASE_MARKERSIZE = 12
    AUX_LINEWIDTH = 1.2
    GRID_LINEWIDTH = 0.8

    plot_pairs = list()
    for network_name, accum_games, accum_b20_c256_eval_queries in accum_node:
        if len(plot_pairs) == 0 or plot_pairs[-1]["name"] != network_name:
            last_pair = None if len(plot_pairs) == 0 else plot_pairs[-1]
            plot_pairs.append(
                { "name" : network_name,
                  "games" : [accum_games],
                  "queries" : [accum_b20_c256_eval_queries]}
            )
            if last_pair:
                plot_pairs[-1]["games"].insert(0, last_pair["games"][-1])
                plot_pairs[-1]["queries"].insert(0, last_pair["queries"][-1])
        else:
            plot_pairs[-1]["games"].append(accum_games)
            plot_pairs[-1]["queries"].append(accum_b20_c256_eval_queries)

    for pair in plot_pairs:
        network_name = pair["name"]
        accum_games = pair["games"]
        accum_b20_c256_eval_queries = pair["queries"]
        plt.plot(
            accum_games,
            accum_b20_c256_eval_queries,
            "-", label="{}".format(network_name),
            linewidth=BASE_LINEWIDTH, markersize=BASE_MARKERSIZE
        )
    plt.xlabel("Games", fontsize=24)
    plt.ylabel("b20xc256 Queries", fontsize=24)
    plt.grid(True, which="minor", linestyle="--", linewidth=GRID_LINEWIDTH, alpha=0.5)
    plt.grid(True, which="major", linestyle="--", linewidth=GRID_LINEWIDTH, alpha=0.9)
    plt.legend(fontsize=24)
    plt.show()

def main(args):
    sort_fn = os.path.getmtime
    files = gather_files(args.path, sort_fn)

    queries_list = list()
    for filename in files:
        queries_list.append(read_queries_file(filename))

    accum_b20_c256_eval_queries = 0
    accum_games = 0
    accum_node = list()
    for line in queries_list[:]:
        network_name, played_games, b20_c256_eval_queries = compute_line_computation(line)
        accum_games += played_games
        accum_b20_c256_eval_queries += b20_c256_eval_queries

        accum_node.append((get_network_size(network_name), accum_games, accum_b20_c256_eval_queries))
        if not args.num_games is None and \
               accum_games >= args.num_games:
            break
    print("Current Network size is \"{}\".".format(get_network_size(network_name)))
    print("Accumulate {} Games".format(accum_games))
    print("Accumulate {:.0f} ({:.4e}) b20xc256 queries".format(accum_b20_c256_eval_queries, accum_b20_c256_eval_queries))

    if args.plot:
        plot_img(accum_node)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", metavar="<string>",
                        help="The root path of queries files.", type=str)
    parser.add_argument("-n", "--num-games", metavar="<int>",
                        help="Select a specific number of games.", type=int)
    parser.add_argument("--plot", help="", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
