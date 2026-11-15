import argparse, re, os
import matplotlib.pyplot as plt

FLOAT_REGEX = "([+-]?(\\d+([.]\\d*)?([eE][+-]?\\d+)?|[.]\\d+([eE][+-]?\\d+)?))" # "([+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?))"
INT_REGEX = "([+-]?\\d+)" # "([+-]?\d+)"

X_AXIS_TYPE_LIST = ["steps", "samples"]
Y_AXIS_TYPE_LIST = ["all", "policy", "optimistic", "ownership", "wdl", "q", "score"]
CUT_FIRST_DEFAULT = 1000

class LineDataParser:
    def __init__(self, line):
        self._match(line)

    def _single_match(self, fmt, as_type=None):
        val = re.search(fmt, self.line).group(1)
        if as_type:
            val = as_type(val)
        return val

    def _match(self, line):
        self.line = line
        self.steps      = self._single_match("steps: {}".format(INT_REGEX), int)
        self.samples    = self._single_match("samples: {}".format(INT_REGEX), int)
        self.speed      = self._single_match("speed: {}".format(FLOAT_REGEX), float)
        self.lr         = self._single_match("learning rate: {}".format(FLOAT_REGEX), float)
        self.batch_size = self._single_match("batch size: {}".format(INT_REGEX), int)

        self.count_dict = dict()
        self.count_dict["steps"] = self.steps
        self.count_dict["samples"] = self.samples

        self.loss_dict = dict()
        self.loss_dict["all"]        = self._single_match("-> all loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["policy"]     = self._single_match(", prob loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["optimistic"] = self._single_match(", optimistic loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["ownership"]  = self._single_match(", ownership loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["wdl"]        = self._single_match(", wdl loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["q"]          = self._single_match(", Q values loss: {}".format(FLOAT_REGEX), float)
        self.loss_dict["score"]      = self._single_match(", scores loss: {}".format(FLOAT_REGEX), float)

class GraphInfo:
    def __init__(self):
        self._data_list = list()

    def reset(self):
        self._data_list.clear()

    def append(self, line):
        self._data_list.append(LineDataParser(line))

    def get_xy_plot(self, *args, **kwargs):
        x_axis = kwargs.get("x_axis", X_AXIS_TYPE_LIST[0])
        y_axis = kwargs.get("y_axis", Y_AXIS_TYPE_LIST[0])
        cut_first = kwargs.get("cut_first", CUT_FIRST_DEFAULT)

        xx, yy = list(), list()
        for line_data in self._data_list:
            if line_data.steps > cut_first:
                xx.append(line_data.count_dict[x_axis])
                yy.append(line_data.loss_dict[y_axis])
        return xx, yy

    def get_avg_speed(self):
        speed_list = list()
        for line_data in self._data_list:
            speed_list.append(line_data.speed)
        return sum(speed_list)/len(speed_list)

def get_graph_info(lines):
    graph_info = GraphInfo()
    for line in lines:
        graph_info.append(line)
    return graph_info

def plot_process(args):
    graph_info_list = list()
    existing_names = set()
    for filename in args.filename:
        lines = readfile(filename)
        graph_info_list.append(
            (get_graph_info(lines), os.path.split(filename)[-1])
        )

    for graph_info, file_tail in graph_info_list:
        xx, yy = graph_info.get_xy_plot(
            x_axis = args.x_axis,
            y_axis = args.y_axis,
            cut_first = args.cut_first
        )
        if len(xx) <= 1:
            print("Not enough lines in the \"{}\", skipping it...".format(file_tail))
            continue
        while file_tail in existing_names:
            fixed_file_tail = file_tail + "-{}".format(len(existing_names))
            print("The file, \"{}\", is already in the list, replacing \"{}\" with \"{}\"".format(
                      file_tail, file_tail, fixed_file_tail))
            file_tail = fixed_file_tail
        existing_names.add(file_tail)
        plt.plot(xx, yy, linestyle="-", linewidth=1, label="{}".format(file_tail))
    plt.ylabel("{} loss".format(args.y_axis))
    plt.xlabel(args.x_axis)
    if args.log_scale:
        plt.xscale("log")
    plt.grid(linestyle="--", linewidth=1, alpha=0.25)
    plt.legend()
    plt.show()

def readfile(filename):
    lines = list()
    with open(filename, "r") as f:
        line = f.readline().strip()
        while len(line) != 0:
            lines.append(line)
            line = f.readline().strip()
    return lines
    
def check(args):
    success = True
    if args.filename == None:
        success = False
        print("Please add argument --filename <str>, -f <str> to load the files.")
    if not args.x_axis in X_AXIS_TYPE_LIST:
        success = False
        print("X axis should be one of {}.".format("/".join([x for x in X_AXIS_TYPE_LIST])))
    if not args.y_axis in Y_AXIS_TYPE_LIST:
        success = False
        print("Loss type should be one of {}.".format("/".join([y for y in Y_AXIS_TYPE_LIST])))
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", nargs="+", metavar="<str>",
                        help="The input file name(s).", type=str)
    parser.add_argument("--cut-first", metavar="<int>", default=CUT_FIRST_DEFAULT,
                        help="Cut the first X steps loss information, default is {}.".format(CUT_FIRST_DEFAULT), type=int)
    parser.add_argument("--log-scale", default=False,
                        help="The x-axis will be log scale.", action="store_true")
    parser.add_argument("--y-axis", "--loss-type", metavar="<str>", default="all",
                        help="Loss type in {}.".format("/".join([y for y in Y_AXIS_TYPE_LIST])))
    parser.add_argument("--x-axis", metavar="<str>", default="steps",
                        help="X axis type in {}.".format("/".join([x for x in X_AXIS_TYPE_LIST])))
    args = parser.parse_args()
    if check(args):
        plot_process(args)
