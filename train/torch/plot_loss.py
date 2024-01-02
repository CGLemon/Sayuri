import argparse, re
import matplotlib.pyplot as plt

FRAMESIZE = 12
FLOAT_REGEX = "([+-]?(\d+([.]\d*)?([eE][+-]?\d+)?|[.]\d+([eE][+-]?\d+)?))"
INT_REGEX = "([+-]?\d+)"

class FrameData:
    def __init__(self):
        self.steps = 0
        self.speed = 0
        self.lr = 0
        self.batch_size = 0
        self.loss = 0

def matchframe(frame, loss_type):
    data = FrameData()
    data.steps = int(re.search("steps: {}".format(INT_REGEX), frame[0]).group(1))
    data.speed = float(re.search("speed: {}".format(FLOAT_REGEX), frame[0]).group(1))
    data.lr = float(re.search("learning rate: {}".format(FLOAT_REGEX), frame[0]).group(1))
    data.batch_size = int(re.search("batch size: {}".format(INT_REGEX), frame[0]).group(1))
    if loss_type == "all":
        data.loss = float(re.search("loss: {}".format(FLOAT_REGEX), frame[1]).group(1))
    elif loss_type == "policy":
        data.loss = float(re.search("loss: {}".format(FLOAT_REGEX), frame[2]).group(1))
    elif loss_type == "wdl":
        data.loss = float(re.search("loss: {}".format(FLOAT_REGEX), frame[8]).group(1))
    else:
        raise Exception("unknow loss type")
    return data

def plot_process(args):
    xxs = list() # steps
    yys = list() # loss value
    verticals_list = list() # lr changed

    for filename in args.filename:
        xx, yy, verticals = gather_data(filename, args.loss_type)
        xxs.append(xx)
        yys.append(yy)
        verticals_list.append(verticals)

    plot_multi_loss(xxs, yys, verticals_list, args.loss_type)
    if args.individual:
        # Print each loss graph.
        for xx, yy, verticals, val in zip(xxs, yys, verticals_list, range(len(xxs))):
            plot_individual_loss(xx, yy, verticals, val, args.loss_type)

def gather_data(filename, loss_type):
    data_list = readfile(filename)
    xx = list()
    yy = list()
    lr_schedule = list()
    verticals = list()

    for i in data_list:
        data = matchframe(i, loss_type)
        x, y, lr = data.steps, data.loss, data.lr
        if len(lr_schedule) == 0:
            lr_schedule.append(lr)
        elif lr_schedule[len(lr_schedule)-1] != lr:
            lr_schedule.append(lr)
            verticals.append(x)
        xx.append(x)
        yy.append(y)
    return xx, yy, verticals

def plot_multi_loss(xxs, yys, verticals_list, loss_type):
    size = len(xxs)
    global_min = min(xxs[0])

    for xx, yy, _, val in zip(xxs, yys, verticals_list, range(size)):
        cnt_n = min(1, len(xx)/2)
        for _ in range(cnt_n):
            xx.pop(0)
            yy.pop(0)
        global_min = min(min(yy), global_min)

        alpha = 1 - (val)/size * 0.5
        plt.plot(xx, yy, linestyle="-", linewidth=1, alpha=alpha, label="loss {}".format(val+1))

    plt.axhline(y=global_min, color="red", alpha=1, label="loss={}".format(global_min))
    plt.ylabel("{} loss".format(loss_type))
    plt.xlabel("steps")
    # plt.xscale("log")
    plt.legend()
    plt.show()

def plot_individual_loss(xx, yy, verticals, val, loss_type):
    cnt_n = min(1, len(xx)/2)
    for _ in range(cnt_n):
        xx.pop(0)
        yy.pop(0)

    y_upper = max(yy)
    y_lower = min(yy)

    plt.plot(xx, yy, linestyle="-", label="loss {}".format(val+1))
    plt.ylabel("{} loss".format(loss_type))
    plt.xlabel("steps")
    plt.ylim([y_lower * 0.95, y_upper * 1.05])
    plt.axhline(y=y_lower, color="red", label="loss={}".format(y_lower))

    printlable = True
    for vertical in verticals:
        vcolor = "blue"
        if printlable:
            printlable = False
            plt.axvline(x=vertical, linestyle="--", linewidth=0.5, color=vcolor, label="lr changed")
        else:
            plt.axvline(x=vertical, linestyle="--", linewidth=0.5, color=vcolor)
    plt.legend()
    plt.show()

# def plot_tangent_line(xx, yy):
#     N = 25
#     R = range(10, len(yy)-N)
#
#     if len(yy) < N:
#         return
#
#     tangent = list()
#     for i in R:
#         val = yy[i] - yy[i+N-1]
#         tangent.append(val/N)
#
#     plt.plot(R, tangent, 'o',  markersize=3, label="tangent line")
#     plt.ylabel("rate")
#     plt.xlabel("scale")
#     plt.axhline(y=0, color="red")
#     plt.legend()
#     plt.show()

def readfile(filename):
    data_list = list()
    with open(filename, 'r') as f:
        line = f.readline().strip()
        while len(line) != 0:
            frame = list()
            frame.append(line)
            for _ in range(FRAMESIZE-1):
                frame.append(f.readline().strip())
            data_list.append(frame)
            line = f.readline().strip()
    return data_list
    
def check(args):
    success = True
    if args.filename == None:
        success = False
        print("Please add argument --filename <string>")
    if not args.loss_type in ["all", "policy", "wdl"]:
        success = False
        print("Loss type should be all/policy.")
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", nargs="+", metavar="<string>",
                        help='The input file name.', type=str)
    parser.add_argument("-i", "--individual", default=False,
                        help="Draw individual loss.", action="store_true")
    parser.add_argument("-t", "--loss-type", metavar="<string>", default="all",
                        help="Loss type in all/policy/wdl.")
    args = parser.parse_args()
    if check(args):
        plot_process(args)
