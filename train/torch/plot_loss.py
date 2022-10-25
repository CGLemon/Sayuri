import argparse, re
import matplotlib.pyplot as plt

def matchline(line):
    regex = re.compile(r'steps: (\d+) -> loss: (\d.\d+), speed: (\d.\d+) \| opt: (\D+), learning rate: ((\d.\d+)|(\de-\d+)), batch size: (\d+)')
    m = regex.match(line)
    steps = 0
    loss = 0
    lr = 0

    if m is not None:
        steps = m.group(1)
        loss = m.group(2)
        lr = m.group(5)

    return int(steps), float(loss), float(lr)

def plot_all(args):
    xxs = list()
    yys = list()
    verticals_list = list()

    for file_name in args.file_name:
        print(file_name)
        xx, yy, verticals = gather_data(file_name)
        xxs.append(xx)
        yys.append(yy)
        verticals_list.append(verticals)

    plot_multi_loss(xxs, yys, verticals_list)
    for xx, yy, verticals, val in zip(xxs, yys, verticals_list, range(len(xxs))):
        plot_loss(xx, yy, verticals, val)


def gather_data(file_name):
    data_list = readfile(file_name)
    xx = list()
    yy = list()
    lr_schedule = list()
    verticals = list()

    for i in data_list:
        x, y, lr = matchline(i)
        if len(lr_schedule) == 0:
            lr_schedule.append(lr)
        elif lr_schedule[len(lr_schedule)-1] != lr:
            lr_schedule.append(lr)
            verticals.append(x)
        xx.append(x)
        yy.append(y)
    return xx, yy, verticals

def plot_multi_loss(xxs, yys, verticals_list):
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
    plt.ylabel("loss")
    plt.xlabel("steps")
    # plt.xscale("log")
    plt.legend()
    plt.show()

def plot_loss(xx, yy, verticals, val):
    cnt_n = min(1, len(xx)/2)
    for _ in range(cnt_n):
        xx.pop(0)
        yy.pop(0)

    y_upper = max(yy)
    y_lower = min(yy)

    plt.plot(xx, yy, linestyle="-", label="loss {}".format(val+1))
    plt.ylabel("loss")
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

def plot_tangent_line(xx, yy):
    N = 25
    R = range(10, len(yy)-N)

    if len(yy) < N:
        return

    tangent = list()
    for i in R:
        val = yy[i] - yy[i+N-1]
        tangent.append(val/N)

    plt.plot(R, tangent, 'o',  markersize=3, label="tangent line")
    plt.ylabel("rate")
    plt.xlabel("scale")
    plt.axhline(y=0, color="red")
    plt.legend()
    plt.show()

def readfile(filename):
    data_list = list()
    with open(filename, 'r') as f:
        line = f.readline()
        while len(line) != 0:
            data_list.append(line)
            line = f.readline()
    return data_list

def check(args):
    success = True
    if args.file_name == None:
        success = False
        print('Please add argument --file-name <string>')
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file-name', nargs="+", metavar='<string>',
                        help='The input file name.', type=str)
    args = parser.parse_args()
    if check(args):
        plot_all(args)
