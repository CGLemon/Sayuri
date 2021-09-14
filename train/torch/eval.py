import argparse

from train import *
from config import *
from nnprocess import *
import numpy as np



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="The json file name", type=str)
    parser.add_argument("-n", "--net", help="", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)
    dataset = DataSet(cfg, "vv-dir")

    '''
    data = dataset[12]
    planes = data[0]
    print(planes[0])
    print()
    print(planes[1])
    print()
    print(planes[2])
    print()

    owner = data[3]
    print(owner)
    '''

    Net = NNProcess(cfg)
    Net = Net.to(torch.device('cuda:0'))
    Net.load_ckpt(args.net + ".ckpt")
    Net.trainable(False)

    # Net.transfer2text(args.net + ".txt")

    length = len(dataset)
    print(length)
    data = dataset[0]
    planes = data[0].unsqueeze(dim=0)
    idx = 0


    def print_planes(plane):
        bsize = 9
        for i in range(bsize):
            for j in range(bsize):
                o = plane[i][j]   
                if o == 0:
                    print(" x ", end=' ')
                else:
                    print("{0:0.1f}".format(o), end=' ')
            print()
        print()
    for p in planes[0]:
        print('plane: ' + str(idx))
        print_planes(p)
        print()
        idx += 1

    m = nn.Softmax(dim=1)
    planes = planes.to(torch.device('cuda:0'))
    out = Net(planes)

    def print_sqrt(outs):
        np.set_printoptions(precision=4)
        o = outs[0].detach().numpy()
        bsize = 9
        for i in range(bsize):
            for j in range(bsize):
                idx = i * bsize + j
                if o[idx] < 0:
                    print("{0:0.4f}".format(o[idx]), end=' ')
                else:
                    print(" {0:0.4f}".format(o[idx]), end=' ')
            print()
        if len(o) == bsize * bsize + 1:
            print("{0:0.4f}".format(o[bsize * bsize]))
        print(np.max(o))
        print()

    print_sqrt(m(out[0].to(torch.device('cpu'))))
    print_sqrt(m(out[1].to(torch.device('cpu'))))
    print_sqrt(out[2].to(torch.device('cpu')))
    print(m(out[3].to(torch.device('cpu'))))
    print((out[4].to(torch.device('cpu'))+1)/2)
    print(out[5].to(torch.device('cpu')) * 20)
