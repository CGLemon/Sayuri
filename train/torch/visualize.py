import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np
import pysayuri
import argparse, colorsys, math, os

class VisualizedNetworkWrap(pysayuri.NetworkWrap):
    def __init__(self, cfg):
        super(VisualizedNetworkWrap, self).__init__(cfg)

    @torch.no_grad()
    def fancy_inference(self, planes):
        # mask buffers
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :]
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        # input layer
        x = self.input_conv(planes, mask)

        # residual tower
        x, _ = self.residual_tower((x, mask_buffers))

        # policy head
        pol = self.policy_conv(x, mask)
        pol_gpool = self.global_pool(pol, mask_buffers)
        pol_inter = self.policy_intermediate_fc(pol_gpool)

        # Add intermediate as biases. It may improve the policy performance.
        b, c = pol_inter.shape
        pol = (pol + pol_inter.view(b, c, 1, 1)) * mask
        pol = torch.flatten(pol, start_dim=2, end_dim=3)

        # value head
        val = self.value_conv(x, mask)
        val = torch.flatten(val, start_dim=2, end_dim=3)

        predict = (
            pol,
            val
        )
        return predict

    @torch.no_grad()
    def shift_tensor(self, tensor):
        mean = tensor.mean()
        std = tensor.std()

    @torch.no_grad()
    def get_output_without_batch(self, features, *args, **kwargs):
        as_numpy = kwargs.get("as_numpy", False)
        board_size = kwargs.get("board_size", None)

        if isinstance(features, np.ndarray):
            planes = torch.from_numpy(features).float().to(self._device)
        elif torch.is_tensor(features):
            planes = features.float().to(self._device)
        else:
            raise Exception("get_output_without_batch(...): Tensor should be torch or numpy array")

        planes = torch.from_numpy(features).float().to(self._device)
        pred = self.fancy_inference(torch.unsqueeze(planes, 0))

        labels = (
            "policy-head",
            "value-head"
        )
        reuslt = dict()
        for tensors, label in zip(pred, labels):
            _, channels, _ = tensors.shape
            tensors = self._get_valid_spat(tensors, board_size)
            # print(tensors.shape)
            for c in range(channels):
                tensor = tensors[:, c, :]
                mean = tensor.mean()
                std = tensor.std()
                tensor = (torch.tanh((tensor - mean)/(std + 1e-5)) + 1.) / 2.
                tensors[:, c, :] = tensor[:]
            reuslt[label] = tensors[0] if not as_numpy else \
                                tensors[0].cpu().detach().numpy()
        return reuslt

class VisualizedAgent(pysayuri.Agent):
    def __init__(self, *args, **kwargs):
        super(VisualizedAgent, self).__init__(*args, **kwargs)

    def load_sgf_string(self, sgf_string):
        sgf = pysayuri.SgfGame()
        sgf.load_string(sgf_string)
        self._board = sgf.last_board.copy()

    def _load_network(self, cfg):
        return VisualizedNetworkWrap(cfg)

class PlotBoard:
    BOARD_COL = (0.85, 0.68, 0.40)
    BLACK_COL = (0.05, 0.05, 0.05)
    WHITE_COL = (0.95, 0.95, 0.95)
    RED_COL = (0.95, 0.45, 0.55)

    def __init__(self, board, imagesize=800):
        self.imagesize = imagesize
        self.board = board
        self.board_size = board.board_size
        self.grid_margin = self.imagesize/20.
        self.gridsize = (self.imagesize - self.grid_margin)/self.board_size
        self.gridpos = [ self.gridsize * (i + 0.5) + self.grid_margin/2. for i in range(self.board_size) ]
        self.board_img = np.ones((self.imagesize, self.imagesize, 3))

    def _draw_board(self):
        self.board_img[:,:,:] = self.BOARD_COL[:]
        for i in range(self.board_size):
            lo = round(self.gridpos[0])
            hi = round(self.gridpos[-1])
            ss = round(self.gridpos[i])
            self.board_img[ss-1:ss+1,lo:hi,:] = 0
            self.board_img[lo:hi,ss-1:ss+1,:] = 0

    def _draw_rect(self, x, y, col, alpha=1., scale=1.):
        sz = scale * self.gridsize
        xstart = round(self.gridpos[x] - sz/2.)
        xend = round(self.gridpos[x] + sz/2.)
        ystart = round(self.gridpos[y] - sz/2.)
        yend = round(self.gridpos[y] + sz/2.)
        alpha_col = self.board_img[ystart:yend, xstart:xend, :]
        alpha_col = (1. - alpha) * alpha_col[:, :] + alpha * np.array(col)
        self.board_img[ystart:yend, xstart:xend, :] = alpha_col[:]

    def _draw_circle(self, x, y, col, line_col=(0.,0.,0.), scale=1.):
        r = scale * self.gridsize
        localsize = round(self.gridsize + 1)
        circlemap = np.zeros((localsize,localsize))
        linemap = np.zeros((localsize,localsize))

        for pixel in range(round(max(1, round(r/10)))):
            pixelscale = 1.0 - (2.0 * pixel/localsize)
            N = round(pixelscale * localsize * 8)
            for i in range(N):
                xpos = round(pixelscale * r * np.cos(2. * np.pi * i/N) + self.gridsize/2.)
                ypos = round(pixelscale * r * np.sin(2. * np.pi * i/N) + self.gridsize/2.)
                linemap[ypos,xpos] = 1
        circlemap[:] = linemap[:]

        ct = round(self.gridsize/2.)
        que=[(ct,ct)]
        circlemap[ct,ct] = 1
        while len(que) != 0:
            xpos, ypos = que.pop(0)
            for xdir, ydir in [(1,0),(0,1),(-1,0),(0,-1)]:
                xnext = xpos + xdir
                ynext = ypos + ydir
                if circlemap[ynext,xnext] == 0:
                    que.append((xnext,ynext))
                    circlemap[ynext,xnext] = 1

        for iy in range(localsize):
            for ix in range(localsize):
                if circlemap[iy,ix] == 1:
                    xpos = round(ix + self.gridpos[x] - self.gridsize/2.)
                    ypos = round(iy + self.gridpos[y] - self.gridsize/2.)
                    self.board_img[ypos,xpos,:] = col[:]
        for iy in range(localsize):
            for ix in range(localsize):
                if linemap[iy,ix] == 1:
                    xpos = round(ix + self.gridpos[x] - self.gridsize/2.)
                    ypos = round(iy + self.gridpos[y] - self.gridsize/2.)
                    self.board_img[ypos,xpos,:] = line_col[:]

    def display(self, *args, **kwargs):
        show = kwargs.get("show", True)
        title = kwargs.get("title", None)
        save_path = kwargs.get("filename", None)
        u_contents = kwargs.get("upper_contents", [])
        l_contents = kwargs.get("lower_contents", [])

        self._draw_board()
        for col, alpha, x, y in l_contents:
            self._draw_rect(x, self.board_size-y-1, col=col, alpha=alpha)
        for y in range(self.board_size):
            for x in range(self.board_size):
                vertex = self.board.get_vertex(x,y)
                color = self.board.state[vertex]
                if color == pysayuri.BLACK:
                    self._draw_circle(x, self.board_size-y-1, col=self.BLACK_COL, scale=0.475)
                if color == pysayuri.WHITE:
                    self._draw_circle(x, self.board_size-y-1, col=self.WHITE_COL, scale=0.475)
                if vertex == self.board.last_move:
                    self._draw_circle(x, self.board_size-y-1, col=self.RED_COL, scale=0.175)
        for col, alpha, x, y in u_contents:
            self._draw_rect(x, self.board_size-y-1, col=col, alpha=alpha)
        if save_path:
            mpimg.imsave(save_path, self.board_img)
        if title:
            plt.title(title)
        if show:
            plt.imshow(self.board_img)
            plt.show()

def value_to_code(val):
    h1, h2 = 145, 215
    w = h2 - h1
    w2 = 20

    h = (1.0 - val) * (242 - w + w2)
    s = 1.0
    v = 1.0

    if (h1 <= h and h <= h1 + w2):
        h = h1 + (h - h1) * w / w2
        m = w / 2
        v -= (m - abs(h - (h1 + m))) * 0.2 / m
    elif h >= h1 + w2:
        h += w - w2

    h0 = 100
    m0 = (h2 - h0) / 2
    if h0 <= h and h <= h2:
        v -= (m0 - abs(h - (h0 + m0))) * 0.2 / m0
    return colorsys.hsv_to_rgb(h/360,s,v)

def main(args):
    agent = VisualizedAgent(
        checkpoint = args.checkpoint,
        use_swa = True,
        scoring_rule = "area"
    )
    if args.sgf:
        agent.load_sgf(args.sgf)
    else:
        sgf = "(;GM[1]FF[4]CA[UTF-8]AP[Sabaki:0.52.0]KM[7.5]" \
              "SZ[19]DT[2024-10-13];B[dc];W[qp];B[cp];W[pd];" \
              "B[op];W[oq];B[nq];W[pq];B[np];W[qn];B[jq];W[ce];B[fd];W[di])"
        agent.load_sgf_string(sgf)
    board = agent.get_board()
    board_size = board.board_size
    plt_board = PlotBoard(board)

    pred_result = agent.get_net_output_without_batch(as_numpy=True)
    root_dir = "result" if args.result_dir is None else args.result_dir
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    policy_head = pred_result["policy-head"]
    NUN_LAYERS, _ = policy_head.shape
    for layer_n in range(NUN_LAYERS):
        contents = list()
        for y in range(board_size):
            for x in range(board_size):
                pval = policy_head[layer_n][x + board_size * y]
                contents.append((value_to_code(pval), 0.5, x, y))
        plt_board.display(
            show=False,
            filename=os.path.join(root_dir, "policyhead-tanh-map-{}.png".format(layer_n+1)),
            title="The {} Policy Layer".format(layer_n+1),
            lower_contents=contents)

    value_head = pred_result["value-head"]
    NUN_LAYERS, _ = value_head.shape
    for layer_n in range(NUN_LAYERS):
        contents = list()
        for y in range(board_size):
            for x in range(board_size):
                pval = value_head[layer_n][x + board_size * y]
                contents.append((value_to_code(pval), 0.5, x, y))
        plt_board.display(
            show=False,
            filename=os.path.join(root_dir, "valuehead-tanh-map-{}.png".format(layer_n+1)),
            title="The {} Value Layer".format(layer_n+1),
            lower_contents=contents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", metavar="<string>",
                        help="The path of checkpoint.", type=str)
    parser.add_argument("-s", "--sgf", metavar="<string>",
                        help="Load the SGF file from here.", type=str)
    parser.add_argument("-r", "--result-dir", metavar="<string>",
                        help="Save the result file here.", type=str)
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print("exception: {}".format(e))
