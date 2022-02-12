import torch
import torch.nn.functional as F
import numpy as np
import random

from symmetry import get_symmetry_plane
from network import Network
from loader import Loader

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Subset

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "NumPy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))

class DataSet():
    # The simple DataSet wrapper.

    def __init__(self, cfg, dirname):
        self.data_loader = Loader(dirname)
        self.nn_board_size = cfg.boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = cfg.input_channels

    def __getitem__(self, idx):
        data = self.data_loader[idx]
        data.unpack_planes()

        symm = int(np.random.choice(8, 1)[0])

        nn_board_size = self.nn_board_size
        nn_num_intersections = self.nn_num_intersections

        board_size = data.board_size
        num_intersections = data.board_size * data.board_size


        # allocate all buffers
        input_planes = np.zeros((self.input_channels, nn_board_size, nn_board_size))
        prob = np.zeros(nn_num_intersections+1)
        aux_prob = np.zeros(nn_num_intersections+1)
        ownership = np.zeros((nn_board_size, nn_board_size))
        wdl = np.zeros(3)
        stm = np.zeros(1)
        final_score = np.zeros(1)

        buf = np.zeros(num_intersections)
        sqr_buf = np.zeros((nn_board_size, nn_board_size))

        # input planes
        for p in range(self.input_channels-4):
            buf[:] = data.planes[p][:]
            buf = get_symmetry_plane(symm, buf, data.board_size)
            input_planes[p, 0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]

        if data.to_move == 1:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] = data.komi/10
        else:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] = -data.komi/10

        input_planes[self.input_channels-3, 0:board_size, 0:board_size] = (data.board_size**2)/100

        if data.to_move == 1:
            input_planes[self.input_channels-2, 0:board_size, 0:board_size] = 1
        else:
            input_planes[self.input_channels-1, 0:board_size, 0:board_size] = 1

        # probabilities
        buf[:] = data.prob[0:num_intersections]
        buf = get_symmetry_plane(symm, buf, data.board_size)
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]

        prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        prob[nn_num_intersections] = data.prob[num_intersections]

        # auxiliary probabilities
        buf[:] = data.aux_prob[0:num_intersections]
        buf = get_symmetry_plane(symm, buf, data.board_size)
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]

        aux_prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        aux_prob[nn_num_intersections] = data.aux_prob[num_intersections]

        # ownership
        buf[:] = data.ownership[:]
        buf = get_symmetry_plane(symm, buf, data.board_size)

        ownership[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]
        ownership = np.reshape(ownership, (nn_num_intersections))

        # winrate
        wdl[1 - data.result] = 1
        stm[0] = data.q_value
        final_score[0] = data.final_score

        data.pack_planes()

        return (
            data.board_size,
            torch.tensor(input_planes).float(),
            torch.tensor(prob).float(),
            torch.tensor(aux_prob).float(),
            torch.tensor(ownership).float(),
            torch.tensor(wdl).float(),
            torch.tensor(stm).float(),
            torch.tensor(final_score).float()
        )

    def __len__(self):
        return len(self.data_loader)

class TrainingPipe():
    # TODO: Support for multi-gpu training.

    def __init__(self, cfg):
        self.cfg = cfg
        self.batchsize =  cfg.batchsize
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir

        self.step_per_epoch =  cfg.step_per_epoch
        self.max_step =  cfg.max_step

        self.learn_rate = cfg.learn_rate
        self.weight_decay = cfg.weight_decay

        self.use_gpu = cfg.use_gpu
        self.device = torch.device('cpu')
        if self.use_gpu:
            self.device = torch.device('cuda:0')

        self.net = Network(self.cfg)
        self.net.trainable(True)
        self.data_set = None

    def setup(self):
        if self.use_gpu:
            self.net = self.net.to(self.device)

        self.adam_opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )

    def prepare_data(self):
        self.data_set = DataSet(self.cfg, self.train_dir)

    def fit(self):
        if self.data_set == None:
            return

        # Be sure the network is on the right device.
        self.setup()

        print("start training...")

        tb_writer = SummaryWriter()

        keep_running = True
        running_loss = 0
        num_step = 0
        verbose_step = 500

        print("data size {}".format(len(self.data_set)))
        indics_list = range(len(self.data_set))

        while keep_running:
            selections = min(self.step_per_epoch * self.batchsize, len(self.data_set))

            subset = Subset(self.data_set, random.sample(indics_list, selections))
            train_data = DataLoader(
                subset,
                num_workers=self.num_workers,
                shuffle=True,
                batch_size=self.batchsize
            )

            for _, batch in enumerate(train_data):
                board_size_list, planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score = batch
                if self.use_gpu:
                    planes = planes.to(self.device)
                    target_prob = target_prob.to(self.device)
                    target_aux_prob = target_aux_prob.to(self.device)
                    target_ownership = target_ownership.to(self.device)
                    target_wdl = target_wdl.to(self.device)
                    target_stm = target_stm.to(self.device)
                    target_score = target_score.to(self.device)

                target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score)

                # update network
                running_loss += self.step(board_size_list, planes, target, self.adam_opt)

                num_step += 1

                if num_step % verbose_step == 0:
                    print("step: {} -> loss {:.4f}".format(
                                                        num_step,
                                                        running_loss/verbose_step
                                                    ))
                    tb_writer.add_scalar('loss', running_loss/verbose_step, num_step)
                    running_loss = 0

                if num_step >= self.max_step:
                    keep_running = False
                    break

    def step(self, board_size_list, planes, target, opt):
        pred = self.net(planes, board_size_list)

        loss = self.compute_loss(pred, target)

        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item()

    def compute_loss(self, pred, target):
        (pred_prob, pred_aux_prob, pred_ownership, pred_wdl, pred_stm, pred_score) = pred
        (target_prob,target_aux_prob, target_ownership, target_wdl, target_stm, target_final_score) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        prob_loss = cross_entropy(pred_prob, target_prob)
        aux_prob_loss = 0.15 * cross_entropy(pred_aux_prob, target_aux_prob)
        ownership_loss = 0.15 * F.mse_loss(pred_ownership, target_ownership)
        wdl_loss = cross_entropy(pred_wdl, target_wdl)
        stm_loss = F.mse_loss(pred_stm, target_stm)

        fina_score_loss = 0.0012 * huber_loss(20 * pred_score, target_final_score, 12)

        loss = prob_loss + aux_prob_loss + ownership_loss + wdl_loss + stm_loss + fina_score_loss

        return loss

    def to_cpu(self):
        self.net = self.net.to(torch.device('cpu'))

    def to_device(self):
        self.net = self.net.to(self.device)

    def save_pt(self, filename):
        self.net.save_pt(filename)

    def load_pt(self, filename):
        self.net.load_pt(filename)
