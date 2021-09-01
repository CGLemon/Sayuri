import torch
import torch.nn.functional as F
import numpy as np

from symmetry import get_symmetry_plane
from nnprocess import NNProcess
from loader import Loader

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))

class DataSet():
    # The simple DataSet wrapper.

    def __init__(self, cfg, dirname):
        self.data_loader = Loader(cfg, dirname)
        self.cfg = cfg
        self.board_size = cfg.boardsize
        self.input_channels = cfg.input_channels

    def __getitem__(self, idx):
        data = self.data_loader[idx]
        data.unpack_planes()

        symm = int(np.random.choice(8, 1)[0])
        num_intersections = data.board_size * data.board_size

        input_planes = np.zeros((self.input_channels, num_intersections))
        prob = np.zeros(num_intersections+1)
        aux_prob = np.zeros(num_intersections+1)
        ownership = np.zeros(num_intersections)
        wdl = np.zeros(3)
        stm = np.zeros(1)
        final_score = np.zeros(1)

        buf = np.zeros(num_intersections)

        # fill input planes
        for p in range(self.input_channels-4):
            buf[:] = data.planes[p][:]
            buf = get_symmetry_plane(symm, buf, data.board_size)
            input_planes[p][:] = buf[:]

        if data.to_move == 1:
            input_planes[self.input_channels-4][:] = data.komi/10
        else:
            input_planes[self.input_channels-4][:] = -data.komi/10

        input_planes[self.input_channels-3][:] = data.board_size/10

        if data.to_move == 1:
            input_planes[self.input_channels-2][:] = 1
        else:
            input_planes[self.input_channels-1][:] = 1

        # fill probabilities
        buf[:] = data.prob[0:num_intersections]
        buf = get_symmetry_plane(symm, buf, data.board_size)

        prob[0:num_intersections] = buf[:]
        prob[num_intersections] = data.prob[num_intersections]

        # fill auxiliary probabilities
        buf[:] = data.aux_prob[0:num_intersections]
        buf = get_symmetry_plane(symm, buf, data.board_size)

        aux_prob[0:num_intersections] = buf[:]
        aux_prob[num_intersections] = data.aux_prob[num_intersections]

        # fill ownership
        buf[:] = data.ownership[:]
        buf = get_symmetry_plane(symm, buf, data.board_size)
        ownership[:] = buf[:]

        # fill winrate
        wdl[1 - data.result] = 1
        stm[0] = data.result
        final_score[0] = data.final_score

        input_planes = np.reshape(input_planes, (self.input_channels, data.board_size, data.board_size))
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

        self.epochs = cfg.epochs
        self.learn_rate = cfg.learn_rate
        self.weight_decay = cfg.weight_decay

        self.use_gpu = cfg.use_gpu
        self.device = torch.device('cpu')
        if self.use_gpu:
            self.device = torch.device('cuda:0')

        self.net = NNProcess(self.cfg)
        self.net.trainable(True)
        self.data_set = DataSet(self.cfg, self.train_dir)
        self.train_data = DataLoader(
            self.data_set,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batchsize
        )

        if self.cfg.misc_verbose:
            dump_dependent_version()

    def setup(self):
        if self.use_gpu:
            self.net = self.net.to(self.device)

        self.adam_opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )

    def fit(self):
        # Be sure the network is on the right device.
        self.setup()

        print("start training...")

        tb_writer = SummaryWriter()

        running_loss = 0
        num_step = 0
        verbose_step = 500

        print("max step {}...".format(self.epochs * len(self.train_data)))

        for e in range(self.epochs):
            for _, batch in enumerate(self.train_data):
                _, planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score = batch
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
                running_loss += self.step(planes, target, self.adam_opt)

                num_step += 1
                if num_step % verbose_step == 0:
                    print("step: {} -> loss {:.4f}".format(
                                                        num_step,
                                                        running_loss/verbose_step
                                                    ))
                    tb_writer.add_scalar('loss', running_loss/verbose_step, num_step)
                    running_loss = 0
            print("epoch {} finished...".format(e+1))

    def step(self, planes, target, opt):
        pred = self.net(planes)

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
