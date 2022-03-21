import torch
import torch.nn.functional as F
import numpy as np
import random, time

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
        self.nn_board_size = cfg.boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = cfg.input_channels

        self.dummy_size = 0

        self.data_loaders = []
        num_workers = max(cfg.num_workers, 1)
        for _ in range(num_workers):
            self.data_loaders.append(Loader(dirname))


    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = None
        if worker_info == None:
            worker_id = 0
        else:
            worker_id = worker_info.id

        data = self.data_loaders[worker_id].next()

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
        # input_planes[self.input_channels-2, 0:board_size, 0:board_size] = 0 # fill zeros
        input_planes[self.input_channels-1, 0:board_size, 0:board_size] = 1 # fill ones

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
        return self.dummy_size

class TrainingPipe():
    # TODO: Support for multi-gpu training.

    def __init__(self, cfg):
        self.cfg = cfg
        self.batchsize =  cfg.batchsize
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir

        self.steps_per_epoch =  cfg.steps_per_epoch
        self.max_steps =  cfg.max_steps

        self.learn_rate = cfg.learn_rate
        self.weight_decay = cfg.weight_decay

        self.use_gpu = cfg.use_gpu
        self.device = torch.device('cpu')
        if self.use_gpu:
            self.device = torch.device('cuda')

        self.net = Network(cfg)
        self.net.trainable(True)

    def setup(self):
        if self.use_gpu:
            self.net = self.net.to(self.device)

        self.adam_opt = torch.optim.Adam(
            self.net.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )

    def fit_and_store(self, filename_prefix):

        # Be sure the network is on the right device.
        self.setup()

        print("start training...")

        tb_writer = SummaryWriter()

        keep_running = True
        running_loss = 0
        num_steps = 0
        verbose_steps = 1000
        clock_time = time.time()

        self.data_set = DataSet(self.cfg, self.train_dir)

        while keep_running:
            self.data_set.dummy_size = self.steps_per_epoch * self.batchsize

            train_data = DataLoader(
                self.data_set,
                num_workers=self.num_workers,
                batch_size=self.batchsize
            )

            if num_steps != 0:
                self.save_pt("{}-s{}.{}".format(filename_prefix, num_steps, "pt"))


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
                num_steps += 1

                if num_steps % verbose_steps == 0:
                    elapsed = time.time() - clock_time
                    clock_time = time.time()

                    print("steps: {} -> loss {:.4f}, speed: {:.2f}".format(
                                                        num_steps,
                                                        running_loss/verbose_steps,
                                                        verbose_steps/elapsed
                                                    ))
                    tb_writer.add_scalar('loss', running_loss/verbose_steps, num_steps)
                    running_loss = 0

                # should stop?
                if num_steps >= self.max_steps:
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

    def save_pt(self, filename):
        self.net.save_pt(filename)

    def load_pt(self, filename):
        self.net.load_pt(filename)
