import torch
import torch.nn.functional as F
import numpy as np
import random, time

from symmetry import get_symmetry_plane
from network import Network
from loader import Loader

from torch.nn import DataParallel
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
    def __init__(self, cfg):
        self.cfg = cfg
        self.batchsize =  cfg.batchsize
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir

        self.opt_name = cfg.optimizer
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
        self.module  = self.net

        if self.use_gpu:
            self.net = self.net.to(self.device)
            self.net = DataParallel(self.net) 
            self.module  = self.net.module

        self.opt = None

        if self.opt_name == "Adam":
            self.opt = torch.optim.Adam(
                self.net.parameters(),
                lr=self.learn_rate,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "SGD":
            self.opt = torch.optim.SGD(
                self.net.parameters(),
                lr=self.learn_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )

        self.swa_model = torch.optim.swa_utils.AveragedModel(self.module)
        self.swa_scheduler = torch.optim.swa_utils.SWALR(self.opt, swa_lr=0.05)

    def fit_and_store(self, filename_prefix, init_steps, log_file):

        # Be sure the network is on the right device.
        self.setup()

        print("start training...")

        keep_running = True
        running_loss = 0
        num_steps = init_steps
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
                running_loss += self.step(board_size_list, planes, target, self.opt)
                num_steps += 1

                if num_steps % verbose_steps == 0:
                    elapsed = time.time() - clock_time
                    clock_time = time.time()

                    log_outs = "steps: {} -> loss: {:.4f}, speed: {:.2f} | opt: {}, learning rate: {}, batch size: {}".format(
                                   num_steps,
                                   running_loss/verbose_steps,
                                   verbose_steps/elapsed,
                                   self.opt_name,
                                   self.learn_rate,
                                   self.batchsize)
                    print(log_outs)
                    with open(log_file, 'a') as f:
                        f.write(log_outs + '\n')

                    running_loss = 0

                # should stop?
                if num_steps >= self.max_steps + init_steps:
                    keep_running = False
                    break

            # update swa
            self.swa_model.update_parameters(self.module)
            self.swa_scheduler.step()

            # save the last network

            # TODO: We should update the batch normalization layer, but seem there
            #       is bug here. Try to fix it.
            # torch.optim.swa_utils.update_bn(train_data, self.swa_model)

            torch.save(self.module.state_dict(), "{}-s{}.pt".format(filename_prefix, num_steps))
        print("Training is over.")

    def step(self, board_size_list, planes, target, opt):
        _, loss = self.net(planes, board_size_list, target)
        loss = loss.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        return loss.item()

    def load_pt(self, filename):
        self.net.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))
