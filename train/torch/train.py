import torch
import torch.nn.functional as F
import numpy as np
import random, time, math, os

from network import Network
from loader import Loader

from torch.nn import DataParallel
from torch.utils.data import DataLoader, Subset

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "NumPy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))

class DataSet():
    def __init__(self, cfg, dirname):
        self.nn_board_size = cfg.boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = cfg.input_channels

        self.dummy_size = 0

        self.data_loaders = []
        num_workers = max(cfg.num_workers, 1)
        for _ in range(num_workers):
            self.data_loaders.append(Loader(dirname))

    def __skip(self, data):
        # Reture true if we want to skip current data. For example,
        # we only want 19 board data, we can use following code...
        #
        # if data.board_size != 19:
        #    return True

        return False

    def __wrap_data(self, worker_id):
        data = self.data_loaders[worker_id].next()

        while self.__skip(data):
            data = self.data_loaders[worker_id].next()

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
            plane = data.planes[p]
            input_planes[p, 0:board_size, 0:board_size] = np.reshape(plane, (board_size, board_size))[:, :]

        if data.to_move == 1:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] =  data.komi/20
            input_planes[self.input_channels-3, 0:board_size, 0:board_size] = -data.komi/20
        else:
            input_planes[self.input_channels-4, 0:board_size, 0:board_size] = -data.komi/20
            input_planes[self.input_channels-3, 0:board_size, 0:board_size] =  data.komi/20

        input_planes[self.input_channels-2, 0:board_size, 0:board_size] = (data.board_size**2)/361
        input_planes[self.input_channels-1, 0:board_size, 0:board_size] = 1 # fill ones

        # probabilities
        buf[:] = data.prob[0:num_intersections]
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]
        prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        prob[nn_num_intersections] = data.prob[num_intersections]

        # auxiliary probabilities
        buf[:] = data.aux_prob[0:num_intersections]
        sqr_buf[0:board_size, 0:board_size] = np.reshape(buf, (board_size, board_size))[:, :]
        aux_prob[0:nn_num_intersections] = np.reshape(sqr_buf, (nn_num_intersections))[:]
        aux_prob[nn_num_intersections] = data.aux_prob[num_intersections]

        # ownership
        ownership[0:board_size, 0:board_size] = np.reshape(data.ownership, (board_size, board_size))[:, :]
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

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = None
        if worker_info == None:
            worker_id = 0
        else:
            worker_id = worker_info.id

        return self.__wrap_data(worker_id)


    def __len__(self):
        return self.dummy_size

class TrainingPipe():
    def __init__(self, cfg):
        self.cfg = cfg

        # mini-batch size, update the network per batch size
        self.batchsize =  cfg.batchsize

        # marco batch size and factor, (marco batch size) * factor = batch size
        self.macrobatchsize = cfg.macrobatchsize
        self.macrofactor = cfg.macrofactor

        # how many cpu does the 'DataLoader' use?
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir

        # Store the last model per epoch. It also define the training data
        # buffer size.
        self.steps_per_epoch =  cfg.steps_per_epoch

        # Max steps per training task.
        self.max_steps =  cfg.max_steps

        # Which optimizer do we use?
        self.opt_name = cfg.optimizer

        # Optimizer's parameters.
        self.learning_rate = cfg.learning_rate
        self.weight_decay = cfg.weight_decay

        self.use_gpu = cfg.use_gpu
        self.device = torch.device('cpu')
        if self.use_gpu:
            self.device = torch.device('cuda')

        self.net = Network(cfg)
        self.net.trainable(True)

        # store root dir
        self.store_path = cfg.store_path

        self.__setup()

    def __setup(self):
        self.module = self.net # linking

        if self.use_gpu:
            self.net = self.net.to(self.device)
            self.net = DataParallel(self.net) 
            self.module  = self.net.module

        # We may fail to load the optimizer. So init
        # it before load it.
        self.opt = None
        if self.opt_name == "Adam":
            self.opt = torch.optim.Adam(
                self.net.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "SGD":
            # Recommanded optimizer, the SGD is better than Adam
            # in this kind training task.
            self.opt = torch.optim.SGD(
                self.net.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                nesterov=True,
                weight_decay=self.weight_decay,
            )

        model_path = os.path.join(self.store_path, "model")
        if not os.path.isdir(model_path):
            os.mkdir(model_path)
            
        opt_path = os.path.join(self.store_path, "opt")
        if not os.path.isdir(opt_path):
            os.mkdir(opt_path)

        info_file = os.path.join(self.store_path, "info.txt")
        with open(info_file, 'w') as f:
            f.write(self.module.simple_info())

    def __lr_scheduler(self):
        pass

    def __optimizer_to(self, optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

    def __load_current_status(self):
        #TODO: Merge optimizer status and model into one file.
        last_steps = 0

        steps_name = os.path.join(self.store_path, "last_steps.txt")
        if os.path.isfile(steps_name):
            with open(steps_name, 'r') as f:
                last_steps = int(f.read())

        model_path = os.path.join(self.store_path, "model")
        model_name = os.path.join(model_path, "s{}.pt".format(last_steps))
        if os.path.isfile(model_name):
            self.module.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
            print("load model: {}".format(model_name))
        else:
            # If we fail to load another model, be sure that
            # init steps is zero.
            assert last_steps==0, ""

        opt_path = os.path.join(self.store_path, "opt")
        opt_name = os.path.join(opt_path, "s{}.pt".format(last_steps))
        if os.path.isfile(opt_name):
            # TODO: We may load different optimizers. Be sure that
            #       program don't crash in any condition.
            self.opt.load_state_dict(torch.load(opt_name, map_location=torch.device('cpu')))
            print("load optimizer: {}".format(opt_name))

        # update to current learning rate...
        self.opt.param_groups[0]["lr"] = self.learning_rate
        self.opt.param_groups[0]["weight_decay"] = self.weight_decay

        self.__optimizer_to(self.opt, self.device)

        return last_steps

    def __store_current_status(self, steps):
        # TODO: Automatically parse the last steps from model's name.

        steps_name = os.path.join(self.store_path, "last_steps.txt")
        with open(steps_name, 'w') as f:
            f.write(str(steps))

        model_path = os.path.join(self.store_path, "model")
        model_name = os.path.join(model_path, "s{}.pt".format(steps))
        torch.save(self.module.state_dict(), model_name)

        opt_path = os.path.join(self.store_path, "opt")
        opt_name = os.path.join(opt_path, "s{}.pt".format(steps))
        torch.save(self.opt.state_dict(), opt_name)

    def fit_and_store(self):
        init_steps = self.__load_current_status()
        print("start training...")

        def get_running_loss_dict():
            # get New dict 
            running_loss_dict = dict()
            running_loss_dict['loss'] = 0
            running_loss_dict['prob_loss'] = 0
            running_loss_dict['aux_prob_loss'] = 0
            running_loss_dict['ownership_loss'] = 0
            running_loss_dict['wdl_loss'] = 0
            running_loss_dict['stm_loss'] = 0
            running_loss_dict['fina_score_loss'] = 0
            return running_loss_dict

        keep_running = True
        running_loss_dict = get_running_loss_dict()
        num_steps = init_steps
        macro_steps = 0

        verbose_steps = 1000
        clock_time = time.time()

        self.data_set = DataSet(self.cfg, self.train_dir)

        while keep_running:
            self.data_set.dummy_size = self.steps_per_epoch * self.batchsize

            # DataLoader or we can call it data buffer. It will prepare
            # some datas before they are used. To Shuffle the buffer is
            # unuseful so we do not open this option.
            train_data = DataLoader(
                self.data_set,
                num_workers=self.num_workers,
                batch_size=self.macrobatchsize
            )

            for _, batch in enumerate(train_data):
                # Fetch the next batch data from disk.
                _, planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score = batch
                if self.use_gpu:
                    planes = planes.to(self.device)
                    target_prob = target_prob.to(self.device)
                    target_aux_prob = target_aux_prob.to(self.device)
                    target_ownership = target_ownership.to(self.device)
                    target_wdl = target_wdl.to(self.device)
                    target_stm = target_stm.to(self.device)
                    target_score = target_score.to(self.device)

                # gather batch data
                target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score)

                # forward and backforwad
                _, all_loss = self.net(planes, target, use_symm=True)

                prob_loss, aux_prob_loss, ownership_loss, wdl_loss, stm_loss, fina_score_loss = all_loss

                # compute loss
                prob_loss = prob_loss.mean() / self.macrofactor
                aux_prob_loss = aux_prob_loss.mean() / self.macrofactor
                ownership_loss = ownership_loss.mean() / self.macrofactor
                wdl_loss = wdl_loss.mean() / self.macrofactor
                stm_loss = stm_loss.mean() / self.macrofactor
                fina_score_loss = fina_score_loss.mean() / self.macrofactor

                loss = prob_loss + aux_prob_loss + ownership_loss + wdl_loss + stm_loss + fina_score_loss
                loss.backward()
                macro_steps += 1

                # accumulate loss
                running_loss_dict['loss'] += loss.item()
                running_loss_dict['prob_loss'] += prob_loss.item()
                running_loss_dict['aux_prob_loss'] += aux_prob_loss.item()
                running_loss_dict['ownership_loss'] += ownership_loss.item()
                running_loss_dict['wdl_loss'] += wdl_loss.item()
                running_loss_dict['stm_loss'] += stm_loss.item()
                running_loss_dict['fina_score_loss'] += fina_score_loss.item()

                if math.isnan(running_loss_dict['loss']):
                    print("The gradient is explosion. Stop training...")
                    keep_running = False
                    break

                if macro_steps % self.macrofactor == 0:
                    # clip grad
                    if self.cfg.fixup_batch_norm:
                        gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 400.0)

                    # update network
                    self.opt.step()
                    self.opt.zero_grad()
                    num_steps += 1

                    # dump the verbose
                    if num_steps % verbose_steps == 0:
                        elapsed = time.time() - clock_time
                        clock_time = time.time()

                        # The dump_outs contains more infomations. The log_outs
                        # only stores short infomations because squeezing it in
                        # one line.
                        dump_outs = "steps: {} -> ".format(num_steps)
                        dump_outs += "speed: {:.2f}, opt: {}, learning rate: {}, batch size: {}\n".format(
                                         verbose_steps/elapsed,
                                         self.opt_name,
                                         self.opt.param_groups[0]["lr"],
                                         self.batchsize)
                        dump_outs += "\tloss: {:.4f}\n".format(running_loss_dict['loss']/verbose_steps)
                        dump_outs += "\tprob loss: {:.4f}\n".format(running_loss_dict['prob_loss']/verbose_steps)
                        dump_outs += "\taux prob loss: {:.4f}\n".format(running_loss_dict['aux_prob_loss']/verbose_steps)
                        dump_outs += "\townership loss: {:.4f}\n".format(running_loss_dict['ownership_loss']/verbose_steps)
                        dump_outs += "\twdl loss: {:.4f}\n".format(running_loss_dict['wdl_loss']/verbose_steps)
                        dump_outs += "\tstm loss: {:.4f}\n".format(running_loss_dict['stm_loss']/verbose_steps)
                        dump_outs += "\tfina score loss: {:.4f}".format(running_loss_dict['fina_score_loss']/verbose_steps)

                        print(dump_outs)
                        log_outs = "steps: {} -> loss: {:.4f}, speed: {:.2f} | opt: {}, learning rate: {}, batch size: {}".format(
                                       num_steps,
                                       running_loss_dict['loss']/verbose_steps,
                                       verbose_steps/elapsed,
                                       self.opt_name,
                                       self.learning_rate,
                                       self.batchsize)
                        log_file = os.path.join(self.store_path, "log.txt")
                        with open(log_file, 'a') as f:
                            f.write(log_outs + '\n')

                        running_loss_dict = get_running_loss_dict()

                # should we stop?
                if num_steps >= self.max_steps + init_steps:
                    keep_running = False
                    break

            # store the last network
            self.__store_current_status(num_steps)
        print("Training is over.")
