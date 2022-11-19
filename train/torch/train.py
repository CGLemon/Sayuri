import torch
import torch.nn.functional as F
import numpy as np
import random, time, math, os, glob, io, gzip

from network import Network
from data import Data, FIXED_DATA_VERSION

from torch.nn import DataParallel
from lazy_loader import LazyLoader

def gather_filenames(root, num_chunks=None, sort_key_fn=None):
    def gather_recursive_files(root):
        l = list()
        for name in glob.glob(os.path.join(root, "*")):
            if os.path.isdir(name):
                l.extend(gather_recursive_files(name))
            else:
                l.append(name)
        return l
    chunks = gather_recursive_files(root)

    if num_chunks is not None:
        if len(chunks) > num_chunks:
            if sort_key_fn is None:
                random.shuffle(chunks)
            else:
                chunks.sort(key=sort_key_fn, reverse=True)
            chunks = chunks[:num_chunks]
    return chunks


class StreamLoader:
    def __init__(self):
        pass

    def func(self, filename):
        stream = None
        if not os.path.isfile(filename):
            return stream

        if filename.find(".gz") >= 0:
            with gzip.open(filename, 'rt') as f:
                stream = io.StringIO(f.read())
        else:
            with open(filename, 'r') as f:
                stream = io.StringIO(f.read())
        return stream

class StreamParser:
    def __init__(self, down_sample_rate):
        # Use a random sample input data read. This helps improve the spread of
        # games in the shuffle buffer.
        self.down_sample_rate = down_sample_rate

    def func(self, stream):
        if stream is None:
            return None

        datalines = Data.get_datalines(FIXED_DATA_VERSION);
        data_str = []

        while True:
            for cnt in range(datalines):
                line = stream.readline()
                if len(line) == 0:
                    return None # stream is end
                else:
                    data_str.append(line)

            if self.down_sample_rate > 1:
                if random.randint(0, self.down_sample_rate-1) != 0:
                    data_str = []
                    continue
            break

        data = Data()

        for cnt in range(datalines):
            line = data_str[cnt]
            data.fill_v1(cnt, line)

        data.apply_symmetry(random.randint(0, 7))

        return data

class BatchGenerator:
    def __init__(self, boardsize, input_channels):
        self.nn_board_size = boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = input_channels

    def __wrap_data(self, data):
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
            input_planes,
            prob,
            aux_prob,
            ownership,
            wdl,
            stm,
            final_score
        )

    def func(self, data_list):
        batch_bsize = list()
        batch_planes = list()
        batch_prob = list()
        batch_aux_prob = list()
        batch_ownership = list()
        batch_wdl = list()
        batch_stm = list()
        batch_score = list()

        for data in data_list:
            bsize, planes, prob, aux_prob, ownership, wdl, stm, score = self.__wrap_data(data)

            batch_bsize.append(bsize)
            batch_planes.append(planes)
            batch_prob.append(prob)
            batch_aux_prob.append(aux_prob)
            batch_ownership.append(ownership)
            batch_wdl.append(wdl)
            batch_stm.append(stm)
            batch_score.append(score)

        return (
            batch_bsize,
            torch.tensor(np.array(batch_planes)).float(),
            torch.tensor(np.array(batch_prob)).float(),
            torch.tensor(np.array(batch_aux_prob)).float(),
            torch.tensor(np.array(batch_ownership)).float(),
            torch.tensor(np.array(batch_wdl)).float(),
            torch.tensor(np.array(batch_stm)).float(),
            torch.tensor(np.array(batch_score)).float()
        )

class TrainingPipe():
    def __init__(self, cfg):
        self.cfg = cfg

        # The mini-batch size, update the network per batch size
        self.batchsize =  cfg.batchsize

        # The marco batch size and factor, (marco batch size) * factor = batch size
        self.macrobatchsize = cfg.macrobatchsize
        self.macrofactor = cfg.macrofactor

        # How many cpu does the 'DataLoader' use?
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir
        self.validation_dir = cfg.validation_dir

        self.num_chunks = cfg.num_chunks

        # The stpes of storing the last model and validating it per epoch.
        self.steps_per_epoch =  cfg.steps_per_epoch

        # Report the information per this steps.
        self.verbose_steps = max(100, cfg.verbose_steps)

        self.validation_steps = cfg.validation_steps

        # Lazy loader options.
        self.train_buffer_size = self.cfg.buffersize
        self.validation_buffer_size = self.cfg.buffersize // 10
        self.down_sample_rate = cfg.down_sample_rate

        # Max steps per training task.
        self.max_steps =  cfg.max_steps

        # Which optimizer do we use?
        self.opt_name = cfg.optimizer

        # Optimizer's parameters.
        self.weight_decay = cfg.weight_decay
        self.lr_schedule = cfg.lr_schedule

        # The training device.
        self.use_gpu = cfg.use_gpu
        self.device = torch.device('cpu')
        if self.use_gpu:
            self.device = torch.device('cuda')
        self.net = Network(cfg)
        self.net.trainable(True)

        # The Store root dir
        self.store_path = cfg.store_path

        self.__setup()

    def __setup(self):
        self.module = self.net # linking

        if self.use_gpu:
            self.net = self.net.to(self.device)
            self.net = DataParallel(self.net) 
            self.module  = self.net.module

        init_lr = self.__get_lr_schedule(0)

        # We may fail to load the optimizer. So init
        # it before loading it.
        self.opt = None
        if self.opt_name == "Adam":
            self.opt = torch.optim.Adam(
                self.net.parameters(),
                lr=init_lr,
                weight_decay=self.weight_decay,
            )
        elif self.opt_name == "SGD":
            # Recommanded optimizer, the SGD is better than Adam
            # in this kind of training task.
            self.opt = torch.optim.SGD(
                self.net.parameters(),
                lr=init_lr,
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

    def __get_lr_schedule(self, num_steps):
        # Get the current learning rate with schedule.
        curr_lr = 0.2
        for s, lr in self.lr_schedule:
            if s <= num_steps:
                curr_lr = lr
            else:
                break
        return curr_lr

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
        curr_lr = self.__get_lr_schedule(last_steps)

        for param in self.opt.param_groups:
            param["lr"] = curr_lr
            param["weight_decay"] = self.weight_decay

        print("Current steps is {}, learning rate is {}".format(last_steps, curr_lr))

        return last_steps

    def __store_current_status(self, steps):
        steps_name = os.path.join(self.store_path, "last_steps.txt")
        with open(steps_name, 'w') as f:
            f.write(str(steps))

        self.validate_the_last_model(steps)

        model_path = os.path.join(self.store_path, "model")
        model_name = os.path.join(model_path, "s{}.pt".format(steps))
        torch.save(self.module.state_dict(), model_name)

        opt_path = os.path.join(self.store_path, "opt")
        opt_name = os.path.join(opt_path, "s{}.pt".format(steps))
        torch.save(self.opt.state_dict(), opt_name)

    def __init_loader(self):
        self.__stream_loader = StreamLoader()
        self.__stream_parser = StreamParser(self.down_sample_rate)
        self.__batch_gen = BatchGenerator(self.cfg.boardsize, self.cfg.input_channels)

        sort_fn = os.path.getmtime
        chunks = gather_filenames(self.train_dir, self.num_chunks, sort_fn)

        self.train_lazy_loader = LazyLoader(
            filenames = chunks,
            stream_loader = self.__stream_loader,
            stream_parser = self.__stream_parser,
            batch_generator = self.__batch_gen,
            down_sample_rate = 0,
            num_workers = self.num_workers,
            buffer_size = self.train_buffer_size,
            batch_size = self.macrobatchsize
        )

        # Try to get the first batch, be sure that the loader is ready.
        batch = next(self.train_lazy_loader)

        if self.validation_dir is not None:
            self.validation_lazy_loader = LazyLoader(
                filenames = gather_filenames(self.validation_dir, len(chunks)//10, sort_fn),
                stream_loader = self.__stream_loader,
                stream_parser = self.__stream_parser,
                batch_generator = self.__batch_gen,
                down_sample_rate = 0,
                num_workers = self.num_workers,
                buffer_size = self.validation_buffer_size,
                batch_size = self.macrobatchsize
            )
            batch = next(self.validation_lazy_loader)
        else:
            self.validation_lazy_loader = None


    def get_new_running_loss_dict(self):
        # Get the new dict.
        running_loss_dict = dict()
        running_loss_dict['loss'] = 0
        running_loss_dict['prob_loss'] = 0
        running_loss_dict['aux_prob_loss'] = 0
        running_loss_dict['ownership_loss'] = 0
        running_loss_dict['wdl_loss'] = 0
        running_loss_dict['stm_loss'] = 0
        running_loss_dict['final_score_loss'] = 0
        return running_loss_dict

    def gather_data_from_loader(self, use_training=True):
        # Fetch the next batch data from disk.
        if use_training:
            batch = next(self.train_lazy_loader)
        else:
            batch = next(self.validation_lazy_loader)

        _, planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score = batch

        # Move the data to the current device.
        if self.use_gpu:
            planes = planes.to(self.device)
            target_prob = target_prob.to(self.device)
            target_aux_prob = target_aux_prob.to(self.device)
            target_ownership = target_ownership.to(self.device)
            target_wdl = target_wdl.to(self.device)
            target_stm = target_stm.to(self.device)
            target_score = target_score.to(self.device)

        # Gather batch data
        target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score)
        return planes, target

    def validate_the_last_model(self, steps):
        if self.validation_lazy_loader is None:
            return

        self.module.trainable(False)
        running_loss_dict = self.get_new_running_loss_dict()
        total_steps = self.validation_steps * self.macrofactor

        clock_time = time.time()

        for _ in range(total_steps):
            planes, target = self.gather_data_from_loader(False)
            _, all_loss = self.net(planes, target, use_symm=True)
            prob_loss, aux_prob_loss, ownership_loss, wdl_loss, stm_loss, final_score_loss = all_loss
            
            loss = prob_loss + aux_prob_loss + ownership_loss + wdl_loss + stm_loss + final_score_loss

            # accumulate loss
            running_loss_dict['loss'] += loss.mean().item()
            running_loss_dict['prob_loss'] += prob_loss.mean().item()
            running_loss_dict['aux_prob_loss'] += aux_prob_loss.mean().item()
            running_loss_dict['ownership_loss'] += ownership_loss.mean().item()
            running_loss_dict['wdl_loss'] += wdl_loss.mean().item()
            running_loss_dict['stm_loss'] += stm_loss.mean().item()
            running_loss_dict['final_score_loss'] += final_score_loss.mean().item()

        log_outs = "steps: {} -> ".format(steps)
        log_outs += "speed: {:.2f}, opt: {}, learning rate: {}, batch size: {}\n".format(
                        self.validation_steps/(time.time() - clock_time),
                        self.opt_name,
                        self.opt.param_groups[0]["lr"],
                        self.batchsize)
        log_outs += "\tloss: {:.4f}\n".format(running_loss_dict['loss']/total_steps)
        log_outs += "\tprob loss: {:.4f}\n".format(running_loss_dict['prob_loss']/total_steps)
        log_outs += "\taux prob loss: {:.4f}\n".format(running_loss_dict['aux_prob_loss']/total_steps)
        log_outs += "\townership loss: {:.4f}\n".format(running_loss_dict['ownership_loss']/total_steps)
        log_outs += "\twdl loss: {:.4f}\n".format(running_loss_dict['wdl_loss']/total_steps)
        log_outs += "\tstm loss: {:.4f}\n".format(running_loss_dict['stm_loss']/total_steps)
        log_outs += "\tfinal score loss: {:.4f}".format(running_loss_dict['final_score_loss']/total_steps)

        log_file = os.path.join(self.store_path, "validation.log")
        with open(log_file, 'a') as f:
            f.write(log_outs + '\n')
        self.module.trainable(True)

    def fit_and_store(self):
        init_steps = self.__load_current_status()

        print("init loader...")
        self.__init_loader()
        print("start training...")

        running_loss_dict = self.get_new_running_loss_dict()
        num_steps = init_steps
        keep_running = True
        macro_steps = 0

        clock_time = time.time()

        while keep_running:
            for _ in range(self.steps_per_epoch):
                planes, target = self.gather_data_from_loader(True)

                # forward and backforwad
                _, all_loss = self.net(planes, target, use_symm=True)

                prob_loss, aux_prob_loss, ownership_loss, wdl_loss, stm_loss, final_score_loss = all_loss

                # compute loss
                prob_loss = prob_loss.mean() / self.macrofactor
                aux_prob_loss = aux_prob_loss.mean() / self.macrofactor
                ownership_loss = ownership_loss.mean() / self.macrofactor
                wdl_loss = wdl_loss.mean() / self.macrofactor
                stm_loss = stm_loss.mean() / self.macrofactor
                final_score_loss = final_score_loss.mean() / self.macrofactor

                loss = prob_loss + aux_prob_loss + ownership_loss + wdl_loss + stm_loss + final_score_loss
                loss.backward()
                macro_steps += 1

                # accumulate loss
                running_loss_dict['loss'] += loss.item()
                running_loss_dict['prob_loss'] += prob_loss.item()
                running_loss_dict['aux_prob_loss'] += aux_prob_loss.item()
                running_loss_dict['ownership_loss'] += ownership_loss.item()
                running_loss_dict['wdl_loss'] += wdl_loss.item()
                running_loss_dict['stm_loss'] += stm_loss.item()
                running_loss_dict['final_score_loss'] += final_score_loss.item()

                if math.isnan(running_loss_dict['loss']):
                    print("The gradient is explosion. Stop the training...")
                    keep_running = False
                    break

                if macro_steps % self.macrofactor == 0:
                    # clip grad
                    if self.cfg.fixup_batch_norm:
                        gnorm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), 400.0)

                    # update network
                    self.opt.step()
                    self.opt.zero_grad()

                    # update learning rate
                    num_steps += 1
                    for param in self.opt.param_groups:
                        param["lr"] = self.__get_lr_schedule(num_steps) 

                    # dump the verbose
                    if num_steps % self.verbose_steps == 0:
                        elapsed = time.time() - clock_time
                        clock_time = time.time()

                        log_outs = "steps: {} -> ".format(num_steps)
                        log_outs += "speed: {:.2f}, opt: {}, learning rate: {}, batch size: {}\n".format(
                                        self.verbose_steps/elapsed,
                                        self.opt_name,
                                        self.opt.param_groups[0]["lr"],
                                        self.batchsize)
                        log_outs += "\tloss: {:.4f}\n".format(running_loss_dict['loss']/self.verbose_steps)
                        log_outs += "\tprob loss: {:.4f}\n".format(running_loss_dict['prob_loss']/self.verbose_steps)
                        log_outs += "\taux prob loss: {:.4f}\n".format(running_loss_dict['aux_prob_loss']/self.verbose_steps)
                        log_outs += "\townership loss: {:.4f}\n".format(running_loss_dict['ownership_loss']/self.verbose_steps)
                        log_outs += "\twdl loss: {:.4f}\n".format(running_loss_dict['wdl_loss']/self.verbose_steps)
                        log_outs += "\tstm loss: {:.4f}\n".format(running_loss_dict['stm_loss']/self.verbose_steps)
                        log_outs += "\tfinal score loss: {:.4f}".format(running_loss_dict['final_score_loss']/self.verbose_steps)
                        print(log_outs)

                        # Also save the current running loss.
                        log_file = os.path.join(self.store_path, "training.log")
                        with open(log_file, 'a') as f:
                            f.write(log_outs + '\n')

                        running_loss_dict = self.get_new_running_loss_dict()

                # should we stop it?
                if num_steps >= self.max_steps + init_steps:
                    keep_running = False
                    break

            # store the last network
            self.__store_current_status(num_steps)
        print("Training is over.")
