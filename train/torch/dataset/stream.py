import torch
import numpy as np
import random, math, os, glob, io, gzip, sys

from .data import Data

def stdout_write(val):
    sys.stdout.write(val)
    sys.stdout.flush()

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

    if num_chunks:
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

        try:
            if filename.find(".gz") >= 0:
                with gzip.open(filename, "rt") as f:
                    stream = io.StringIO(f.read())
            else:
                with open(filename, "r") as f:
                    stream = io.StringIO(f.read())
        except:
            stdout_write("Could not open the file: {}\n".format(filename))
        return stream

class StreamParser:
    def __init__(self, down_sample_rate, policy_surprise_factor=0.0):
        # Use a random sample input data read. This helps improve the spread of
        # games in the shuffle buffer.
        self.down_sample_rate = down_sample_rate

        # We assume each game has 50 positions on average.
        self.virtual_buffsize = 8000 * 50
        self.running_kld_mean = 1.0
        self.policy_surprise_factor = policy_surprise_factor
        self.num_samples_per_proc = 0

    def _sample(self, data):
        # quick find out the KLD running average so we increase the factor value in
        # the early samples.
        gamma_factor = math.exp((max(self.virtual_buffsize - self.num_samples_per_proc, 0))/ \
                            (self.virtual_buffsize/2.71828182846))
        gamma = (1.0/self.virtual_buffsize) * gamma_factor
        self.running_kld_mean = (1.0 - gamma) * self.running_kld_mean + \
                                    gamma * data.kld
        self.num_samples_per_proc += 1

        # compute policy surprise sample probability
        surprise_freq = (1.0 - self.policy_surprise_factor) + \
                              self.policy_surprise_factor * (data.kld/self.running_kld_mean)
        sample_prob = surprise_freq * (1.0 / self.down_sample_rate)

        if self.num_samples_per_proc < self.virtual_buffsize:
            # be sure each worker sees enough games
            return False
        return sample_prob > random.uniform(0.0, 1.0)

    def func(self, stream):
        if not stream:
            return None

        while True:
            data = Data()

            if not data.load_from_stream(stream):
                return None # stream is end

            if self._sample(data):
                data.parse()
                break
        data.apply_symmetry(random.randint(0, 7))
        return data

class BatchGenerator:
    def __init__(self, boardsize, input_channels):
        self.nn_board_size = boardsize
        self.nn_num_intersections = self.nn_board_size * self.nn_board_size
        self.input_channels = input_channels

    def _wrap_data(self, data):
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
        all_q_vals = np.zeros(5)
        all_scores = np.zeros(5)

        buf = np.zeros(num_intersections)
        sqr_buf = np.zeros((nn_board_size, nn_board_size))

        # input planes
        for p in range(self.input_channels-6):
            plane = data.planes[p]
            input_planes[p, 0:board_size, 0:board_size] = np.reshape(plane, (board_size, board_size))[:, :]

        input_planes[self.input_channels-6, 0:board_size, 0:board_size] = data.rule
        input_planes[self.input_channels-5, 0:board_size, 0:board_size] = data.wave
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

        # all q values
        all_q_vals[0] = data.result
        all_q_vals[1] = data.avg_q
        all_q_vals[2] = data.short_avg_q
        all_q_vals[3] = data.mid_avg_q
        all_q_vals[4] = data.long_avg_q

        # all scores
        all_scores[0] = data.final_score
        all_scores[1] = data.avg_score
        all_scores[2] = data.short_avg_score
        all_scores[3] = data.mid_avg_score
        all_scores[4] = data.long_avg_score

        # weight value
        weight = data.weight

        return (
            input_planes,
            prob,
            aux_prob,
            ownership,
            wdl,
            all_q_vals,
            all_scores,
            weight
        )

    def func(self, data_list):
        batch_planes = list()
        batch_prob = list()
        batch_aux_prob = list()
        batch_ownership = list()
        batch_wdl = list()
        batch_q_vals = list()
        batch_scores = list()
        batch_weight = list()

        for data in data_list:
            planes, prob, aux_prob, ownership, wdl, q_vals, scores, weight = self._wrap_data(data)

            batch_planes.append(planes)
            batch_prob.append(prob)
            batch_aux_prob.append(aux_prob)
            batch_ownership.append(ownership)
            batch_wdl.append(wdl)
            batch_q_vals.append(q_vals)
            batch_scores.append(scores)
            batch_weight.append(weight)

        batch_dict = {
            "planes"    : torch.from_numpy(np.array(batch_planes)).float(),
            "prob"      : torch.from_numpy(np.array(batch_prob)).float(),
            "aux_prob"  : torch.from_numpy(np.array(batch_aux_prob)).float(),
            "ownership" : torch.from_numpy(np.array(batch_ownership)).float(),
            "wdl"       : torch.from_numpy(np.array(batch_wdl)).float(),
            "q_vals"    : torch.from_numpy(np.array(batch_q_vals)).float(),
            "scores"    : torch.from_numpy(np.array(batch_scores)).float(),
            "weight"    : torch.from_numpy(np.array(batch_weight)).float(),
        }
        return batch_dict
