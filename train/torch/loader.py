import numpy as np
import glob, os, random, time, io, gzip

FIXED_DATA_VERSION = 1
DATA_LINES = 45

'''
------- claiming -------
 L1       : Version
 L2       : Mode
 
 ------- Inputs data -------
 L3       : Board size
 L4       : Komi
 L5  - L38: Binary features
 L39      : Current Player

 ------- Prediction data -------
 L40      : Probabilities
 L41      : Auxiliary probabilities
 L42      : Ownership
 L43      : Result
 L44      : Q value
 L45      : Final score

'''

class Data():
    def __init__(self):
        super().__init__()
        self.version = FIXED_DATA_VERSION

        self.board_size = None
        self.komi = None
        self.planes = []
        self.to_move = None
        self.prob = []
        self.aux_prob = []
        self.ownership = []

        self.result = None
        self.q_value = None
        self.final_score = None

    def hex_to_int(self, h):
        if h == '0':
            return [0, 0, 0, 0]
        elif h == '1':
            return [1, 0, 0, 0]
        elif h == '2':
            return [0, 1, 0, 0]
        elif h == '3':
            return [1, 1, 0, 0]
        elif h == '4':
            return [0, 0, 1, 0]
        elif h == '5':
            return [1, 0, 1, 0]
        elif h == '6':
            return [0, 1, 1, 0]
        elif h == '7':
            return [1, 1, 1, 0]
        elif h == '8':
            return [0, 0, 0, 1]
        elif h == '9':
            return [1, 0, 0, 1]
        elif h == 'a':
            return [0, 1, 0, 1]
        elif h == 'b':
            return [1, 1, 0, 1]
        elif h == 'c':
            return [0, 0, 1, 1]
        elif h == 'd':
            return [1, 0, 1, 1]
        elif h == 'e':
            return [0, 1, 1, 1]
        elif h == 'f':
            return [1, 1, 1, 1]

    def int2_to_int(self, v):
        if v == '0':
            return 0
        elif v == '1':
            return 1
        elif v == '2':
            return -2
        elif v == '3':
            return -1

    def gether_binary_plane(self, board_size, readline):
        plane = []
        size = (board_size * board_size) // 4
        for i in range(size):
            plane.extend(self.hex_to_int(readline[i]))

        if board_size % 2 == 1:
            plane.append(int(readline[size]))

        plane = np.array(plane, dtype=np.int8)
        return plane

    def get_probabilities(self, board_size, readline):
        value = readline.split()
        prob = np.zeros(board_size * board_size + 1, dtype=np.float32)

        if len(value) == 1:
             index = int(value[0])
             prob[index] = 1
        else:
             assert len(value) == board_size * board_size + 1, ""
             for index in range(len(value)):
                 prob[index] = float(value[index])
        return prob

    def get_ownership(self, board_size, readline):
        ownership = np.zeros(board_size * board_size, dtype=np.int8)

        for index in range(board_size * board_size):
            ownership[index] = self.int2_to_int(readline[index])
        return ownership

    def fill_v1(self, linecnt, readline):
        if linecnt == 0:
            v = int(readline)
            assert v == FIXED_DATA_VERSION, "The data is not correct version."
        elif linecnt == 1:
            m = int(readline)
        elif linecnt == 2:
            self.board_size = int(readline)
        elif linecnt == 3:
            self.komi = float(readline)
        elif linecnt >= 4 and linecnt <= 37:
            plane = self.gether_binary_plane(self.board_size, readline)
            self.planes.append(plane)
        elif linecnt == 38:
            self.planes = np.array(self.planes)
            self.to_move = int(readline)
        elif linecnt == 39:
            self.prob = self.get_probabilities(self.board_size, readline)
        elif linecnt == 40:
            self.aux_prob = self.get_probabilities(self.board_size, readline)
        elif linecnt == 41:
            self.ownership = self.get_ownership(self.board_size, readline)
        elif linecnt == 42:
            self.result = int(readline)
        elif linecnt == 43:
            self.q_value = float(readline)
        elif linecnt == 44:
            self.final_score = float(readline)

    @staticmethod
    def get_datalines(version):
        if version == 1:
            return DATA_LINES
        return 0

    def __str__(self):
        out = str()
        out += "Board size: {}\n".format(self.board_size)
        out += "Side to move: {}\n".format(self.to_move)
        out += "Komi: {}\n".format(self.komi)
        out += "Result: {}\n".format(self.result)
        out += "Q value: {}\n".format(self.q_value)
        out += "Final score: {}\n".format(self.final_score)

        for p in range(len(self.planes)):
            out += "Plane: {}".format(p+1)
            out += str(self.planes[p])
            out += "\n"

        out += "Probabilities: "
        out += str(self.prob)
        out += "\n"

        out += "Auxiliary probabilities: "
        out += str(self.aux_prob)
        out += "\n"

        out += "Ownership: "
        out += str(self.ownership)
        out += "\n"

        return out

class LazyLoader:
    def __init__(self, dirname):
        self.dirname = dirname
        self.chunks = []

        self.stream = None
        self.buffer_size = 10000
        self.reload_size = 1000
        self.reload_thes = self.buffer_size - self.reload_size

        self.shuf_buf = [] # shuffle buffer.

        # Use a random sample input data read. This helps improve the spread of
        # games in the shuffle buffer.
        self.down_sample_rate = 16

        def gather_recursive_files(root):
            l = list()
            for name in glob.glob(os.path.join(root, '*')):
                if os.path.isdir(name):
                    l.extend(gather_recursive_files(name))
                else:
                    l.append(name)
            return l

        self.done = gather_recursive_files(self.dirname)

    def __fill_buf(self, stream):
        while len(self.shuf_buf) < self.buffer_size:
            datalines = Data.get_datalines(FIXED_DATA_VERSION);
            data_str = []

            for cnt in range(datalines):
                line = stream.readline()
                if len(line) == 0:
                    return True # stream is end
                else:
                    data_str.append(line)

            if self.down_sample_rate > 1:
                if random.randint(0, self.down_sample_rate-1) == 0:
                    self.shuf_buf.append(data_str)
            else:
                self.shuf_buf.append(data_str)
        return False # stream is not end

    def __open_new_stream(self):
        if len(self.chunks) == 0:
            self.chunks, self.done = self.done, self.chunks
            random.shuffle(self.chunks)

        filename = self.chunks.pop()
        self.done.append(filename)

        stream = None
        if filename.find(".gz") >= 0:
            with gzip.open(filename, 'rt') as f:
                stream = io.StringIO(f.read())
        else:
            with open(filename, 'r') as f:
                stream = io.StringIO(f.read())
        return stream

    def __reload(self):
        while len(self.shuf_buf) < self.buffer_size:
            if self.stream is None:
                self.stream = self.__open_new_stream()

            # TODO: Performance is bad if the network is small. Maybe we load too
            #       many data strings at the same time.
            stream_end = self.__fill_buf(self.stream)

            if stream_end:
                # Current stream is end. Open the new one
                # next turn.
                self.stream = None

        if len(self.shuf_buf) > 1:
            random.shuffle(self.shuf_buf)

    def next(self):
        if len(self.shuf_buf) < self.reload_thes:
            self.__reload()

        data_str = self.shuf_buf.pop()
        data = Data()

        for cnt in range(data.get_datalines(FIXED_DATA_VERSION)):
            line = data_str[cnt]
            data.fill_v1(cnt, line)

        return data
