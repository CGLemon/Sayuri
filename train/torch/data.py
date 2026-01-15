import io
import numpy as np
from symmetry import numpy_symmetry_planes, numpy_symmetry_plane, numpy_symmetry_prob

V2_DATA_LINES = 53

'''
    The data format is defined below. Each v2 data package consists of 53 lines.

    ------- Version -------
     L1        : Version
     L2        : Mode
     
     ------- Inputs data -------
     L3        : Board Size
     L4        : Komi
     L5        : Rule
     L6        : Wave
     L7  - L43 : Binary Features
     L44       : Current Player
    
     ------- Prediction data -------
     L45       : Probabilities
     L46       : Auxiliary Probabilities
     L47       : Ownership
     L48       : Result
     L49       : Average Q Value, Short, Middel, Long
     L50       : Final Score
     L51       : Average Score Lead, Short, Middel, Long

     ------- Misc data -------
     L52       : Q Stddev, Score Stddev
     L53       : KLD
'''

class Data():
    STATE_UNINITIALIZED = 0   # Not initialized; no processing has started
    STATE_PROCESSING    = 1   # Data is currently being processed
    STATE_READY         = 2   # Processing finished; data is ready for use

    def __init__(self):
        self.status = self.STATE_UNINITIALIZED
        self.lines = list()
        self.weight = 1.0

        self.version = None
        self.mode = None

        self.board_size = None
        self.komi = None
        self.rule = None
        self.wave = None
        self.planes = list()
        self.to_move = None

        self.prob = list()
        self.aux_prob = list()
        self.ownership = list()

        self.result = None
        self.avg_q = None
        self.short_avg_q = None
        self.mid_avg_q = None
        self.long_avg_q = None

        self.final_score = None
        self.avg_score = None
        self.short_avg_score = None
        self.mid_avg_score = None
        self.long_avg_score = None

        self.q_stddev = None
        self.score_stddev = None

        self.kld = None

    def _hex_to_int(self, h):
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

    def _int2_to_int(self, v):
        if v == '0':
            return 0
        elif v == '1':
            return 1
        elif v == '2':
            return -2
        elif v == '3':
            return -1

    def _gether_binary_plane(self, board_size, readline):
        plane = []
        size = (board_size * board_size) // 4
        for i in range(size):
            plane.extend(self._hex_to_int(readline[i]))

        if board_size % 2 == 1:
            plane.append(int(readline[size]))

        plane = np.array(plane, dtype=np.int8)
        return plane

    def _get_probabilities(self, board_size, readline):
        values = readline.split()
        size = board_size * board_size + 1
        prob = np.zeros(size, dtype=np.float32)

        for index in range(size):
            prob[index] = float(values[index])
        return prob

    def _get_ownership(self, board_size, readline):
        size = board_size * board_size
        buf = np.zeros(size, dtype=np.int8)

        for index in range(size):
            buf[index] = self._int2_to_int(readline[index])
        return buf

    def _get_vals_list(self, readline):
        values = readline.split()
        size = len(values)
        buf = np.zeros(size, dtype=np.float32)

        for index in range(size):
            buf[index] = float(values[index])
        return buf

    def _fill_v2(self, idx, readline):
        linecnt = idx + 1

        if linecnt == 1:
            self.version = int(readline)
        elif linecnt == 2:
            self.mode = int(readline)
        elif linecnt == 3:
            self.board_size = int(readline)
        elif linecnt == 4:
            self.komi = float(readline)
        elif linecnt == 5:
            self.wave = float(readline)
        elif linecnt == 6:
            self.rule = float(readline)
        elif linecnt >= 7 and linecnt <= 43:
            plane = self._gether_binary_plane(self.board_size, readline)
            self.planes.append(plane)
        elif linecnt == 44:
            self.planes = np.array(self.planes)
            self.to_move = int(readline)
        elif linecnt == 45:
            self.prob = self._get_probabilities(self.board_size, readline)
        elif linecnt == 46:
            self.aux_prob = self._get_probabilities(self.board_size, readline)
        elif linecnt == 47:
            self.ownership = self._get_ownership(self.board_size, readline)
        elif linecnt == 48:
            self.result = int(readline)
        elif linecnt == 49:
            vals_list = self._get_vals_list(readline)
            self.avg_q = vals_list[0]
            self.short_avg_q = vals_list[1]
            self.mid_avg_q = vals_list[2]
            self.long_avg_q = vals_list[3]
        elif linecnt == 50:
            self.final_score = float(readline)
        elif linecnt == 51:
            vals_list = self._get_vals_list(readline)
            self.avg_score = vals_list[0]
            self.short_avg_score = vals_list[1]
            self.mid_avg_score = vals_list[2]
            self.long_avg_score = vals_list[3]
        elif linecnt == 52:
            vals_list = self._get_vals_list(readline)
            self.q_stddev = vals_list[0]
            self.score_stddev = vals_list[1]
        elif linecnt == 53:
            self.kld = float(readline)

    def apply_symmetry(self, symm):
        if self.status != self.STATE_READY:
            raise Exception("Data is not ready. Processing must be completed before applying symmetry.")

        channels       = self.planes.shape[0]
        bsize          = self.board_size

        self.planes    = np.reshape(self.planes, (channels, bsize, bsize))
        self.planes    = numpy_symmetry_planes(symm, self.planes)
        self.planes    = np.reshape(self.planes, (channels, bsize * bsize))

        self.ownership = np.reshape(self.ownership, (bsize, bsize))
        self.ownership = numpy_symmetry_plane(symm, self.ownership)
        self.ownership = np.reshape(self.ownership, (bsize * bsize))

        self.prob      = numpy_symmetry_prob(symm, self.prob)
        self.aux_prob  = numpy_symmetry_prob(symm, self.aux_prob)

    def load_from_stream(self, stream):
        # The load_from_stream() reads a single data entry from the input stream and
        # stores the raw strings into an internal buffer. At this stage, only the minimal
        # required information is parsed (e.g., version and KLD). The version is used to
        # verify whether the data format is supported, while KLD is required for policy
        # surprise sampler. Since the data may be discarded in later stages, fully parsing
        # all fields at this point would introduce unnecessary performance overhead. Therefore,
        # full parsing is deferred to a later stage.

        if self.status == self.STATE_READY:
            raise Exception("Data is already processed and ready, reloading is not allowed.")

        line = stream.readline()
        if len(line) == 0:
            return False # stream is end

        self.version = int(line)
        if self.version == 2:
            datalines = V2_DATA_LINES
        else:
            raise Exception("Unsupported data version {}. Should be version 2.".format(self.version))

        self.lines.clear()
        self.lines.append(line)
        for _ in range(1, datalines):
            self.lines.append(stream.readline())

        self.kld = float(self.lines[52])
        self.status = self.STATE_PROCESSING
        return True

    def parse(self):
        # After confirming that the current data will be used, all data is fully parsed.
        if self.status != self.STATE_PROCESSING:
            raise Exception("The data should load the stream first.")

        datalines = len(self.lines)
        for cnt in range(1, datalines):
            self._fill_v2(cnt, self.lines[cnt])
        self.status = self.STATE_READY

    def load_from_stream_and_parse(self, stream):
        if not self.load_from_stream(stream):
            return False
        self.parse()
        return True

if __name__ == "__main__":
    raw_data = """2
0
9
7
0
0
004081012620e84320000
0080021e18c8119404000
000000000008000000000
004081012620e84320000
0080021e18c0119404000
000000000000080000000
004081012620e04320000
0080021e18c0119404000
000000000000001000000
004081012620e04320000
0080021e18c0118404000
000080000000000000000
004001012620e04320000
0080021e18c0118404000
000002000000000000000
004001012620e04320000
0080001e18c0118404000
004000000000000000000
000001012620e04320000
0080001e18c0118404000
008000000000000000000
000001012620e04320000
0000001e18c0118404000
000000000000000020000
000000000000000000000
000000000000000000000
000000000000000000000
004081012620e84320000
0080021e18c8119404000
000000002000008000000
008000000000000400000
0000001000c0090024000
000000000020e04300000
000000002000000000000
008000000000000000000
000100000000000000000
000000004000000000000
0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0025 0.96 0 0.02375 0 0 0 0 0 0 0.01375 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0.0025 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.0025 0 0 0 0 0 0 0 0 0 0 0.0025 0 0 0 0 0 0 0 0 0.04125 0 0 0.94875 0 0 0 0 0 0 0 0 0.0025 0 0 0
333333333313333333311333333113333333111311333133311133113311113111111111111111111
1
0.779271 0.780061 0.781635 0.786311
2
1.53194 1.53094 1.52894 1.52294
0.125479 0.860285
0.344604"""

    stream = io.StringIO(raw_data)
    while 1:
        data = Data()
        success = data.load_from_stream_and_parse(stream)
        print("=================================")
        print("Load Success: {}".format(success))
        if not success:
          break
        print("Version: {}".format(data.version))
        print("Board Size: {}".format(data.board_size))
        print("Plane Shape: {}".format(data.planes.shape))
        print("Policy Shape: {}".format(data.prob.shape))
