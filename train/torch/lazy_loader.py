import multiprocessing as mp
import random

class ShuffleBuffer:
    def __init__(self, buf_size):
        self.__buf = list()
        self.buf_size = buf_size
        assert self.buf_size > 4, ""

    def insert_item_and_pop(self, item):
        size = len(self.__buf)

        if size > 4:
            i = random.randint(0, size-1)
            self.__buf[i], item = item, self.__buf[i]

        if size < self.buf_size:
            self.__buf.append(item)
            return None
        return item

class DataLoader:
    def __init__(self, filenames, data_queue, down_sample_rate, stream_loader, stream_parser):
        self.done = filenames
        self.tasks = list()

        self.parser = stream_parser
        self.loader = stream_loader
        self.queue = data_queue
        self.stream = None

        # Use a random sample input data read. This helps improve the spread of
        # games in the shuffle buffer.
        self.rate = down_sample_rate

        assert len(filenames) != 0, ""

    def __open_new_stream(self):
        if len(self.tasks) == 0:
            self.tasks, self.done = self.done, self.tasks
            random.shuffle(self.tasks)

        filename = self.tasks.pop()
        self.done.append(filename)

        return self.loader.func(filename)

    def next(self):
        while True:
            if self.stream is None:
                self.stream = self.__open_new_stream()

            data = self.parser.func(self.stream)

            if data is None:
                self.stream = None
                continue

            if self.rate > 1:
                if random.randint(0, self.rate-1) != 0:
                    continue

            self.queue.put(data, block=True, timeout=None)
            break

class LoaderConfig:
    def __init__(self):
        self.filenames = list()
        self.stream_loader = None
        self.stream_parser = None
        self.batch_generator = None
        self.down_sample_rate = 16
        self.num_workers = 0
        self.buffer_size = 0
        self.batch_size = 0

    def valid(self):
        if len(self.filenames) <= 0 or \
            self.stream_loader is None or \
            self.stream_parser is None or \
            self.batch_generator is None or \
            self.num_workers <= 0 or \
            self.buffer_size <= 0 or \
            self.batch_size <= 0:
            return False
        return True

def __load_from_files(config, data_queue):
    loader = DataLoader(
                 filenames = config.filenames,
                 data_queue = data_queue,
                 down_sample_rate = config.down_sample_rate,
                 stream_loader = config.stream_loader,
                 stream_parser = config.stream_parser
             )

    while True:
        loader.next()

def __gather_batch(config, data_queue, batch_writer):
    shuf_buff = ShuffleBuffer(config.buffer_size)
    batch_gen = config.batch_generator

    while True:
        # fill the buffer until it is full
        item = data_queue.get(block=True, timeout=None)
        outs = shuf_buff.insert_item_and_pop(item)
        if outs is not None:
            break

    while True:
        data_list = list()

        while len(data_list) < config.batch_size:
            item = data_queue.get(block=True, timeout=None)
            outs = shuf_buff.insert_item_and_pop(item)
            if outs is not None:
                data_list.append(outs)

        batch = batch_gen.func(data_list)
        # batch_queue.put(batch, block=True, timeout=None)
        batch_writer.send(batch)

def LazyLoader(*args, **kwargs):
    config = LoaderConfig()

    config.filenames = kwargs.get("filenames", list())
    config.stream_loader = kwargs.get("stream_loader", None)
    config.stream_parser = kwargs.get("stream_parser", None)
    config.batch_generator = kwargs.get("batch_generator",None)
    config.down_sample_rate = kwargs.get("down_sample_rate",  0)
    config.num_workers = kwargs.get("num_workers", 0)
    config.buffer_size = kwargs.get("buffer_size", 0)
    config.batch_size = kwargs.get("batch_size", 0)
    data_queue_factor = kwargs.get("data_queue_factor", 16)
    batch_queue_factor = kwargs.get("batch_queue_factor", 16)

    if not config.valid():
        return None

    data_que_size = data_queue_factor * config.batch_size * config.num_workers
    data_queue = mp.Queue(maxsize=data_que_size)

    reader, write = mp.Pipe(duplex=False)
    # batch_queue = mp.Queue(maxsize=batch_queue_factor)

    for _ in range(config.num_workers):
        # N workers read the data from files and write the data
        # to queue.
        mp.Process(
            target=__load_from_files,
            args=(config, data_queue),
            daemon=True
        ).start()

    # One worker read the data from queue.
    mp.Process(
        target=__gather_batch,
        args=(config, data_queue, write),
        daemon=True
    ).start()

    while True:
        # batch = batch_queue.get(block=True, timeout=None)
        batch = reader.recv()
        yield batch
