import multiprocessing as mp
import threading
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
    def __init__(self, filenames, data_writer, down_sample_rate, stream_loader, stream_parser):
        self.done = filenames
        self.tasks = list()

        self.parser = stream_parser
        self.loader = stream_loader
        self.writer = data_writer
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

            self.writer.send(data)
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

def __load_from_files(config, data_writer):
    loader = DataLoader(
                 filenames = config.filenames,
                 data_writer = data_writer,
                 down_sample_rate = config.down_sample_rate,
                 stream_loader = config.stream_loader,
                 stream_parser = config.stream_parser
             )

    while True:
        loader.next()

def __gather_batch(config, data_readers, batch_writer):
    shuf_buff = ShuffleBuffer(config.buffer_size)
    batch_gen = config.batch_generator

    stop = False
    while not stop:
        # fill the buffer until it is full
        for r in data_readers:
            item = r.recv()
            outs = shuf_buff.insert_item_and_pop(item)
            if outs is not None:
                stop = True

    while True:
        data_list = list()

        while len(data_list) < config.batch_size:
            for r in data_readers:
                item = r.recv()
                outs = shuf_buff.insert_item_and_pop(item)
                if outs is not None:
                    data_list.append(outs)

        batch = batch_gen.func(data_list)
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

    if not config.valid():
        return None

    data_readers = list()
    batch_reader, batch_writer = mp.Pipe(duplex=False)

    for _ in range(config.num_workers):
        data_reader, data_writer = mp.Pipe(duplex=False)
        data_readers.append(data_reader)

        mp.Process(
            target=__load_from_files,
            args=(config, data_writer),
            daemon=True
        ).start()
        data_writer.close()

    threading.Thread(
        target=__gather_batch,
        args=(config, data_readers, batch_writer),
        daemon=True
    ).start()
    # batch_writer.close()

    while True:
        batch = batch_reader.recv()
        yield batch
