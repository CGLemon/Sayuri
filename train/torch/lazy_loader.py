import multiprocessing as mp
import threading
import random

class ShuffleBuffer:
    def __init__(self, buf_size):
        self._buf = list()
        self._max_buf_size = buf_size
        assert self._max_buf_size >= 1, "Buffer size must be greater than zero."

    def insert_item_and_pop(self, item):
        size = len(self._buf)

        if size > 4:
            i = random.randint(0, size-1)

            # Apply Fisher-Yates shuffle algorithm. Efficiently shuffle
            # the random buffer.
            self._buf[i], item = item, self._buf[i]

        if size < self._max_buf_size:
            self._buf.append(item)
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

        # Use a random sample input data reader. This helps improve the spread of
        # games in the shuffle buffer.
        self.rate = down_sample_rate

        assert len(filenames) != 0, ""

    def _open_new_stream(self):
        if len(self.tasks) == 0:
            self.tasks, self.done = self.done, self.tasks
            random.shuffle(self.tasks)

        filename = self.tasks.pop()
        self.done.append(filename)

        return self.loader.func(filename)

    def next(self):
        while True:
            if self.stream is None:
                self.stream = self._open_new_stream()

            data = self.parser.func(self.stream)

            if data is None:
                # The stream is end. Open the new stream next time.
                self.stream = None
                continue

            if self.rate > 1:
                # Apply the down-sample.
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
        self.flag = None

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

class LoaderFlag:
    NONE = 0
    STOP = 1

    def __init__(self):
        self.flag = mp.Value("i", self.NONE)

    def is_stop(self):
        with self.flag.get_lock():
            v = self.flag.value
        return v == self.STOP

    def reset_flag(self):
        with self.flag.get_lock():
            self.flag.value = self.NONE

    def set_stop_flag(self):
        with self.flag.get_lock():
            self.flag.value = self.STOP

def _load_from_files(config, data_writer):
    # Load the data from disk. Suggest to design a heavy stream parser instead 
    # of heavy batch generator. It is because that N workers execute the 
    # parser function, only one worker executes generator function.

    loader = DataLoader(
                 filenames = config.filenames,
                 data_writer = data_writer,
                 down_sample_rate = config.down_sample_rate,
                 stream_loader = config.stream_loader,
                 stream_parser = config.stream_parser
             )

    while True:
        if config.flag.is_stop():
            data_writer.close()
            break
        loader.next()

def _gather_batch(config, data_readers, batch_writer):
    shuf_buff = ShuffleBuffer(config.buffer_size)
    batch_gen = config.batch_generator

    stop = False
    while not stop:
        # Fill the buffer until it is full.
        for r in data_readers:
            try:
                item = r.recv()
                outs = shuf_buff.insert_item_and_pop(item)
                if outs is not None:
                    stop = True
            except:
                return

    # Now, start to prepare the batch. It significantly improve
    # the loader performanc.
    while True:
        if config.flag.is_stop():
            batch_writer.close()
            break

        data_list = list()

        while len(data_list) < config.batch_size:
            for r in data_readers:
                try:
                    item = r.recv()
                    outs = shuf_buff.insert_item_and_pop(item)
                    if outs is not None:
                        data_list.append(outs)
                except:
                    return

        # Send the batch.
        batch = batch_gen.func(data_list)
        try:
            batch_writer.send(batch)
        except:
            return

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
    config.flag = kwargs.get("flag", LoaderFlag())

    if not config.valid():
        print("Config is invalid. Please check your setting.")
        return None

    proc_list = list()
    data_readers = list()
    batch_reader, batch_writer = mp.Pipe(duplex=False)

    for _ in range(config.num_workers):
        # One process uses one pipe.
        data_reader, data_writer = mp.Pipe(duplex=False)
        data_readers.append(data_reader)

        # Create one SMP process.
        p = mp.Process(
                target=_load_from_files,
                args=(config, data_writer),
                daemon=True
            )
        p.start()
        proc_list.append(p)
        data_writer.close()

    t = threading.Thread(
            target=_gather_batch,
            args=(config, data_readers, batch_writer),
            daemon=True
        )
    t.start()

    # Do not close it because the thread and main thread share the same
    # writer.
    # batch_writer.close()

    while True:
        if config.flag.is_stop():
            for idx, p in enumerate(proc_list):
                while p.is_alive():
                    try:
                        _ = data_readers[idx].recv()
                        continue
                    except:
                        pass
            t.join()
            batch_reader.close()
            return
        try:
            batch = batch_reader.recv()
            yield batch
        except:
            return
