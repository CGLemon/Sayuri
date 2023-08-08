import json
import os
import torch

class Config:
    def __init__(self):
        self.num_workers = None
        self.use_gpu = None
        self.batchsize = None
        self.buffersize = None
        self.macrobatchsize = None
        self.macrofactor = None
        self.lr_schelude = None
        self.weight_decay = None
        self.train_dir = None
        self.validation_dir = None
        self.verbose_steps = None
        self.steps_per_epoch = None
        self.validation_steps = None
        self.max_steps = None
        self.fixup_batch_norm = None
        self.store_path = None
        self.down_sample_rate = None
        self.stack = []
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None
        self.optimizer = None
        self.nntype = None
        self.input_channels = None
        self.input_features = None
        self.boardsize = None
        self.num_chunks = None

def parse_training_config(json_data, config):
    train = json_data.get("Train", None)

    config.optimizer = train.get("Optimizer", "SGD")
    config.use_gpu = train.get("UseGPU", None)
    config.weight_decay = train.get("WeightDecay", 1e-4)
    config.lr_schedule = train.get("LearningRateSchedule", [[0, 0.2]])

    config.train_dir = train.get("TrainDirectory", None)
    config.validation_dir = train.get("ValidationDirectory", None)
    config.store_path = train.get("StorePath", None)
    config.batchsize = train.get("BatchSize", 512)
    config.buffersize = train.get("BufferSize", 16 * 1000)
    config.macrofactor = train.get("MacroFactor", 1)
    config.macrobatchsize = config.batchsize // config.macrofactor

    config.num_workers = train.get("Workers", max(os.cpu_count()-2, 1))
    config.steps_per_epoch = train.get("StepsPerEpoch", 1000)
    config.validation_steps = train.get("ValidationSteps", 100)
    config.verbose_steps = train.get("VerboseSteps", 1000)
    config.max_steps = train.get("MaxStepsPerRunning", 16384000)
    config.fixup_batch_norm = train.get("FixUpBatchNorm", False)
    config.down_sample_rate = train.get("DownSampleRate", 16)
    config.num_chunks  = train.get("NumberChunks", None)

    assert config.train_dir != None, ""
    assert config.store_path != None, ""

    if config.use_gpu == None:
        if torch.cuda.is_available():
            config.use_gpu = True

    return config

def parse_nn_config(json_data, config):
    network = json_data.get("NeuralNetwork", None)

    config.boardsize = network.get("MaxBoardSize", 19)

    config.nntype = network.get("NNType", None)
    config.input_channels = network.get("InputChannels", 43)
    config.residual_channels = network.get("ResidualChannels", None)
    config.policy_extract = network.get("PolicyExtract", None)
    config.value_extract = network.get("ValueExtract", None)

    assert config.input_channels != None, ""
    assert config.residual_channels != None, ""
    assert config.policy_extract != None, ""
    assert config.value_extract != None, ""

    stack = network.get("Stack", None)
    for s in stack:
        config.stack.append(s)
    return config

def json_loader(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def gather_config(filename):
    cfg = Config()
    if filename != None:
        data = json_loader(filename)
        cfg = parse_training_config(data, cfg)
        cfg = parse_nn_config(data, cfg)
    return cfg
