import json
import os
import torch

class Config:
    def __init__(self):
        # Training options
        self.num_workers = None
        self.use_gpu = None
        self.batchsize = None
        self.macrobatchsize = None
        self.macrofactor = None
        self.lr_schelude = None
        self.weight_decay = None
        self.train_dir = None
        self.steps_per_epoch = None
        self.max_steps = None
        self.fixup_batch_norm = None
        self.store_path = None

        # Adjustable values
        self.stack = []
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None
        self.optimizer = None

        # Fixed values but flexible
        self.nntype = None
        self.input_channels = None
        self.input_features = None

        self.boardsize = None
        self.value_misc = None

def parse_training_config(json_data, config):
    # We assume that every value is valid.
    train = json_data["Train"]

    config.optimizer = train["Optimizer"]
    config.use_gpu = train["UseGPU"]
    config.weight_decay = train["WeightDecay"]
    config.lr_schedule = train["LearningRateSchedule"]

    config.train_dir = train["TrainDirectory"]
    config.batchsize = train["BatchSize"]
    config.macrofactor = train["MacroFactor"]
    config.macrobatchsize = config.batchsize // config.macrofactor

    config.num_workers = train["Workers"]
    config.steps_per_epoch = train["StepsPerEpoch"]
    config.max_steps = train["MaxStepsPerRunning"]
    config.fixup_batch_norm = train["FixUpBatchNorm"]
    config.store_path = train["StorePath"]

    assert config.max_steps != None, ""

    if config.steps_per_epoch == None:
        config.steps_per_epoch = 500

    if config.weight_decay == None:
        config.weight_decay = 1e-4

    if config.num_workers == None:
        config.num_workers = os.cpu_count()

    if config.use_gpu == None:
        if torch.cuda.is_available():
            config.use_gpu = True

    return config

def parse_nn_config(json_data, config):
    # We assume that every value is valid.
    network = json_data["NeuralNetwork"]

    config.boardsize = network["MaxBoardSize"]
    
    config.nntype = network["NNType"]
    config.input_channels = network["InputChannels"]
    config.residual_channels = network["ResidualChannels"]
    config.policy_extract = network["PolicyExtract"]
    config.value_extract = network["ValueExtract"]
    config.value_misc = network["ValueMisc"]

    assert config.input_channels != None, ""
    assert config.residual_channels != None, ""
    assert config.policy_extract != None, ""
    assert config.value_extract != None, ""
    assert config.value_misc != None, ""

    stack = network["Stack"]
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
        d = json_loader(filename)
        cfg = parse_training_config(d, cfg)
        cfg = parse_nn_config(d, cfg)
    return cfg
