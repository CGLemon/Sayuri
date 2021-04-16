import json
import os
import torch

CONFIG_KEYWOED = [
    "NeuralNetwork",    # claiming
    "NNType",           # the type of net
    "Version",          # net version(implies the net structure)
    "DefaultBoardSize",
    "InputChannels",    # Input planes channels
    "PolicyExtract",    # policy shared head channels
    "ValueExtract",     # value shared head channels
    "ValueMisc",

    "UseOwnership",
    "UseFinalScore",
    "UseAuxiliaryPolicy",

    "Stack",            # the net structure(also implies the block number)
    "ResidualChannels", # each resnet block channels
    "ResidualBlock",    # resnet block without variant
    "ResidualBlock-SE", # resnet block with SE structure

    "Train",            # claiming
    "GPUs",
    "Epochs",
    "LearningRate",     # the learning rate
    "WeightDecay",      # the net weight decay
    "TrainDirectory",
    "ValidationDirectory",
    "TestDirectory"
]

class Config:
    def __init__(self):
        # Verbose option  
        self.debug_verbose = False
        self.misc_verbose = False

        # Training option
        self.num_workers = None
        self.gpus = None
        self.epochs = None
        self.batchsize = None
        self.learn_rate = None
        self.weight_decay = None
        self.train_dir = None
        self.val_dir = None
        self.test_dir = None

        # Adjustable values
        self.stack = []
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None

        # Options
        self.use_ownership = None
        self.use_finalscore = None
        self.use_auxiliary_policy = None

        # Fixed values but flexible
        self.nntype = None
        self.input_channels = None
        self.input_features = None

        self.boardsize = None
        self.value_misc = None

def trainparser(json_data, config):
    # We assume that every value is valid.
    train = json_data["Train"]

    config.gpus = train["GPUs"]
    config.learn_rate = train["LearningRate"]
    config.weight_decay = train["WeightDecay"]

    config.train_dir = train["TrainDirectory"]
    config.val_dir = train["ValidationDirectory"]
    config.test_dir = train["TestDirectory"]
    config.epochs = train["Epochs"]
    config.batchsize = train["BatchSize"]
    config.num_workers = train["Workers"]

    if config.epochs == None:
        config.epochs = 1000 # the lightning default epochs

    if config.num_workers == None:
        config.num_workers = os.cpu_count()

    if config.gpus == None:
        if torch.cuda.is_available():
            config.gpus = torch.cuda.device_count()

    return config

def nnparser(json_data, config):
    # We assume that every value is valid.
    resnet = json_data["NeuralNetwork"]


    config.boardsize = resnet["DefaultBoardSize"]
    config.use_ownership = resnet["UseOwnership"]
    config.use_finalscore = resnet["UseFinalScore"]
    config.use_auxiliary_policy = resnet["UseAuxiliaryPolicy"]
    
    config.nntype = resnet["NNType"]
    config.input_channels = resnet["InputChannels"]
    config.residual_channels = resnet["ResidualChannels"]
    config.policy_extract = resnet["PolicyExtract"]
    config.value_extract = resnet["ValueExtract"]
    config.value_misc = resnet["ValueMisc"]

    stack = resnet["Stack"]
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
        cfg = trainparser(d, cfg)
        cfg = nnparser(d, cfg)
    return cfg
