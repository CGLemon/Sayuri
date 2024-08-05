import json
import os
import torch

class Config:
    def __init__(self, inputs, is_file=True):
        if is_file:
            self.read(inputs)
        else:
            self.parse(inputs)

    def read(self, filename):
        with open(filename, "r") as f:
           json_str = f.read()
        self.parse(json_str)

    def parse(self, json_str):
        self.json_str = json_str
        jdata = json.loads(self.json_str)
        self.parse_training_config(jdata)
        self.parse_nn_config(jdata)

    def parse_training_config(self, json_data):
        train = json_data.get("Train", None)

        self.optimizer = train.get("Optimizer", "SGD")
        self.use_gpu = train.get("UseGPU", None)
        self.weight_decay = train.get("WeightDecay", 1e-4)
        self.lr_schedule = train.get("LearningRateSchedule", [[0, 0.2]])

        self.train_dir = train.get("TrainDirectory", None)
        self.validation_dir = train.get("ValidationDirectory", None)
        self.store_path = train.get("StorePath", None)
        self.batchsize = train.get("BatchSize", 512)
        self.buffersize = train.get("BufferSize", 16 * 1000)
        self.macrofactor = train.get("MacroFactor", 1)
        self.macrobatchsize = self.batchsize // self.macrofactor

        self.num_workers = train.get("Workers", max(os.cpu_count()-2, 1))
        self.steps_per_epoch = train.get("StepsPerEpoch", 1000)
        self.validation_steps = train.get("ValidationSteps", 100)
        self.verbose_steps = train.get("VerboseSteps", 1000)
        self.max_steps = train.get("MaxStepsPerRunning", 16384000)
        self.down_sample_rate = train.get("DownSampleRate", 16)
        self.num_chunks  = train.get("NumberChunks", None)
        self.soft_loss_weight  = train.get("SoftLossWeight", 0.1)
        self.swa_max_count  = train.get("SwaMaxCount", 16)
        self.swa_steps  = train.get("SwaSteps", 100)

        assert self.train_dir != None, ""
        assert self.store_path != None, ""

        if self.use_gpu == None:
            if torch.cuda.is_available():
                self.use_gpu = True

    def parse_nn_config(self, json_data):
        network = json_data.get("NeuralNetwork", None)

        self.boardsize = network.get("MaxBoardSize", 19)
        self.nntype = network.get("NNType", None)
        self.activation = network.get("Activation", "relu")
        self.input_channels = network.get("InputChannels", 43)
        self.residual_channels = network.get("ResidualChannels", None)
        self.policy_extract = network.get("PolicyExtract", None)
        self.value_extract = network.get("ValueExtract", None)
        self.se_ratio = network.get("SeRatio", 1)
        self.stack = network.get("Stack", [])

        assert self.input_channels != None, ""
        assert self.residual_channels != None, ""
        assert self.policy_extract != None, ""
        assert self.value_extract != None, ""
