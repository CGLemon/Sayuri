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
        self.use_gpu = torch.cuda.is_available() if train.get("UseGPU", None) is None else train.get("UseGPU")
        self.use_fp16 = train.get("UseFp16", False) if self.use_gpu else False
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
        self.max_steps_per_running = train.get("MaxStepsPerRunning", 16384000)
        self.down_sample_rate = train.get("DownSampleRate", 16)
        self.num_chunks = train.get("NumberChunks", None)
        self.chunks_increasing_c = train.get("ChunksIncreasingC", None)
        self.chunks_increasing_scale = train.get("ChunksIncreasingScale", 1.0)
        self.chunks_increasing_alpha = train.get("ChunksIncreasingAlpha", 0.75)
        self.chunks_increasing_beta = train.get("ChunksIncreasingBeta", 0.4)
        self.soft_loss_weight = train.get("SoftLossWeight", 0.1)
        self.swa_max_count = train.get("SwaMaxCount", 16)
        self.swa_steps = train.get("SwaSteps", 100)
        self.warmup_steps = train.get("WarmUpSteps", 0)
        self.renorm_max_r = train.get("RenormMaxR", 1)
        self.renorm_max_d = train.get("RenormMaxD", 0)
        self.policy_surprise_factor = train.get("PolicySurpriseFactor", 0.0)

        assert self.train_dir != None, ""
        assert self.store_path != None, ""

    def parse_nn_config(self, json_data):
        network = json_data.get("NeuralNetwork", None)

        self.boardsize = network.get("MaxBoardSize", 19)
        self.nntype = network.get("NNType", None)
        self.activation = network.get("Activation", "relu")
        self.input_channels = network.get("InputChannels", 43)
        self.residual_channels = network.get("ResidualChannels", None)

        self.policy_head_type = network.get("PolicyHeadType", "normal")
        self.policy_head_channels = network.get("PolicyExtract", None) # v1 ~ v4 net
        if self.policy_head_channels is None:
            self.policy_head_channels = network.get("PolicyHeadChannels", None) # since v5 net
        self.value_head_channels = network.get("ValueExtract", None) # v1 ~ v4 net
        if self.value_head_channels is None:
            self.value_head_channels = network.get("ValueHeadChannels", None) # since v5 net
        self.se_ratio = network.get("SeRatio", 2)
        self.stack = network.get("Stack", [])

        assert self.input_channels != None, ""
        assert self.residual_channels != None, ""
        assert self.policy_head_type in ["normal", "RepLK"], ""
        assert self.policy_head_channels != None, ""
        assert self.value_head_channels != None, ""
