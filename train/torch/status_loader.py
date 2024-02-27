import torch
import os

class StatusLoader:
    def __init__(self):
        self.state_dict = dict()
        self.model_name = "model"
        self.swa_model_name = "swa_model"
        self.optimizer_name = "optimizer"

        self.steps_name = "steps"
        self.swa_count_name = "swa_count"

    def reset(self):
        self.state_dict = dict()

    def load(self, filename, device=torch.device("cpu")):
        if os.path.isfile(filename):
            self.state_dict = torch.load(filename, map_location=device)

    def save(self, filename):
        torch.save(self.state_dict, filename)

    def get_model_dict(self):
        return self.state_dict.get(self.model_name, None)

    def get_swa_model_dict(self):
        return self.state_dict.get(self.swa_model_name, None)

    def get_optimizer_dict(self):
        return self.state_dict.get(self.optimizer_name, None)

    def get_steps(self):
        return self.state_dict.get(self.steps_name, 0)

    def get_swa_count(self):
        return self.state_dict.get(self.swa_count_name, 0)

    def set_model_dict(self, model_dict):
        self.state_dict[self.model_name] = model_dict

    def set_swa_model_dict(self, model_dict):
        self.state_dict[self.swa_model_name] = model_dict

    def set_optimizer_dict(self, optimizer_dict):
        self.state_dict[self.optimizer_name] = optimizer_dict

    def set_steps(self, s):
        self.state_dict[self.steps_name] = s

    def set_swa_count(self, c):
        self.state_dict[self.swa_count_name] = c

    def load_model(self, network):
        model_dict = self.get_model_dict()
        if model_dict is not None:
            network.load_state_dict(model_dict)

    def load_swa_model(self, network):
        model_dict = self.get_swa_model_dict()
        if model_dict is not None:
            network.load_state_dict(model_dict)

    def load_optimizer(self, optimizer):
        optimizer_dict = self.get_optimizer_dict()
        if optimizer_dict is not None:
            optimizer.load_state_dict(optimizer_dict)

    def save_model(self, network):
        self.set_model_dict(network.state_dict())

    def save_swa_model(self, network):
        self.set_swa_model_dict(network.state_dict())

    def save_optimizer(self, optimizer):
        self.set_optimizer_dict(optimizer.state_dict())
