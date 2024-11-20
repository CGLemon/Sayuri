from functools import partial
import sys, math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def stderr_write(val):
    sys.stderr.write(val)
    sys.stderr.flush()

class LoRAParameters:
    def __init__(self):
        self.lora_a_params = list()
        self.lora_b_params = list()
        self.layers = list()
        self.lora_setting = list()

    def lora_fn(self, idx, x):
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        lora_A = self.lora_a_params[idx]
        lora_B = self.lora_b_params[idx]
        scaling = self.lora_setting[idx]["scaling"]
        return F.linear(F.linear(x, lora_A), lora_B) * scaling

    def add_lora_fn(self, fc_layer, alpha, rank):
        idx = len(self.layers)
        in_size = fc_layer.in_size
        out_size = fc_layer.out_size
        scaling = alpha / rank
        weight = fc_layer.linear.weight
        lora_A = nn.Parameter(weight.new_zeros((rank, in_size)))
        lora_B = nn.Parameter(weight.new_zeros((out_size, rank)))

        nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
        nn.init.zeros_(lora_B)

        self.lora_a_params.append(lora_A)
        self.lora_b_params.append(lora_B)
        self.layers.append(fc_layer)
        self.lora_setting.append( {"alpha" : alpha, "rank" : rank, "scaling" : scaling} )

        hook_fn = partial(self.lora_fn, idx)
        fc_layer.set_rola_hook(hook_fn)

    def merge_lora_layers(self):
        # https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
        # def T(w):
        #     return w.transpose(0, 1)
        for idx in range(len(self.layers)):
           fc_layer = self. layers[idx]
           lora_A = self.lora_a_params[idx]
           lora_B = self.lora_b_params[idx]
           scaling = self.lora_setting[idx]["scaling"]
           fc_layer.linear.weight.data += torch.matmul(lora_B, lora_A) * scaling

    def freeze_bone_params(self, net):
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

    def set_lora_layers(self, net):
        for block in net.residual_tower:
            if not block.use_se:
                continue
            self.add_lora_fn(block.se_module.squeeze, 1.0, net.residual_channels // 16)
            self.add_lora_fn(block.se_module.excite, 1.0, net.residual_channels // 8)

    def get_parameters(self):
        module_parameters = list()
        for lora_A, lora_B in zip(self.lora_a_params, self.lora_b_params):
            module_parameters.append(lora_A)
            module_parameters.append(lora_B)
        for param in module_parameters:
            yield param

class FineTuningDataset(Dataset):
    def __init__(self, data_buffer):
        self.data_buffer = list()

        for planes, target in data_buffer:
            our_policy, opp_policy, ownership, wdl, q_vals, scores, global_weight = target

            self.data_buffer.append((
                torch.from_numpy(planes).float(),
                (
                    torch.from_numpy(our_policy).float(),
                    torch.from_numpy(opp_policy).float(),
                    torch.from_numpy(ownership).float(),
                    torch.from_numpy(wdl).float(),
                    torch.from_numpy(q_vals).float(),
                    torch.from_numpy(scores).float(),
                    global_weight
                )
            ))

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        return self.data_buffer[idx]

def fine_tuning(net, data_buffer, use_gpu=False):
    def handle_loss(all_loss_dict):
        loss = all_loss_dict["prob_loss"].new_zeros(1).mean()
        for _, v in all_loss_dict.items():
            loss += v
        return loss, all_loss_dict

    def handel_data(planes, target, device):
        our_policy, opp_policy, ownership, wdl, q_vals, scores, global_weight = target
        for tensor in [planes, our_policy, opp_policy, ownership, wdl, q_vals, scores]:
            tensor = tensor.to(device)
        target = (our_policy, opp_policy, ownership, wdl, q_vals, scores, global_weight)
        return planes, target

    def compute_avg_loss(running_loss):
        return sum(running_loss)/len(running_loss)

    device = torch.device("cuda") \
                 if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    lora = LoRAParameters()
    lora.set_lora_layers(net)
    lora.freeze_bone_params(net)

    train_set = FineTuningDataset(data_buffer)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    opt = torch.optim.SGD(
        lora.get_parameters(),
        lr=5e-5,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    epoches = 20
    for e in range(epoches):
        main_running_loss = list()
        prob_running_loss = list()
        for _, batch in enumerate(train_dataloader):
            planes, target = batch
            planes, target = handel_data(planes, target, device)

            pred, all_loss_dict = net.forward(planes, target, use_symm=True)
            loss, all_loss_dict = handle_loss(all_loss_dict)

            loss.backward()

            main_running_loss.append(loss.item())
            prob_running_loss.append(all_loss_dict["prob_loss"].item())
            opt.step()
            opt.zero_grad()
        stderr_write("[{}/{}] loss: {:.4f}, prob loss: {:.4f}\n".format(
                         e+1, epoches, compute_avg_loss(main_running_loss), compute_avg_loss(prob_running_loss)));
    lora.merge_lora_layers()
