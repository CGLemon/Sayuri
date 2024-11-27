from network import FullyConnect, Convolve, ConvBlock, ResidualBlock, BottleneckBlock
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

def linear_algo_wrapper(x, w):
    return F.linear(x, w)

def conv2d_algo_wrapper(padding, groups, weight_shape, x, w):
    return F.conv2d(x, w.view(weight_shape), padding=padding, groups=groups)

class LoRAParameters:
    def __init__(self):
        self.lora_a_params = list()
        self.lora_b_params = list()
        self.layers = list()
        self.lora_setting = list()

    def lora_fn(self, idx, algo, x):
        lora_A = self.lora_a_params[idx]
        lora_B = self.lora_b_params[idx]
        scaling = self.lora_setting[idx]["scaling"]
        return algo(x, torch.matmul(lora_B, lora_A) * scaling)

    def add_lora_fn(self, layer, alpha, rank):
        if isinstance(layer, FullyConnect):
            in_size = layer.in_size
            out_size = layer.out_size
            weight = layer.linear.weight

            # set up lora parameters
            lora_A = nn.Parameter(weight.new_zeros((rank, in_size)))
            lora_B = nn.Parameter(weight.new_zeros((out_size, rank)))
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

            # push into stack
            self.lora_a_params.append(lora_A)
            self.lora_b_params.append(lora_B)
            self.layers.append(layer)
            self.lora_setting.append( {"alpha" : alpha, "rank" : rank, "scaling" : alpha / rank} )

            # set the hook
            algo = linear_algo_wrapper
            hook_fn = partial(self.lora_fn, len(self.layers) - 1, algo)
            layer.set_rola_hook(hook_fn)
        elif isinstance(layer, Convolve) or \
                 isinstance(layer, ConvBlock):
            in_channels = layer.in_channels
            out_channels = layer.out_channels
            kernel_size = layer.kernel_size
            weight = layer.conv.weight

            # set up lora parameters
            lora_A = nn.Parameter(weight.new_zeros((rank * kernel_size, in_channels * kernel_size)))
            lora_B = nn.Parameter(weight.new_zeros((out_channels * kernel_size, rank * kernel_size)))
            nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
            nn.init.zeros_(lora_B)

            # push into stack
            self.lora_a_params.append(lora_A)
            self.lora_b_params.append(lora_B)
            self.layers.append(layer)
            self.lora_setting.append( {"alpha" : alpha, "rank" : rank, "scaling" : alpha / rank} )

            # set the hook
            algo = partial(conv2d_algo_wrapper, "same", 1, weight.shape)
            hook_fn = partial(self.lora_fn, len(self.layers) - 1, algo)
            layer.set_rola_hook(hook_fn)
        else:
            stderr_write("The {} layer does not support for LoRA.".format(type(layer)))

    def merge_lora_layers(self):
        for idx in range(len(self.layers)):
           layer = self. layers[idx]
           lora_A = self.lora_a_params[idx]
           lora_B = self.lora_b_params[idx]
           scaling = self.lora_setting[idx]["scaling"]

           if isinstance(layer, FullyConnect):
               layer.linear.weight.data += torch.matmul(lora_B, lora_A) * scaling
           elif isinstance(layer, Convolve) or \
                    isinstance(layer, ConvBlock):
               layer.conv.weight.data += torch.matmul(lora_B, lora_A).view(layer.conv.weight.shape) * scaling
           else:
               stderr_write("Unsupported layer {}".format(type(layer)))

    def freeze_bone_params(self, net):
        net.eval()
        for param in net.parameters():
            param.requires_grad = False

    def set_lora_layers(self, net):
        for block in net.residual_tower:
            alpha, rank = 4, 4
            if block.use_se:
                self.add_lora_fn(block.se_module.squeeze, alpha, rank)
                self.add_lora_fn(block.se_module.excite, alpha, rank)
            if isinstance(block, ResidualBlock):
                # self.add_lora_fn(block.conv1, alpha, rank)
                self.add_lora_fn(block.conv2, alpha, rank)
            else:
                pass

    def get_parameters(self):
        module_parameters = list()
        for lora_A, lora_B in zip(self.lora_a_params, self.lora_b_params):
            module_parameters.append(lora_A)
            module_parameters.append(lora_B)
        for param in module_parameters:
            yield param

class FineTuningDataset(Dataset):
    def __init__(self, data_buffer):
        super(FineTuningDataset, self).__init__()
        self.data_buffer = list()

        for planes, target in data_buffer:
            our_policy, opp_policy, ownership, wdl, q_vals, scores = target

            self.data_buffer.append((
                torch.from_numpy(planes).float(),
                (
                    torch.from_numpy(our_policy).float(),
                    torch.from_numpy(opp_policy).float(),
                    torch.from_numpy(ownership).float(),
                    torch.from_numpy(wdl).float(),
                    torch.from_numpy(q_vals).float(),
                    torch.from_numpy(scores).float()
                )
            ))

    def __len__(self):
        return len(self.data_buffer)

    def __getitem__(self, idx):
        return self.data_buffer[idx]

def fine_tuning(net, data_buffer, device):
    def handle_loss(all_loss_dict):
        loss = all_loss_dict["prob_loss"].new_zeros(1).mean()
        for _, v in all_loss_dict.items():
            loss += v
        return loss, all_loss_dict

    def handel_data(planes, target, device):
        our_policy, opp_policy, ownership, wdl, q_vals, scores = target

        planes = planes.to(device)
        our_policy = our_policy.to(device)
        opp_policy = opp_policy.to(device)
        ownership = ownership.to(device)
        wdl = wdl.to(device)
        q_vals = q_vals.to(device)
        scores = scores.to(device)

        target = (our_policy, opp_policy, ownership, wdl, q_vals, scores, 1.0)
        return planes, target

    def compute_avg_loss(running_loss):
        return sum(running_loss)/len(running_loss)

    lora = LoRAParameters()
    lora.set_lora_layers(net)
    lora.freeze_bone_params(net)

    train_set = FineTuningDataset(data_buffer)
    train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

    opt = torch.optim.SGD(
        lora.get_parameters(),
        lr=5e-4,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-4,
    )
    epoches = 200
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
