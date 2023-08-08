import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import struct
import sys

from symmetry import torch_symmetry

CRAZY_NEGATIVE_VALUE = -5000.0

def conv_to_text(in_channels, out_channels, kernel_size):
    return "Convolution {iC} {oC} {KS}\n".format(
               iC=in_channels,
               oC=out_channels,
               KS=kernel_size)

def fullyconnect_to_text(in_size, out_size):
    return "FullyConnect {iS} {oS}\n".format(iS=in_size, oS=out_size)

def bn_to_text(channels):
    return "BatchNorm {C}\n".format(C=channels)

def float_to_bin(num, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.pack(fmt, num)

def bin_to_float(bnum, big_endian):
    fmt = 'f'
    if big_endian:
        fmt = '!' + fmt
    return struct.unpack(fmt, bnum)[0]

def str_to_bin(st):
    return bytearray(st, 'utf-8')

def ffffffff_nan():
    return b'\xff\xff\xff\xff'

def tensor_to_bin(t: torch.Tensor):
    return b''.join([float_to_bin(w, False) for w in t.detach().numpy().ravel()]) + ffffffff_nan()

def tensor_to_text(t: torch.Tensor, use_bin):
    if use_bin:
        return tensor_to_bin(t)
    return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"

# It is imported from KataGo.
class SoftPlusWithGradientFloorFunction(torch.autograd.Function):
    """
    Same as softplus, except on backward pass, we never let the gradient decrease below grad_floor.
    Equivalent to having a dynamic learning rate depending on stop_grad(x) where x is the input.
    If square, then also squares the result while halving the input, and still also keeping the same gradient.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, grad_floor: float, square: bool):
        ctx.save_for_backward(x)
        ctx.grad_floor = grad_floor # grad_floor is not a tensor
        if square:
            return torch.square(F.softplus(0.5 * x))
        else:
            return F.softplus(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        grad_floor = ctx.grad_floor
        grad_x = None
        grad_grad_floor = None
        grad_square = None
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * (grad_floor + (1.0 - grad_floor) / (1.0 + torch.exp(-x)))
        return grad_x, grad_grad_floor, grad_square

class GlobalPool(nn.Module):
    def __init__(self, is_value_head=False):
        super(GlobalPool, self).__init__()
        self.b_avg = (19 + 9) / 2
        self.b_variance = 0.1

        self.is_value_head = is_value_head

    def forward(self, x, mask_buffers):
        mask, mask_sum_hw, mask_sum_hw_sqrt = mask_buffers
        b, c, h, w = x.size()

        div = torch.reshape(mask_sum_hw, (-1,1))
        div_sqrt = torch.reshape(mask_sum_hw_sqrt, (-1,1))

        layer_raw_mean = torch.sum(x, dim=(2,3), keepdims=False) / div
        b_diff = div_sqrt - self.b_avg

        if self.is_value_head:
            # According to KataGo, we compute three orthogonal values. There
            # are 1, (x-14)/10, and (x-14)^2/100 - 0.1. They may improve the value
            # head performance. That because the win-rate and score lead heads consist
            # of komi and intersections.

            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_mean * (torch.square(b_diff) / 100.0 - self.b_variance)

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)
        else:
            # Apply CRAZY_NEGATIVE_VALUE to out of board area. I guess that 
            # -5000 is large enough.
            raw_x = x + (1.0-mask) * CRAZY_NEGATIVE_VALUE

            layer_raw_max = torch.max(torch.reshape(raw_x, (b,c,h*w)), dim=2, keepdims=False)[0]
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_max

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)

        return layer_pooled

class SqueezeAndExcitation(nn.Module):
    def __init__(self, channels,
                       se_size,
                       collector=None):
        super(SqueezeAndExcitation, self).__init__()
        self.global_pool = GlobalPool(is_value_head=False)
        self.channels = channels

        self.squeeze = FullyConnect(
            in_size=3*self.channels,
            out_size=se_size,
            relu=True,
            collector=collector
        )
        self.excite = FullyConnect(
            in_size=se_size,
            out_size=2*self.channels,
            relu=False,
            collector=collector
        )

    def fixup_adjust(self, fixup_scale):
        self.squeeze.linear.weight.data *= fixup_scale
        self.excite.linear.weight.data *= fixup_scale

    def forward(self, x, mask_buffers):
        b, c, _, _ = x.size()
        mask, _, _ = mask_buffers

        seprocess = self.global_pool(x, mask_buffers)
        seprocess = self.squeeze(seprocess)
        seprocess = self.excite(seprocess)

        gammas, betas = torch.split(seprocess, self.channels, dim=1)
        gammas = torch.reshape(gammas, (b, c, 1, 1))
        betas = torch.reshape(betas, (b, c, 1, 1))

        out = torch.sigmoid(gammas) * x + betas
        return out * mask

class BatchNorm2d(nn.Module):
    def __init__(self, num_features,
                       eps=1e-5,
                       momentum=0.01,
                       use_gamma=False,
                       fixup=False):
        # According to the paper "Batch Renormalization: Towards Reducing Minibatch Dependence
        # in Batch-Normalized Models", Batch-Renormalization is much faster and steady than 
        # traditional Batch-Normalized when it is small batch size.

        super(BatchNorm2d, self).__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.long)
        )

        self.gamma = None
        if use_gamma:
            self.gamma = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )

        self.beta = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )

        # Enable the batch renorm if we set true, else will use the batch norm.
        self.use_renorm = True

        self.use_gamma = use_gamma
        self.num_features = num_features
        self.eps = eps
        self.momentum = self.__clamp(momentum)

        # Fix up Batch Normalization layer. According to kata Go, Batch Normalization may cause
        # some wierd reuslts becuse the inference and training computation results are different.
        # Fix up can avoid the weird forwarding result. Fix up also speeds up the performance. May
        # improve around x1.6 ~ x1.8.
        self.fixup = fixup

    def __clamp(self, x):
        x = max(0, x)
        x = min(1, x)
        return x

    @property
    def rmax(self):
        # first 5k training step, rmax = 1
        # after 25k training step, rmax = 3
        return (2 / 35000 * self.num_batches_tracked + 25 / 35).clamp_(
            1.0, 3.0
        )

    @property
    def dmax(self):
        # first 5k training step, dmax = 0
        # after 25k training step, dmax = 5
        return (5 / 20000 * self.num_batches_tracked - 25 / 20).clamp_(
            0.0, 5.0
        )

    def __apply_renorm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        running_std = torch.sqrt(self.running_var+self.eps).view(1, self.num_features, 1, 1)
        running_mean = self.running_mean.view(1, self.num_features, 1, 1)

        r = (
            std.detach() / running_std
        ).clamp_(1 / self.rmax, self.rmax)


        d = (
            (mean.detach() - running_mean) / running_std
        ).clamp_(-self.dmax, self.dmax)

        x = (x-mean)/std * r + d
        return x

    def __apply_norm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        x = (x-mean)/std
        return x

    def forward(self, x, mask):
        #TODO: Improve the performance.

        if self.training and not self.fixup:
            mask_sum = torch.sum(mask) # global sum

            batch_mean = torch.sum(x, dim=(0,2,3)) / mask_sum
            zmtensor = x - batch_mean.view(1, self.num_features, 1, 1)
            batch_var = torch.sum(torch.square(zmtensor * mask), dim=(0,2,3)) / mask_sum

            if self.use_renorm:
                # In the first 5k training steps, the batch renorm is equal
                # to the batch norm.
                x = self.__apply_renorm(x , batch_mean, batch_var)
            else:
                x = self.__apply_norm(x , batch_mean, batch_var)

            # Update moving averages.
            self.running_mean += self.momentum * (batch_mean.detach() - self.running_mean)
            self.running_var += self.momentum * (batch_var.detach() - self.running_var)
            self.num_batches_tracked += 1
        else:
            # Inference step or fixup, they are equal.
            x = self.__apply_norm(x, self.running_mean, self.running_var)

        if self.gamma is not None:
            x = x * self.gamma.view(1, self.num_features, 1, 1)

        if self.beta is not None:
            x = x + self.beta.view(1, self.num_features, 1, 1)

        return x * mask

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       relu=True,
                       collector=None):
        super(FullyConnect, self).__init__()
        self.relu = relu
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_size,
            out_size,
            bias=True
        )
        self.__init_weights()
        self.__try_collect(collector)

    def __init_weights(self):
        nn.init.xavier_normal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def __try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        return fullyconnect_to_text(self.in_size, self.out_size)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.linear.weight, use_bin)
        out += tensor_to_text(self.linear.bias, use_bin)
        return out

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True) if self.relu else x


class Convolve(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       relu=True,
                       collector=None):
        super(Convolve, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=True,
        )
        self.__init_weights()
        self.__try_collect(collector)

    def __init_weights(self):
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)
        nn.init.zeros_(self.conv.bias)

        # nn.init.kaiming_normal_(self.conv.weight,
        #                         mode="fan_out",
        #                         nonlinearity="relu")

    def __try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        return conv_to_text(self.in_channels, self.out_channels, self.kernel_size)

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        out += tensor_to_text(self.conv.bias, use_bin)
        return out

    def forward(self, x, mask):
        x = self.conv(x) * mask
        return F.relu(x, inplace=True) if self.relu else x


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       use_gamma,
                       fixup,
                       relu=True,
                       collector=None):
        super(ConvBlock, self).__init__()
 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=False,
        )

        self.bn = BatchNorm2d(
            num_features=out_channels,
            eps=1e-5,
            use_gamma=use_gamma,
            fixup=fixup
        )
        self.__init_weights()
        self.__try_collect(collector)

    def __init_weights(self):
        nn.init.xavier_normal_(self.conv.weight, gain=1.0)

        # nn.init.kaiming_normal_(self.conv.weight,
        #                         mode="fan_out",
        #                         nonlinearity="relu")

    def __try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        out = str()
        out += conv_to_text(self.in_channels, self.out_channels, self.kernel_size)
        out += bn_to_text(self.out_channels)
        return out

    def tensors_to_text(self, use_bin):
        bn_mean = torch.zeros(self.out_channels)
        bn_std = torch.zeros(self.out_channels)
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        out += tensor_to_text(torch.zeros(self.out_channels), use_bin) # fill zero

        # Merge four tensors (mean, variance, gamma, beta) into two tensors (
        # mean, variance).
        bn_mean[:] = self.bn.running_mean[:]
        bn_std[:] = torch.sqrt(self.bn.eps + self.bn.running_var)[:]

        # Original format: gamma * ((x-mean) / std) + beta
        # Target format: (x-mean) / std
        #
        # Solve the following equation:
        #     gamma * ((x-mean) / std) + beta = (x-tgt_mean) / tgt_std
        #
        # We will get:
        #     tgt_std = std / gamma
        #     tgt_mean = mean - beta * (std / gamma)

        if self.bn.gamma is not None:
            bn_std = bn_std / self.bn.gamma
        if self.bn.beta is not None:
            bn_mean = bn_mean - self.bn.beta * bn_std

        out += tensor_to_text(bn_mean, use_bin)
        out += tensor_to_text(bn_std, use_bin)
        return out

    def forward(self, x, mask):
        x = self.conv(x) * mask
        x = self.bn(x, mask)
        return F.relu(x, inplace=True) if self.relu else x

class ResBlock(nn.Module):
    def __init__(self, blocks,
                       channels,
                       fixup,
                       *args,
                       **kwargs):
        super(ResBlock, self).__init__()

        # The inner layers channels.
        self.inner_channels = channels

        # The main ResBlock channels. We say a 15x192
        # resnet. The 192 is outer_channel.
        self.outer_channels = channels

        bottleneck_channels = kwargs.get("bottleneck_channels", None)
        se_size = kwargs.get("se_size", None)
        collector = kwargs.get("collector", None)

        self.use_btl = bottleneck_channels is not None
        self.use_se = se_size is not None

        if self.use_btl:
            self.inner_channels = bottleneck_channels
            self.pre_conv_btl = ConvBlock(
                in_channels=self.outer_channels,
                out_channels=self.inner_channels,
                kernel_size=1,
                use_gamma=False,
                fixup=fixup,
                relu=True,
                collector=collector
            )
        self.conv1 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            fixup=fixup,
            relu=True,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False if self.use_btl else True,
            fixup=fixup,
            relu=True if self.use_btl else False,
            collector=collector
        )
        if self.use_btl:
            self.post_conv_btl = ConvBlock(
                in_channels=self.inner_channels,
                out_channels=self.outer_channels,
                kernel_size=1,
                use_gamma=True,
                fixup=fixup,
                relu=False,
                collector=collector
            )

        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=se_size,
                collector=collector
            )

        if fixup:
            # re-scale the weights
            fixup_scale2 = 1.0 / math.sqrt(blocks)
            if self.use_btl:
                self.pre_conv_btl.conv.weight.data *= fixup_scale2
                self.conv1.conv.weight.data *= fixup_scale2
                self.conv2.conv.weight.data *= fixup_scale2
                self.post_conv_btl.conv.weight.data *= 0
            else:
                self.conv1.conv.weight.data *= fixup_scale2
                self.conv2.conv.weight.data *= 0
            if self.use_se:
                fixup_scale4 = 1.0 / (blocks ** (1.0 / 4.0)) 
                self.se_module.fixup_adjust(fixup_scale4)

    def forward(self, inputs):
        x, mask_buffers = inputs

        mask, _, _ = mask_buffers

        out = x
        if self.use_btl:
            out = self.pre_conv_btl(out, mask)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        if self.use_btl:
            out = self.post_conv_btl(out, mask)

        if self.use_se:
            out = self.se_module(out, mask_buffers)
        out = out + x

        return F.relu(out, inplace=True), mask_buffers

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        # Layer Collector will collect all weights except for ingnore
        # heads. According to multi-task learning, some heads are just
        # auxiliary. These heads may improve the performance but are useless
        # in the inference time. We do not output them. 
        self.layers_collector = []
        self.unused_collector = []

        self.nntype = cfg.nntype

        self.fixup = cfg.fixup_batch_norm

        self.input_channels = cfg.input_channels
        self.residual_channels = cfg.residual_channels
        self.xsize = cfg.boardsize
        self.ysize = cfg.boardsize
        self.policy_extract = cfg.policy_extract
        self.value_extract = cfg.value_extract
        self.value_misc = 15
        self.policy_outs = 5
        self.stack = cfg.stack
        self.version = 3

        self.construct_layers()

    def construct_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            use_gamma=True,
            fixup=self.fixup,
            relu=True,
            collector=self.layers_collector
        )

        # residual tower
        main_channels = self.residual_channels
        nn_stack = []
        for s in self.stack:
            has_basic_block = False
            use_fixup = self.fixup
            se_size = None
            bottleneck_channels = None

            for component in s.strip().split('-'):
                if component == "ResidualBlock":
                    has_basic_block = True
                elif component == "BottleneckBlock":
                    bottleneck_channels = main_channels//2
                    assert main_channels%2 == 0, ""
                    has_basic_block = True
                elif component == "SE":
                    se_size = main_channels
                elif component == "FixUp":
                    use_fixup = True
                else:
                    raise Exception("Invalid NN structure.")

            if not has_basic_block:
                raise Exception("There is no basic block.")

            nn_stack.append(ResBlock(blocks=len(self.stack),
                                     channels=main_channels,
                                     fixup=use_fixup,
                                     bottleneck_channels=bottleneck_channels,
                                     se_size=se_size,
                                     collector=self.layers_collector))

        if main_channels != self.residual_channels:
            raise Exception("Invalid block stack.")

        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_extract,
            kernel_size=1,
            use_gamma=False,
            fixup=self.fixup,
            relu=True,
            collector=self.layers_collector
        )
        self.policy_intermediate_fc = FullyConnect(
            in_size=self.policy_extract * 3,
            out_size=self.policy_extract,
            relu=True,
            collector=self.layers_collector
        )
        self.pol_misc = Convolve(
            in_channels=self.policy_extract,
            out_channels=self.policy_outs,
            kernel_size=1,
            relu=False,
            collector=self.layers_collector
        )
        self.pol_misc_pass_fc = FullyConnect(
            in_size=self.policy_extract,
            out_size=self.policy_outs,
            relu=False,
            collector=self.layers_collector
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_extract,
            kernel_size=1,
            use_gamma=False,
            fixup=self.fixup,
            relu=True,
            collector=self.layers_collector
        )
        self.value_intermediate_fc = FullyConnect(
            in_size=self.value_extract * 3,
            out_size=self.value_extract * 3,
            relu=True,
            collector=self.layers_collector
        )
        self.ownership_conv = Convolve(
            in_channels=self.value_extract,
            out_channels=1,
            kernel_size=1,
            relu=False,
            collector=self.layers_collector
        )
        self.value_misc_fc = FullyConnect(
            in_size=self.value_extract * 3,
            out_size=self.value_misc,
            relu=False,
            collector=self.layers_collector
        )

    def forward(self, planes, target=None, use_symm=False):
        symm = int(np.random.choice(8, 1)[0])
        if use_symm:
            planes = torch_symmetry(symm, planes, invert=False)

        # mask buffers
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :]
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        policy_mask = torch.flatten(mask, start_dim=1, end_dim=3)
        b, _ = policy_mask.shape
        policy_mask = torch.cat((policy_mask, mask.new_ones((b, 1))), dim=1)

        # input layer
        x = self.input_conv(planes, mask)

        # residual tower
        x, _ = self.residual_tower((x, mask_buffers))

        # policy head
        pol = self.policy_conv(x, mask)
        pol_gpool = self.global_pool(pol, mask_buffers)
        pol_inter = self.policy_intermediate_fc(pol_gpool)

        # Add intermediate as biases. It may improve the policy performance.
        b, c = pol_inter.shape
        pol = (pol + pol_inter.view(b, c, 1, 1)) * mask

        # Apply CRAZY_NEGATIVE_VALUE on out of board area. This position
        # policy will be zero after softmax 
        pol_without_pass = self.pol_misc(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE

        if use_symm:
            pol_without_pass = torch_symmetry(symm, pol_without_pass, invert=True)
        pol_without_pass = torch.flatten(pol_without_pass, start_dim=2, end_dim=3) # b, c, h*w
        pol_pass = self.pol_misc_pass_fc(pol_inter)  # b, c
        b, c = pol_pass.shape
        pol_misc = torch.cat((pol_without_pass, pol_pass.view(b, c, 1)), dim=2)

        prob, aux_prob, soft_prob, soft_aux_prob, optimistic_prob = torch.split(pol_misc, [1, 1, 1, 1, 1], dim=1)
        prob            = torch.flatten(prob, start_dim=1, end_dim=2)
        aux_prob        = torch.flatten(aux_prob, start_dim=1, end_dim=2)
        soft_prob       = torch.flatten(soft_prob, start_dim=1, end_dim=2)
        soft_aux_prob   = torch.flatten(soft_aux_prob, start_dim=1, end_dim=2)
        optimistic_prob = torch.flatten(optimistic_prob, start_dim=1, end_dim=2)

        # value head
        val = self.value_conv(x, mask)
        val_gpool = self.global_pool_val(val, mask_buffers)
        val_inter = self.value_intermediate_fc(val_gpool)

        ownership = self.ownership_conv(val, mask)
        if use_symm:
            ownership = torch_symmetry(symm, ownership, invert=True)
        ownership = torch.flatten(ownership, start_dim=1, end_dim=3)
        ownership = torch.tanh(ownership)

        val_misc = self.value_misc_fc(val_inter)
        wdl, all_q_vals, all_scores, all_errors = torch.split(val_misc, [3, 5, 5, 2], dim=1)
        all_q_vals = torch.tanh(all_q_vals)
        all_errors = SoftPlusWithGradientFloorFunction.apply(all_errors, 0.05, True)

        short_term_q_error, short_term_score_error = torch.split(all_errors, [1, 1], dim=1)
        all_scores = 20 * all_scores
        short_term_q_error = 0.25 * short_term_q_error
        short_term_score_error = 150 * short_term_score_error
        all_errors = torch.cat((short_term_q_error, short_term_score_error), dim=1)

        predict = (
            prob, # logits
            aux_prob, # logits
            soft_prob, # logits
            soft_aux_prob, # logits
            optimistic_prob, # logits
            ownership,
            wdl, # logits
            all_q_vals, # {final, current, short, middle, long}
            all_scores, # {final, current, short, middle, long}
            all_errors # {q error, score error}
        )

        all_loss_dict = dict()
        if target is not None:
            all_loss_dict = self.compute_loss(predict, target, mask_sum_hw)

        return predict, all_loss_dict

    def compute_loss(self, pred, target, mask_sum_hw, soft_weight=0.1):
        p_prob, p_aux_prob, p_soft_prob, p_soft_aux_prob, p_optimistic_prob, p_ownership, p_wdl, p_q_vals, p_scores, p_errors = pred
        t_prob, t_aux_prob, t_ownership, t_wdl, t_q_vals, t_scores = target

        def make_soft_porb(prob, t=4):
            soft_prob = torch.pow(prob, 1/t)
            soft_prob /= torch.sum(soft_prob, dim=1, keepdim=True)
            return soft_prob

        def cross_entropy(pred, target, weight=1):
            loss_sum = -torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1)
            return torch.mean(weight * loss_sum, dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        def mse_loss_spat(pred, target):
            spat_sum = torch.sum(torch.square(pred - target), dim=1) / mask_sum_hw
            return torch.mean(spat_sum, dim=0)

        def square_huber_loss(pred, x, y, delta, eps):
            sqerror = torch.square(x - y) + eps
            loss = huber_loss(pred, sqerror, delta=delta)
            return loss

        # will use these values later
        _, short_term_q_pred, _ = torch.split(p_q_vals, [2, 1, 2], dim=1)
        _, short_term_q_target, _ = torch.split(t_q_vals, [2, 1, 2], dim=1)
        _, short_term_score_pred, _ = torch.split(p_scores, [2, 1, 2], dim=1)
        _, short_term_score_target, _ = torch.split(t_scores, [2, 1, 2], dim=1)
        short_term_q_error, short_term_score_error = torch.split(p_errors, [1, 1], dim=1)

        # current player's probabilities loss
        prob_loss = 1 * cross_entropy(p_prob, t_prob)

        # opponent's probabilities loss
        aux_prob_loss = 0.15 * cross_entropy(p_aux_prob, t_aux_prob)

        # current player's soft probabilities loss
        soft_prob_loss = 1 * soft_weight * cross_entropy(p_soft_prob, make_soft_porb(t_prob))

        # opponent's soft probabilities loss
        soft_aux_prob_loss = 0.15 * soft_weight * cross_entropy(p_soft_aux_prob, make_soft_porb(t_aux_prob))

        # short-term optimistic probabilities loss
        z_short_term_q = (short_term_q_target - short_term_q_pred.detach()) / torch.sqrt(short_term_q_error.detach() + 0.0001)
        z_short_term_score = (short_term_score_target - short_term_score_pred.detach()) / torch.sqrt(short_term_score_error.detach() + 0.25)

        optimistic_weight = torch.clamp(
            torch.sigmoid((z_short_term_q - 1.5) * 3.0) + torch.sigmoid((z_short_term_score - 1.5) * 3.0),
            min=0.0,
            max=1.0,
        )
        b, _ = optimistic_weight.shape
        optimistic_weight = torch.reshape(optimistic_weight, (b, ))
        optimistic_loss = 1 * cross_entropy(p_optimistic_prob, t_prob, optimistic_weight)

        # ownership loss
        ownership_loss = 1.5 * mse_loss_spat(p_ownership, t_ownership)

        # win-draw-lose loss
        wdl_loss = cross_entropy(p_wdl, t_wdl)

        # all Q values loss
        q_vals_loss = F.mse_loss(p_q_vals, t_q_vals)

        # all scores loss
        scores_loss = 0.0012 * huber_loss(p_scores, t_scores, 12.)

        # all short term square error loss
        q_error_loss = 2 * square_huber_loss(
            short_term_q_error,
            short_term_q_pred.detach(),
            short_term_q_target,
            delta=0.4, eps=1.0e-8
        )
        score_error_loss = 0.00002 * square_huber_loss(
            short_term_score_error,
            short_term_score_pred.detach(),
            short_term_score_target,
            delta=100.0, eps=1.0e-4
        )
        errors_loss = q_error_loss + score_error_loss

        all_loss_dict = {
            "prob_loss"          : prob_loss,
            "aux_prob_loss"      : aux_prob_loss,
            "soft_prob_loss"     : soft_prob_loss,
            "soft_aux_prob_loss" : soft_aux_prob_loss,
            "optimistic_loss"    : optimistic_loss,
            "ownership_loss"     : ownership_loss,
            "wdl_loss"           : wdl_loss,
            "q_vals_loss"        : q_vals_loss,
            "scores_loss"        : scores_loss,
            "errors_loss"        : errors_loss
        }
        return all_loss_dict

    def save_pt(self, filename):
        torch.save(self.state_dict(), filename)

    def load_pt(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def simple_info(self):
        info = str()
        info += "NN Type: {type}\n".format(type=self.nntype)
        info += "NN size [x,y]: [{xsize}, {ysize}]\n".format(xsize=self.xsize, ysize=self.ysize)
        info += "Input channels: {channels}\n".format(channels=self.input_channels)
        info += "Residual channels: {channels}\n".format(channels=self.residual_channels)
        info += "Residual tower: size -> {s} [\n".format(s=len(self.stack))
        for s in self.stack:
            info += "  {}\n".format(s)
        info += "]\n"
        info += "Policy extract channels: {policyextract}\n".format(policyextract=self.policy_extract)
        info += "Value extract channels: {valueextract}\n".format(valueextract=self.value_extract)
        info += "Value misc size: {valuemisc}\n".format(valuemisc=self.value_misc)

        return info

    def transfer_to_bin(self, filename):
        def write_stack(f, stack):
            f.write(str_to_bin("get stack\n"))
            for s in stack:
                f.write(str_to_bin("{}\n".format(s)))
            f.write(str_to_bin("end stack\n"))

        def write_struct(f, layers_collector):
            f.write(str_to_bin("get struct\n"))
            for layer in layers_collector:
                f.write(str_to_bin(layer.shape_to_text()))
            f.write(str_to_bin("end struct\n"))

        def write_params(f, layers_collector):
            f.write(str_to_bin("get parameters\n"))
            for layer in layers_collector:
                f.write(layer.tensors_to_text(True))
            f.write(str_to_bin("end parameters\n"))

        with open(filename, 'wb') as f:
            f.write(str_to_bin("get main\n"))

            f.write(str_to_bin("get info\n"))
            f.write(str_to_bin("NNType {}\n".format(self.nntype)))
            f.write(str_to_bin("Version {}\n".format(self.version)))
            f.write(str_to_bin("FloatType {}\n".format("float32bin")))
            f.write(str_to_bin("InputChannels {}\n".format(self.input_channels)))
            f.write(str_to_bin("ResidualChannels {}\n".format(self.residual_channels)))
            f.write(str_to_bin("ResidualBlocks {}\n".format(len(self.stack))))
            f.write(str_to_bin("PolicyExtract {}\n".format(self.policy_extract)))
            f.write(str_to_bin("ValueExtract {}\n".format(self.value_extract)))
            f.write(str_to_bin("ValueMisc {}\n".format(self.value_misc)))
            f.write(str_to_bin("end info\n"))

            write_stack(f, self.stack)
            write_struct(f, self.layers_collector)
            write_params(f, self.layers_collector)

            f.write(str_to_bin("end main"))

    def transfer_to_text(self, filename):
        def write_stack(f, stack):
            f.write("get stack\n")
            for s in stack:
                f.write("{}\n".format(s))
            f.write("end stack\n")

        def write_struct(f, layers_collector):
            f.write("get struct\n")
            for layer in layers_collector:
                f.write(layer.shape_to_text())
            f.write("end struct\n")

        def write_params(f, layers_collector):
            f.write("get parameters\n")
            for layer in layers_collector:
                f.write(layer.tensors_to_text(False))
            f.write("end parameters\n")

        with open(filename, 'w') as f:
            f.write("get main\n")

            f.write("get info\n")
            f.write("NNType {}\n".format(self.nntype))
            f.write("Version {}\n".format(self.version))
            f.write("FloatType {}\n".format("float32"))
            f.write("InputChannels {}\n".format(self.input_channels))
            f.write("ResidualChannels {}\n".format(self.residual_channels))
            f.write("ResidualBlocks {}\n".format(len(self.stack)))
            f.write("PolicyExtract {}\n".format(self.policy_extract))
            f.write("ValueExtract {}\n".format(self.value_extract))
            f.write("ValueMisc {}\n".format(self.value_misc))
            f.write("end info\n")

            write_stack(f, self.stack)
            write_struct(f, self.layers_collector)
            write_params(f, self.layers_collector)

            f.write("end main")
