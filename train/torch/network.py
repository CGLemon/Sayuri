import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

def tensor_to_text(t: torch.Tensor):
    return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"

class GlobalPool(nn.Module):
    def __init__(self, is_value_head=False):
        super().__init__()
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
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)

            # Accordiog to KataGo, computing the board size variance can 
            # improve the value head performance. That because the winrate
            # and score lead heads consist of komi and intersections.
            layer2 = layer_raw_mean * (torch.square(b_diff) / 100.0 - self.b_variance)

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)
        else:
            # Apply CRAZY_NEGATIVE_VALUE to out of board area. We guess that 
            # -5000 is large enough.
            raw_x = x + (1.0-mask) * CRAZY_NEGATIVE_VALUE

            layer_raw_max = torch.max(torch.reshape(raw_x, (b,c,h*w)), dim=2, keepdims=False)[0]
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_max

            layer_pooled = torch.cat((layer0, layer1, layer2), 1)

        return layer_pooled

class BatchNorm2d(nn.Module):
    def __init__(self, num_features,
                       eps=1e-5,
                       momentum=0.01,
                       use_gamma=False,
                       fixup=False):
        # According to the paper "Batch Renormalization: Towards Reducing Minibatch Dependence
        # in Batch-Normalized Models", Batch-Renormalization is much faster and steady than 
        # traditional Batch-Normalized. Will improve the performance in the small batch size
        # case.

        super().__init__()
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

        # TODO: Test Batch Renormalization.
        self.use_renorm = False

        self.use_gamma = use_gamma
        self.num_features = num_features
        self.eps = eps
        self.momentum = self.__clamp(momentum)

        # Fix up Batch Normalization layer. According to kata Go, Batch Normalization may cause
        # some wierd reuslts that becuse the inference and training computation results are different.
        # Fix up also speeds up the performance. May improve around x1.6 ~ x1.8 thae becuse we use
        # customized Batch Normalization layer. The performance of customized layer is very slow.
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
            # Inference step or fixup, they are same.
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
        super().__init__()
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

    def tensors_to_text(self):
        out = str()
        out += tensor_to_text(self.linear.weight)
        out += tensor_to_text(self.linear.bias)
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
        super().__init__()

        assert kernel_size in (1, 3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
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

    def tensors_to_text(self):
        out = str()
        out += tensor_to_text(self.conv.weight)
        out += tensor_to_text(self.conv.bias)
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
        super().__init__()
 
        assert kernel_size in (1, 3)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
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

    def tensors_to_text(self):
        bn_mean = torch.zeros(self.out_channels)
        bn_std = torch.zeros(self.out_channels)

        out = str()
        out += tensor_to_text(self.conv.weight)
        out += tensor_to_text(torch.zeros(self.out_channels)) # fill zero

        # Merge four tensors(mean, variance, gamma, beta) into two tensors (
        # mean, variance).
        bn_mean[:] = self.bn.running_mean[:]
        bn_std[:] = torch.sqrt(self.bn.eps + self.bn.running_var)[:]

        # Original format: gamma * ((x-mean) / std) + beta
        # Target format: (x-mean) / std
        #
        # Solve the following equation:
        #     gamma * ((x-mean) / std) + beta = (x-tgt_mean) / tgt_std
        #
        # We get:
        #     tgt_std = std / gamma
        #     tgt_mean = (mean - beta) * (std / gamma)

        if self.bn.gamma is not None:
            bn_std = bn_std / self.bn.gamma
        if self.bn.beta is not None:
            bn_mean = bn_mean - self.bn.beta * bn_std
        bn_var = torch.square(bn_std) - self.bn.eps

        out += tensor_to_text(bn_mean)
        out += tensor_to_text(bn_var)
        return out

    def forward(self, x, mask):
        x = self.conv(x) * mask
        x = self.bn(x, mask)
        return F.relu(x, inplace=True) if self.relu else x

class ResBlock(nn.Module):
    def __init__(self, blocks,
                       channels,
                       fixup,
                       se_size=None,
                       collector=None):
        super().__init__()
        self.use_se=False
        self.channels=channels

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            use_gamma=False,
            fixup=fixup,
            relu=True,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            use_gamma=True,
            fixup=fixup,
            relu=False,
            collector=collector
        )

        if se_size is not None:
            self.use_se = True
            self.global_pool = GlobalPool(is_value_head=False)

            self.squeeze = FullyConnect(
                in_size=3*channels,
                out_size=se_size,
                relu=True,
                collector=collector
            )
            self.excite = FullyConnect(
                in_size=se_size,
                out_size=2*channels,
                relu=False,
                collector=collector
            )

        if fixup:
            # re-scale the weights
            fixup_scale2 = 1.0 / math.sqrt(blocks)
            self.conv1.conv.weight.data *= fixup_scale2
            self.conv2.conv.weight.data *= 0
            if se_size is not None:
                fixup_scale4 = 1.0 / (blocks ** (1.0 / 4.0)) 
                self.squeeze.linear.weight.data *= fixup_scale4
                self.excite.linear.weight.data *= fixup_scale4

    def forward(self, inputs):
        x, mask_buffers = inputs

        mask, _, _ = mask_buffers

        identity = x

        out = self.conv1(x, mask)
        out = self.conv2(out, mask)
        
        if self.use_se:
            b, c, _, _ = out.size()
            seprocess = self.global_pool(out, mask_buffers)
            seprocess = self.squeeze(seprocess)
            seprocess = self.excite(seprocess)

            gammas, betas = torch.split(seprocess, self.channels, dim=1)
            gammas = torch.reshape(gammas, (b, c, 1, 1))
            betas = torch.reshape(betas, (b, c, 1, 1))
            out = torch.sigmoid(gammas) * out + betas + identity
            out = out * mask
        else:
            out += identity

        return F.relu(out, inplace=True), mask_buffers


class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Layer Collector will collect all weights except for ingnore
        # heads. According to multi-task learning, some heads are just
        # auxiliary. These heads may improve the performance but are useless
        # in the inference time. We do not collect them. 
        self.layer_collector = []

        self.cfg = cfg
        self.nntype = cfg.nntype

        self.fixup = cfg.fixup_batch_norm

        self.input_channels = cfg.input_channels
        self.residual_channels = cfg.residual_channels
        self.xsize = cfg.boardsize
        self.ysize = cfg.boardsize
        self.policy_extract = cfg.policy_extract
        self.value_extract = cfg.value_extract
        self.value_misc = cfg.value_misc
        self.stack = cfg.stack

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
            collector=self.layer_collector
        )

        # residual tower
        nn_stack = []
        for s in self.stack:
            if s == "ResidualBlock":
                nn_stack.append(ResBlock(blocks=len(self.stack),
                                         channels=self.residual_channels,
                                         fixup=self.fixup,
                                         se_size=None,
                                         collector=self.layer_collector))
            elif s == "ResidualBlock-SE":
                nn_stack.append(ResBlock(blocks=len(self.stack),
                                         channels=self.residual_channels,
                                         fixup=self.fixup,
                                         se_size=self.residual_channels,
                                         collector=self.layer_collector))

        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_extract,
            kernel_size=1,
            use_gamma=False,
            fixup=self.fixup,
            relu=True,
            collector=self.layer_collector
        )
        self.policy_intermediate_fc = FullyConnect(
            in_size=self.policy_extract * 3,
            out_size=self.policy_extract,
            relu=True,
            collector=self.layer_collector
        )
        self.prob = Convolve(
            in_channels=self.policy_extract,
            out_channels=1,
            kernel_size=1,
            relu=False,
            collector=self.layer_collector
        )
        self.prob_pass_fc = FullyConnect(
            in_size=self.policy_extract,
            out_size=1,
            relu=False,
            collector=self.layer_collector
        )

        # Ingnore auxiliary policy.
        self.aux_prob = Convolve(
            in_channels=self.policy_extract,
            out_channels=1,
            kernel_size=1,
            relu=False
        )

        # Ingnore auxiliary pass policy.
        self.aux_prob_pass_fc = FullyConnect(
            in_size=self.policy_extract,
            out_size=1,
            relu=False,
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_extract,
            kernel_size=1,
            use_gamma=False,
            fixup=self.fixup,
            relu=True,
            collector=self.layer_collector
        )
        self.value_intermediate_fc = FullyConnect(
            in_size=self.policy_extract * 3,
            out_size=self.policy_extract,
            relu=True,
            collector=self.layer_collector
        )
        self.ownership_conv = Convolve(
            in_channels=self.value_extract,
            out_channels=1,
            kernel_size=1,
            relu=False,
            collector=self.layer_collector
        )
        self.value_misc_fc = FullyConnect(
            in_size=self.value_extract,
            out_size=self.value_misc,
            relu=False,
            collector=self.layer_collector
        )

    def forward(self, planes, target=None, use_symm=False):
        symm = int(np.random.choice(8, 1)[0])
        if use_symm:
            planes = torch_symmetry(symm, planes, inves=False)

        # mask buffers
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :]
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        # input layer
        x = self.input_conv(planes, mask)

        # residual tower
        x, _ = self.residual_tower((x, mask_buffers))

        # policy head
        pol = self.policy_conv(x, mask)
        pol_gpool = self.global_pool(pol, mask_buffers)
        pol_inter = self.policy_intermediate_fc(pol_gpool)

        # Add intermediate as biases. It may improve the performance.
        b, c = pol_inter.size()
        pol = (pol + pol_inter.view(b, c, 1, 1)) * mask

        # Apply CRAZY_NEGATIVE_VALUE on out of board area. This position
        # policy will be zero after softmax 
        prob_without_pass = self.prob(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE

        if use_symm:
            prob_without_pass = torch_symmetry(symm, prob_without_pass, inves=True)
        prob_without_pass = torch.flatten(prob_without_pass, start_dim=1, end_dim=3)
        prob_pass = self.prob_pass_fc(pol_inter)
        prob = torch.cat((prob_without_pass, prob_pass), 1)

        # Apply CRAZY_NEGATIVE_VALUE on out of board area. This position
        # policy will be zero after softmax 
        aux_prob_without_pass = self.aux_prob(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE
        if use_symm:
            aux_prob_without_pass = torch_symmetry(symm, aux_prob_without_pass, inves=True)
        aux_prob_without_pass = torch.flatten(aux_prob_without_pass, start_dim=1, end_dim=3)
        aux_prob_pass = self.aux_prob_pass_fc(pol_inter)
        aux_prob = torch.cat((aux_prob_without_pass, aux_prob_pass), 1)

        # value head
        val = self.value_conv(x, mask)
        val_gpool = self.global_pool_val(val, mask_buffers)
        val_inter = self.value_intermediate_fc(val_gpool)

        b, c = val_inter.size()
        val = (val + val_inter.view(b, c, 1, 1)) * mask

        ownership = self.ownership_conv(val, mask)
        if use_symm:
            ownership = torch_symmetry(symm, ownership, inves=True)
        ownership = torch.flatten(ownership, start_dim=1, end_dim=3)
        ownership = torch.tanh(ownership)

        val_misc = self.value_misc_fc(val_inter)
        wdl, stm, score = torch.split(val_misc, [3, 1, 1], dim=1)
        stm = torch.tanh(stm)

        predict = (prob, aux_prob, ownership, wdl, stm, score)

        all_loss = None
        if target is not None:
            all_loss = self.compute_loss(predict, target, mask_sum_hw)

        return predict, all_loss

    def compute_loss(self, pred, target, mask_sum_hw):
        (pred_prob, pred_aux_prob, pred_ownership, pred_wdl, pred_stm, pred_score) = pred
        (target_prob,target_aux_prob, target_ownership, target_wdl, target_stm, target_final_score) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        def mse_loss_spat(pred, target, mask_sum_hw):
            spat_sum = torch.sum(torch.square(pred - target), dim=1) / mask_sum_hw
            return torch.mean(spat_sum, dim=0)

        prob_loss = cross_entropy(pred_prob, target_prob)
        aux_prob_loss = 0.15 * cross_entropy(pred_aux_prob, target_aux_prob)
        ownership_loss = 1.5 * mse_loss_spat(pred_ownership, target_ownership, mask_sum_hw)
        wdl_loss = cross_entropy(pred_wdl, target_wdl)
        stm_loss = F.mse_loss(pred_stm, target_stm)

        fina_score_loss = 0.0012 * huber_loss(20 * pred_score, target_final_score, 12)


        return (prob_loss, aux_prob_loss, ownership_loss, wdl_loss, stm_loss, fina_score_loss)

    def trainable(self, t=True):
        if t==True:
            self.train()
        else:
            self.eval()

    def save_pt(self, filename):
        torch.save(self.state_dict(), filename)

    def load_pt(self, filename):
        self.load_state_dict(torch.load(filename, map_location=torch.device('cpu')))

    def dump_info(self):
        print("NN Type: {type}".format(type=self.nntype))
        print("NN size [x,y]: [{xsize}, {ysize}] ".format(xsize=self.xsize, ysize=self.ysize))
        print("Input channels: {channels}".format(channels=self.input_channels))
        print("Residual channels: {channels}".format(channels=self.residual_channels))
        print("Residual tower: size -> {s} [".format(s=len(self.stack)))
        for s in self.stack:
            print("  {}".format(s))
        print("]")
        print("Policy extract channels: {policyextract}".format(policyextract=self.policy_extract))
        print("Value extract channels: {valueextract}".format(valueextract=self.value_extract))
        print("Value misc size: {valuemisc}".format(valuemisc=self.value_misc))

    def transfer2text(self, filename):
        def write_struct(f, layer_collector):
            f.write("get struct\n")
            for layer in layer_collector:
                f.write(layer.shape_to_text())
            f.write("end struct\n")

        def write_params(f, layer_collector):
            f.write("get parameters\n")
            for layer in layer_collector:
                f.write(layer.tensors_to_text())
            f.write("end parameters\n")

        with open(filename, 'w') as f:
            f.write("get main\n")

            f.write("get info\n")
            f.write("NNType {}\n".format(self.nntype))
            f.write("InputChannels {}\n".format(self.input_channels))
            f.write("ResidualChannels {}\n".format(self.residual_channels))
            f.write("ResidualBlocks {}\n".format(len(self.stack)))
            f.write("PolicyExtract {}\n".format(self.policy_extract))
            f.write("ValueExtract {}\n".format(self.value_extract))
            f.write("ValueMisc {}\n".format(self.value_misc))
            f.write("end info\n")

            write_struct(f, self.layer_collector)
            write_params(f, self.layer_collector)

            f.write("end main")
