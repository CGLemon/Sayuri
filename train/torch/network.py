import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

CRAZY_NEGATIVE_VALUE = -5000.0

class GlobalPool(nn.Module):
    def __init__(self, is_value_head=False):
        super().__init__()
        self.b_avg = (19 + 9) / 2
        self.b_varinat = 0.1

        self.is_value_head = is_value_head

    def forward(self, x, mask_buffers):
        mask, mask_sum_hw, mask_sum_hw_sqrt = mask_buffers
        b, c, h, w = x.size()

        div = torch.reshape(mask_sum_hw, (-1,1))
        div_sqrt = torch.reshape(mask_sum_hw_sqrt, (-1,1))

        layer_raw_mean = torch.sum(x, dim=(2,3), keepdims=False) / div
        b_diff = div_sqrt - self.b_avg

        layer0 = None
        layer1 = None
        layer2 = None

        if self.is_value_head:
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_mean * (torch.square(b_diff) / 100.0 - self.b_varinat)
        else:
            raw_x = x + (1.0-mask) * CRAZY_NEGATIVE_VALUE
            layer_raw_max = torch.max(torch.reshape(raw_x, (b,c,h*w)), dim=2, keepdims=False)[0]
            layer0 = layer_raw_mean
            layer1 = layer_raw_mean * (b_diff / 10.0)
            layer2 = layer_raw_max

        return torch.cat((layer0, layer1, layer2), 1)

class BatchNorm2d(nn.Module):
    def __init__(self, num_features,
                       eps = 1e-5,
                       momentum = 0.02,
                       affine = True,
                       fixup = False):
        # TODO: According to the paper "Batch Renormalization: Towards Reducing Minibatch Dependence
        #       in Batch-Normalized Models", Batch-Renormalization is much faster and steady than 
        #       traditional Batch-Normalized.

        super().__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )

        if affine:
            self.weight = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(num_features, dtype=torch.float)
            )
        self.affine = affine
        self.num_features = num_features
        self.eps = eps
        self.momentum = self._clamp(momentum)
        self.fixup = fixup

    def _clamp(self, x):
        x = max(0, x)
        x = min(1, x)
        return x

    def forward(self, x, mask):
        if self.training and not self.fixup:
            mask_sum = torch.sum(mask) # global sum

            batch_mean = torch.sum(x, dim=(0,2,3)) / mask_sum
            zmtensor = x - batch_mean.view(1, self.num_features, 1, 1)
            batch_var = torch.sum(torch.square(zmtensor * mask), dim=(0,2,3)) / mask_sum

            m = batch_mean.view(1, self.num_features, 1, 1)
            v = batch_var.view(1, self.num_features, 1, 1)
            x = (x-m)/torch.sqrt(self.eps+v)

            # Update running mean and variant.
            self.running_mean += self.momentum * (batch_mean - self.running_mean)
            self.running_var += self.momentum * (batch_var - self.running_var)
        else:
            m = self.running_mean.view(1, self.num_features, 1, 1)
            v = self.running_var.view(1, self.num_features, 1, 1)
            x = (x-m)/torch.sqrt(self.eps+v)

        if self.affine:
            x = self.weight * x + self.bias

        return x * mask

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       relu=True,
                       collector=None):
        super().__init__()
        self.relu = relu
        self.linear = nn.Linear(in_size, out_size)

        if collector != None:
            collector.append(self.linear.weight)
            collector.append(self.linear.bias)

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
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=True,
        )
        if collector != None:
            collector.append(self.conv.weight)
            collector.append(self.conv.bias)

        nn.init.kaiming_normal_(self.conv.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    def forward(self, x, mask):
        x = self.conv(x) * mask
        return F.relu(x, inplace=True) if self.relu else x


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       fixup,
                       relu=True,
                       collector=None):
        super().__init__()
 
        assert kernel_size in (1, 3)
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
            affine=False,
            fixup=fixup
        )

        if collector != None:
            collector.append(self.conv.weight)
            collector.append(torch.zeros(out_channels))
            collector.append(self.bn.running_mean)
            collector.append(self.bn.running_var)

        nn.init.kaiming_normal_(self.conv.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    def forward(self, x, mask):
        x = self.conv(x) * mask
        x = self.bn(x, mask)
        return F.relu(x, inplace=True) if self.relu else x

class ResBlock(nn.Module):
    def __init__(self, channels, fixup, se_size=None, collector=None):
        super().__init__()
        self.use_se=False
        self.channels=channels

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            fixup=fixup,
            relu=True,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            fixup=fixup,
            relu=False,
            collector=collector
        )

        if se_size != None:
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

        # Tensor Collector will collect all weights which we want automatically.
        # According to multi-task learning, some heads are only auxiliary. These
        # heads may be not useful in inference process. So they will not be 
        # collected by Tensor Collector.
        self.tensor_collector = []

        self.cfg = cfg
        self.nntype = cfg.nntype

        self.fixup = cfg.fixup_batch_norm

        self.input_channels = cfg.input_channels
        self.residual_channels = cfg.residual_channels
        self.xsize = cfg.boardsize
        self.ysize = cfg.boardsize
        self.plane_size = self.xsize * self.ysize
        self.policy_extract = cfg.policy_extract
        self.value_extract = cfg.value_extract
        self.value_misc = cfg.value_misc
        self.stack = cfg.stack

        self.set_layers()

    def set_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            fixup=self.fixup,
            relu=True,
            collector=self.tensor_collector
        )

        # residual tower
        nn_stack = []
        for s in self.stack:
            if s == "ResidualBlock":
                nn_stack.append(ResBlock(self.residual_channels,
                                         self.fixup,
                                         None,
                                         self.tensor_collector))
            elif s == "ResidualBlock-SE":
                nn_stack.append(ResBlock(self.residual_channels,
                                         self.fixup,
                                         self.residual_channels,
                                         self.tensor_collector))
        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_extract,
            kernel_size=1,
            fixup=self.fixup,
            relu=True,
            collector=self.tensor_collector
        )
        self.prob = Convolve(
            in_channels=self.policy_extract,
            out_channels=1,
            kernel_size=1,
            relu=False,
            collector=self.tensor_collector
        )
        self.prob_pass_fc = FullyConnect(
            in_size=self.policy_extract * 3,
            out_size=1,
            relu=False,
            collector=self.tensor_collector
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
            in_size=self.policy_extract * 3,
            out_size=1,
            relu=False,
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_extract,
            kernel_size=1,
            fixup=self.fixup,
            relu=True,
            collector=self.tensor_collector
        )

        self.ownership_conv = Convolve(
            in_channels=self.value_extract,
            out_channels=1,
            kernel_size=1,
            relu=False,
            collector=self.tensor_collector
        )

        self.value_misc_fc = FullyConnect(
            in_size=self.value_extract * 3,
            out_size=self.value_misc,
            relu=False,
            collector=self.tensor_collector
        )

    def forward(self, planes, target = None):
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

        prob_without_pass = self.prob(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE
        prob_without_pass = torch.flatten(prob_without_pass, start_dim=1, end_dim=3)

        pol_gpool = self.global_pool(pol, mask_buffers)

        prob_pass = self.prob_pass_fc(pol_gpool)
        prob = torch.cat((prob_without_pass, prob_pass), 1)

        # auxiliary policy
        aux_prob_without_pass = self.aux_prob(pol, mask) + (1.0-mask) * CRAZY_NEGATIVE_VALUE
        aux_prob_without_pass = torch.flatten(aux_prob_without_pass, start_dim=1, end_dim=3)
        aux_prob_pass = self.aux_prob_pass_fc(pol_gpool)
        aux_prob = torch.cat((aux_prob_without_pass, aux_prob_pass), 1)

        # value head
        val = self.value_conv(x, mask)

        ownership = self.ownership_conv(val, mask)
        ownership = torch.flatten(ownership, start_dim=1, end_dim=3)
        ownership = torch.tanh(ownership)

        val_gpool = self.global_pool_val(val, mask_buffers)

        val_misc = self.value_misc_fc(val_gpool)

        wdl, stm, score = torch.split(val_misc, [3, 1, 1], dim=1)
        stm = torch.tanh(stm)

        predict = (prob, aux_prob, ownership, wdl, stm, score)

        loss = None
        if target != None:
            loss = self.compute_loss(predict, target)

        return predict, loss

    def compute_loss(self, pred, target):
        (pred_prob, pred_aux_prob, pred_ownership, pred_wdl, pred_stm, pred_score) = pred
        (target_prob,target_aux_prob, target_ownership, target_wdl, target_stm, target_final_score) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        prob_loss = cross_entropy(pred_prob, target_prob)
        aux_prob_loss = 0.15 * cross_entropy(pred_aux_prob, target_aux_prob)
        ownership_loss = 0.15 * F.mse_loss(pred_ownership, target_ownership)
        wdl_loss = cross_entropy(pred_wdl, target_wdl)
        stm_loss = F.mse_loss(pred_stm, target_stm)

        fina_score_loss = 0.0012 * huber_loss(20 * pred_score, target_final_score, 12)

        loss = prob_loss + aux_prob_loss + ownership_loss + wdl_loss + stm_loss + fina_score_loss

        return loss

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

            f.write("get struct\n")
            f.write(Network.conv2text(self.input_channels, self.residual_channels, 3))
            f.write(Network.bn2text(self.residual_channels))

            for s in self.stack:
                if s == "ResidualBlock" or s == "ResidualBlock-SE":
                    f.write(Network.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(Network.bn2text(self.residual_channels))
                    f.write(Network.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(Network.bn2text(self.residual_channels))
                    if s == "ResidualBlock-SE":
                        f.write(Network.fullyconnect2text(self.residual_channels * 3, self.residual_channels))
                        f.write(Network.fullyconnect2text(self.residual_channels, self.residual_channels * 2))

            # policy head
            f.write(Network.conv2text(self.residual_channels, self.policy_extract, 1))
            f.write(Network.bn2text(self.policy_extract))
            f.write(Network.conv2text(self.policy_extract, 1, 1))
            f.write(Network.fullyconnect2text(self.policy_extract * 3, 1))

            # value head
            f.write(Network.conv2text(self.residual_channels, self.value_extract, 1))
            f.write(Network.bn2text(self.value_extract))
            f.write(Network.conv2text(self.value_extract, 1, 1))

            f.write(Network.fullyconnect2text(self.value_extract * 3, self.value_misc))
            f.write("end struct\n")
            f.write("get parameters\n")
            for tensor in self.tensor_collector:
                f.write(Network.tensor2text(tensor))
            f.write("end parameters\n")
            f.write("end main")

    @staticmethod
    def fullyconnect2text(in_size, out_size):
        return "FullyConnect {iS} {oS}\n".format(iS=in_size, oS=out_size)

    @staticmethod
    def conv2text(in_channels, out_channels, kernel_size):
        return "Convolution {iC} {oC} {KS}\n".format(
                   iC=in_channels,
                   oC=out_channels,
                   KS=kernel_size)

    @staticmethod
    def bn2text(channels):
        return "BatchNorm {C}\n".format(C=channels)
    
    @staticmethod
    def tensor2text(t: torch.Tensor):
        return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"
