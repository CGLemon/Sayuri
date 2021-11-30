import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True) if self.relu else x


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
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
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-5,
            affine=False,
        )

        if collector != None:
            collector.append(self.conv.weight)
            collector.append(torch.zeros(out_channels))
            collector.append(self.bn.running_mean)
            collector.append(self.bn.running_var)

        nn.init.kaiming_normal_(self.conv.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) if self.relu else x

class ResBlock(nn.Module):
    def __init__(self, channels, se_size=None, collector=None):
        super().__init__()
        self.with_se=False
        self.channels=channels

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            relu=False,
            collector=collector
        )

        if se_size != None:
            self.with_se = True
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.extend = FullyConnect(
                in_size=channels,
                out_size=se_size,
                relu=True,
                collector=collector
            )
            self.squeeze = FullyConnect(
                in_size=se_size,
                out_size=2 * channels,
                relu=False,
                collector=collector
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.with_se:
            b, c, _, _ = out.size()
            seprocess = self.avg_pool(out)
            seprocess = torch.flatten(seprocess, start_dim=1, end_dim=3)
            seprocess = self.extend(seprocess)
            seprocess = self.squeeze(seprocess)

            gammas, betas = torch.split(seprocess, self.channels, dim=1)
            gammas = torch.reshape(gammas, (b, c, 1, 1))
            betas = torch.reshape(betas, (b, c, 1, 1))
            out = torch.sigmoid(gammas) * out + betas
            
        out += identity

        return F.relu(out, inplace=True)


class NNProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Tensor Collector will collect all weights which we want automatically.
        # According to multi-task learning, some heads are only auxiliary. These
        # heads may be not useful in inference process. So they will not be 
        # collected by Tensor Collector.
        self.tensor_collector = []

        self.cfg = cfg
        self.nntype = cfg.nntype

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
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            relu=True,
            collector=self.tensor_collector
        )

        # residual tower
        nn_stack = []
        for s in self.stack:
            if s == "ResidualBlock":
                nn_stack.append(ResBlock(self.residual_channels,
                                         None,
                                         self.tensor_collector))
            elif s == "ResidualBlock-SE":
                nn_stack.append(ResBlock(self.residual_channels,
                                         self.residual_channels * 4,
                                         self.tensor_collector))
        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_extract,
            kernel_size=1,
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
            in_size=self.policy_extract,
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
            in_size=self.policy_extract,
            out_size=1,
            relu=False,
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_extract,
            kernel_size=1,
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
            in_size=self.value_extract,
            out_size=self.value_misc,
            relu=False,
            collector=self.tensor_collector
        )

    def forward(self, planes):
        x = self.input_conv(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)

        prob_without_pass = self.prob(pol)
        prob_without_pass = torch.flatten(prob_without_pass, start_dim=1, end_dim=3)
        pol_gpool = self.avg_pool(pol)
        pol_gpool = torch.flatten(pol_gpool, start_dim=1, end_dim=3)
        prob_pass = self.prob_pass_fc(pol_gpool)

        prob = torch.cat((prob_without_pass, prob_pass), 1)

        # auxiliary policy
        aux_prob_without_pass = self.aux_prob(pol)
        aux_prob_without_pass = torch.flatten(aux_prob_without_pass, start_dim=1, end_dim=3)
        aux_pol_gpool = self.avg_pool(pol)
        aux_pol_gpool = torch.flatten(aux_pol_gpool, start_dim=1, end_dim=3)
        aux_prob_pass = self.aux_prob_pass_fc(aux_pol_gpool)

        aux_prob = torch.cat((aux_prob_without_pass, aux_prob_pass), 1)

        # value head
        val = self.value_conv(x)

        ownership = self.ownership_conv(val)
        ownership = torch.flatten(ownership, start_dim=1, end_dim=3)
        ownership = torch.tanh(ownership)

        val_gpool = self.avg_pool(val)
        val_gpool = torch.flatten(val_gpool, start_dim=1, end_dim=3)
        val_misc = self.value_misc_fc(val_gpool)

        wdl, stm, score = torch.split(val_misc, [3, 1, 1], dim=1)
        stm = torch.tanh(stm)

        return prob, aux_prob, ownership, wdl, stm, score

    def trainable(self, t=True):
        if t==True:
            self.train()
        else:
            self.eval()

    def save_pt(self, filename):
        torch.save(self.state_dict(), filename)

    def load_pt(self, filename):
        self.load_state_dict(torch.load(filename))

    def dump_info(self):
        print("NN Type: {type}".format(type=self.nntype))
        print("Plane size [x,y]: [{xsize}, {ysize}] ".format(xsize=self.xsize, ysize=self.ysize))
        print("Input channels: {channels}".format(channels=self.input_channels))
        print("Residual channels: {channels}".format(channels=self.residual_channels))
        print("Residual tower: size -> {s} [".format(s=len(self.stack)))
        for s in self.stack:
            print("  {}".format(s))
        print("]")
        print("Policy Extract: {policyextract}".format(policyextract=self.policy_extract))
        print("Value Extract: {valueextract}".format(valueextract=self.value_extract))
        print("Value Misc: {valuemisc}".format(valuemisc=self.value_misc))

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
            f.write(NNProcess.conv2text(self.input_channels, self.residual_channels, 3))
            f.write(NNProcess.bn2text(self.residual_channels))

            for s in self.stack:
                if s == "ResidualBlock" or s == "ResidualBlock-SE":
                    f.write(NNProcess.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(NNProcess.bn2text(self.residual_channels))
                    f.write(NNProcess.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(NNProcess.bn2text(self.residual_channels))
                    if s == "ResidualBlock-SE":
                        f.write(NNProcess.fullyconnect2text(self.residual_channels * 1, self.residual_channels * 4))
                        f.write(NNProcess.fullyconnect2text(self.residual_channels * 4, self.residual_channels * 2))

            # policy head
            f.write(NNProcess.conv2text(self.residual_channels, self.policy_extract, 1))
            f.write(NNProcess.bn2text(self.policy_extract))
            f.write(NNProcess.conv2text(self.policy_extract, 1, 1))
            f.write(NNProcess.fullyconnect2text(self.policy_extract, 1))

            # value head
            f.write(NNProcess.conv2text(self.residual_channels, self.value_extract, 1))
            f.write(NNProcess.bn2text(self.value_extract))
            f.write(NNProcess.conv2text(self.value_extract, 1, 1))

            f.write(NNProcess.fullyconnect2text(self.value_extract, self.value_misc))
            f.write("end struct\n")
            f.write("get parameters\n")
            for tensor in self.tensor_collector:
                f.write(NNProcess.tensor2text(tensor))
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
