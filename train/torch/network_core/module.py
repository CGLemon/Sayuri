import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .utils import (
    CRAZY_NEGATIVE_VALUE,
    DEFAULT_ACTIVATION,
    activation_func,
    compute_gain,
    dwconv_to_text,
    conv_to_text,
    fullyconnect_to_text,
    bn_to_text,
    tensor_to_text,
)

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
                       activation,
                       collector=None):
        super(SqueezeAndExcitation, self).__init__()

        self.activation = activation
        self.global_pool = GlobalPool(is_value_head=False)
        self.channels = channels

        self.squeeze = FullyConnect(
            in_size=self.channels * 3,
            out_size=se_size,
            activation=self.activation,
            collector=collector
        )
        self.excite = FullyConnect(
            in_size=se_size,
            out_size=self.channels * 2,
            activation="identity",
            collector=collector
        )

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
                       mode="renorm",
                       renorm_clipping={"rmax" : 1, "dmax" : 0},
                       momentum_basic_batchsize=None):
        super(BatchNorm2d, self).__init__()
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float)
        )

        self.gamma = None
        if use_gamma:
            self.gamma = torch.nn.Parameter(
                torch.ones(num_features, dtype=torch.float)
            )

        self.beta = torch.nn.Parameter(
            torch.zeros(num_features, dtype=torch.float)
        )

        self.use_gamma = use_gamma
        self.num_features = num_features
        self.eps = eps
        self.momentum = self._clamp(momentum)
        self.momentum_basic_batchsize = momentum_basic_batchsize

        self.mode = mode
        assert self.mode in ["norm", "renorm", "fixup"]

        # According to the paper "Batch Renormalization: Towards Reducing Minibatch Dependence
        # in Batch-Normalized Models", Batch-Renormalization is much faster and steady than
        # traditional Batch-Normalized when batch size is very small, eg bs=4.
        self.use_renorm = mode == "renorm"
        self.rmax = renorm_clipping["rmax"]
        self.dmax = renorm_clipping["dmax"]

        # Fixup Batch Normalization layer. According to kataGo, Batch Normalization may cause
        # some wierd reuslts becuse the inference and training computation results are different.
        # Fixup can avoid the weird forwarding result. Fixup also speeds up the performance. The
        # improvement may be around x1.6 ~ x1.8 faster.
        self.fixup = mode == "fixup"

    def get_merged_params(self):
        bn_mean = torch.zeros(self.num_features)
        bn_std = torch.zeros(self.num_features)

        # Merge four tensors (mean, variance, gamma, beta) into two tensors (
        # mean, variance).
        bn_mean[:] = self.running_mean[:]
        bn_std[:] = torch.sqrt(self.eps + self.running_var)[:]

        # Original format: gamma * ((x-mean) / std) + beta
        # Target format: (x-mean) / std
        #
        # Solve the following equation:
        #     gamma * ((x-mean) / std) + beta = (x-tgt_mean) / tgt_std
        #
        # We will get:
        #     tgt_std = std / gamma
        #     tgt_mean = mean - beta * (std / gamma)

        if self.gamma is not None:
            bn_std = bn_std / self.gamma
        if self.beta is not None:
            bn_mean = bn_mean - self.beta * bn_std
        return bn_mean, bn_std

    def _clamp(self, x, lower=0., upper=1.):
        x = max(lower, x)
        x = min(upper, x)
        return x

    def _get_momentum(self, x):
        if self.momentum_basic_batchsize is None:
            return self.momentum
        b, _, _, _ = x.shape
        return self.momentum * math.sqrt(b / self.momentum_basic_batchsize)

    def _apply_renorm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        running_std = torch.sqrt(self.running_var+self.eps).view(1, self.num_features, 1, 1)
        running_mean = self.running_mean.view(1, self.num_features, 1, 1)

        r = (
            std.detach() / running_std
        ).clamp(1 / self.rmax, self.rmax)

        d = (
            (mean.detach() - running_mean) / running_std
        ).clamp(-self.dmax, self.dmax)

        x = (x-mean)/std * r + d
        return x

    def _apply_norm(self, x, mean, var):
        mean = mean.view(1, self.num_features, 1, 1)
        std = torch.sqrt(var+self.eps).view(1, self.num_features, 1, 1)
        x = (x-mean)/std
        return x

    def forward(self, x, mask):
        if self.training and not self.fixup:
            mask_sum = torch.sum(mask) # global sum

            batch_mean = torch.sum(x, dim=(0,2,3)) / mask_sum
            zmtensor = x - batch_mean.view(1, self.num_features, 1, 1)
            batch_var = torch.sum(torch.square(zmtensor * mask), dim=(0,2,3)) / mask_sum

            if self.use_renorm:
                x = self._apply_renorm(x , batch_mean, batch_var)
            else:
                x = self._apply_norm(x , batch_mean, batch_var)

            # Update moving averages.
            momentum = self._get_momentum(x)
            self.running_mean += momentum * (batch_mean.detach() - self.running_mean)
            self.running_var += momentum * (batch_var.detach() - self.running_var)
        else:
            # Inference step or fixup, they are equal.
            x = self._apply_norm(x, self.running_mean, self.running_var)

        if self.gamma is not None:
            x = x * self.gamma.view(1, self.num_features, 1, 1)

        if self.beta is not None:
            x = x + self.beta.view(1, self.num_features, 1, 1)

        return x * mask

class BroadcastDepthwiseConv2d(nn.Module):
    def __init__(self, channels,
                       kernel_size,
                       padding="same",
                       bias=True):
        super(BroadcastDepthwiseConv2d, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.use_bias = bias

        self.weight = nn.Parameter(
            torch.randn((self.channels, 1, self.kernel_size, self.kernel_size), dtype=torch.float)
        )
        self.gamma = nn.Parameter(
            torch.ones(self.channels) / math.sqrt(self.channels)
        )
        if self.use_bias:
            self.bias = nn.Parameter(
                torch.zeros(self.channels, dtype=torch.float)
            )

    def _compute_equivalent_weight(self):
        return self.weight + torch.sum(self.weight * self.gamma.view(self.channels, 1, 1, 1), dim=0, keepdim=True)

    def get_merged_params(self):
        weight = torch.zeros_like(self.weight)
        bias = torch.zeros(self.channels)

        weight[:] = self._compute_equivalent_weight().detach()[:]
        if self.use_bias:
            bias[:] = self.bias[:]
        return weight, bias

    def forward(self, x):
        weight = self._compute_equivalent_weight()
        x = F.conv2d(
            x,
            weight,
            padding=self.padding,
            groups=self.channels
        )
        if self.use_bias:
            x = x + self.bias.view(1, self.channels, 1, 1)
        return x

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       activation,
                       collector=None):
        super(FullyConnect, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.linear = nn.Linear(
            in_size,
            out_size,
            bias=True
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.linear.weight, gain=compute_gain(self.activation))
        nn.init.zeros_(self.linear.bias)

    def _try_collect(self, collector):
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
        x = self.act(x)
        return x

class Convolve(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       activation,
                       collector=None):
        super(Convolve, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding="same",
            bias=True,
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))
        nn.init.zeros_(self.conv.bias)

    def _try_collect(self, collector):
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
        x = self.act(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       use_gamma,
                       renorm_clipping,
                       activation,
                       collector=None):
        super(ConvBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
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
            mode="renorm",
            renorm_clipping=renorm_clipping,
            momentum_basic_batchsize=256
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        out = str()
        out += conv_to_text(self.in_channels, self.out_channels, self.kernel_size)
        out += bn_to_text(self.out_channels)
        return out

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()
        out += tensor_to_text(self.conv.weight, use_bin)
        out += tensor_to_text(torch.zeros(self.out_channels), use_bin) # fill zero

        bn_mean, bn_std = self.bn.get_merged_params()
        out += tensor_to_text(bn_mean, use_bin)
        out += tensor_to_text(bn_std, use_bin)
        return out

    def forward(self, x, mask):
        x = self.conv(x) * mask
        x = self.bn(x, mask)
        x = self.act(x)
        return x

class DepthwiseConvBlock(nn.Module):
    def __init__(self, channels,
                       kernel_size,
                       use_gamma,
                       renorm_clipping,
                       activation,
                       collector=None):
        # Implement it based on "Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design
        # in CNNs".

        assert kernel_size >= 5, ""
        assert kernel_size % 2 == 1, ""
        super(DepthwiseConvBlock, self).__init__()

        self.channels = channels
        self.kernel_size = kernel_size
        self.groups = self.channels
        self.conv = BroadcastDepthwiseConv2d(
            self.channels,
            self.kernel_size,
            padding="same",
            bias=True
        )
        self.rep3x3 = BroadcastDepthwiseConv2d(
            self.channels,
            3,
            padding="same",
            bias=True
        )
        self.bn = BatchNorm2d(
            num_features=self.channels,
            eps=1e-5,
            use_gamma=use_gamma,
            mode="renorm",
            renorm_clipping=renorm_clipping,
            momentum_basic_batchsize=256
        )
        self.activation = activation
        self.act = activation_func(self.activation, inplace=True)
        self._init_weights()
        self._try_collect(collector)

    def _init_weights(self):
        nn.init.xavier_normal_(
            self.conv.weight, gain=compute_gain(self.activation))
        nn.init.xavier_normal_(
            self.rep3x3.weight, gain=compute_gain(self.activation))

    def tensors_to_text(self, use_bin):
        if use_bin:
            out = bytes()
        else:
            out = str()

        weights, biases = self.conv.get_merged_params()

        ps = int((self.kernel_size - 3) / 2)
        rep3x3_weights, rep3x3_biases = self.rep3x3.get_merged_params()
        weights += F.pad(rep3x3_weights, (ps, ps, ps, ps), "constant", 0)
        biases += rep3x3_biases

        out += tensor_to_text(weights, use_bin)
        out += tensor_to_text(biases, use_bin)

        bn_mean, bn_std = self.bn.get_merged_params()
        out += tensor_to_text(bn_mean, use_bin)
        out += tensor_to_text(bn_std, use_bin)
        return out

    def _try_collect(self, collector):
        if collector is not None:
            collector.append(self)

    def shape_to_text(self):
        out = str()
        out += dwconv_to_text(self.channels // self.groups, self.channels, self.kernel_size)
        out += bn_to_text(self.channels)
        return out

    def forward(self, x, mask):
        x = (self.conv(x) + self.rep3x3(x)) * mask
        x = self.bn(x, mask)
        x = self.act(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(ResidualBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.se_size = kwargs.get("se_size", None)
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        self.conv1 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        if self.use_se:
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class BottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(BottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        collector = kwargs.get("collector", None)

        assert self.bottleneck_channels is not None, ""
        self.use_se = self.se_size is not None

        # The inner layers channels.
        self.inner_channels = self.bottleneck_channels

        # The main ResidualBlock channels. We say a 15x192
        # resnet. The 192 is outer_channel.
        self.outer_channels = channels

        self.pre_btl_conv = ConvBlock(
            in_channels=self.outer_channels,
            out_channels=self.inner_channels,
            kernel_size=1,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv1 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.inner_channels,
            kernel_size=3,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        out = self.pre_btl_conv(out, mask)
        out = self.conv1(out, mask)
        out = self.conv2(out, mask)
        out = self.post_btl_conv(out, mask)
        if self.use_se:
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class NestedBottleneckBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(NestedBottleneckBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.bottleneck_channels = kwargs.get("bottleneck_channels", None)
        self.se_size = kwargs.get("se_size", None)
        collector = kwargs.get("collector", None)

        assert self.bottleneck_channels is not None, ""
        self.use_se = self.se_size is not None

        # The inner layers channels.
        self.inner_channels = self.bottleneck_channels

        # The main ResidualBlock channels. We say a 15x192
        # resnet. The 192 is outer_channel.
        self.outer_channels = channels

        self.pre_btl_conv = ConvBlock(
            in_channels=self.outer_channels,
            out_channels=self.inner_channels,
            kernel_size=1,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.block1 = ResidualBlock(
            channels=self.inner_channels,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.block2 = ResidualBlock(
            channels=self.inner_channels,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.post_btl_conv = ConvBlock(
            in_channels=self.inner_channels,
            out_channels=self.outer_channels,
            kernel_size=1,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.outer_channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        out = x
        out = self.pre_btl_conv(out, mask)
        out = self.block1(out, mask_buffers)
        out = self.block2(out, mask_buffers)
        out = self.post_btl_conv(out, mask)
        if self.use_se:
            out = self.se_module(out, mask_buffers)
        out = out + x
        out = self.act(out)
        return out

class MixerBlock(nn.Module):
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(MixerBlock, self).__init__()

        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        self.renorm_clipping = kwargs.get("renorm_clipping", {"rmax" : 1, "dmax" : 0})
        self.se_size = kwargs.get("se_size", None)
        self.kernel_size = kwargs.get("kernel_size", 7)
        self.ffn_expansion_ratio = kwargs.get("ffn_expansion_ratio", 1.5)
        self.version = kwargs.get("version", 1)
        collector = kwargs.get("collector", None)

        self.channels = channels
        self.use_se = self.se_size is not None
        assert self.version in [1, 2], ""

        self.depthwise_conv = DepthwiseConvBlock(
            channels=self.channels,
            kernel_size=self.kernel_size,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )

        self.ffn_channels = int(self.ffn_expansion_ratio * self.channels)
        self.ffn1 = ConvBlock(
            in_channels=self.channels,
            out_channels=self.ffn_channels,
            kernel_size=1,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=collector
        )
        self.ffn2 = ConvBlock(
            in_channels=self.ffn_channels,
            out_channels=self.channels,
            kernel_size=1,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation="identity",
            collector=collector
        )
        if self.use_se:
            self.se_module = SqueezeAndExcitation(
                channels=self.channels,
                se_size=self.se_size,
                activation=self.activation,
                collector=collector
            )
        self.act = activation_func(self.activation, inplace=True)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        if self.version == 1:
            x = self.depthwise_conv(x, mask) + x
            out = x
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se:
                out = self.se_module(out, mask_buffers)
            out = out + x
            out = self.act(out)
        elif self.version == 2:
            out = x
            out = self.depthwise_conv(out, mask)
            out = self.ffn1(out, mask)
            out = self.ffn2(out, mask)
            if self.use_se:
                out = self.se_module(out, mask_buffers)
            out = out + x
            out = self.act(out)
        return out

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _apply_norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._apply_norm(x.float()).type_as(x)
        return output * self.weight

class RoPE(nn.Module):
    # Learnable 2D rotary position embedding. One (omega_x, omega_y) frequency
    # pair per head, per dimension pair; applied to a pair of (B, S, H, D)
    # tensors (q, k) given the board's row-major sequence layout.
    def __init__(self, head_dim, num_heads, pos_len):
        super(RoPE, self).__init__()
        assert head_dim % 4 == 0, ""

        self.head_dim = head_dim
        self.num_heads = num_heads
        self.pos_len = pos_len

        # Geometric init from 1 rad/square to 1/50 rad/square.
        num_pairs = head_dim // 2
        log_lo, log_hi = math.log(1.0 / 50.0), math.log(1.0)
        init_freqs = (
            torch.exp(torch.empty(num_heads, num_pairs, 2).uniform_(log_lo, log_hi))
            * (torch.randint(0, 2, (num_heads, num_pairs, 2)) * 2 - 1).float()
        )
        self.rope_freqs = nn.Parameter(init_freqs)  # (num_heads, P, 2)

    def _compute_cos_sin(self, s_x, s_y):
        # s_x, s_y: (S,) column/row positions. Returns (cos, sin), both (S, H, P).
        angles = s_x.unsqueeze(-1).unsqueeze(-1) * self.rope_freqs[:, :, 0] + \
            s_y.unsqueeze(-1).unsqueeze(-1) * self.rope_freqs[:, :, 1]
        return torch.cos(angles), torch.sin(angles)

    def _rotate(self, x, cos, sin):
        # x: (B, S, H, D); cos, sin: (S, H, D/2).
        b, s, h, d = x.shape
        p = d // 2
        x0, x1 = x.view(b, s, h, p, 2).unbind(dim=-1)
        cos = cos.unsqueeze(0)  # (1, S, H, P)
        sin = sin.unsqueeze(0)
        out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], dim=-1)
        return out.reshape(b, s, h, d).type_as(x)

    def forward(self, xq, xk):
        # xq, xk: (B, S, H, D)
        seq_len = xq.shape[1]
        s_idx = torch.arange(seq_len, device=xq.device)
        s_y = (s_idx // self.pos_len).float()  # row
        s_x = (s_idx % self.pos_len).float()   # col
        cos, sin = self._compute_cos_sin(s_x, s_y)
        return self._rotate(xq, cos, sin), self._rotate(xk, cos, sin)

class MultiHeadAttention(nn.Module):
    # Multi-head self-attention sublayer (pre-norm). Always uses learnable 2D
    # RoPE with num_heads == num_kv_heads. Returns the attention output only,
    # (B, C, H, W); caller is responsible for adding the residual.
    def __init__(self, channels,
                       pos_len,
                       num_heads,
                       collector=None):
        super(MultiHeadAttention, self).__init__()

        self.pos_len = pos_len
        self.num_heads = num_heads
        self.q_head_dim = channels // self.num_heads
        self.v_head_dim = channels // self.num_heads

        self.q_proj = FullyConnect(
            channels, self.num_heads * self.q_head_dim, activation="identity", collector=collector)
        self.k_proj = FullyConnect(
            channels, self.num_heads * self.q_head_dim, activation="identity", collector=collector)
        self.v_proj = FullyConnect(
            channels, self.num_heads * self.v_head_dim, activation="identity", collector=collector)
        self.out_proj = FullyConnect(
            self.num_heads * self.v_head_dim, channels, activation="identity", collector=collector)

        self.rope = RoPE(head_dim=self.q_head_dim, num_heads=self.num_heads, pos_len=self.pos_len)

        self.norm1 = RMSNorm(channels, eps=1e-6)

    def forward(self, x, mask):
        # x: (B, C, H, W); mask: (B, 1, H, W).
        b, c, h, w = x.shape
        seq_len = h * w

        x_in = x.view(b, c, -1).permute(0, 2, 1)  # NSC

        x_norm = self.norm1(x_in)
        q = self.q_proj(x_norm).view(b, seq_len, self.num_heads, self.q_head_dim)
        k = self.k_proj(x_norm).view(b, seq_len, self.num_heads, self.q_head_dim)
        v = self.v_proj(x_norm).view(b, seq_len, self.num_heads, self.v_head_dim)

        q, k = self.rope(q, k)

        q = q.permute(0, 2, 1, 3)  # (B, H, S, Dq)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        mask_flat = mask.reshape(b, 1, 1, seq_len)
        attn_mask = torch.zeros_like(mask_flat, dtype=q.dtype)
        attn_mask.masked_fill_(mask_flat == 0, float("-inf"))

        scale = 1.0 / math.sqrt(self.q_head_dim)
        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, scale=scale)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(b, seq_len, self.num_heads * self.v_head_dim)
        attn_output = self.out_proj(attn_output)
        return attn_output.permute(0, 2, 1).view(b, c, h, w)

class TransformerBlock(nn.Module):
    # A complete transformer block: a self-attention sublayer followed by a
    # SwiGLU feed-forward sublayer, each pre-norm and wrapped with its own
    # residual connection, so this already returns x + block(x).
    def __init__(self, channels,
                       *args,
                       **kwargs):
        super(TransformerBlock, self).__init__()
        self.ffn_expansion_ratio = kwargs.get("ffn_expansion_ratio", 1.5)
        self.pos_len = kwargs.get("pos_len", 19)
        self.num_heads = kwargs.get("num_heads", 3)
        self.activation = kwargs.get("activation", DEFAULT_ACTIVATION)
        collector = None

        self.mha = MultiHeadAttention(
            channels, pos_len=self.pos_len, num_heads=self.num_heads, collector=collector)

        self.ffn_dim = int(channels * self.ffn_expansion_ratio)
        self.ffn_linear1 = FullyConnect(
            channels, self.ffn_dim, activation=self.activation, collector=collector)
        self.ffn_linear_gate = FullyConnect(
            channels, self.ffn_dim, activation="identity", collector=collector)
        self.ffn_linear2 = FullyConnect(
            self.ffn_dim, channels, activation="identity", collector=collector)
        self.norm2 = RMSNorm(channels, eps=1e-6)

    def forward(self, x, mask_buffers):
        mask, _, _ = mask_buffers

        # Self-attention sublayer (pre-norm, residual).
        x = x + self.mha(x, mask)

        # Feed-forward sublayer (pre-norm, residual, gate).
        b, c, h, w = x.shape
        x_in = x.view(b, c, -1).permute(0, 2, 1)  # NSC

        xn = self.norm2(x_in)
        ffn_output = self.ffn_linear1(xn) * self.ffn_linear_gate(xn)
        ffn_output = self.ffn_linear2(ffn_output)
        ffn_output = ffn_output.permute(0, 2, 1).view(b, c, h, w)

        return x + ffn_output
