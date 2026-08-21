import torch
import torch.nn as nn
import numpy as np
import sys

from network_core.symmetry import torch_symmetry
from network_core.utils import (
    CRAZY_NEGATIVE_VALUE,
    write_network_to_file,
    network_info_to_text,
    make_soft_porb,
    cross_entropy,
    huber_loss,
    mse_loss,
    mse_loss_spat,
    square_huber_loss,
)
from network_core.module import (
    SoftPlusWithGradientFloorFunction,
    GlobalPool,
    FullyConnect,
    Convolve,
    ConvBlock,
    DepthwiseConvBlock,
    ResidualBlock,
    BottleneckBlock,
    NestedBottleneckBlock,
    MixerBlock,
    TransformerBlock,
)

class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()

        self.layers_collector = list()

        self.nntype = cfg.nntype

        self.activation = cfg.activation.lower()
        self.input_channels = cfg.input_channels
        self.residual_channels = cfg.residual_channels
        self.xsize = cfg.boardsize
        self.ysize = cfg.boardsize
        self.policy_head_channels = cfg.policy_head_channels
        self.value_head_channels = cfg.value_head_channels
        self.se_ratio = cfg.se_ratio
        self.policy_head_type = cfg.policy_head_type
        if type(self.policy_head_type) == str:
            self.policy_head_type = { "Type" : self.policy_head_type }
        self.renorm_clipping = {"rmax" : cfg.renorm_max_r, "dmax" : cfg.renorm_max_d}
        self.value_misc = 15
        self.policy_outs = 5
        self.stack = cfg.stack
        self.version = 5

        self.construct_layers()

    def create_policy_head(self):
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_head_channels,
            kernel_size=1,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        if self.policy_head_type["Type"] == "Normal":
            pass
        elif self.policy_head_type["Type"] == "RepLK":
            dw_kernel_size = max(self.policy_head_type.get("KernelSize", 7), 7)
            self.policy_depthwise_conv = DepthwiseConvBlock(
                channels=self.policy_head_channels,
                kernel_size=dw_kernel_size,
                use_gamma=False,
                renorm_clipping=self.renorm_clipping,
                activation=self.activation,
                collector=self.layers_collector
            )
            self.policy_pointwise_conv = ConvBlock(
                in_channels=self.policy_head_channels,
                out_channels=self.policy_head_channels,
                kernel_size=1,
                use_gamma=True,
                renorm_clipping=self.renorm_clipping,
                activation=self.activation,
                collector=self.layers_collector
            )
        else:
            raise Exception("Invalid policy head type.")

        self.policy_intermediate_fc = FullyConnect(
            in_size=self.policy_head_channels * 3,
            out_size=self.policy_head_channels,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.pol_misc = Convolve(
            in_channels=self.policy_head_channels,
            out_channels=self.policy_outs,
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.pol_misc_pass_fc = FullyConnect(
            in_size=self.policy_head_channels,
            out_size=self.policy_outs,
            activation="identity",
            collector=self.layers_collector
        )

    def create_value_head(self):
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_head_channels,
            kernel_size=1,
            use_gamma=False,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.value_intermediate_fc = FullyConnect(
            in_size=self.value_head_channels * 3,
            out_size=self.value_head_channels * 3,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.ownership_conv = Convolve(
            in_channels=self.value_head_channels,
            out_channels=1,
            kernel_size=1,
            activation="identity",
            collector=self.layers_collector
        )
        self.value_misc_fc = FullyConnect(
            in_size=self.value_head_channels * 3,
            out_size=self.value_misc,
            activation="identity",
            collector=self.layers_collector
        )

    def parse_blocksetting(self, blocksetting, blockargs):
        components = list()
        if type(blocksetting) == str:
            components = blocksetting.strip().split('-')
            setting_args = dict()
        else:
            components = blocksetting["Block"].strip().split('-')
            setting_args = blocksetting["Args"]

        block = None
        channels = self.residual_channels
        for component in components:
            if component == "ResidualBlock":
                block = ResidualBlock
            elif component == "BottleneckBlock":
                blockargs["bottleneck_channels"] = channels // 2
                assert channels % 2 == 0, ""
                block = BottleneckBlock
            elif component == "NestedBottleneckBlock":
                blockargs["bottleneck_channels"] = channels // 2
                assert channels % 2 == 0, ""
                block = NestedBottleneckBlock
            elif component == "MixerBlock":
                block = MixerBlock
            elif component == "TransformerBlock":
                blockargs["pos_len"] = max(self.xsize, self.ysize)
                blockargs["num_heads"] = max(1, channels // 32)
                block = TransformerBlock
            elif component == "SE":
                blockargs["se_size"] = channels // self.se_ratio
                assert channels % self.se_ratio == 0, ""
            else:
                raise Exception("Invalid block structure.")

        if block is None:
            raise Exception("There is no basic block.")

        # overwrite default settings
        for key, value in setting_args.items():
            if key == "BottleneckChannels" :
                blockargs["bottleneck_channels"] = value
            elif key == "SeRatio" :
                blockargs["se_size"] = channels // value
                assert channels % self.se_ratio == 0, ""
            elif key == "KernelSize":
                blockargs["kernel_size"] = value
            elif key == "FfnExpansionRatio":
                blockargs["ffn_expansion_ratio"] = value
            elif key == "NumHeads":
                blockargs["num_heads"] = value
            else:
                raise Exception("Invalid block setting.")
        return block, channels, blockargs

    def create_residual_tower(self):
        self.residual_tower = nn.ModuleList()

        for blocksetting in self.stack:
            blockargs = {
                "se_size" : None,
                "activation" : self.activation,
                "collector" : self.layers_collector
            }
            block, channels, blockargs = self.parse_blocksetting(blocksetting, blockargs)
            self.residual_tower.append(block(channels=channels, **blockargs))

    def construct_layers(self):
        self.global_pool = GlobalPool(is_value_head=False)
        self.global_pool_val = GlobalPool(is_value_head=True)

        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            use_gamma=True,
            renorm_clipping=self.renorm_clipping,
            activation=self.activation,
            collector=self.layers_collector
        )
        self.create_residual_tower()
        self.create_policy_head()
        self.create_value_head()

    def forward(self, planes, *args, **kwargs):
        target = kwargs.get("target", None)
        use_symm = kwargs.get("use_symm", False)
        loss_weight_dict = kwargs.get("loss_weight_dict", None)

        symm = int(np.random.choice(8, 1)[0])
        if use_symm:
            planes = torch_symmetry(symm, planes, invert=False)

        # mask buffers
        mask = planes[:, (self.input_channels-1):self.input_channels , :, :]
        mask_sum_hw = torch.sum(mask, dim=(1,2,3))
        mask_sum_hw_sqrt = torch.sqrt(mask_sum_hw)
        mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        # input layer
        x = self.input_conv(planes, mask)

        # residual tower
        for block in self.residual_tower:
            x = block(x, mask_buffers)

        # policy head
        pol = self.policy_conv(x, mask)
        if self.policy_head_type["Type"] == "RepLK":
            pol = self.policy_depthwise_conv(pol, mask)
            pol = self.policy_pointwise_conv(pol, mask)
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
        if use_symm:
            mask = torch_symmetry(symm, mask, invert=True)
            mask_buffers = (mask, mask_sum_hw, mask_sum_hw_sqrt)

        all_loss_dict = dict()
        if target is not None:
            all_loss_dict = self.compute_loss(predict, target, mask_buffers, loss_weight_dict)

        return predict, all_loss_dict

    def compute_loss(self, pred, target, mask_buffers, loss_weight_dict):
        mask, mask_sum_hw, _ = mask_buffers
        policy_mask = torch.flatten(mask, start_dim=1, end_dim=3)
        b, _ = policy_mask.shape
        policy_mask = torch.cat((policy_mask, mask.new_ones((b, 1))), dim=1)

        if loss_weight_dict is None:
            soft_weight = 0.1
        else:
            soft_weight = loss_weight_dict["soft"]

        p_prob, p_aux_prob, p_soft_prob, p_soft_aux_prob, p_optimistic_prob, p_ownership, p_wdl, p_q_vals, p_scores, p_errors = pred
        t_prob, t_aux_prob, t_ownership, t_wdl, t_q_vals, t_scores, global_weight = target

        # will use these values later
        _, short_term_q_pred, _ = torch.split(p_q_vals, [2, 1, 2], dim=1)
        _, short_term_q_target, _ = torch.split(t_q_vals, [2, 1, 2], dim=1)
        _, short_term_score_pred, _ = torch.split(p_scores, [2, 1, 2], dim=1)
        _, short_term_score_target, _ = torch.split(t_scores, [2, 1, 2], dim=1)
        short_term_q_error, short_term_score_error = torch.split(p_errors, [1, 1], dim=1)

        # current player's probabilities loss
        prob_loss = 1. * cross_entropy(p_prob, t_prob, global_weight)

        # opponent's probabilities loss
        aux_prob_loss = 0.15 * cross_entropy(p_aux_prob, t_aux_prob, global_weight)

        # current player's soft probabilities loss
        soft_prob_loss = 1. * soft_weight * cross_entropy(p_soft_prob, make_soft_porb(t_prob, policy_mask), global_weight)

        # opponent's soft probabilities loss
        soft_aux_prob_loss = 0.15 * soft_weight * cross_entropy(p_soft_aux_prob, make_soft_porb(t_aux_prob, policy_mask), global_weight)

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
        ownership_loss = 1.5 * mse_loss_spat(p_ownership, t_ownership, mask_sum_hw, global_weight)

        # win-draw-lose loss
        wdl_loss = cross_entropy(p_wdl, t_wdl)

        # all Q values loss
        q_vals_loss = mse_loss(p_q_vals, t_q_vals, global_weight)

        # all scores loss
        scores_loss = 0.0012 * huber_loss(p_scores, t_scores, 12., global_weight)

        # all short term square error loss
        q_error_loss = 2 * square_huber_loss(
            short_term_q_error,
            short_term_q_pred.detach(),
            short_term_q_target,
            delta=0.4, eps=1.0e-8,
            weight=global_weight
        )
        score_error_loss = 0.00002 * square_huber_loss(
            short_term_score_error,
            short_term_score_pred.detach(),
            short_term_score_target,
            delta=100.0, eps=1.0e-4,
            weight=global_weight
        )
        errors_loss = q_error_loss + score_error_loss

        # add all loss
        loss = prob_loss + \
                   aux_prob_loss + \
                   soft_prob_loss + \
                   soft_aux_prob_loss + \
                   optimistic_loss + \
                   ownership_loss + \
                   wdl_loss + \
                   q_vals_loss + \
                   scores_loss + \
                   errors_loss

        # make loss dictionary
        all_loss_dict = {
            "loss"               : loss,
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

    def update_parameters(self, curr_steps):
        pass

    def accumulate_swa(self, other_network, swa_count):
        def accum_weights(v, w, n):
            # EMA formula
            if n <= 0:
                decay = 0.
            else:
                decay = n / (n + 1.)
            return decay * v.detach() + (1. - decay) * w.detach()

        for a, b in zip(self.parameters(), other_network.parameters()):
            a.data = accum_weights(a.data, b.data, swa_count)

        for a, b in zip(self.buffers(), other_network.buffers()):
            a.data = accum_weights(a.data, b.data, swa_count)

    def get_simple_info(self):
        return network_info_to_text(self)

    def get_name(self):
        blocks = len(self.stack)
        channels = self.residual_channels
        return "sayuri-b{}xc{}".format(blocks, channels)

    def get_stack_name(self, stack):
        stackname = list()
        for blocksetting in self.stack:
            if type(blocksetting) == str:
                blockname = blocksetting
            else:
                blockname = blocksetting["Block"]
            stackname.append(blockname)
        return stackname                

    def transfer_to_bin(self, filename):
        write_network_to_file(self, filename, use_bin=True)

    def transfer_to_text(self, filename):
        write_network_to_file(self, filename, use_bin=False)
