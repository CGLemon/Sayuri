import argparse

from network import Network, FullyConnect, Convolve, ConvBlock
from config import gather_config

from train import StreamLoader, StreamParser, BatchGenerator, gather_filenames
from lazy_loader import LazyLoader
import torch

def recompute_bnorm(cfg, net):
    stream_loader = StreamLoader()
    stream_parser = StreamParser(cfg.down_sample_rate)
    batch_gen = BatchGenerator(cfg.boardsize, cfg.input_channels)

    lazy_loader = LazyLoader(
        filenames = gather_filenames(cfg.train_dir),
        stream_loader = stream_loader,
        stream_parser = stream_parser,
        batch_generator = batch_gen,
        down_sample_rate = 0,
        num_workers = cfg.num_workers,
        buffer_size = cfg.buffersize,
        batch_size = cfg.batchsize
    )

    print("init loader...")
    batch = next(lazy_loader)
    print("start to recompute the batch normalize layers...")

    net.trainable(True)
    torch.set_grad_enabled(False)

    device = torch.device('cpu')
    if cfg.use_gpu:
        device = torch.device('cuda')
        net = net.to(device)

    running_loss = 0
    for steps in range(4000):
        batch = next(lazy_loader)
        _, planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score = batch

        # Move to the current device.
        if cfg.use_gpu:
            planes = planes.to(device)
            target_prob = target_prob.to(device)
            target_aux_prob = target_aux_prob.to(device)
            target_ownership = target_ownership.to(device)
            target_wdl = target_wdl.to(device)
            target_stm = target_stm.to(device)
            target_score = target_score.to(device)

        # gather batch data
        target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_stm, target_score)

        # forward and backforwad
        _, all_loss = net(planes, target, use_symm=True)

        prob_loss, aux_prob_loss, ownership_loss, wdl_loss, stm_loss, final_score_loss = all_loss
        loss = prob_loss + ownership_loss + wdl_loss + stm_loss + final_score_loss # The aux_prob_loss is not used.

        running_loss += loss.item()

        if (steps+1) % 100 == 0:
            print("steps {} -> {:.4f}".format(steps+1, running_loss/100))
            running_loss = 0


def swa(cfg, input_weights, outs_name, recompute):
    avarage_list = []
    out_net = Network(cfg)

    inputs_size = 0
    for filename in input_weights:
        avarage_list.append(Network(cfg));
        avarage_list[inputs_size].load_pt(filename)
        inputs_size += 1

    lsize = len(out_net.layer_collector)
    for i in range(lsize):
        out_layer = out_net.layer_collector[i]

        # set zeros
        if isinstance(out_layer, FullyConnect):
            out_layer.linear.weight.data *= 0
            out_layer.linear.bias.data *= 0
        elif isinstance(out_layer, Convolve):
            out_layer.conv.weight.data *= 0
            out_layer.conv.bias.data *= 0
        elif isinstance(out_layer, ConvBlock):
            out_layer.conv.weight.data *= 0
            out_layer.bn.running_mean.data *= 0
            out_layer.bn.running_var.data *= 0
            if out_layer.bn.gamma is not None:
                out_layer.bn.gamma.data *= 0
            out_layer.bn.beta.data *= 0

        # average
        for j in range(inputs_size):
            input_layer = avarage_list[j].layer_collector[i]

            if isinstance(out_layer, FullyConnect):
                out_layer.linear.weight.data += (input_layer.linear.weight.data / inputs_size)
                out_layer.linear.bias.data += (input_layer.linear.bias.data / inputs_size)
            elif isinstance(out_layer, Convolve):
                out_layer.conv.weight.data += (input_layer.conv.weight.data / inputs_size)
                out_layer.conv.bias.data += (input_layer.conv.bias.data / inputs_size)
            elif isinstance(out_layer, ConvBlock):
                out_layer.conv.weight.data += (input_layer.conv.weight.data / inputs_size)
                out_layer.bn.running_mean.data += (input_layer.bn.running_mean.data / inputs_size)
                out_layer.bn.running_var.data += (input_layer.bn.running_var.data / inputs_size)
                if out_layer.bn.gamma is not None:
                    out_layer.bn.gamma.data += (input_layer.bn.gamma.data / inputs_size)
                out_layer.bn.beta.data += (input_layer.bn.beta.data / inputs_size)

    if recompute and not cfg.fixup_batch_norm:
        recompute_bnorm(cfg, out_net)
        out_net.save_pt('temp.pt')
        out_net = out_net.to(torch.device('cpu'))
        out_net.load_pt('temp.pt')

    out_net.transfer_to_bin(args.network + ".bin.txt")
    # out_net.transfer_to_text(args.network + ".txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-j", "--json", metavar="<string>",
                        help="The setting json file name.", type=str)
    parser.add_argument("-n", "--network", metavar="<string>",
                        help="The output network name prefix.", type=str)
    parser.add_argument("-i", "--inputs", nargs="+",
                        help="The input networks list.")
    args = parser.parse_args()


    if args.json == None:
        print("Please give the setting json file.")
    elif args.network == None:
        print("Plase give the network name.")
    elif args.inputs == None:
        print("")
    else:
        swa(gather_config(args.json), args.inputs, args.network, False)
