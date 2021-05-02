import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from symmetry import *
from nnprocess import NNProcess
from chunkparser import ChunkParser

from torch.utils.data import DataLoader

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))
    print("Name : {name} ->  Version : {ver}".format(name =  "Pytorch Lightning", ver = pl.__version__))

class DataSet():
    def __init__(self, cfg, dirname):
        self.parser = ChunkParser(cfg, dirname)
        self.cfg = cfg
        self.board_size = cfg.boardsize
        self.input_channels = cfg.input_channels
        self.symmetry = Symmetry(self.boardsize)

    def __getitem__(self, idx):
        data = self.parser[idx]

        symm = int(np.random.choice(8, 1)[0])

        assert self.board_size == data.board_size, ""
        num_intersections = self.board_size * self.board_size

        input_planes = np.zeros((self.input_channels, num_intersections))
        prob = np.zeros(num_intersections + 1)
        aux_prob = np.zeros(num_intersections + 1)
        ownership = np.zeros(num_intersections)
        result = np.zeros(1)
        final_score = np.zeros(1)

        # input planes
        for p in range(self.input_channels-2):
            for index in range(num_intersections):
                symm_index = self.symmetry.get_symmetry(symm, index)
                input_planes[p][symm_index] = data.planes[p][index]

        if data.to_move == 1:
            input_planes[self.input_channels-2][:] = 1
        else:
            input_planes[self.input_channels-1][:] = 1

        # probabilities, auxiliary probabilities, ownership
        for index in range(num_intersections):
            symm_index = self.symmetry.get_symmetry(symm, index)
            prob[symm_index] = data.prob[index]
            aux_prob[symm_index] = data.aux_prob[index]
            ownership[symm_index] = data.ownership[index]

        # winrate
        result[0] = data.result
        final_score[0] = data.final_score

        return (
            torch.tensor(input_planes).float(),
            torch.tensor(prob).float(),
            torch.tensor(aux_prob).float(),
            torch.tensor(ownership).float(),
            torch.tensor(result).float()
            torch.tensor(final_score).float()
        )

    def __len__(self):
        return len(self.parser)

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.batchsize =  cfg.batchsize
        self.num_workers = cfg.num_workers
        self.train_dir = cfg.train_dir
        self.val_dir = cfg.val_dir
        self.test_dir = cfg.test_dir

    def setup(self, stage):
        if stage == 'fit':
            self.train_data = DataSet(self.cfg, self.train_dir)
            self.val_data = DataSet(self.cfg, self.val_dir)

        if stage == 'test':
            self.test_data = DataSet(self.cfg, self.test_dir)

    def train_dataloader(self):
        return DataLoader(self.train_data, num_workers=self.num_workers, batch_size=self.batchsize)

    def val_dataloader(self):
        return DataLoader(self.val_data, num_workers=self.num_workers, batch_size=self.batchsize)

    def test_dataloader(self):
        return DataLoader(self.test_data, num_workers=self.num_workers , batch_size=self.batchsize)


class Network(NNProcess, pl.LightningModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trainable(True)

        self.lr = cfg.lr
        self.weight_decay = cfg.weight_decay

        # metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    
    def compute_loss(self, pred, target):
        (pred_prob, pred_aux_prob, pred_ownership, pred_result, pred_score) = pred
        (target_prob,target_aux_prob, target_ownership, target_result, target_score) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            return torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)

        porb_loss = cross_entropy(pred_pol, target_pol)
        aux_porb_loss = cross_entropy(pred_wdl, target_wdl)
        # ownership_loss = 
        # result_loss = 
        score_loss = huber_loss(pred_score, target_score, 12)


        stm_loss = F.mse_loss(pred_stm.squeeze(), target_stm.squeeze())

        return pol_loss, wdl_loss, stm_loss

    def training_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))

        loss = pol_loss + wdl_loss + stm_loss
        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(
            {
                "train_pol_loss": pol_loss,
                "train_wdl_loss": wdl_loss,
                "train_stm_loss": stm_loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))
        loss = pol_loss + wdl_loss + stm_loss
        self.log_dict(
            {
                "val_loss": loss,
                "val_pol_loss": pol_loss,
                "val_wdl_loss": wdl_loss,
                "val_stm_loss": stm_loss,
            }
        )

    def test_step(self, batch, batch_idx):
        planes, features, target_pol, target_wdl, target_stm = batch
        pred_pol, pred_wdl, pred_stm = self(planes, features)
        pol_loss, wdl_loss, stm_loss = self.compute_loss((pred_pol, pred_wdl, pred_stm), (target_pol, target_wdl, target_stm))
        loss = pol_loss + wdl_loss + stm_loss
        self.log_dict(
            {
                "test_loss": loss,
                "test_pol_loss": pol_loss,
                "test_wdl_loss": wdl_loss,
                "test_stm_loss": stm_loss,
            }
        )

    def configure_optimizers(self):
        adam_opt = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": adam_opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                adam_opt, verbose=self.cfg.misc_verbose, min_lr=5e-6
            ),
            "monitor": "val_loss",
        }
