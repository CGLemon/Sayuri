import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from symmetry import *
from nnprocess import NNProcess
from loader import Loader

from torch.utils.data import DataLoader

def dump_dependent_version():
    print("Name: {name} ->  Version: {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name: {name} ->  Version: {ver}".format(name =  "Torch", ver = torch.__version__))
    print("Name : {name} ->  Version : {ver}".format(name =  "Pytorch Lightning", ver = pl.__version__))

class DataSet():
    def __init__(self, cfg, dirname):
        self.data_loader = Loader(cfg, dirname)
        self.cfg = cfg
        self.board_size = cfg.boardsize
        self.input_channels = cfg.input_channels
        self.symmetry = Symmetry()

    def __getitem__(self, idx):
        data = self.data_loader[idx]

        symm = int(np.random.choice(8, 1)[0])

        num_intersections = data.board_size * data.board_size

        input_planes = np.zeros((self.input_channels, num_intersections))
        prob = np.zeros(num_intersections+1)
        aux_prob = np.zeros(num_intersections+1)
        ownership = np.zeros(num_intersections)
        result = np.zeros(3)
        final_score = np.zeros(1)

        buf = np.zeros(num_intersections)

        # input planes
        for p in range(self.input_channels-2):
            buf[:] = data.planes[p][:]
            buf = self.symmetry.get_transform_planes(symm, buf, data.board_size)
            input_planes[p][:] = buf[:]

        if data.to_move == 1:
            input_planes[self.input_channels-2][:] = data.komi/10
        else:
            input_planes[self.input_channels-1][:] = data.komi/10

        # probabilities
        buf[:] = data.prob[0:num_intersections]
        buf = self.symmetry.get_transform_planes(symm, buf, data.board_size)

        prob[0:num_intersections] = buf[:]
        prob[num_intersections] = data.prob[num_intersections]

        # auxiliary probabilities
        buf[:] = data.aux_prob[0:num_intersections]
        buf = self.symmetry.get_transform_planes(symm, buf, data.board_size)

        aux_prob[0:num_intersections] = buf[:]
        aux_prob[num_intersections] = data.aux_prob[num_intersections]

        # ownership
        buf[:] = data.ownership[:]
        buf = self.symmetry.get_transform_planes(symm, buf, data.board_size)
        ownership[:] = buf[:]

        # winrate
        result[1 - data.result] = 1
        final_score[0] = data.final_score

        input_planes = np.reshape(input_planes, (self.input_channels, data.board_size, data.board_size))

        return (
            torch.tensor(input_planes).float(),
            torch.tensor(prob).float(),
            torch.tensor(aux_prob).float(),
            torch.tensor(ownership).float(),
            torch.tensor(result).float(),
            torch.tensor(final_score).float()
        )

    def __len__(self):
        return len(self.data_loader)

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

        self.learn_rate = cfg.learn_rate
        self.weight_decay = cfg.weight_decay

        # metrics
        self.train_accuracy = pl.metrics.Accuracy()
        self.val_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

    def compute_loss(self, pred, target):
        (pred_prob, pred_aux_prob, pred_ownership, pred_result, pred_score) = pred
        (target_prob,target_aux_prob, target_ownership, target_result, target_final_score) = target

        def cross_entropy(pred, target):
            return torch.mean(-torch.sum(torch.mul(F.log_softmax(pred, dim=-1), target), dim=1), dim=0)

        def huber_loss(x, y, delta):
            absdiff = torch.abs(x - y)
            loss = torch.where(absdiff > delta, (0.5 * delta*delta) + delta * (absdiff - delta), 0.5 * absdiff * absdiff)
            return torch.mean(torch.sum(loss, dim=1), dim=0)

        porb_loss = cross_entropy(pred_prob, target_prob)
        aux_porb_loss = 0.15 * cross_entropy(pred_aux_prob, target_aux_prob)
        ownership_loss = 0.15 * F.mse_loss(pred_ownership, target_ownership)
        wdl_loss = cross_entropy(pred_result, target_result)

        pred_final_score, pred_score_width = torch.split(pred_score, [1, 1], dim=1)
        pred_final_score = 20 * pred_final_score
        fina_score_loss = 0.0012 * huber_loss(pred_final_score, target_final_score, 12)

        target_score_width = torch.pow(pred_final_score - target_final_score, 2)
        score_width_loss = 0.0012 * huber_loss(pred_score_width, target_score_width, 12)

        return porb_loss, aux_porb_loss, ownership_loss, wdl_loss, fina_score_loss, score_width_loss

    def training_step(self, batch, batch_idx):
        planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_score = batch
        target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_score)
        predict = self(planes)

        porb_loss, aux_porb_loss, ownership_loss, wdl_loss, fina_score_loss, score_width_loss = self.compute_loss(predict, target)
        loss = porb_loss + aux_porb_loss + ownership_loss + wdl_loss + fina_score_loss + score_width_loss

        self.log("train_loss", loss, prog_bar=True)
        self.log_dict(
            {
                "train_prob_loss": porb_loss,
                "train_aux_prob_loss": aux_porb_loss,
                "train_ownership_loss": ownership_loss,
                "train_wdl_loss": wdl_loss,
                "train_fina_score_loss": fina_score_loss,
                "train_fscore_width_loss": score_width_loss,
            }
        )
        return loss

    def validation_step(self, batch, batch_idx):
        planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_score = batch
        target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_score)
        predict = self(planes)

        porb_loss, aux_porb_loss, ownership_loss, wdl_loss, fina_score_loss, score_width_loss = self.compute_loss(predict, target)
        loss = porb_loss + aux_porb_loss + ownership_loss + wdl_loss + fina_score_loss + score_width_loss

        self.log_dict(
            {
                "val_loss": loss,
                "val_prob_loss": porb_loss,
                "val_aux_prob_loss": aux_porb_loss,
                "val_ownership_loss": ownership_loss,
                "val_wdl_loss": wdl_loss,
                "val_fina_score_loss": fina_score_loss,
                "val_fscore_width_loss": score_width_loss,
            }
        )

    def test_step(self, batch, batch_idx):
        planes, target_prob, target_aux_prob, target_ownership, target_wdl, target_score = batch
        target = (target_prob, target_aux_prob, target_ownership, target_wdl, target_score)
        predict = self(planes)

        porb_loss, aux_porb_loss, ownership_loss, wdl_loss, fina_score_loss, score_width_loss = self.compute_loss(predict, target)
        loss = porb_loss + aux_porb_loss + ownership_loss + wdl_loss + fina_score_loss + score_width_loss

        self.log_dict(
            {
                "test_loss": loss,
                "test_prob_loss": porb_loss,
                "test_aux_prob_loss": aux_porb_loss,
                "test_ownership_loss": ownership_loss,
                "test_wdl_loss": wdl_loss,
                "test_fina_score_loss": fina_score_loss,
                "test_fscore_width_loss": score_width_loss,
            }
        )

    def configure_optimizers(self):
        adam_opt = torch.optim.Adam(
            self.parameters(),
            lr=self.learn_rate,
            weight_decay=self.weight_decay,
        )
        return {
            "optimizer": adam_opt,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                adam_opt, verbose=self.cfg.misc_verbose, min_lr=5e-6
            ),
            "monitor": "val_loss",
        }

