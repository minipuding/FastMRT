import random

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from fastmrt.models.runet import Unet
from fastmrt.models.resunet import UNet as ResUnet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.normalize import denormalize
from fastmrt.data.prf import PrfFunc
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
from typing import Sequence, Dict


class UNetModule(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            level_num: int = 4,
            drop_prob: float = 0.0,
            leakyrelu_slope: float = 0.4,
            last_layer_with_act: bool = False,
            lr: float = 1e-3,
            lr_step_size: int = 40,
            lr_gamma: float = 0.1,
            weight_decay: float = 0.0,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_max_temp_thresh: int = 45,
            tmap_ablation_thresh: int = 43,
            log_images_frame_idx: int = 5,  # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(UNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                         tmap_patch_rate=tmap_patch_rate,
                                         tmap_max_temp_thresh=tmap_max_temp_thresh,
                                         tmap_ablation_thresh=tmap_ablation_thresh,
                                         log_images_frame_idx=log_images_frame_idx,
                                         log_images_freq=log_images_freq)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.level_num = level_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope
        self.last_layer_with_act = last_layer_with_act
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.model = Unet(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          base_channels=self.base_channels,
                          level_num=self.level_num,
                          drop_prob=self.drop_prob,
                          leakyrelu_slope=self.leakyrelu_slope,
                          last_layer_with_act=self.last_layer_with_act,
                          )
        # self.model = ResUnet(inout_ch=2, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)

    def training_step(self, batch, batch_idx):
        train_loss, recon_loss, cl_loss = self._cl_training_v2(batch)

        return {"loss": train_loss,
                "recon_loss": recon_loss,
                "cl_loss": cl_loss}

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        train_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        recon_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        cl_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in train_logs:
            train_loss += log["loss"]
            recon_loss += log["recon_loss"]
            cl_loss += log["cl_loss"]
        self.log("loss", train_loss, on_epoch=True, on_step=False)
        self.log("recon_loss", recon_loss, on_epoch=True, on_step=False)
        self.log("cl_loss", cl_loss, on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        output = self.model(batch.input)
        output_ref = self.model(batch.input_ref)
        val_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": denormalize(batch.input, mean, std),
            "label": denormalize(batch.label, mean, std),
            "output": denormalize(output, mean, std),
            "input_ref": denormalize(batch.input_ref, mean, std),
            "label_ref": denormalize(batch.label_ref, mean, std),
            "output_ref": denormalize(output_ref, mean, std),
            "tmap_mask": batch.tmap_mask,
            "val_loss": val_loss,
            "file_name": batch.file_name,
            "frame_idx": batch.frame_idx,
            "slice_idx": batch.slice_idx,
            "coil_idx": batch.coil_idx,
        }

    def test_step(self, batch):
        output = self.model(batch.input)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": (batch.input * std + mean),
            "output": (output * std + mean),
            "target": (batch.label * std + mean),
        }

    def predict_step(self, batch):
        output = self.model(batch.input)
        pred_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "output": denormalize(output, mean, std),
            "label": denormalize(batch.label, mean, std),
            "pred_loss": pred_loss,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
        #                                             step_size=self.lr_step_size,
        #                                             gamma=self.lr_gamma)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=300,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def cl_processing(self, inputs, labels):
        batch_size = inputs.shape[0]
        assert batch_size % 2 == 0
        half_bs = batch_size // 2
        for sample_idx in range(half_bs):
            input_phase = rt2ct(inputs[sample_idx])
            input_phase = input_phase / (torch.abs(input_phase) + torch.tensor(1e-12))
            inputs[sample_idx + half_bs] = ct2rt(torch.abs(rt2ct(inputs[sample_idx+half_bs])) * input_phase)
            label_phase = rt2ct(labels[sample_idx])
            label_phase = label_phase / (torch.abs(label_phase) + torch.tensor(1e-12))
            labels[sample_idx + half_bs] = ct2rt(torch.abs(rt2ct(labels[sample_idx+half_bs])) * label_phase)
        return inputs, labels

    def info_nce_loss(self, features, n_views=2, cl_temp=0.2):

        batch_size = features.shape[0]
        labels = torch.cat([torch.arange(batch_size // n_views) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to("cuda")

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to("cuda")
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives (modified)
        positives = torch.sum(similarity_matrix[labels.bool()].view(labels.shape[0], -1), dim=1, keepdim=True)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to("cuda")

        logits = logits / torch.tensor(cl_temp)
        return F.cross_entropy(logits, labels)

    def _normal_training(self, batch):
        output = self.model(batch.input)
        train_loss = F.l1_loss(output, batch.label)
        return train_loss

    def _cl_training(self, batch):
        alpha = 0
        output, cl_embedding = self.model.cl_forward(batch.input)
        train_loss = (1 - alpha) * F.l1_loss(output, batch.label) + alpha * self.info_nce_loss(cl_embedding)
        return train_loss

    def _cl_training_v2(self, batches):
        alpha = 0.01
        n_views = len(batches)
        # random.shuffle(batches)  # shuffle inputs
        output = self.model(batches[0].input)
        if alpha != 0:
            for idx, batch in enumerate(batches):
                if idx == 0:
                    cl_embedding = self.model.cl_forward_v2(batch.input)
                else:
                    cl_embedding = torch.cat((cl_embedding, self.model.cl_forward_v2(batch.input)), dim=0)
            cl_loss = self.info_nce_loss(cl_embedding, n_views)
        else:
            cl_loss = 0
        recon_loss = F.l1_loss(output, batches[0].label)
        # recon_loss = self.phase_loss(output, batches[0].label, batches[0].tmap_mask.unsqueeze(-3))
        train_loss = recon_loss + alpha * cl_loss

        return train_loss, recon_loss, cl_loss

    def phase_loss(self, output, label, tmap_mask):
        # sin / 2 .abs()
        # return torch.sum((tmap_mask * torch.sin(torch.angle(rt2ct(output) * torch.conj(rt2ct(label))) / torch.tensor(2))).abs(), dim=[1, 2]).mean()
        # sin / 2 ** 2
        # return torch.sum((tmap_mask * torch.sin(torch.angle(rt2ct(output) * torch.conj(rt2ct(label))) / torch.tensor(2))) ** 2, dim=[1, 2]).mean()
        # sin ** 2
        # return torch.sum((tmap_mask * torch.sin(torch.angle(rt2ct(output) * torch.conj(rt2ct(label))))).abs(), dim=[1, 2]).mean()

        # sin / 2 ** 2
        return torch.mean((tmap_mask * torch.sin(torch.angle(rt2ct(output) * torch.conj(rt2ct(label))) / torch.tensor(2))) ** 2)

