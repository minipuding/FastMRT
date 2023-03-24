import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from fastmrt.models.cunet import Unet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.normalize import denormalize
from fastmrt.data.prf import PrfFunc
from fastmrt.utils.trans import complex_tensor_to_real_tensor as ct2rt
import pdb


class CUNetModule(BaseModule):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            base_channels: int = 32,
            level_num: int = 4,
            drop_prob: float = 0.0,
            leakyrelu_slope: float = 0.1,
            last_layer_with_act: bool = False,
            lr: float = 4e-5,
            weight_decay: float = 1e-4,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_max_temp_thresh: int = 45,
            tmap_ablation_thresh: int = 43,
            log_images_frame_idx: int = 5,  # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(CUNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
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
        self.weight_decay = weight_decay
        self.model = Unet(in_channels=self.in_channels,
                          out_channels=self.out_channels,
                          base_channels=self.base_channels,
                          level_num=self.level_num,
                          drop_prob=self.drop_prob,
                          leakyrelu_slope=self.leakyrelu_slope,
                          last_layer_with_act=self.last_layer_with_act,
                          )

    def training_step(self, batch):
        train_loss = self._decoupled_loss(batch)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        output = self.model(batch.input)
        output_ref = self.model(batch.input_ref)
        val_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": ct2rt(denormalize(batch.input, mean, std).squeeze(1)),
            "label": ct2rt(denormalize(batch.label, mean, std).squeeze(1)),
            "output": ct2rt(denormalize(output, mean, std).squeeze(1)),
            "input_ref": ct2rt(denormalize(batch.input_ref, mean, std).squeeze(1)),
            "label_ref": ct2rt(denormalize(batch.label_ref, mean, std).squeeze(1)),
            "output_ref": ct2rt(denormalize(output_ref, mean, std).squeeze(1)),
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
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.lr,
                                      weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=200,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def _l1_loss(self, batch):
        output = self.model(batch.input)
        train_loss = F.l1_loss(output.real, batch.label.real) + F.l1_loss(output.imag, batch.label.imag)
        return train_loss
    
    def _decoupled_loss(self, batch):
        output = self.model(batch.input)

        # amplitude loss
        amp_loss = F.l1_loss(output.abs(), batch.label.abs())

        # phase loss
        phs_loss = (torch.angle(output * torch.conj(batch.label)) * batch.label.abs()).abs().sum(0).mean()

        # train_loss
        train_loss = amp_loss + phs_loss / torch.pi

        return train_loss