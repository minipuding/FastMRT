import torch
import torch.nn.functional as F
from fastmrt.models.swtnet import SwinIR
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.trans import real_tensor_to_complex_tensor as rt2ct
from fastmrt.utils.normalize import denormalize
from fastmrt.data.prf import PrfFunc
from typing import Sequence, Dict


class SwtNetModule(BaseModule):
    def __init__(
            self,
            upscale=1,
            in_channels=2,
            img_size=[96, 96],
            patch_size=1,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2.0,
            upsampler='',
            resi_connection='1conv',
            lr: float = 1e-3,
            weight_decay: float = 1e-4,
            tmap_prf_func: PrfFunc = None,
            tmap_patch_rate: int = 4,
            tmap_max_temp_thresh: int = 45,
            tmap_ablation_thresh: int = 43,
            log_images_frame_idx: int = 5,  # recommend 4 ~ 8
            log_images_freq: int = 50,
    ):
        super(SwtNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                           tmap_patch_rate=tmap_patch_rate,
                                           tmap_max_temp_thresh=tmap_max_temp_thresh,
                                           tmap_ablation_thresh=tmap_ablation_thresh,
                                           log_images_frame_idx=log_images_frame_idx,
                                           log_images_freq=log_images_freq)
        self.in_channels = in_channels
        self.upscale = upscale
        self.img_size = img_size
        self.patch_size = patch_size
        self.window_size = window_size
        self.img_range = img_range
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.upsampler = upsampler
        self.resi_connection = resi_connection
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = SwinIR(upscale=self.upscale,
                            in_chans=self.in_channels,
                            img_size=self.img_size,
                            patch_size=self.patch_size,
                            window_size=self.window_size,
                            img_range=self.img_range,
                            depths=self.depths,
                            embed_dim=self.embed_dim,
                            num_heads=self.num_heads,
                            mlp_ratio=self.mlp_ratio,
                            upsampler=self.upsampler,
                            resi_connection=self.resi_connection
                            )
        # self.model = ResUnet(inout_ch=2, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)

    def training_step(self, batch, batch_idx):
        train_loss = self._normal_training(batch)

        return {"loss": train_loss}

    def training_epoch_end(self, train_logs: Sequence[Dict]) -> None:
        train_loss = torch.tensor(0, dtype=torch.float32, device='cuda')
        for log in train_logs:
            train_loss += log["loss"]
        self.log("loss", train_loss, on_epoch=True, on_step=False)

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
                                                               T_max=200,
                                                               last_epoch=-1)
        return [optimizer], [scheduler]

    def _normal_training(self, batch):
        output = self.model(batch.input)
        train_loss = F.l1_loss(output, batch.label)
        return train_loss

    def _decoupled_loss(self, batch):
        output = self.model(batch.input)
        output_complex = rt2ct(output)
        label_complex = rt2ct(batch.label)

        # amplitude loss
        amp_loss = F.l1_loss(output_complex.abs(), label_complex.abs())

        # phase loss
        phs_loss = (torch.angle(output_complex * torch.conj(label_complex)) * label_complex.abs()).abs().sum(0).mean()

        return amp_loss + phs_loss / torch.pi


