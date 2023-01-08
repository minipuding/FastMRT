import torch
import torch.nn.functional as F
from fastmrt.models.casnet import CasNet
from fastmrt.modules.base_module import BaseModule
from fastmrt.utils.normalize import denormalize
from fastmrt.data.prf import PrfFunc


class CasNetModule(BaseModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        res_block_num: int = 5,
        res_conv_ksize: int = 3,
        res_conv_num: int = 5,
        drop_prob: float = 0.0,
        leakyrelu_slope = 0.4,
        lr: float = 1e-3,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        tmap_prf_func: PrfFunc = None,
        tmap_patch_rate: int = 4,
        tmap_max_temp_thresh = 45,
        tmap_ablation_thresh = 43,
        log_images_frame_idx: int = 5, # recommend 4 ~ 8
        log_images_freq: int = 50,
    ):
        super(CasNetModule, self).__init__(tmap_prf_func=tmap_prf_func,
                                           tmap_patch_rate=tmap_patch_rate,
                                           tmap_max_temp_thresh=tmap_max_temp_thresh,
                                           tmap_ablation_thresh=tmap_ablation_thresh,
                                           log_images_frame_idx=log_images_frame_idx,
                                           log_images_freq=log_images_freq)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.res_block_num = res_block_num
        self.res_conv_ksize = res_conv_ksize
        self.res_conv_num = res_conv_num
        self.drop_prob = drop_prob
        self.leakyrelu_slope = leakyrelu_slope
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.model = CasNet(in_channels=self.in_channels,
                            out_channels=self.out_channels,
                            base_channels=self.base_channels,
                            res_block_num=self.res_block_num,
                            res_conv_ksize=self.res_conv_ksize,
                            res_conv_num=self.res_conv_num,
                            drop_prob=self.drop_prob,
                            leakyrelu_slope=self.leakyrelu_slope,
        )

    def training_step(self, batch, batch_idx):
        outputs = self.model(batch.input, batch.origin_shape, batch.mask)
        train_loss = torch.tensor(0, dtype=torch.float32, device="cuda")
        for output in outputs:
            train_loss += F.l1_loss(output, batch.label)
        train_loss = train_loss / len(outputs)
        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):
        outputs = self.model(batch.input, batch.origin_shape, batch.mask)
        output_refs = self.model(batch.input_ref, batch.origin_shape, batch.mask)
        val_loss = F.l1_loss(outputs[-1], batch.label)
        return {
            "input": batch.input,
            "label": batch.label,
            "output": outputs[-1],
            "input_ref": batch.input_ref,
            "label_ref": batch.label_ref,
            "output_ref": output_refs[-1],
            "tmap_mask": batch.tmap_mask,
            "val_loss": val_loss,
            "file_name": batch.file_name,
            "frame_idx": batch.frame_idx,
            "slice_idx": batch.slice_idx,
            "coil_idx": batch.coil_idx,
        }

    def test_step(self, batch):
        output = self.model(batch.input, batch.origin_shape, batch.mask, batch.mean, batch.std)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "input": (batch.input * std + mean),
            "output": (output * std + mean),
            "target": (batch.label * std + mean),
        }

    def predict_step(self, batch):
        output = self.model(batch.input, batch.origin_shape, batch.mask)
        pred_loss = F.l1_loss(output, batch.label)
        mean = batch.mean.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        std = batch.std.unsqueeze(1).unsqueeze(2).unsqueeze(3)
        return {
            "output": denormalize(output, mean, std),
            "label": denormalize(batch.label, mean, std),
            "pred_loss": pred_loss,
        }

    def configure_optimizers(self):
        # optimizer = torch.optim.SGD(self.parameters(),
        #                              lr=self.lr,
        #                              weight_decay=self.weight_decay,
        #                             momentum=0.9,
        #                             nesterov=True)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay,
                                     betas=(0.9, 0.999),
                                     eps=1e-8)
        # optimizer = torch.optim.RMSprop(self.parameters(),
        #                                 lr=self.lr,
        #                                 weight_decay=self.weight_decay,
        #                                 momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    step_size=self.lr_step_size,
                                                    gamma=self.lr_gamma)
        return [optimizer], [scheduler]



