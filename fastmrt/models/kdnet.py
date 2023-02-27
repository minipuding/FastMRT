import torch
import torch.nn as nn


class KDNet(nn.Module):

    def __init__(self,
                 tea_net: nn.Module,
                 stu_net: nn.Module,
                 use_ema: bool = False,
                 soft_label_weight: float = 2.0):
        super(KDNet, self).__init__()
        self.tea_net = tea_net.eval()
        self.stu_net = stu_net

        self.use_ema = use_ema
        self.soft_label_weight = soft_label_weight

    def forward(self, input_stu: torch.Tensor, input_tea=None):
        if input_tea is not None:
            with torch.no_grad():
                output_tea = self.tea_net(input_tea)
        else:
            output_tea = None
        output_stu = self.stu_net(input_stu)

        return output_tea, output_stu
