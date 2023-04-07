import torch
import torch.nn as nn


class KDNet(nn.Module):
    """
    This is a Knowledge Distillation Network that supports different inputs for the student and teacher networks.

    Args:
        tea_net: nn.Module, the teacher network.
        stu_net: nn.Module, the student network.

    """


    def __init__(self,
                 tea_net: nn.Module,
                 stu_net: nn.Module,):
        super(KDNet, self).__init__()
        self.tea_net = tea_net.eval()
        self.stu_net = stu_net


    def forward(self, input_stu: torch.Tensor, input_tea=None):
        with torch.no_grad():
            if input_tea is not None:
                output_tea = self.tea_net(input_tea)
            else:
                output_tea = self.tea_net(input_stu)
        output_stu = self.stu_net(input_stu)

        return output_tea, output_stu
