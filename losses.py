import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.softplus = nn.Softplus()

    # def forward(self, output1, output2, target, alpha=0.1):
    def forward(self, loss, target):
        # loss = F.cosine_similarity(output1, output2, dim=1).mean()
        # loss = loss * (1-target.float().mean())
        # loss = self.softplus((-1) * loss)
        # loss = (-1) * self.softplus(loss)
        return loss