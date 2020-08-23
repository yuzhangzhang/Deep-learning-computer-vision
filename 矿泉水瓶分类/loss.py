import torch.nn as nn
import torch
import torch.nn.functional as F


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2, reduction="mean"):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0]
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = 2 * torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise Exception("Unexpected reduction format.")


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, 1], target[:, 1])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], 'Expect weight shape and real shape not match.'
                    dice_loss *= self.weight[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]

#
# def make_one_hot(input, num_classes):
#     shape = np.array(input.shape)
#     shape[1] = num_classes
#     shape = tuple(shape)
#     result = torch.zeros(shape)
#     result = result.scatter_(1, input.cpu(), 1)
#
#     return result


def calculate_loss(predicts,gt,label):
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(predicts,label)
    dice_loss = DiceLoss()(predicts, gt)
    loss = ce_loss+dice_loss
    return loss
