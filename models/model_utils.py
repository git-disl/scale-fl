import torch.nn as nn


class Scaler(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, input):
        output = input / self.rate
        return output


class KDLoss(nn.Module):
    def __init__(self, args):
        super(KDLoss, self).__init__()

        self.kld_loss = nn.KLDivLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.softmax = nn.Softmax(dim=1)

        self.T = args.KD_T
        self.gamma = args.KD_gamma

    def loss_fn_kd(self, pred, target, soft_target, gamma_active=True):
        _ce = self.ce_loss(pred, target)
        T = self.T
        if self.gamma and gamma_active:
            # _ce = (1. - self.gamma) * _ce
            _kld = self.kld_loss(self.log_softmax(pred / T), self.softmax(soft_target / T)) * self.gamma * T * T
        else:
            _kld = 0
        loss = _ce + _kld
        return loss


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
