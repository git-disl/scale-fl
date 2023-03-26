# Original resource before modifications: https://github.com/kuangliu/pytorch-cifar/blob/master/models/efficientnet.py

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import Classifier
from models.model_utils import Scaler


def swish(x):
    return x * x.sigmoid()


def drop_connect(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x


class SE(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels, scale=1.):
        super(SE, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(self.scaler(out)))
        out = self.scaler(self.se2(out)).sigmoid()
        out = torch.mul(x, out)
        return out


class Block(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.,
                 trs=False,
                 scale=1.):
        super(Block, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels, track_running_stats=trs)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels, track_running_stats=trs)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE(channels, se_channels, scale)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, track_running_stats=trs)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.scaler(self.conv1(x))))
        out = swish(self.bn2(self.scaler(self.conv2(out))))
        out = self.se(out)
        out = self.bn3(self.scaler(self.conv3(out)))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect(out, self.drop_rate)
            out = out + x
        return out


class EfficientNet(nn.Module):
    def __init__(self, cfg, num_classes, ee_layer_locations=[], scale=1., trs=False):
        super(EfficientNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        self.cfg = cfg
        self.ee_layer_locations = ee_layer_locations
        self.scale = scale
        self.trs = trs
        self.num_classes = num_classes

        if scale < 1:
            self.scaler = Scaler(scale)
        else:
            self.scaler = nn.Identity()

        in_channels = int(32 * self.scale)
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels, track_running_stats=trs)
        layers, self.ee_classifiers = self._make_layers(in_channels=in_channels)
        self.layers = nn.ModuleList(layers)
        self.linear = nn.Linear(int(cfg['out_channels'][-1] * self.scale), num_classes)

    def _make_layers(self, in_channels):
        layers = []
        ee_classifiers = []

        cfg = [self.cfg[k] for k in ['expansion', 'out_channels', 'num_blocks', 'kernel_size',
                                     'stride']]
        b = 0
        blocks = sum(self.cfg['num_blocks'])
        for i, (expansion, out_channels, num_blocks, kernel_size, stride) in enumerate(zip(*cfg)):
            strides = [stride] + [1] * (num_blocks - 1)
            sub_layers = []
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                sub_layers.append(
                    Block(in_channels,
                          int(out_channels * self.scale),
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate,
                          trs=self.trs,
                          scale=self.scale))
                in_channels = int(out_channels * self.scale)
            layers.append(nn.Sequential(*sub_layers))
            if i + 1 in self.ee_layer_locations and i < len(self.cfg['num_blocks']):
                ee_classifiers.append(Classifier(int(out_channels * self.scale), num_classes=self.num_classes,
                                                 reduction=6, scale=self.scale))

        return layers, nn.Sequential(*ee_classifiers)

    def forward(self, x, manual_early_exit_index=0):
        out = swish(self.bn1(self.scaler(self.conv1(x))))
        ee_counter = 0
        preds = []

        for i in range(len(self.layers)-1):
            if out is not None:
                out = self.layers[i](out)
                if i + 1 in self.ee_layer_locations:
                    pred = self.ee_classifiers[ee_counter](out)
                    ee_counter += 1
                    preds.append(pred)
                    if manual_early_exit_index == ee_counter:
                        out = None

        if out is not None:
            out = self.layers[-1](out)
            out = F.adaptive_avg_pool2d(out, 1)
            out = out.view(out.size(0), -1)
            dropout_rate = self.cfg['dropout_rate']
            if self.training and dropout_rate > 0:
                out = F.dropout(out, p=dropout_rate)
            out = self.linear(out)
            preds.append(out)

        return preds


def effnetb4_1(args, params):
    cfg = {
        'num_blocks': [2, 4, 4, 6, 6, 4, 4, 2],
        'expansion': [1, 6, 6, 6, 6, 6, 6, 6],
        'out_channels': [24, 32, 56, 112, 160, 272, 272, 448],
        'kernel_size': [3, 3, 5, 3, 5, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 2, 1],
        'dropout_rate': 0.4,
        'drop_connect_rate': 0.2,
    }
    return EfficientNet(cfg, num_classes=args.num_classes, scale=params.get('scale', 1), trs=True)


def effnetb4_4(args, params):
    cfg = {
        'num_blocks': [2, 4, 4, 6, 6, 4, 4, 2],
        'expansion': [1, 6, 6, 6, 6, 6, 6, 6],
        'out_channels': [24, 32, 56, 112, 160, 272, 272, 448],
        'kernel_size': [3, 3, 5, 3, 5, 5, 5, 3],
        'stride': [1, 2, 2, 2, 1, 2, 2, 1],
        'dropout_rate': 0.4,
        'drop_connect_rate': 0.2,
    }
    ee_layer_locations = params['ee_layer_locations']
    return EfficientNet(cfg, num_classes=args.num_classes, ee_layer_locations=ee_layer_locations, scale=params.get('scale', 1), trs=args.track_running_stats)
