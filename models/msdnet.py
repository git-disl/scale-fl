import copy

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import pdb
import random


class ConvBasic(nn.Module):
    def __init__(self, nIn, nOut, kernel=3, stride=1, padding=1, trs=False):
        super(ConvBasic, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(nIn, nOut, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(nOut, track_running_stats=trs),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.net(x)


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck, bnWidth, trs=False):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = max(min(nInner, bnWidth * nOut), 1)
            layer.append(nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner, track_running_stats=trs))
            layer.append(nn.ReLU(True))

        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False))
        else:
            raise ValueError

        layer.append(nn.BatchNorm2d(nOut, track_running_stats=trs))
        layer.append(nn.ReLU(True))

        self.net = nn.Sequential(*layer)

    def forward(self, x):

        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2, trs=False):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, max(nOut // 2, 1), 'down', bottleneck, bnWidth1, trs=trs)
        self.conv_normal = ConvBN(nIn2, max(nOut // 2, 1), 'normal', bottleneck, bnWidth2, trs=trs)

    def forward(self, x):
        res = [x[1], self.conv_down(x[0]), self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth, trs=False):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth, trs=trs)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]

        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, args, params, trs=False):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if args.data.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * params['growth_factor'][0], kernel=3, stride=1, padding=1,
                                         trs=trs))
        elif args.data == 'imagenet':
            conv = ConvBasic(nIn, nOut * params['growth_factor'][0], kernel=3, stride=2, padding=1, trs=trs)
            self.layers.append(conv)

        nIn = nOut * params['growth_factor'][0]

        for i in range(1, params['num_scales']):
            self.layers.append(ConvBasic(nIn, nOut * params['growth_factor'][i],
                                         kernel=3, stride=2, padding=1, 
                                         trs=trs))
            nIn = nOut * params['growth_factor'][i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)

        return res


class MSDNLayer(nn.Module):
    # Original resource: https://github.com/kalviny/MSDNet-PyTorch/blob/master/models/msdnet.py
    def __init__(self, nIn, nOut, params, in_scales=None, out_scales=None, trs=False):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.in_scales = in_scales if in_scales is not None else params['num_scales']
        self.out_scales = out_scales if out_scales is not None else params['num_scales']

        self.num_scales = params['num_scales']
        self.discard = self.in_scales - self.out_scales

        self.offset = self.num_scales - self.out_scales
        self.layers = nn.ModuleList()

        if self.discard > 0:
            nIn1 = nIn * params['growth_factor'][self.offset - 1]
            nIn2 = nIn * params['growth_factor'][self.offset]
            _nOut = nOut * params['growth_factor'][self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, params['bottleneck'],
                                              params['bn_factor'][self.offset - 1],
                                              params['bn_factor'][self.offset],
                                              trs=trs))
        else:
            self.layers.append(ConvNormal(nIn * params['growth_factor'][self.offset],
                                          nOut * params['growth_factor'][self.offset],
                                          params['bottleneck'],
                                          params['bn_factor'][self.offset],
                                          trs=trs))

        for i in range(self.offset + 1, self.num_scales):
            nIn1 = nIn * params['growth_factor'][i - 1]
            nIn2 = nIn * params['growth_factor'][i]
            _nOut = nOut * params['growth_factor'][i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, params['bottleneck'],
                                              params['bn_factor'][i - 1],
                                              params['bn_factor'][i],
                                              trs=trs))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.out_scales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.out_scales):
                inp.append([x[i - 1], x[i]])

        res = []
        for i in range(self.out_scales):
            res.append(self.layers[i](inp[i]))

        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))

        return res


class ClassifierModule(nn.Module):
    def __init__(self, m, channel, num_classes):
        super(ClassifierModule, self).__init__()
        self.m = m
        self.linear = nn.Linear(channel, num_classes)

    def forward(self, x):
        res = self.m(x[-1])
        res = res.view(res.size(0), -1)
        return self.linear(res)


class MSDNet(nn.Module):
    def __init__(self, args, params):
        super(MSDNet, self).__init__()
        self.stored_inp_kwargs = copy.deepcopy(locals())
        del self.stored_inp_kwargs['self']
        del self.stored_inp_kwargs['__class__']

        self.blocks = nn.ModuleList()
        self.classifier = nn.ModuleList()
        self.num_blocks = len(params['ee_layer_locations'])
        self.steps = params['ee_layer_locations']
        self.args = args
        self.scale = params.get('scale', 1)
        self.trs = args.track_running_stats

        self.growth_rate = int(params['growth_rate'] * self.scale)

        n_layers_all, n_layer_curr = self.steps[-1], 0

        nIn = int(params['num_channels'] * self.scale)
        for i in range(self.num_blocks):
            # print(' ********************** Block {} '
            #       ' **********************'.format(i + 1))
            if i == 0:
                step = self.steps[0]
            else:
                step = self.steps[i] - self.steps[i-1]
            m, nIn = \
                self._build_block(nIn, args, params, step, n_layers_all, n_layer_curr)
            self.blocks.append(m)
            n_layer_curr = self.steps[i]

            self.classifier.append(self._build_classifier_cifar(nIn * params['growth_factor'][-1], args.num_classes))

        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

        for m in self.classifier:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, args, params, step, n_layer_all, n_layer_curr):

        layers = [MSDNFirstLayer(3, nIn, args, params)] \
            if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            if params['prune'] == 'min':
                in_scales = min(params['num_scales'], n_layer_all - n_layer_curr + 2)
                out_scales = min(params['num_scales'], n_layer_all - n_layer_curr + 1)
            elif params['prune'] == 'max':
                interval = math.ceil(1.0 * n_layer_all / params['num_scales'])
                in_scales = params['num_scales'] - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                out_scales = params['num_scales'] - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError

            growth_factor = params['growth_factor']
            growth_rate = self.growth_rate
            layers.append(MSDNLayer(nIn, growth_rate, params, in_scales, out_scales, self.trs))
            # print('|\t\tin_scales {} out_scales {} inChannels {} outChannels {}\t\t|'.format(in_scales, out_scales, nIn, growth_rate))

            nIn += growth_rate
            if params['prune'] == 'max' and in_scales > out_scales and params['reduction'] > 0:
                offset = params['num_scales'] - out_scales
                layers.append(self._build_transition(nIn, max(math.floor(1.0 * params['reduction'] * nIn), 1), out_scales, offset, growth_factor))
                _t = nIn
                nIn = max(math.floor(1.0 * params['reduction'] * nIn), 1)
                # print('|\t\tTransition layer inserted! (max), inChannels {}, outChannels {}\t|'.format(_t, math.floor(
                #     1.0 * params['reduction'] * _t)))
            elif params['prune'] == 'min' and params['reduction'] > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = params['num_scales'] - out_scales
                layers.append(self._build_transition(nIn, max(math.floor(1.0 * params['reduction'] * nIn), 1), out_scales, offset, growth_factor))

                nIn = max(math.floor(1.0 * params['reduction'] * nIn), 1)
            #     print('|\t\tTransition layer inserted! (min)\t|')
            # print("")

        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, out_scales, offset, growth_factor):
        net = []
        for i in range(out_scales):
            net.append(ConvBasic(nIn * growth_factor[offset + i],
                                 nOut * growth_factor[offset + i],
                                 kernel=1, stride=1, padding=0, 
                                 trs=self.trs))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        interChannels1, interChannels2 = int(128 * self.scale), int(128 * self.scale)
        conv = nn.Sequential(
            ConvBasic(nIn, interChannels1, kernel=3, stride=2, padding=1, 
                      trs=self.trs),
            ConvBasic(interChannels1, interChannels2, kernel=3, stride=2, padding=1, 
                      trs=self.trs),
            # nn.AvgPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        return ClassifierModule(conv, interChannels2, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        conv = nn.Sequential(
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1, 
                      trs=self.trs),
            ConvBasic(nIn, nIn, kernel=3, stride=2, padding=1, 
                      trs=self.trs),
            nn.AvgPool2d(2)
        )
        return ClassifierModule(conv, nIn, num_classes)

    def forward(self, x, manual_early_exit_index=0):

        res = []

        num_iter = self.num_blocks
        if manual_early_exit_index != 0:
            num_iter = min(manual_early_exit_index, self.num_blocks)

        for i in range(num_iter):
            x = self.blocks[i](x)

            pred = self.classifier[i](x)
            res.append(pred)

        return res

def msdnet24_1(args, params):
    return MSDNet(args, params)

def msdnet24_4(args, params):
    return MSDNet(args, params)
