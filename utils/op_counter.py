from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from functools import reduce
import operator

'''
    Calculate the FLOPS of each exit without lazy prediction pruning"
    https://github.com/kalviny/MSDNet-PyTorch/blob/master/op_counter.py
'''

count_ops = 0
count_params = 0
cls_ops = []
cls_params = []

def get_num_gen(gen):
    return sum(1 for x in gen)


def is_leaf(model):
    return get_num_gen(model.children()) == 0


def get_layer_info(layer):
    layer_str = str(layer)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


def get_layer_param(model):
    return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


### The input batch size should be 1 to call this function
def measure_layer(layer, *x):
    global count_ops, count_params, cls_ops, cls_params
    delta_ops = 0
    delta_params = 0
    multi_add = 1
    type_name = get_layer_info(layer)

    x = x[0]

    ### ops_conv
    if type_name in ['Conv2d']:
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        delta_ops = layer.in_channels * layer.out_channels * layer.kernel_size[0] *  \
                layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        delta_params = get_layer_param(layer)

    ### ops_nonlinearity
    elif type_name in ['ReLU', 'Tanh', 'GELUActivation']:
        delta_ops = x.numel()
        delta_params = get_layer_param(layer)

    ### ops_pooling
    elif type_name in ['AvgPool2d', 'MaxPool2d']:
        in_w = x.size()[2]
        kernel_ops = layer.kernel_size * layer.kernel_size
        out_w = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        out_h = int((in_w + 2 * layer.padding - layer.kernel_size) / layer.stride + 1)
        delta_ops = x.size()[0] * x.size()[1] * out_w * out_h * kernel_ops
        delta_params = get_layer_param(layer)

    elif type_name in ['AdaptiveAvgPool2d']:
        delta_ops = x.size()[0] * x.size()[1] * x.size()[2] * x.size()[3]
        delta_params = get_layer_param(layer)

    elif type_name in ['BertSelfAttention']:
        delta_ops = x.shape[1] ** 2 * x.shape[-1] + layer.num_attention_heads * x.shape[1] ** 3

    elif type_name in ['SE']:
        in_w = x.size()[2]
        kernel_ops = in_w * in_w
        delta_ops = x.size()[0] * x.size()[1] * kernel_ops + np.prod(x.shape)

    elif type_name in ['Linear']:
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer)

    elif type_name in ['BatchNorm2d', 'BatchNorm1d', 'LayerNorm', 'Dropout2d', 'DropChannel', 'Dropout',
                       'MSDNFirstLayer', 'ConvBasic', 'ConvBN',
                       'ParallelModule', 'MSDNet', 'Sequential',
                       'MSDNLayer', 'ConvDownNormal', 'ConvNormal', 'ClassifierModule', 'Flatten', 
                       'Softmax', 'Identity', 'Scaler', 'Embedding']:
        delta_params = get_layer_param(layer)

    else:
        raise TypeError('unknown layer type: %s' % type_name)

    count_ops += delta_ops
    count_params += delta_params
    if type_name == 'Linear' and layer.out_features in [2, 10, 100, 1000]:
        print('---------------------')
        print('FLOPs: %.2fM, Params: %.2fM' % (count_ops / 1e6, count_params / 1e6))
        cls_ops.append(count_ops)
        cls_params.append(count_params)
        
    return


def measure_model(model, H, W, exit_idx=0):
    global count_ops, count_params, cls_ops, cls_params
    count_ops = 0
    count_params = 0
    cls_ops = []
    cls_params = []

    exceptions = ['BertSelfAttention', 'SE']

    if 'bert' in str(model.__class__):
        data = torch.randint(1, 100, size=(H, W)).long()
    else:
        data = Variable(torch.zeros(1, 3, H, W))

    def should_measure(x):
        if any([x_ in str(type(x)) for x_ in exceptions]):
            return True
        return is_leaf(x) and 'measure_model' not in str(x.forward)

    def modify_forward(model):
        for child in model.children():
            if should_measure(child):
                def new_forward(m):
                    def lambda_forward(*x):
                        measure_layer(m, *x)
                        return m.old_forward(*x)
                    return lambda_forward
                child.old_forward = child.forward
                child.forward = new_forward(child)
                if any([x_ in str(type(child)) for x_ in exceptions]):
                    modify_forward(child)
            else:
                modify_forward(child)

    def restore_forward(model):
        for child in model.children():
            # leaf node
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
            else:
                restore_forward(child)

    modify_forward(model)
    model.forward(data, manual_early_exit_index=exit_idx)
    restore_forward(model)
    return cls_ops, cls_params
