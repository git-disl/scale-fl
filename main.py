#!/usr/bin/env python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle as pkl

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import models
from args import arg_parser, modify_args
from config import Config
from data_tools.dataloader import get_dataloaders, get_datasets, get_user_groups
from fed import Federator
from models.model_utils import KDLoss
from predict import validate, local_validate
from utils.utils import load_checkpoint, measure_flops, load_state_dict, save_user_groups, load_user_groups

np.set_printoptions(precision=2)

args = arg_parser.parse_args()
args = modify_args(args)
torch.manual_seed(args.seed)


def main():
    global args

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    config = Config()

    if args.ee_locs:
        config.model_params[args.data][args.arch]['ee_layer_locations'] = args.ee_locs

    model = getattr(models, args.arch)(args, {**config.model_params[args.data][args.arch]})
    args.num_exits = config.model_params[args.data][args.arch]['num_blocks']

    if args.use_gpu:
        model = model.cuda()
        criterion = KDLoss(args).cuda()
    else:
        criterion = KDLoss(args)

    if args.resume:
        checkpoint = load_checkpoint(args, load_best=False)
        if checkpoint is not None:
            args.start_round = checkpoint['round'] + 1
            model.load_state_dict(checkpoint['state_dict'])

    cudnn.benchmark = True

    batch_size = args.batch_size if args.batch_size else config.training_params[args.data][args.arch]['batch_size']
    train_set, val_set, test_set = get_datasets(args)
    _, val_loader, test_loader = get_dataloaders(args, batch_size, (train_set, val_set, test_set))
    if val_set is None:
        val_set = val_loader.dataset

    train_user_groups, val_user_groups, test_user_groups = get_user_groups(train_set, val_set, test_set, args)

    prev_user_groups = load_user_groups(args)
    if prev_user_groups is None:
        if args.resume:
            print('Could not find user groups')
            raise RuntimeError
        user_groups = (train_user_groups, val_user_groups, test_user_groups)
        save_user_groups(args, (train_user_groups, val_user_groups, test_user_groups))
    else:
        user_groups = prev_user_groups

    if args.evalmode is not None:
        load_state_dict(args, model)
        if 'global' in args.evalmode:
            validate(model, test_loader, criterion, args)
            return
        elif 'local' in args.evalmode:
            train_args = eval('argparse.' + open(os.path.join(args.save_path, 'args.txt')).readlines()[0])
            if os.path.exists(os.path.join(args.save_path, 'client_groups.pkl')):
                client_groups = pkl.load(open(os.path.join(args.save_path, 'client_groups.pkl'), 'rb'))
            else:
                client_groups = []
            federator = Federator(model, train_args, client_groups)
            local_validate(federator, test_set, user_groups[1], criterion, args, batch_size)
            return
        else:
            raise NotImplementedError

    with open(os.path.join(args.save_path, 'args.txt'), 'w') as f:
        print(args, file=f)

    federator = Federator(model, args)
    best_acc1, best_round = federator.fed_train(train_set, val_set, user_groups, criterion, args, batch_size,
                                                 config.training_params[args.data][args.arch])

    print('Best val_acc1: {:.4f} at round {}'.format(best_acc1, best_round))
    validate(federator.global_model, test_loader, criterion, args, save=True)

    return


if __name__ == '__main__':
    main()
