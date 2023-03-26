import copy
import datetime as dt
import os
import pickle as pkl

import numpy as np
import torch
import torch.multiprocessing as mp

from data_tools.dataloader import get_client_dataloader
from predict import local_validate
from train import execute_epoch
from utils.grad_traceback import get_downscale_index
from utils.utils import save_checkpoint

mp.set_start_method('spawn', force=True)


class Federator:
    def __init__(self, global_model, args, client_groups=[]):
        self.global_model = global_model

        self.vertical_scale_ratios = args.vertical_scale_ratios
        self.horizontal_scale_ratios = args.horizontal_scale_ratios
        self.client_split_ratios = args.client_split_ratios

        assert len(self.vertical_scale_ratios) == len(self.horizontal_scale_ratios) == len(self.client_split_ratios)

        self.num_rounds = args.num_rounds
        self.num_clients = args.num_clients
        self.sample_rate = args.sample_rate
        self.alpha = args.alpha
        self.num_levels = len(self.vertical_scale_ratios)
        self.idx_dicts = [get_downscale_index(self.global_model, args, s) for s in self.vertical_scale_ratios]
        self.client_groups = client_groups

        self.use_gpu = args.use_gpu

    def fed_train(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params):

        scores = ['epoch\ttrain_loss\tval_loss\tval_acc1\tval_acc5\tlocal_val_acc1\tlocal_val_acc5' +
                  '\tlocal_val_acc1' * self.num_levels]
        best_acc1, best_round = 0.0, 0

        # pre-assignment of levels to clients (needs to be saved for inference)
        if not self.client_groups:
            client_idxs = np.arange(self.num_clients)
            np.random.seed(args.seed)
            shuffled_client_idxs = np.random.permutation(client_idxs)
            client_groups = []
            s = 0
            for ratio in self.client_split_ratios:
                e = s + int(len(shuffled_client_idxs) * ratio)
                client_groups.append(shuffled_client_idxs[s: e])
                s = e
            self.client_groups = client_groups

            with open(os.path.join(args.save_path, 'client_groups.pkl'), 'wb') as f:
                pkl.dump(self.client_groups, f)

        for round_idx in range(args.start_round, self.num_rounds):

            print(f'\n | Global Training Round : {round_idx + 1} |\n')

            train_loss, val_results, local_val_results = \
                self.execute_round(train_set, val_set, user_groups, criterion, args, batch_size,
                                   train_params, round_idx)

            val_loss, val_acc1, val_acc5, _, _ = val_results

            scores.append(('{}' + '\t{:.4f}' * int(6 + self.num_levels))
                          .format(round_idx, train_loss, val_loss, val_acc1, val_acc5,
                                  local_val_results[-1][1], local_val_results[-1][2],
                                  *[l[1] for l in local_val_results[:-1]]))

            is_best = val_acc1 > best_acc1
            if is_best:
                best_acc1 = val_acc1
                best_round = round_idx
                print('Best var_acc1 {}'.format(best_acc1))

            model_filename = 'checkpoint_%03d.pth.tar' % round_idx
            save_checkpoint({
                'round': round_idx,
                'arch': args.arch,
                'state_dict': self.global_model.state_dict(),
                'best_acc1': best_acc1,
            }, args, is_best, model_filename, scores)

        return best_acc1, best_round

    def get_level(self, client_idx):
        # Return the complexity level of given client, starts with 0
        try:
            level = np.where([client_idx in c for c in self.client_groups])[0][0]
        except:
            # client will be skipped
            level = -1

        return level

    def execute_round(self, train_set, val_set, user_groups, criterion, args, batch_size, train_params, round_idx):
        self.global_model.train()
        m = max(int(self.sample_rate * self.num_clients), 1)
        client_idxs = np.random.choice(range(self.num_clients), m, replace=False)

        client_train_loaders = [get_client_dataloader(train_set, user_groups[0][client_idx], args, batch_size) for
                                client_idx in client_idxs]
        levels = [self.get_level(client_idx) for client_idx in client_idxs]
        scales = [self.vertical_scale_ratios[level] for level in levels]
        local_models = [self.get_local_split(levels[i], scales[i]) for i in range(len(client_idxs))]
        h_scale_ratios = [self.horizontal_scale_ratios[level] for level in levels]

        pool_args = [train_set, user_groups, criterion, args, batch_size, train_params, round_idx]
        local_weights = []
        local_losses = []
        local_grad_flags = []

        pool_args.append(None)

        for i, client_idx in enumerate(client_idxs):
            client_args = pool_args + [local_models[i], client_train_loaders[i], levels[i], scales[i], h_scale_ratios[i], client_idx]
            result = execute_client_round(client_args)

            if args.use_gpu:
                for k, v in result[0].items():
                    result[0][k] = v.cuda(0)

            local_weights.append(result[0])
            local_grad_flags.append(result[1])
            local_losses.append(result[2])
            print(f'Client {i+1}/{len(client_idxs)} completely finished')

        train_loss = sum(local_losses) / len(client_idxs)

        # Update the global model
        global_weights = self.average_weights(local_weights, local_grad_flags, levels, self.global_model)
        self.global_model.load_state_dict(global_weights)

        # Validation for all clients
        if self.client_split_ratios[-1] == 0:
            level = np.where(self.client_split_ratios)[0].tolist()[-1]
            scale = self.vertical_scale_ratios[level]
            global_model = self.get_local_split(level, scale)
            if self.use_gpu:
                global_model = global_model.cuda()
        else:
            global_model = copy.deepcopy(self.global_model)

        val_results, local_val_results = local_validate(self, val_set, user_groups[1], criterion, args, 512,
                                                        global_model)

        return train_loss, val_results, local_val_results

    def average_weights(self, w, grad_flags, levels, model):
        w_avg = copy.deepcopy(model.state_dict())
        for key in w_avg.keys():

            if 'num_batches_tracked' in key:
                w_avg[key] = w[0][key]
                continue

            if 'running' in key:
                w_avg[key] = sum([w_[key] for w_ in w]) / len(w)
                continue

            tmp = torch.zeros_like(w_avg[key])
            count = torch.zeros_like(tmp)
            for i in range(len(w)):
                if grad_flags[i][key]:
                    idx = self.idx_dicts[levels[i]][key]
                    idx = self.fix_idx_array(idx, w[i][key].shape)
                    tmp[idx] += w[i][key].flatten()
                    count[idx] += 1
            w_avg[key][count != 0] = tmp[count != 0]
            count[count == 0] = 1
            w_avg[key] = w_avg[key] / count
        return w_avg

    def get_idx_shape(self, inp, local_shape):
        # Return the output shape for binary mask input
        # [[1, 1, 0], [1, 1, 0], [0, 0, 0,]] -> [2, 2]
        if any([s == 0 for s in inp.shape]):
            print('Indexing error')
            raise RuntimeError

        if len(local_shape) == 4:
            dim_1 = inp.shape[2] // 2
            dim_2 = inp.shape[3] // 2
            idx_shape = (inp[:, 0, dim_1, dim_2].sum().item(),
                         inp[0, :, dim_1, dim_2].sum().item(), *local_shape[2:])
        elif len(local_shape) == 2:
            idx_shape = (inp[:, 0].sum().item(),
                         inp[0, :].sum().item())
        else:
            idx_shape = (inp.sum(),)

        return idx_shape

    def fix_idx_array(self, idx_array, local_shape):
        idx_shape = self.get_idx_shape(idx_array, local_shape)
        if all([idx_shape[i] >= local_shape[i] for i in range(len(local_shape))]):
            pass
        else:
            idx_array = idx_array[idx_array.sum(dim=1).argmax()].repeat((idx_array.shape[0], 1))
            idx_shape = self.get_idx_shape(idx_array, local_shape)

        ind_list = [slice(None)] * len(idx_array.shape)
        for i in range(len(local_shape)):

            lim = idx_array.shape[i]
            while idx_shape[i] != local_shape[i]:
                lim -= 1
                ind_list[i] = slice(0, lim)
                idx_shape = self.get_idx_shape(idx_array[tuple(ind_list)], local_shape)

        tmp = torch.zeros_like(idx_array, dtype=bool)
        tmp[tuple(ind_list)] = idx_array[tuple(ind_list)]
        idx_array = tmp

        if len(idx_array.shape) == 4:
            dim_1 = idx_array.shape[2] // 2
            dim_2 = idx_array.shape[3] // 2
            if idx_array.sum(dim=0).sum(dim=0)[0, 0] != idx_array.sum(dim=0).sum(dim=0)[dim_1, dim_2]:
                idx_array = idx_array[:, :, dim_1, dim_2].repeat(idx_array.shape[2], idx_array.shape[3], 1, 1).permute(
                    2, 3, 0, 1)
        return idx_array

    def get_local_split(self, level, scale):
        model = copy.deepcopy(self.global_model)

        if scale == 1:
            return model

        model_kwargs = model.stored_inp_kwargs
        if 'scale' in model_kwargs.keys():
            model_kwargs['scale'] = scale
        else:
            model_kwargs['params']['scale'] = scale

        local_model = type(self.global_model)(**model_kwargs)
        if 'bert' in str(type(local_model)):
            local_model.add_exits(model_kwargs['ee_layer_locations'])

        local_state_dict = local_model.state_dict()

        for n, p in self.global_model.state_dict().items():

            if 'num_batches_tracked' in n:
                local_state_dict[n] = p
                continue

            global_shape = p.shape
            local_shape = local_state_dict[n].shape

            if len(global_shape) != len(local_shape):
                print('Models are not alignable!')
                raise RuntimeError

            idx_array = self.fix_idx_array(self.idx_dicts[level][n], local_shape)
            local_state_dict[n] = p[idx_array].reshape(local_shape)

        local_model.load_state_dict(local_state_dict)

        return local_model


def execute_client_round(args):
    train_set, user_groups, criterion, args, batch_size, train_params, round_idx, global_model, \
    local_model, client_train_loader, level, scale, h_scale_ratio, client_idx = args

    if args.use_gpu:
        local_model = local_model.cuda()

    base_params = [v for k, v in local_model.named_parameters() if 'ee_' not in k]
    exit_params = [v for k, v in local_model.named_parameters() if 'ee_' in k]

    optimizer = torch.optim.SGD([{'params': base_params},
                                 {'params': exit_params}],
                                lr=train_params['lr'],
                                momentum=train_params['momentum'],
                                weight_decay=train_params['weight_decay'])

    loss = 0.0
    for epoch in range(train_params['num_epoch']):
        print(f'{client_idx}-{epoch}-{dt.datetime.now()}')
        iter_idx = round_idx
        loss = execute_epoch(local_model, client_train_loader, criterion, optimizer, iter_idx, epoch,
                             args, train_params, h_scale_ratio, level, global_model)

    print(f'Finished epochs for {client_idx}')
    local_weights = {k: v.cpu() for k, v in local_model.state_dict(keep_vars=True).items()}
    local_grad_flags = {k: v.grad is not None for k, v in local_model.state_dict(keep_vars=True).items()}

    del local_model
    torch.cuda.empty_cache()

    return local_weights, local_grad_flags, loss