import math
import random
import numpy as np


def create_noniid_users(train_set, val_set, test_set, args, alpha=100):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :param alpha:
    :return:
    """

    # Original resource: https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/datasets/partition_data.py
    def build_non_iid_by_dirichlet(
            random_state, indices2targets, non_iid_alpha, num_classes, num_indices, n_workers
    ):
        n_auxi_workers = 10
        assert n_auxi_workers <= n_workers

        # random shuffle targets indices.
        random_state.shuffle(indices2targets)

        # partition indices.
        from_index = 0
        splitted_targets = []
        num_splits = math.ceil(n_workers / n_auxi_workers)
        split_n_workers = [
            n_auxi_workers
            if idx < num_splits - 1
            else n_workers - n_auxi_workers * (num_splits - 1)
            for idx in range(num_splits)
        ]
        split_ratios = [_n_workers / n_workers for _n_workers in split_n_workers]
        for idx, ratio in enumerate(split_ratios):
            to_index = from_index + int(n_auxi_workers / n_workers * num_indices)
            splitted_targets.append(
                indices2targets[
                from_index: (num_indices if idx == num_splits - 1 else to_index)
                ]
            )
            from_index = to_index

        #
        idx_batch = []
        for _targets in splitted_targets:
            # rebuild _targets.
            _targets = np.array(_targets)
            _targets_size = len(_targets)

            # use auxi_workers for this subset targets.
            _n_workers = min(n_auxi_workers, n_workers)
            n_workers = n_workers - n_auxi_workers

            # get the corresponding idx_batch.
            min_size = 0
            while min_size < int(0.50 * _targets_size / _n_workers):
                _idx_batch = [[] for _ in range(_n_workers)]
                for _class in range(num_classes):
                    # get the corresponding indices in the original 'targets' list.
                    idx_class = np.where(_targets[:, 1] == _class)[0]
                    idx_class = _targets[idx_class, 0]

                    # sampling.
                    try:
                        proportions = random_state.dirichlet(
                            np.repeat(non_iid_alpha, _n_workers)
                        )
                        # balance
                        proportions = np.array(
                            [
                                p * (len(idx_j) < _targets_size / _n_workers)
                                for p, idx_j in zip(proportions, _idx_batch)
                            ]
                        )
                        proportions = proportions / proportions.sum()
                        proportions = (np.cumsum(proportions) * len(idx_class)).astype(int)[
                                      :-1
                                      ]
                        _idx_batch = [
                            idx_j + idx.tolist()
                            for idx_j, idx in zip(
                                _idx_batch, np.split(idx_class, proportions)
                            )
                        ]
                        sizes = [len(idx_j) for idx_j in _idx_batch]
                        min_size = min([_size for _size in sizes])
                    except ZeroDivisionError:
                        pass
            idx_batch += _idx_batch
        return {i: v for i, v in enumerate(idx_batch)}

    if args.data == 'sst2' or args.data == 'ag_news':
        dict_users = build_non_iid_by_dirichlet(
            random_state=np.random.RandomState(1),
            indices2targets=np.array(
                [
                    (idx, target)
                    for idx, target in enumerate(train_set['label'] + val_set['label'] + test_set['label'])
                ]
            ),
            non_iid_alpha=alpha,
            num_classes=args.num_classes,
            num_indices=len(train_set) + len(val_set) + len(test_set),
            n_workers=args.num_clients
        )
    else:
        dict_users = build_non_iid_by_dirichlet(
            random_state=np.random.RandomState(1),
            indices2targets=np.array(
                [
                    (idx, target)
                    for idx, target in enumerate(train_set.targets + val_set.targets + test_set.targets)
                ]
            ),
            non_iid_alpha=alpha,
            num_classes=args.num_classes,
            num_indices=len(train_set) + len(val_set) + len(test_set),
            n_workers=args.num_clients
        )

    train_dict = {k: [v for v in d if v < len(train_set)] for k, d in dict_users.items()}
    val_dict = {k: [v - len(train_set) for v in d if len(train_set) + len(val_set) > v >= len(train_set)] for k, d in dict_users.items()}
    test_dict = {k: [v - len(train_set) - len(val_set) for v in d if v >= len(train_set) + len(val_set)] for k, d in dict_users.items()}

    return train_dict, val_dict, test_dict
