class Config:
    def __init__(self):
        self.training_params = {
            'cifar10': {
                'msdnet24_1': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 4e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'msdnet24_4': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 4e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'resnet110_1': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'},
                'resnet110_4': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'}
            },
            'cifar100': {
                'msdnet24_1': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'msdnet24_4': {'batch_size': 16,
                               'num_epoch': 5,
                               'lr': 0.1,
                               'lr_type': 'multistep',
                               'decay_rate': 0.1,
                               'decay_epochs': [100, 200],
                               'weight_decay': 5e-4,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'resnet110_1': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'},
                'resnet110_4': {'batch_size': 16,
                                'num_epoch': 5,
                                'lr': 0.1,
                                'lr_type': 'multistep',
                                'decay_rate': 0.1,
                                'decay_epochs': [100, 200],
                                'weight_decay': 5e-4,
                                'momentum': 0.9,
                                'optimizer': 'sgd'}
            },
            # lr details in submission for effnet were reported with some errors, we use exponential decay as suggested in the original paper.
            'imagenet': {
                'effnetb4_1': {'batch_size': 64,
                               'num_epoch': 5,
                               'lr': 0.2,
                               'lr_type': 'exp',
                               'decay_rate': 0.98,
                               'decay_epochs': [30, 60],
                               'weight_decay': 1e-5,
                               'momentum': 0.9,
                               'optimizer': 'sgd'},
                'effnetb4_4': {'batch_size': 64,
                               'num_epoch': 5,
                               'lr': 0.2,
                               'lr_type': 'exp',
                               'decay_rate': 0.98,
                               'decay_epochs': [30, 60],
                               'weight_decay': 1e-5,
                               'momentum': 0.9,
                               'optimizer': 'sgd'}
            },
            'sst2': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                }
            },
            'ag_news': {
                'bert_1': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                },
                'bert_4': {
                    'batch_size': 16,
                    'num_epoch': 5,
                    'lr': 3e-5,
                    'lr_type': 'none',
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    'optimizer': 'sgd'
                }
            }
        }
        self.model_params = {
            'cifar10': {
                'msdnet24_1': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 1,
                               'reduction': 0.5},
                'msdnet24_4': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 4,
                               'reduction': 0.5},
                'resnet110_1': {'ee_layer_locations': [],
                                'ee_num_conv_layers': [],
                                'num_blocks': 1},
                'resnet110_4': {'ee_layer_locations': [30, 38, 46],
                                'ee_num_conv_layers': [3, 3, 3],
                                'num_blocks': 4},
            },
            'cifar100': {
                'msdnet24_1': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 1,
                               'reduction': 0.5},
                'msdnet24_4': {'base': 6,
                               'step': 6,
                               'num_scales': 3,
                               'step_mode': 'even',
                               'num_channels': 16,
                               'growth_rate': 6,
                               'growth_factor': [1, 2, 4],
                               'prune': 'max',
                               'bn_factor': [1, 2, 4],
                               'bottleneck': True,
                               'compression': 0.5,
                               'num_blocks': 4,
                               'reduction': 0.5},
                'resnet110_1': {'ee_layer_locations': [],
                                'ee_num_conv_layers': [],
                                'num_blocks': 1},
                'resnet110_4': {'ee_layer_locations': [30, 38, 46],
                                'ee_num_conv_layers': [3, 3, 3],
                                'num_blocks': 4}
            },
            'imagenet': {
                'effnetb4_1': {'ee_layer_locations': [],
                               'ee_num_conv_layers': [],
                               'num_blocks': 1},
                'effnetb4_4': {'ee_layer_locations': [5, 6, 7],
                               'ee_num_conv_layers': [3, 3, 3],
                               'num_blocks': 4}
            },
            'sst2': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [4, 6, 9],
                           'num_blocks': 4},
            },
            'ag_news': {
                'bert_1': {'ee_layer_locations': [],
                           'num_blocks': 1},
                'bert_4': {'ee_layer_locations': [4, 6, 9],
                           'num_blocks': 4},
            }
        }