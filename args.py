import argparse
import datetime
import os


def modify_args(args):
    if args.use_gpu and args.gpu_idx:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_idx

    if args.use_valid:
        args.splits = ['train', 'val', 'test']
    else:
        args.splits = ['train', 'val']

    if args.data == 'cifar10':
        args.num_classes = 10
        args.image_size = (32, 32)
    elif args.data == 'cifar100':
        args.num_classes = 100
        args.image_size = (32, 32)
    elif args.data == 'imagenet':
        args.num_classes = 1000
        if 'effnetb4' in args.arch:
            args.image_size = (224, 224)
        else:
            args.image_size = (64, 64)
    elif args.data == 'sst2':
        args.num_classes = 2
        args.image_size = (1, 64)
    elif args.data == 'ag_news':
        args.num_classes = 4
        args.image_size = (1, 64)
    else:
        raise NotImplementedError

    if not hasattr(args, "save_path") or args.save_path is None:
        args.save_path = f"outputs/{args.arch}_{args.evalmode}_{args.data}_{format(str(datetime.datetime.now()))}_{args.num_clients}_{args.num_rounds}_{args.sample_rate}_{args.alpha}"

    return args


model_names = ['msdnet24_1', 'msdnet24_4',
               'resnet110_1', 'resnet110_4',
               'effnetb4_1', 'effnetb4_4',
               'bert_1', 'bert_4']

arg_parser = argparse.ArgumentParser(
    description='Image classification PK main script')

exp_group = arg_parser.add_argument_group('exp', 'experiment setting')
exp_group.add_argument('--save_path', default=None,
                       type=str, metavar='SAVE',
                       help='path to the experiment logging directory')
exp_group.add_argument('--resume', action='store_true',
                       help='path to latest checkpoint (default: none)')
exp_group.add_argument('--evalmode', default=None,
                       choices=['local', 'global'],
                       help='which mode to evaluate')
exp_group.add_argument('--evaluate_from', default=None, type=str, metavar='PATH',
                       help='path to saved checkpoint (default: none)')
exp_group.add_argument('--print-freq', '-p', default=10, type=int,
                       metavar='N', help='print frequency (default: 100)')
exp_group.add_argument('--seed', default=0, type=int,
                       help='random seed')
exp_group.add_argument('--gpu_idx', default=None, type=str, help='Index of available GPU')
exp_group.add_argument('--use_gpu', default=1, type=int, help='Use CPU if zero')

# dataset related
data_group = arg_parser.add_argument_group('data', 'dataset setting')
data_group.add_argument('--data', metavar='D', default='cifar100',
                        choices=['cifar10', 'cifar100', 'imagenet', 'sst2', 'ag_news'],
                        help='data to work on')
data_group.add_argument('--data-root', metavar='DIR', default='data',
                        help='path to dataset (default: data)')
data_group.add_argument('--use-valid', action='store_true',
                        help='use validation set or not')
data_group.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 0)')
data_group.add_argument('-jj', '--num_fed_workers', default=1, type=int, metavar='N',
                        help='number of fl workers (default: 1)')
# model arch related
arch_group = arg_parser.add_argument_group('arch', 'model architecture setting')
arch_group.add_argument('--arch', '-a', metavar='ARCH', default='resnet110_4',
                        type=str, choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet110_4)')
arch_group.add_argument('--ee_locs', type=int, nargs='*', default=[], help='ee locations')

# training related
optim_group = arg_parser.add_argument_group('optimization', 'optimization setting')

optim_group.add_argument('--start_round', default=0, type=int, metavar='N',
                         help='manual round number (useful on restarts)')
optim_group.add_argument('-b', '--batch-size', type=int, help='mini-batch size')
optim_group.add_argument('--KD_gamma', type=float, default=0, help='KD gamma')
optim_group.add_argument('--KD_T', type=int, default=3, help='KD T')

# FL related
fl_group = arg_parser.add_argument_group('fl', 'FL setting')
fl_group.add_argument('--vertical_scale_ratios', type=float, nargs='*', default=[0.7, 0.7, 0.75, 1],
                      help='model split ratio vertically for each complexity level')
fl_group.add_argument('--horizontal_scale_ratios', type=int, nargs='*', default=[1, 2, 3, 4],
                      help='model horizontal split indices for each complexity level')
fl_group.add_argument('--client_split_ratios', type=float, nargs='*', default=[0.25, 0.25, 0.25, 0.25],
                      help='client ratio at each complexity level')
fl_group.add_argument('--num_rounds', type=int, default=2,
                      help='number of rounds')
fl_group.add_argument('--num_clients', type=int, default=100,
                      help='number of clients')
fl_group.add_argument('--sample_rate', type=float, default=0.1,
                      help='client sample rate')
fl_group.add_argument('--alpha', type=int, default=100,
                      help='data nonIID alpha')
fl_group.add_argument('-trs', '--track_running_stats', action='store_true',
                      help='trs')