from datasets import load_dataset
from transformers import AutoTokenizer

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
import torchvision.datasets as tvdatasets
import os
from data_tools.sampling import *


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def get_datasets(args):
    val_set = None
    if args.data == 'cifar10':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        train_set = tvdatasets.CIFAR10(args.data_root, train=True, download=True,
                                       transform=transforms.Compose([
                                           transforms.RandomCrop(32, padding=4),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
        test_set = tvdatasets.CIFAR10(args.data_root, train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          normalize
                                      ]))
    elif args.data == 'cifar100':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        train_set = tvdatasets.CIFAR100(args.data_root, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))
        test_set = tvdatasets.CIFAR100(args.data_root, train=False,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           normalize
                                       ]))
    elif args.data == 'imagenet':
        # imagenet
        traindir = os.path.join(args.data_root, 'train')
        valdir = os.path.join(args.data_root, 'val')
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.25, 0.25, 0.25])
        im_size = args.image_size[0]
        train_set = tvdatasets.ImageFolder(traindir, transforms.Compose([
            transforms.Resize(int(im_size * 9 / 8)),
            transforms.CenterCrop(im_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]))
        test_set = tvdatasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(int(im_size * 9 / 8)),
            transforms.CenterCrop(im_size),
            transforms.ToTensor(),
            normalize
        ]))
    elif args.data == 'sst2':
        task_to_keys = {
            "cola": ("sentence", None),
            "mnli": ("premise", "hypothesis"),
            "mnli-mm": ("premise", "hypothesis"),
            "mrpc": ("sentence1", "sentence2"),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "rte": ("sentence1", "sentence2"),
            "sst2": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
            "wnli": ("sentence1", "sentence2"),
        }

        task = "sst2"
        model_checkpoint = "bert-base-uncased"
        dataset = load_dataset("glue", task)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        sentence1_key, sentence2_key = task_to_keys[task]

        def preprocess_function(examples):
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        sentence1_key, sentence2_key = task_to_keys[task]

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_set = encoded_dataset['train']
        val_set = encoded_dataset['validation']
        test_set = encoded_dataset['test']
        test_df = pd.read_csv('datasets/sst2/test.tsv', header=None, sep='\t')
        synch_list = [test_set['sentence'].index(s.lower().replace('-lrb-', '(').replace('-rrb-', ')'))
                      for s in test_df.iloc[:, 0].tolist()]
        synch_list = [synch_list.index(i) for i in range(len(synch_list))]
        new_labels = [test_df.iloc[:, -1].tolist()[x] for x in synch_list]

        def change_label(data, idx):
            data['label'] = new_labels[idx]
            return data

        test_set = test_set.map(change_label, with_indices=True)

    elif args.data == 'ag_news':
        model_checkpoint = "bert-base-uncased"
        dataset = load_dataset(args.data)
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
        sentence1_key, sentence2_key = 'text', None

        def preprocess_function(examples):
            if sentence2_key is None:
                return tokenizer(examples[sentence1_key], truncation=True)
            return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

        encoded_dataset = dataset.map(preprocess_function, batched=True)
        train_set = encoded_dataset['train']
        test_set = encoded_dataset['test']
    else:
        raise NotImplementedError

    return train_set, val_set, test_set


def get_user_groups(train_set, val_set, test_set, args):
    train_user_groups, val_user_groups, test_user_groups = create_noniid_users(train_set, val_set, test_set, args, args.alpha)
    return train_user_groups, val_user_groups, test_user_groups


def get_dataloaders(args, batch_size, dataset):
    train_loader, val_loader, test_loader = None, None, None
    train_set, val_set, test_set = dataset

    if args.use_valid:
        if val_set is None:
            train_set_index = torch.randperm(len(train_set))
            if os.path.exists(os.path.join(args.save_path, 'index.pth')):
                train_set_index = torch.load(os.path.join(args.save_path, 'index.pth'))
            else:
                torch.save(train_set_index, os.path.join(args.save_path, 'index.pth'))
            if args.data.startswith('cifar'):
                num_sample_valid = 0
            elif args.data == 'imagenet':
                num_sample_valid = 0
            elif args.data == 'sst2':
                num_sample_valid = 872
            elif args.data == 'ag_news':
                num_sample_valid = 0
            else:
                raise NotImplementedError

            train_indices = train_set_index[:-num_sample_valid]
            val_indices = train_set_index[-num_sample_valid:]
            val_set = train_set
        else:
            train_indices = torch.arange(len(train_set))
            val_indices = torch.arange(len(val_set))

        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    train_indices),
                num_workers=args.workers, pin_memory=True)
        if 'val' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                val_set, batch_size=batch_size,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(
                    val_indices),
                num_workers=args.workers, pin_memory=True)
        if 'test' in args.splits:
            test_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
    else:
        if 'train' in args.splits:
            train_loader = torch.utils.data.DataLoader(
                train_set,
                batch_size=batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        if 'val' or 'test' in args.splits:
            val_loader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)
            test_loader = val_loader

    if 'train' not in args.splits:
        if len(val_loader.dataset.transform.transforms) > 2:
            val_loader.dataset.transform.transforms = val_loader.dataset.transform.transforms[-2:]

    if 'bert' in args.arch:
        train_loader.collate_fn = collate_fn
        val_loader.collate_fn = collate_fn
        test_loader.collate_fn = collate_fn

    return train_loader, val_loader, test_loader


def get_client_dataloader(dataset, idxs, args, batch_size):
    """
    Returns train, validation and test dataloaders for a given dataset
    and user indexes.
    """
    if 'bert' in args.arch:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                           num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    else:
        return torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(idxs)),
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
                                           num_workers=args.workers, pin_memory=True)


def collate_fn(data):
    return (pad_sequence([torch.tensor(d['input_ids']) for d in data], batch_first=True, padding_value=0),
            torch.tensor([d['label'] for d in data]))
