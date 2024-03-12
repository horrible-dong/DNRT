# Copyright (c) QIU Tian. All rights reserved.

from .cifar import CIFAR10, CIFAR100
from .folder import ImageFolder

num_classes = {
    # all in lowercase !!!
    'cifar10': 10,
    'cifar100': 100,
    'imagenet100': 100,
    'imagenet1k': 1000,
}


def build_dataset(args, split, download=True):
    """
    split: 'train', 'val', 'test' or others
    """
    import os
    from timm.data import create_transform

    split = split.lower()
    dataset_name = args.dataset.lower()
    dataset_path = os.path.join(args.data_root, dataset_name)

    if dataset_name == 'cifar10':
        if split == 'val':
            split = 'test'

        image_size = 32 if args.image_size is None else args.image_size
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std)

        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return CIFAR10(root=dataset_path,
                       split=split,
                       transform=transform,
                       download=download)

    if dataset_name == 'cifar100':
        if split == 'val':
            split = 'test'

        image_size = 32 if args.image_size is None else args.image_size
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)

        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std)

        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'test': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return CIFAR100(root=dataset_path,
                        split=split,
                        transform=transform,
                        download=download)

    if dataset_name in ['imagenet100', 'imagenet1k']:
        image_size = 224 if args.image_size is None else args.image_size
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        aug_kwargs = build_timm_aug_kwargs(args, image_size, mean, std)

        transform = {
            'train': create_transform(**aug_kwargs['train_aug_kwargs']),
            'val': create_transform(**aug_kwargs['eval_aug_kwargs']),
        }

        return ImageFolder(root=dataset_path,
                           split=split,
                           transform=transform)


def build_timm_aug_kwargs(args, image_size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    train_aug_kwargs = dict(input_size=image_size, is_training=True, use_prefetcher=False, no_aug=False,
                            scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), hflip=0.5, vflip=0., color_jitter=0.4,
                            auto_augment='rand-m9-mstd0.5-inc1', interpolation='random', mean=mean, std=std,
                            re_prob=0.25, re_mode='pixel', re_count=1, re_num_splits=0, separate=False)

    eval_aug_kwargs = dict(input_size=image_size, is_training=False, use_prefetcher=False, no_aug=False, crop_pct=0.875,
                           interpolation='bilinear', mean=mean, std=std)

    train_aug_kwargs.update(args.train_aug_kwargs)
    eval_aug_kwargs.update(args.eval_aug_kwargs)

    return {
        'train_aug_kwargs': train_aug_kwargs,
        'eval_aug_kwargs': eval_aug_kwargs,
    }
