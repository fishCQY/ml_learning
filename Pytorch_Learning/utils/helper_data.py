# @file    helper_data.py
# @author  cqy 3049623863@qq.com
# @date    2025/04/02 19:50:14
# @version 1.0
# @brief

import torch
from torch.utils.data import sampler  # 定义数据加载时的采样策略，控制数据被访问的顺序和频率。
# Sampler：所有采样器的基类。
# RandomSampler：随机采样（默认）。
# SequentialSampler：顺序采样。
# SubsetRandomSampler：从子集索引中随机采样
from torchvision import datasets  # 提供预置的公开数据集​（如 MNIST、CIFAR-10、ImageNet），简化数据下载和加载
from torch.utils.data import DataLoader  # 将数据集包装为可迭代的批量数据生成器，支持多进程加速和随机打乱。
from torch.utils.data import SubsetRandomSampler  # 从给定的索引列表中随机采样数据，常用于划分训练集和验证集
from torchvision import transforms  # 提供数据预处理和增强的标准化操作（如归一化、裁剪、翻转）


# ​训练集：6万张手写数字图片。
# ​验证集：从6万张中分出1.2万张（validation_fraction=0.2）。
# ​测试集：1万张独立图片，​全程不参与训练和验证

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # 逆归一化：tensor = (tensor * std) + mean
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_dataloaders_mnist(batch_size, num_workers=0,
                          validation_fraction=None,
                          train_transforms=None,
                          test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=train_transforms,
                                   download=True)
    valid_dataset = datasets.MNIST(root='data',
                                   train=True,
                                   transform=test_transforms)
    ## 测试集官方已经划分好，测试集已经与训练集完全独立，无需再次划分
    test_dataset = datasets.MNIST(root='data',
                                  train=False,
                                  transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 60000)  # MNIST训练集共6万样本
        train_indices = torch.arange(0, 60000 - num)
        valid_indices = torch.arange(60000 - num, 60000)
        # 使用SubsetRandomSampler随机采样索引
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)  # drop_last=True 丢弃不完整批次
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader


def get_dataloaders_cifar10(batch_size, num_workers=0,
                            validation_fraction=None,
                            train_transforms=None,
                            test_transforms=None):
    if train_transforms is None:
        train_transforms = transforms.ToTensor()
    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
    valid_dataset = datasets.CIFAR10(root='data',
                                     train=True,
                                     transform=test_transforms)
    test_dataset = datasets.CIFAR10(root='data',
                                    train=False,
                                    transform=test_transforms)

    if validation_fraction is not None:
        num = int(validation_fraction * 50000)
        train_indices = torch.arange(0, 50000 - num)
        valid_indices = torch.arange(50000 - num, 50000)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        valid_loader = DataLoader(dataset=valid_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=valid_sampler)
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)
    if validation_fraction is None:
        return train_loader, test_loader
    else:
        return train_loader, valid_loader, test_loader
