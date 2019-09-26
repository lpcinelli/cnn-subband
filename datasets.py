
import numpy as np
from torchvision import datasets, transforms

import torch
from torch.utils.data.sampler import SubsetRandomSampler


class Cifar10():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    default_transform = [transforms.ToTensor(), normalize]
    def __init__(self,
                 data_dir='./data',
                 batch_size=32,
                 split=0.1,
                 rnd_seed=0,
                 num_workers=4,
                 grayscale=False,
                 augmentation=False):
        """Create dataloaders for training and test sets of CIFAR-10 dataset. Download it if
        necessary.

        Args:
            batch_size (int, optional): Nb of img elements of each mini-batch. Defaults to 32.
            num_workers (int, optional): Nb of works to fetch data. Defaults to 4.
            extra_transforms (list, optional): Transformations to apply to the imgs. Defaults to [].
        """

        extra = []
        default_transforms = transforms.Compose(self.default_transform)

        if grayscale:
            extra += [transforms.Grayscale(num_output_channels=3)]
        if augmentation:
            extra += [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip()]

        extra_transforms = transforms.Compose(extra + self.default_transform)

        train_set = datasets.CIFAR10(root=data_dir,
                                     train=True,
                                     download=True,
                                     transform=extra_transforms)
        val_set = datasets.CIFAR10(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=default_transforms)

        self.train_loader, self.valid_loader = get_train_valid_loader(
            train_set,
            val_set,
            valid_size=split,
            batch_size=batch_size,
            num_workers = num_workers,
            rnd_seed=rnd_seed)

        test_set = datasets.CIFAR10(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=default_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers)

class Cifar100():
    normalize = transforms.Normalize(
        mean=(0.5071, 0.4865, 0.4409),
        std=(0.2673, 0.2564, 0.2762)
    )

    default_transform = [transforms.ToTensor(), normalize]
    def __init__(self,
                 data_dir='./data',
                 batch_size=32,
                 split=0.1,
                 rnd_seed=0,
                 num_workers=4,
                 extra_transforms=[]):
        """Create dataloaders for training and test sets of CIFAR-100 dataset. Download it if
        necessary.

        Args:
            batch_size (int, optional): Nb of img elements of each mini-batch. Defaults to 32.
            num_workers (int, optional): Nb of works to fetch data. Defaults to 4.
            extra_transforms (list, optional): Transformations to apply to the imgs. Defaults to [].
        """

        extra = []
        default_transforms = transforms.Compose(self.default_transform)

        if grayscale:
            extra += [transforms.Grayscale(num_output_channels=3)]
        if augmentation:
            extra += [transforms.RandomCrop(32, padding=4),
                                 transforms.RandomHorizontalFlip()]

        extra_transforms = transforms.Compose(extra +
                                              self.default_transform)

        train_set = datasets.CIFAR100(root=data_dir,
                                     train=True,
                                     download=True,
                                     transform=extra_transforms)
        val_set = datasets.CIFAR100(root=data_dir,
                                   train=True,
                                   download=True,
                                   transform=default_transforms)

        self.train_loader, self.valid_loader = get_train_valid_loader(
            train_set,
            val_set,
            valid_size=split,
            batch_size=batch_size,
            num_workers = num_workers,
            rnd_seed=rnd_seed)

        test_set = datasets.CIFAR100(root=data_dir,
                                    train=False,
                                    download=True,
                                    transform=default_transforms)
        self.test_loader = torch.utils.data.DataLoader(test_set,
                                                       batch_size=batch_size,
                                                       shuffle=False,
                                                       num_workers=num_workers)


def get_train_valid_loader(train_dataset,
                           valid_dataset,
                           valid_size,
                           batch_size,
                           num_workers,
                           rnd_seed=None,
                           pin_memory=False,
                           shuffle=True):

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(rnd_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        sampler=valid_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader


dataset = {
    'cifar10': Cifar10,
    'cifar100': Cifar100,
}
