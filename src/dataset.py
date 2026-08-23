"""Load CIFAR 10 data."""

import ssl
from pathlib import Path

from torch import Tensor
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# Disable SSL certificate validation to avoid [SSL: CERTIFICATE_VERIFY_FAILED] error while downloading dataset
ssl._create_default_https_context = ssl._create_unverified_context

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR 10 class mapping
root_dir = Path(__file__).parent.parent

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

def cifar10_train_val_dataloaders(batch_size: int) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Returns train, test data loaders for CIFAR 10 data (with train & test transforms applied). 
    
    Saves CIFAR 10 dataset to data/ folder in current repo (git ignored). 
    Download takes a few minutes on first run. It's cached on disk subsequently.

    >>> train_loader, val_loader = cifar10_train_val_dataloaders(batch_size=64)
    """
    # Training needs data augmentation; Testing only needs conversion and normalization
    train_set = torchvision.datasets.CIFAR10(
        root=root_dir / 'data', 
        train=True, 
        download=True, 
        transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=root_dir / 'data', 
        train=False, 
        download=True, 
        transform=val_transform
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2
    )

    return train_loader, val_loader


if __name__ == '__main__':
    import doctest 
    doctest.testmod()
