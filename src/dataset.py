"""Load CIFAR 10 data."""

import ssl
from pathlib import Path
from io import BytesIO

from torch import Tensor
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np

# Disable SSL certificate validation to avoid [SSL: CERTIFICATE_VERIFY_FAILED] error while downloading dataset
ssl._create_default_https_context = ssl._create_unverified_context
root_dir = Path(__file__).parent.parent
class_labels = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']  # CIFAR 10 class mapping

augment_transforms = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
]
preprocess_transforms = [
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
]
train_transform = transforms.Compose(augment_transforms + preprocess_transforms)
val_transform = transforms.Compose(preprocess_transforms)

def cifar10_train_val_dataloaders(batch_size: int) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    """Returns train, test data loaders for CIFAR 10 data (with train & test transforms applied). 
    
    Saves CIFAR 10 dataset to data/ folder in current repo (git ignored). 
    Download takes a few minutes on first run. It's cached on disk subsequently.

    >>> train_loader, val_loader = cifar10_train_val_dataloaders(batch_size=64)
    """
    # in kubernetes this file ends up at /app/src/dataset.py
    data_path = (root_dir / 'data').resolve()
    print('Data Path Folder:', data_path, end=', ')
    print('Has Contents:', list(data_path.iterdir()))

    train_set = torchvision.datasets.CIFAR10(
        root=data_path, 
        train=True, 
        transform=train_transform
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_path, 
        train=False, 
        transform=val_transform
    )

    train_loader = DataLoader(
        train_set, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0      # 2 -- set to 0 else kubernetes gives weird memory errors
    )
    val_loader = DataLoader(
        val_set, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0      # 2 -- set to 0 else kubernetes gives weird memory errors
    )
    return train_loader, val_loader

def load_tensor_from_image_bytes(image_bytes: bytes) -> Tensor:
    """Load tensor (for giving input to model) from image bytes.
    
    >>> from pathlib import Path
    >>> image_bytes = Path('test_images/car.jpeg').read_bytes()
    >>> image_tensor = load_tensor_from_image_bytes(image_bytes)
    >>> image_tensor.shape
    torch.Size([3, 32, 32])
    """
    # Pillow is needed only at serve time, not train. 
    # So put import inside function (instead of with rest of module imports) so that during training we don't get import error for Pillow
    from PIL import Image      
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))
    image_np = np.array(image)
    transformed_tensor: Tensor = val_transform(image_np)
    return transformed_tensor


if __name__ == '__main__':
    import doctest 
    doctest.testmod()
