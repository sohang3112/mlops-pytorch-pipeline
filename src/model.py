"""Define architecture of CNN model to train on CIFAR 10 dataset."""

from torch import nn

def _conv_batchnorm_relu_maxpool_block(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]

def cnn_model() -> nn.Sequential:
    """CNN model for CIFAR-10 classification.
    
    >>> import torch
    >>> model = cnn_model()
    >>> mock_input = torch.randn(4, 3, 32, 32)
    >>> logits = model(mock_input)
    >>> logits.shape        # (num_batches, num_output_classes)
    torch.Size([4, 10])
    """
    model = nn.Sequential(
        *_conv_batchnorm_relu_maxpool_block(3, 32),
        *_conv_batchnorm_relu_maxpool_block(32, 64),
        *_conv_batchnorm_relu_maxpool_block(64, 128),
        nn.Flatten(),
        nn.Linear(128 * 4 * 4, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3),  # Prevents overfitting
        nn.Linear(512, 10),  # 10 output classes for CIFAR-10
        # Softmax layer omitted because CrossEntropyLoss() automatically does softmax
    )
    return model


if __name__ == '__main__':
    import doctest
    doctest.testmod()