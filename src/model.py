"""Define architecture of CNN model to train on CIFAR 10 dataset."""

from torch import nn, Tensor
import torch

from dataset import class_labels, load_tensor_from_image_bytes

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
        # Softmax layer omitted because CrossEntropyLoss() expects logits, not softmax probabilities
    )
    return model

def predict_labels(model: nn.Module, batch_tensor: Tensor) -> list[str]:
    """Predict labels using model and input tensor."""
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor)
        probabs = torch.softmax(logits, dim=1)
        label_indices = torch.argmax(probabs, dim=1)
        return [class_labels[idx] for idx in label_indices]

def predict_label_for_single_image(model: nn.Module, image_bytes: bytes, device: torch.device) -> str:
    """Predict label for a single image.
    
    Trained model should correctly predict for car test image:

    >>> from pathlib import Path
    >>> device = torch.device('cpu')
    >>> model = cnn_model().to(device)
    >>> model.load_state_dict(torch.load('checkpoints/classifier_v1.pt', weights_only=True))
    <All keys matched successfully>
    >>> image_bytes = Path('test_images/car.jpeg').read_bytes()
    >>> predict_label_for_single_image(model, image_bytes, device)    # BUG: instead of 'car', trained model inference is giving 'frog'
    'car'
    """
    image_tensor = load_tensor_from_image_bytes(image_bytes)
    batch_tensor = image_tensor.unsqueeze(0).to(device)      # add batch dimension at start
    labels = predict_labels(model, batch_tensor)
    return labels[0]

if __name__ == '__main__':
    import doctest
    doctest.testmod()