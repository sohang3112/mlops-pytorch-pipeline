"""Train model at specified path.

$ python train.py path/to/model.pth
"""

from pathlib import Path

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

root_dir = Path(__file__).parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def load_hyperparams() -> dict[str, int | float]:
    """Load model hyper-parameters.
    
    >>> load_hyperparams().keys()
    dict_keys(['batch_size', 'epochs', 'patience', 'lr', 'weight_decay'])
    """
    with (root_dir / 'configs' / 'training_config.yaml').open() as f:
        return yaml.safe_load(f)

def train(
    model: nn.Module, train_loader: DataLoader[tuple[Tensor, Tensor]],
    batch_size: int, epochs: int, patience: int,
    lr: float, weight_decay: float,       # for AdamW optimizer
) -> None:
    """Train model with Early Stopping (according to patience)."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_epoch = 0
    best_loss = float('inf')
    for epoch in range(epochs):
        # train for current epoch
        model.train()
        for (inputs, label_indices) in tqdm(train_loader):
            inputs, label_indices = inputs.to(device), label_indices.to(device)
            optimizer.zero_grad() 
            logits = model(inputs)   
            loss = loss_fn(logits, label_indices)   
            loss.backward()       
            optimizer.step()  

        # evaluate loss for current epoch
        model.eval()
        with torch.no_grad():
            eval_loss = 0
            correct = 0
            total = 0
            for (inputs, label_indices) in val_loader:
                inputs, label_indices = inputs.to(device), label_indices.to(device)
                logits = model(inputs)
                probabs = torch.softmax(logits, dim=1)
                output_indices = torch.argmax(probabs, dim=1)
                eval_loss += loss_fn(logits, label_indices)
                correct += (output_indices == label_indices).sum()
                total += batch_size
            accuracy = correct / total
            print({'epoch': epoch, 'loss': float(eval_loss), 'accuracy': float(accuracy)})

        if eval_loss < best_loss:
            best_epoch, best_loss = epoch, eval_loss
        elif epoch - best_epoch > patience:
            print('Train: Early Stop at epoch', epoch)


if __name__ == '__main__':
    from argparse import ArgumentParser

    import torchinfo

    from model import cnn_model
    from dataset import cifar10_train_val_dataloaders

    parser = ArgumentParser(__doc__)
    parser.add_argument('model_path', type=Path, help='*.pth file path where trained model should be saved.')
    args = parser.parse_args()

    hyperparams = load_hyperparams()
    model = cnn_model()
    torchinfo.summary(model, input_size=(4,3,32,32))
    print('Loading CIFAR 10 data...')
    train_loader, val_loader = cifar10_train_val_dataloaders(hyperparams['batch_size'])
    print('Starting training...')
    train(model, train_loader, **hyperparams)
    torch.save(model.state_dict(), args.model_path)
    print('Trained model saved to', args.model_path)
