# mlops-pytorch-pipeline

ML Ops course assignment 3 (in 3rd trimester of MTech AI from IIT Madras)

## Install

Install requirements inside a venv:

```bash
$ python -m venv .venv/    
$ source .venv/bin/activate
$ pip install -r requirements/train.txt
$ pip install -r requirements/serve.txt
```

## Run

Training:

```bash
$ cat configs/training_config.yaml
batch_size: 512
epochs: 4
patience: 2

# AdamW optimizer params
lr: 0.001
weight_decay: 0.01

$ python src/train.py src/model.pth       # on first run, downloads CIFAR dataset to data/ folder
Using device: cpu
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [4, 10]                   --
├─Conv2d: 1-1                            [4, 32, 32, 32]           896
├─BatchNorm2d: 1-2                       [4, 32, 32, 32]           64
├─ReLU: 1-3                              [4, 32, 32, 32]           --
├─MaxPool2d: 1-4                         [4, 32, 16, 16]           --
├─Conv2d: 1-5                            [4, 64, 16, 16]           18,496
├─BatchNorm2d: 1-6                       [4, 64, 16, 16]           128
├─ReLU: 1-7                              [4, 64, 16, 16]           --
├─MaxPool2d: 1-8                         [4, 64, 8, 8]             --
├─Conv2d: 1-9                            [4, 128, 8, 8]            73,856
├─BatchNorm2d: 1-10                      [4, 128, 8, 8]            256
├─ReLU: 1-11                             [4, 128, 8, 8]            --
├─MaxPool2d: 1-12                        [4, 128, 4, 4]            --
├─Flatten: 1-13                          [4, 2048]                 --
├─Linear: 1-14                           [4, 512]                  1,049,088
├─ReLU: 1-15                             [4, 512]                  --
├─Dropout: 1-16                          [4, 512]                  --
├─Linear: 1-17                           [4, 10]                   5,130
==========================================================================================
Total params: 1,147,914
Trainable params: 1,147,914
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 45.74
==========================================================================================
Input size (MB): 0.05
Forward/backward pass size (MB): 3.69
Params size (MB): 4.59
Estimated Total Size (MB): 8.33
==========================================================================================
Loading CIFAR 10 data...
Starting training...
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:52<00:00,  1.86it/s]
{'epoch': 0, 'loss': tensor(26.4423), 'accuracy': tensor(0.5029)}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:58<00:00,  1.68it/s]
{'epoch': 1, 'loss': tensor(22.9558), 'accuracy': tensor(0.5798)}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:56<00:00,  1.73it/s]
{'epoch': 2, 'loss': tensor(19.7099), 'accuracy': tensor(0.6290)}
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 98/98 [00:58<00:00,  1.67it/s]
{'epoch': 3, 'loss': tensor(17.2747), 'accuracy': tensor(0.6751)}
Trained model saved to model.pth
```

## Development Details

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) are required - i.e., Git commit messages should follow this standard format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

To automatically follow Conventional Commits standard, `npm install --global git-cz` is installed, and instead of `git commit`, `git cz` command is used.



