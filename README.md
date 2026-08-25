# mlops-pytorch-pipeline

Solution for ML Ops course assignment 3 (in 3rd trimester of MTech AI from IIT Madras).
Full assignment question is given in [assignment3.pdf](assignment3.pdf) .

## With Docker

Instructions adapted from assignment PDF:

```bash
# Build training image
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
# Run training with mounted volumes
$ docker run --rm \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    -v $(pwd)/configs:/app/configs \
    mlops-train:v1

# Build serving image
$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
# Run serving
$ docker run --rm -p 8080:8080 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  mlops-serve:v1
# Test prediction endpoint
$ curl -X 'POST' \
  'http://127.0.0.1:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@src/test_images/car.jpeg;type=image/jpeg'
```

## Without Docker

### Install

After cloning this repo, pull objects (eg. trained model files) from Git LFS:

```bash
$ git lfs install

Install requirements inside a venv:

```bash
$ python -m venv .venv/    
$ source .venv/bin/activate
$ pip install -r requirements/train.txt
$ pip install -r requirements/serve.txt
```

### Run

Train using *configs/training_config.yaml* and save trained model to *checkpoints/model.pth*:

```bash
$ python src/train.py checkpoints/model.pth       # on first run, downloads CIFAR dataset to data/ folder
```

Serving this trained *checkpoints/model.pth* on FastAPI server:

```bash
$ python src/serve.py
```

Server starts, now sending POST request to *http://127.0.0.1:8080/predict* with image upload *test_images/car.jpeg* correctly gives label 'car'.

## Run Automated Tests

`doctest` is utilized to ensure all code examples given in docstrings run correctly. Run all tests like this:

```bash
$ cd src/
$ python -m doctest *.py
```

## Development Details

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) are required - i.e., Git commit messages should follow this standard format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

To automatically follow Conventional Commits standard, `npm install --global git-cz` is installed, and instead of `git commit`, `git cz` command is used.



