# mlops-pytorch-pipeline

Solution for ML Ops course assignment 3 (in 3rd trimester of MTech AI from IIT Madras).
Full assignment question is given in [assignment3.pdf](assignment3.pdf) .

## With Kubernetes (local cluster setup)

```bash
# Start minikube (local kubernetes cluster) inside a docker container -> don't need to keep open seperate terminal for it
$ minikube start --driver=docker --cpus=4 --memory=6144
😄  minikube v1.38.1 on Ubuntu 26.04
✨  Using the docker driver based on existing profile
❗  You cannot change the memory size for an existing minikube cluster. Please first delete the cluster.
❗  You cannot change the CPUs for an existing minikube cluster. Please first delete the cluster.
👍  Starting "minikube" primary control-plane node in "minikube" cluster
🚜  Pulling base image v0.0.50 ...
🏃  Updating the running docker "minikube" container ...
🐳  Preparing Kubernetes v1.35.1 on Docker 29.2.1 ...
🔎  Verifying Kubernetes components...
    ▪ Using image gcr.io/k8s-minikube/storage-provisioner:v5
🌟  Enabled addons: storage-provisioner, default-storageclass
🏄  Done! kubectl is now configured to use "minikube" cluster and "default" namespace by default

# configure any further docker containers (terminal) to run inside minikube's environment (which is itself a docker container!) instead of on host directly
# NOTE: running docker-in-docker is usually bad idea, but Minikube is an exception
$ eval $(minikube docker-env)        # outputs nothing
$ echo $MINIKUBE_ACTIVE_DOCKERD     # env var set by the above command
minikube

# Build the training image directly inside Minikube's environment --- same command as before (direct in docker)
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
[+] Building 1.8s (11/11) FINISHED                                                                                                                                      docker:default
 ...
$ minikube image ls  # list images inside minikube - proves the docker build ran inside it 
...
docker.io/library/mlops-train:v1

# starts new terminal, must keep it open for subsequent commands
$ minikube mount $(pwd):/host-project        # hosts a host folder onto minikube's environment
📁  Mounting host path /home/sohang/Projects/mlops-pytorch-pipeline into VM as /host-project ...
    ▪ Mount type:   9p
    ▪ User ID:      docker
    ▪ Group ID:     docker
    ▪ Version:      9p2000.L
    ▪ Message Size: 262144
    ▪ Options:      map[]
    ▪ Bind Address: 192.168.49.1:38353
🚀  Userspace file server: 
ufs starting
✅  Successfully mounted /home/sohang/Projects/mlops-pytorch-pipeline to /host-project

📌  NOTE: This process must stay alive for the mount to be accessible ...

# Apply Kubernetes manifests in order
$ kubectl apply -f k8s/namespace.yaml
namespace/ml-training created
$ kubectl apply -f k8s/configmap.yaml
configmap/training-config created
$ kubectl apply -f k8s/training-job.yaml
persistentvolumeclaim/ml-training-pvc created
job.batch/ml-training-job created

# Monitor Job execution and stream training logs
$ kubectl -n ml-training get pods -w
$ kubectl -n ml-training logs -f job/ml-training-job

# --- Optional / Cleanup ---
# Unset the Minikube Docker environment in your current shell
$ eval $(minikube docker-env -u)

# delete existing training job
# can reapply training-config.yaml only after this to avoid immutable job error
$ kubectl -n ml-training delete job ml-training-job     
job.batch "ml-training-job" deleted from ml-training namespace

# Delete the Minikube cluster when finished
$ minikube delete

#### Failing command outs
$ kubectl -n ml-training get pods -l job-name=ml-training-job       
No resources found in ml-training namespace.
$ kubectl -n ml-training get jobs
NAME              STATUS   COMPLETIONS   DURATION   AGE
ml-training-job   Failed   0/1           3m40s      3m40s

$ kubectl -n ml-training run train-interactive \
  --image=mlops-train:v1 \
  --image-pull-policy=IfNotPresent \
  --restart=Never \
  --attach \
  --rm \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "trainer",
        "image": "mlops-train:v1",
        "volumeMounts": [
          {"name": "config-volume", "mountPath": "/app/configs"},
          {"name": "storage-volume", "mountPath": "/app/data", "subPath": "data"},
          {"name": "storage-volume", "mountPath": "/app/checkpoints", "subPath": "checkpoints"}
        ]
      }],
      "volumes": [
        {"name": "config-volume", "configMap": {"name": "training-config"}},
        {"name": "storage-volume", "persistentVolumeClaim": {"claimName": "ml-training-pvc"}}
      ]
    }
  }'
All commands and output from this session will be recorded in container logs, including credentials and sensitive information passed through the command prompt.
If you don't see a command prompt, try pressing enter.
Using device: cpu
Hyperparams loaded: {'model': {'architecture': 'resnet18', 'num_classes': 10}, 'training': {'epochs': 10, 'batch_size': 64, 'learning_rate': 0.001, 'early_stopping_patience': 3}, 'data': {'dataset': 'cifar10', 'data_dir': '/app/data'}, 'output': {'checkpoint_dir': '/app/checkpoints', 'model_name': 'classifier_v1.pt'}}
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
Traceback (most recent call last):
  File "/app/src/train.py", line 90, in <module>
    train_loader, val_loader = cifar10_train_val_dataloaders(hyperparams['batch_size'])
                                                             ~~~~~~~~~~~^^^^^^^^^^^^^^
KeyError: 'batch_size'
pod "train-interactive" deleted from ml-training namespace
pod ml-training/train-interactive terminated (Error)
```

## With Docker

Instructions adapted from assignment PDF:

```bash
# Build training image
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
# Run training with mounted volumes; specify current user so that checkpoints/model.pth is created NOT owned by root
# reason for deleting & recreating checkpoints directory is to prevent docker from creating checkpoints/ folder as root
$ rm -rf checkpoints/ && mkdir checkpoints/ && docker run --rm \
    --user "$(id -u):$(id -g)" \
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
# Check health status of the started serve docker container 
$ docker ps
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
$ python -m doctest src/*.py
```

## Development Details

[Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/#summary) are required - i.e., Git commit messages should follow this standard format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

To automatically follow Conventional Commits standard, `npm install --global git-cz` is installed, and instead of `git commit`, `git cz` command is used.