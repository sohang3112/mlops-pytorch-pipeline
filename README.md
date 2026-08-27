# mlops-pytorch-pipeline

Solution for ML Ops course assignment 3 (in 3rd trimester of MTech AI from IIT Madras).
Full assignment question is given in [assignment3.pdf](assignment3.pdf) .


## With Kubernetes (local cluster setup using Minikube)

Kubernetes Training:

```bash
# Start minikube (local kubernetes cluster) inside a docker container -> don't need to keep open separate terminal for it
$ minikube start --driver=docker --cpus=4 --memory=6144

# IMPORTANT: run following in a separate dedicated terminal or tmux (or else run it as a background process)
# 2 parts of mounting volumes: host -> minikube (done by this command), and minikube -> training docker (that's done in training-job.yaml)
$ minikube mount $(pwd):/host-project

$ eval $(minikube docker-env)        # outputs nothing - configures docker builds to run in minikube environment not host docker
$ echo $MINIKUBE_ACTIVE_DOCKERD      # env var should be set by the above command
minikube
$ kubectl apply -f k8s/namespace.yaml        # create namespace
$ kubectl apply -f k8s/configmap.yaml        # configure hyper parameters for training
$ ./kubernetes_retrain.sh    # build docker, kill existing job & start new using kubectl apply -f k8s/training-job.yaml, show running logs
$ minikube delete     # Delete the Minikube cluster when finished
```

Most important parts of the applied Kubernetes YAML files are (note: except for *namespace.yaml*, all other YAML files explicitly mention `project: mlops-pytorch-pipeline`):

* [namespace.yaml](k8s/namespace.yaml) creates `kind: Namespace` having `name: ml-training`.
* [configmap.yaml](k8s/configmap.yaml) creates `kind: ConfigMap` having `name: training-config` in existing Namespace (`namespace: ml-training`) and specifies training hyper-parameters YAML (in section `data:   training_config.yaml:`). This overwrites existing contents of *configs/training_config.yaml*, so hyper-parameters must be configured in *k8s/configmap.yaml* only to have effect in Kubernetes.
* [training-job.yaml](k8s/training-job.yaml) has 2 jobs separated by `--` (both use `namespace: ml-training`) :
  * `kind: PersistentVolumeClaim` having `name: ml-training-pvc` requires disk space `storage: 10Gi` .
  * **Training** `kind: Job` having `name: ml-training-job` runs Docker `image: mlops-train:v1` that we built, constrained by limits `cpu: "2"` and `memory: "4Gi"`. */app/data/*, */app/configs/*, */app/checkpoints/* are all mounted as volumes using the PersistentValueClaim (PVC) we created.
    * IMPORTANT: `hostPath:  path: /host-project/data` refers to minikube environment's file system.

**NOTE**: Misnomer: Although namespace is called `ml-training`, it is in fact used in common for BOTH training and serving.s

Kubernetes Model Serving:

```bash
$ eval $(minikube docker-env)
$ docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
$ kubectl apply -f k8s/serving-deployment.yaml
$ kubectl apply -f k8s/serving-service.yaml
# When rebuilding the same image tag, force pods to use the new local image.
$ kubectl rollout restart deployment/ml-serving-deployment -n ml-training
# Wait until all replicas are ready.
$ kubectl rollout status deployment/ml-serving-deployment -n ml-training
# Verify pods are running and healthy:
$ kubectl get deployment,pods,svc -n ml-training -o wide

# If a serving pod is in CrashLoopBackOff, inspect Kubernetes events and FastAPI
# stdout. `--previous` retrieves the terminated container's logs after a restart.
$ kubectl get events -n ml-training --sort-by=.lastTimestamp
$ kubectl logs -n ml-training -l app=ml-serving --all-containers --tail=200 --prefix
$ kubectl logs -n ml-training -l app=ml-serving --all-containers --previous --tail=200 --prefix
$ kubectl describe deployment ml-serving-deployment -n ml-training

# Confirm the training job completed and the PVC it writes to is bound.
$ kubectl logs -n ml-training job/ml-training-job --tail=250
$ kubectl get pvc,pv -n ml-training -o wide

# The serving Deployment mounts the PVC at /app/checkpoints with
# `subPath: checkpoints`. The current training image saves classifier_v1.pt;
# an earlier image saved model.pth. Serving accepts either artifact so a
# persistent PVC from an earlier training run remains deployable. After
# changing the Deployment, apply it and wait for the rollout before testing.
$ kubectl apply -f k8s/serving-deployment.yaml
$ kubectl rollout status deployment/ml-serving-deployment -n ml-training
$ kubectl get pods -n ml-training -l app=ml-serving

# Test the prediction endpoint:
$ curl -X 'POST' \
  'http://127.0.0.1:8080/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@test_images/car.jpeg;type=image/jpeg'
```

The rollout can stall when a new serving pod enters `CrashLoopBackOff`. One
observed cause was a checkpoint-name mismatch: the persistent PVC contained
`checkpoints/model.pth` from an earlier training image while serving expected
`checkpoints/classifier_v1.pt`. `src/serve.py` now prefers `classifier_v1.pt`
and falls back to `model.pth`. Training and serving both mount the PVC's
`checkpoints` subdirectory.

## With Docker

Instructions adapted from assignment PDF:

```bash
# Build training image
$ docker build -f docker/Dockerfile.train -t mlops-train:v1 .
# Run training with mounted volumes; specify current user so that checkpoints/classifier_v1.pt is created NOT owned by root
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
  -F 'file=@test_images/car.jpeg;type=image/jpeg'
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

Train using *configs/training_config.yaml* and save trained model to *checkpoints/classifier_v1.pt*:

```bash
$ python src/train.py checkpoints/classifier_v1.pt      # on first run, downloads CIFAR dataset to data/ folder
```

Serving this trained *checkpoints/classifier_v1.pt* on FastAPI server:

```bash
$ python src/serve.py
```

Server starts, now sending POST request to *http://127.0.0.1:8080/predict* with image upload *test_images/car.jpeg* correctly gives label 'car'.

Server also has a GET */health* endpoint.

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
