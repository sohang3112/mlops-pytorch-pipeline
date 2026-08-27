#! /bin/bash

# if any command in the sequence fails, immediately fail & exit script
set -e

eval $(minikube docker-env)          # configure all docker builds to happen inside minikube, not host
[[ "$MINIKUBE_ACTIVE_DOCKERD" == "minikube" ]] || { echo "Not building dockers in minikube environment!!"; exit 1; }
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
kubectl -n ml-training delete job ml-training-job --ignore-not-found
kubectl apply -f k8s/training-job.yaml        # re-create training job using latest image after deleting existing job
kubectl -n ml-training logs -f job/ml-training-job         # see live logs