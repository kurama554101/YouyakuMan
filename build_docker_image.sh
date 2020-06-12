#!/bin/bash

# set default parameter
VERSION=1.0
PUSH_ECR=false
DOCKERFILE_PATH="docker/server/Dockerfile.server"

# check parameter
if [ $# != 3 ]; then
    echo "parameter count is not match!! setting parameter = $*"
    exit 1
fi

# get parameter( arg1=AWS_ACCOUNT_ID, arg2=ECR_RESION, arg3=DEVICE )
AWS_ACCOUNT_ID=$1
ECR_RESION=$2
DEVICE=$3  # cpu or gpu
ECR_REPO_NAME="${AWS_ACCOUNT_ID}.dkr.ecr.${ECR_RESION}.amazonaws.com"

# set base docker image url
if [ $DEVICE = "gpu" ]; then
    TRAIN_BASE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-gpu-py36-cu101-ubuntu16.04"
    INFER_BASE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.4.0-gpu-py36-cu101-ubuntu16.04"
else
    TRAIN_BASE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.4.0-cpu-py36-ubuntu16.04"
    INFER_BASE_IMAGE="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.4.0-cpu-py36-ubuntu16.04"
fi

# login ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# build docker image of train
docker build -t youyakuman:${DEVICE}_training_${VERSION} --build-arg BASE_IMAGE=${TRAIN_BASE_IMAGE} -f ${DOCKERFILE_PATH} .

# build docker image of inference
docker build -t youyakuman:${DEVICE}_inference_${VERSION} --build-arg BASE_IMAGE=${INFER_BASE_IMAGE} -f ${DOCKERFILE_PATH} .

# login ECR for push
aws ecr get-login-password --region ${ECR_RESION} | docker login --username AWS --password-stdin ${ECR_REPO_NAME}

# push docker images into ECR
if [ $PUSH_ECR ]; then
    docker tag youyakuman:${DEVICE}_training_${VERSION} ${ECR_REPO_NAME}/youyakuman:${DEVICE}_training_${VERSION}
    docker tag youyakuman:${DEVICE}_inference_${VERSION} ${ECR_REPO_NAME}/youyakuman:${DEVICE}_inference_${VERSION}
    docker push ${ECR_REPO_NAME}/youyakuman:${DEVICE}_training_${VERSION}
    docker push ${ECR_REPO_NAME}/youyakuman:${DEVICE}_inference_${VERSION}
fi
