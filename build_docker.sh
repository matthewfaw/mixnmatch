#!/bin/bash
set -e

PROJECT=$1
COMMIT_FIRST_7=$2
BRANCH=$3
BUILD_NUMBER=$4

TAG=${BRANCH}-${BUILD_NUMBER}-${COMMIT_FIRST_7}

echo "Building Dockerfile with tag ${TAG}"

if [[ "$BRANCH" = "master" ]]; then
    LATEST_TAG="latest"
else
    LATEST_TAG="latest-${BRANCH}"
fi
docker build -t gcr.io/${PROJECT}/main:${LATEST_TAG} -t gcr.io/${PROJECT}/main:${TAG} .
docker push gcr.io/${PROJECT}/main:${LATEST_TAG}
docker push gcr.io/${PROJECT}/main:${TAG}
