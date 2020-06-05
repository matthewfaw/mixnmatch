#!/bin/bash
set -e

if [[ -z "$GCLOUD_PROJECT" ]]; then
    echo "Must have GCLOUD_PROJECT env var set. Cannot proceed."
    exit 1
else
    echo "Using Gcloud project: $GCLOUD_PROJECT"
fi
if [[ -z "$GCLOUD_DATASET_BUCKET" ]]; then
    echo "Must have GCLOUD_DATASET_BUCKET env var set. Cannot proceed."
    exit 1
else
    echo "Using Gcloud project: $GCLOUD_DATASET_BUCKET"
fi

SRC_TYPE=$1
SRC=$2
KAGGLE_API_TYPE=$3
DATASET_ID=$4
MODE=$5
DATASET_NAME=$6
CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURR_BRANCH" = "master" ]]; then
    TAG=latest
else
    TAG="latest-$CURR_BRANCH"
fi
TRANSFORMED_CSV=${7:-transformed.csv}
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

kubectl delete cm transformed-${DATASET_ID}-cm || true

cat dataset_transformation.yaml |\
    sed "s|<SRC_TYPE>|$SRC_TYPE|g" |\
    sed "s|<SRC>|$SRC|g" |\
    sed "s/<KAGGLE_API_TYPE>/$KAGGLE_API_TYPE/g" |\
    sed "s/<DATASET_NAME>/$DATASET_NAME/g" |\
    sed "s/<DATASET_ID>/$DATASET_ID/g" |\
    sed "s/<GCLOUD_PROJECT>/$GCLOUD_PROJECT/g" |\
    sed "s/<TAG>/$TAG/g" |\
    sed "s/<DATE>/$DATE/g" |\
    sed "s/<TRANSFORMED_CSV>/$TRANSFORMED_CSV/g" |\
    sed "s/<MODE>/$MODE/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    kubectl apply -f -