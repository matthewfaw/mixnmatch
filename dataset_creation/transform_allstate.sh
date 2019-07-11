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

KAGGLE_SRC=allstate-purchase-prediction-challenge
KAGGLE_API_TYPE=competitions
DATASET_ID=allstate
DATASET_NAME=train.csv
TAG=latest

kubectl delete cm transformed-cm || true

cat dataset_transformation.yaml |\
    sed "s|<KAGGLE_SRC>|$KAGGLE_SRC|g" |\
    sed "s/<KAGGLE_API_TYPE>/$KAGGLE_API_TYPE/g" |\
    sed "s/<DATASET_NAME>/$DATASET_NAME/g" |\
    sed "s/<DATASET_ID>/$DATASET_ID/g" |\
    sed "s/<GCLOUD_PROJECT>/$GCLOUD_PROJECT/g" |\
    sed "s/<TAG>/$TAG/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    kubectl apply -f -
