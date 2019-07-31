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

DOWNLOAD_TRANSFORMED=True
DATASET_PATH=/transfer/transformed.csv
DATASET_ID=allstate
IS_CATEGORICAL=True
SPLIT_KEY=state
TARGET="G_final"
#COLS_TO_DROP="customer_ID,cost_final"
COLS_TO_DROP="customer_ID"
#for key in {A..G}; do
#    if [[ "${key}_final" != "${TARGET}" ]]; then
#        COLS_TO_DROP+=",${key}_final"
#    fi
#done
VAL_TRAIN_VAL_TEST="FL:0.9934,0.0004,0.005,0.0012|CT:0.7,0.01875,0.225,0.05625|OH:0.0225,0.001875,0.0225,0.953125"
TAG=latest
SIZE=aimk-alt2-small-val

cat dataset_creation.yaml |\
    sed "s/<DOWNLOAD_TRANSFORMED>/${DOWNLOAD_TRANSFORMED}/g" |\
    sed "s|<DATASET_PATH>|${DATASET_PATH}|g" |\
    sed "s/<DATASET_ID>/$DATASET_ID/g" |\
    sed "s/<DATASET_ID_LOWER>/$(echo $DATASET_ID | tr '[:upper:]' '[:lower:]')/g" |\
    sed "s/<IS_CATEGORICAL>/$IS_CATEGORICAL/g" |\
    sed "s/<SPLIT_KEY>/$SPLIT_KEY/g" |\
    sed "s/<TARGET>/$TARGET/g" |\
    sed "s/<COLS_TO_DROP>/$COLS_TO_DROP/g" |\
    sed "s/<VAL_TRAIN_VAL_TEST>/$VAL_TRAIN_VAL_TEST/g" |\
    sed "s/<GCLOUD_PROJECT>/$GCLOUD_PROJECT/g" |\
    sed "s/<TAG>/$TAG/g" |\
    sed "s/<SIZE>/$SIZE/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    kubectl apply -f -

