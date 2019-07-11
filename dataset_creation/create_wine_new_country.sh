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
DATASET_ID=wine
# Note: must pass empty string in order to have
# is-categorical evaluate to false,
# since in python only the empty string is falsy
IS_CATEGORICAL=""
SPLIT_KEY=country
TARGET=price
COLS_TO_DROP=Other
# US: 54265
# Italy: 16914
# Portugal: 4875
# Spain: 6573
# France: 17776
# New Zealand: 1378
VAL_TRAIN_VAL_TEST="Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|France:1.0,0.0,0.0,0.0|US:0.0,0.5,0.5,0.0"
TAG=latest
SIZE=new-country

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
