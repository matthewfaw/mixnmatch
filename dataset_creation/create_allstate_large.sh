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
TARGET="F_final"
COLS_TO_DROP="customer_ID,cost_final"
for key in {A..G}; do
    if [[ "${key}_final" != "${TARGET}" ]]; then
        COLS_TO_DROP+=",${key}_final"
    fi
done
#NY -- almost all 0 -- 91627
#MO -- largely 2 -- 15243
#OK -- largely 2 -- 13779
#FL -- largely 2 -- 106287
#KS -- largely 2 -- 5585
#CT -- almost all 0 -- 19353
VAL_TRAIN_VAL_TEST="NY:0.01,0.495,0.495,0.0|MO:0.6,0.2,0.2,0.0|OK:0.2,0.4,0.4,0.0|FL:0.9,0.05,0.05,0.0|KS:0.4,0.3,0.3,0.0|CT:0.01,0.495,0.495,0.0"
TAG=latest
SIZE=large

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
