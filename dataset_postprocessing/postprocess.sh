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

CURR_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURR_BRANCH" = "master" ]]; then
    TAG=latest
else
    TAG="latest-$CURR_BRANCH"
fi
DATASET_ID=$1
DATASET_ID_LOWER=$(echo $DATASET_ID | tr '[:upper:]' '[:lower:]')
DATASET_FILENAME="$2"
POSTPROCESS_STEP="$3"
MAX_LR_ITERS="$4"
SIZE=$5
DATE=$6

RAND_ALPHNUM=$(openssl rand -hex 12)
POSTPROCESS_GROUPING="post-iw-${DATE}"
UNIQUE_ID=$(echo $RAND_ALPHNUM | cut -c -$((63 - $(echo $POSTPROCESS_GROUPING | wc -c) + 1)) )
JOB_NAME="${POSTPROCESS_GROUPING}-${UNIQUE_ID}"

if [[ "$POSTPROCESS_STEP" = "importance-weight" ]]; then
    DATASET_CREATION_FOLDER="created"
else
    echo "Postprocessing step $POSTPROCESS_STEP dataset creation folder has not been specified. Cannot proceed"
    exit 1
fi

cat dataset_postprocess.yaml |\
    sed "s/<GCLOUD_PROJECT>/${GCLOUD_PROJECT}/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    sed "s/<TAG>/${TAG}/g" |\
    sed "s/<DATASET_ID>/${DATASET_ID}/g" |\
    sed "s/<DATASET_ID_LOWER>/${DATASET_ID_LOWER}/g" |\
    sed "s/<DATASET_FILENAME>/${DATASET_FILENAME}/g" |\
    sed "s/<DATASET_CREATION_FOLDER>/${DATASET_CREATION_FOLDER}/g" |\
    sed "s/<POSTPROCESS_STEP>/${POSTPROCESS_STEP}/g" |\
    sed "s/<MAX_LR_ITERS>/${MAX_LR_ITERS}/g" |\
    sed "s/<SIZE>/${SIZE}/g" |\
    sed "s/<POSTPROCESS_GROUPING>/${POSTPROCESS_GROUPING}/g" |\
    sed "s/<UNIQUE_ID>/${UNIQUE_ID}/g" |\
    sed "s/<JOB_NAME>/${JOB_NAME}/g" |\
    tee /dev/tty |\
    kubectl apply -f -