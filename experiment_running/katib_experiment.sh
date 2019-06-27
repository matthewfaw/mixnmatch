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
TAG=latest
DATASET_ID=$1
OPT_GOAL=$2
DATASET_ID_LOWER=$(echo $DATASET_ID | tr '[:upper:]' '[:lower:]')
DATASET_FILENAME=$3
EXPERIMENT_TYPE=tree
BUDGET_MIN=$4
BUDGET_MAX=$((BUDGET_MIN + 1))
BUDGET_STEP=1000
NUM_REPEATS=3
INNER_LAYER_MULT=$5
COLUMNS_TO_CENSOR=None
ACTUAL_MIXTURES_AND_BUDGETS_CM=dummy-mixtures-and-budgets-cm
SIZE=$6
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

cat katib_run_experiment.yaml |\
    sed "s/<GCLOUD_PROJECT>/${GCLOUD_PROJECT}/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    sed "s/<TAG>/${TAG}/g" |\
    sed "s/<DATASET_ID>/${DATASET_ID}/g" |\
    sed "s/<DATASET_ID_LOWER>/${DATASET_ID_LOWER}/g" |\
    sed "s/<OPT_GOAL>/${OPT_GOAL}/g" |\
    sed "s/<DATASET_FILENAME>/${DATASET_FILENAME}/g" |\
    sed "s/<EXPERIMENT_TYPE>/${EXPERIMENT_TYPE}/g" |\
    sed "s/<BUDGET_MIN>/${BUDGET_MIN}/g" |\
    sed "s/<BUDGET_MAX>/${BUDGET_MAX}/g" |\
    sed "s/<BUDGET_STEP>/${BUDGET_STEP}/g" |\
    sed "s/<INNER_LAYER_MULT>/${INNER_LAYER_MULT}/g" |\
    sed "s/<NUM_REPEATS>/${NUM_REPEATS}/g" |\
    sed "s/<COLUMNS_TO_CENSOR>/${COLUMNS_TO_CENSOR}/g" |\
    sed "s/<ACTUAL_MIXTURES_AND_BUDGETS_CM>/${ACTUAL_MIXTURES_AND_BUDGETS_CM}/g" |\
    sed "s/<DATE>/${DATE}/g" |\
    sed "s/<SIZE>/${SIZE}/g" |\
    tee /dev/tty |\
    kubectl apply -f -
