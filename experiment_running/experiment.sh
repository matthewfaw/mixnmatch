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
DATASET_ID_LOWER=$(echo $DATASET_ID | tr '[:upper:]' '[:lower:]')
DATASET_FILENAME="$2"
EXPERIMENT_TYPE=$3
OPT_BUDGET=$4
OPT_BUDGET_MULTIPLIER=$5
OPT_BUDGET_HEIGHT_CAP=$6
BUDGET_MIN=$7
BUDGET_MAX=$8
BUDGET_STEP=$9
NUM_REPEATS=${10}
MEM_REQ=${11}
MEM_LIMIT=${12}
INNER_LAYER_MULT=${13}
BATCH_SIZE=${14}
NU=${15}
RHO=${16}
ETA=${17}
RETURN_BEST_DEEPEST_NODE=${18}
MIXTURE_SELECTION_STRATEGY=${19}
COLUMNS_TO_CENSOR=${20}
ACTUAL_MIXTURES_AND_BUDGETS_CM=${21}
PUBLISH_CM=${22}
SIZE=${23}
DATE=${24}
GROUP_WITH=${25}
if [[ "$PUBLISH_CM" = "true" ]]; then
    CM_TO_PUBLISH=${DATASET_ID_LOWER}-${SIZE}-${EXPERIMENT_TYPE}-${OPT_BUDGET}-${MIXTURE_SELECTION_STRATEGY}-${DATE}
else
    CM_TO_PUBLISH=""
fi
RECORD_TEST_ERROR=True
RAND_ALPHNUM=$(openssl rand -hex 12)
EXP_GROUPING="exp-run-${DATE}"
UNIQUE_ID=$(echo $RAND_ALPHNUM | cut -c -$((63 - $(echo $EXP_GROUPING | wc -c) + 1)) )
JOB_NAME="${EXP_GROUPING}-${UNIQUE_ID}"

cat run_experiment.yaml |\
    sed "s/<GCLOUD_PROJECT>/${GCLOUD_PROJECT}/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    sed "s/<TAG>/${TAG}/g" |\
    sed "s/<DATASET_ID>/${DATASET_ID}/g" |\
    sed "s/<DATASET_ID_LOWER>/${DATASET_ID_LOWER}/g" |\
    sed "s/<DATASET_FILENAME>/${DATASET_FILENAME}/g" |\
    sed "s/<EXPERIMENT_TYPE>/${EXPERIMENT_TYPE}/g" |\
    sed "s/<OPT_BUDGET>/${OPT_BUDGET}/g" |\
    sed "s/<OPT_BUDGET_MULTIPLIER>/${OPT_BUDGET_MULTIPLIER}/g" |\
    sed "s/<OPT_BUDGET_HEIGHT_CAP>/${OPT_BUDGET_HEIGHT_CAP}/g" |\
    sed "s/<BUDGET_MIN>/${BUDGET_MIN}/g" |\
    sed "s/<BUDGET_MAX>/${BUDGET_MAX}/g" |\
    sed "s/<BUDGET_STEP>/${BUDGET_STEP}/g" |\
    sed "s/<MEM_REQ>/${MEM_REQ}/g" |\
    sed "s/<MEM_LIMIT>/${MEM_LIMIT}/g" |\
    sed "s/<INNER_LAYER_MULT>/${INNER_LAYER_MULT}/g" |\
    sed "s/<NUM_REPEATS>/${NUM_REPEATS}/g" |\
    sed "s/<BATCH_SIZE>/${BATCH_SIZE}/g" |\
    sed "s/<NU>/${NU}/g" |\
    sed "s/<RHO>/${RHO}/g" |\
    sed "s/<ETA>/${ETA}/g" |\
    sed "s/<RETURN_BEST_DEEPEST_NODE>/${RETURN_BEST_DEEPEST_NODE}/g" |\
    sed "s/<MIXTURE_SELECTION_STRATEGY>/${MIXTURE_SELECTION_STRATEGY}/g" |\
    sed "s/<COLUMNS_TO_CENSOR>/${COLUMNS_TO_CENSOR}/g" |\
    sed "s/<ACTUAL_MIXTURES_AND_BUDGETS_CM>/${ACTUAL_MIXTURES_AND_BUDGETS_CM}/g" |\
    sed "s/<CM_TO_PUBLISH>/${CM_TO_PUBLISH}/g" |\
    sed "s/<RECORD_TEST_ERROR>/${RECORD_TEST_ERROR}/g" |\
    sed "s/<SIZE>/${SIZE}/g" |\
    sed "s/<EXP_GROUPING>/${EXP_GROUPING}/g" |\
    sed "s/<UNIQUE_ID>/${UNIQUE_ID}/g" |\
    sed "s/<JOB_NAME>/${JOB_NAME}/g" |\
    sed "s/<GROUP_WITH>/${GROUP_WITH}/g" |\
    tee /dev/tty |\
    kubectl apply -f -
