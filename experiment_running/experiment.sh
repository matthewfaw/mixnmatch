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
INNER_LAYER_SIZE=${14}
BATCH_SIZE=${15}
NU=${16}
RHO=${17}
ETA=${18}
#ETA_DECAY_STEP=30000
#ETA_DECAY_MULT=0.5
ETA_DECAY_STEP=0
ETA_DECAY_MULT=1
RETURN_BEST_DEEPEST_NODE=${19}
MIXTURE_SELECTION_STRATEGY=${20}
CUSTOM_MIXTURE=${21}
COLUMNS_TO_CENSOR=${22}
ACTUAL_MIXTURES_AND_BUDGETS_CM=${23}
PUBLISH_CM=${24}
SIZE=${25}
DATE=${26}
GROUP_WITH=${27}
TREE_SEARCH_OBJECTIVE=${28}
EVALUATE_BEST_RESULT_AGAIN=${29}
MODEL_MODE=${30:-torch}
SKLEARN_LOSS=${31:-hinge}
SKLEARN_LOSS_PENALTY=${32:-l2}
SKLEARN_LEARNING_RATE=${33:-optimal}
SKLEARN_LEARNING_RATE_ALPHA=${34:-0.0001}
SKLEARN_KERNEL=${35:-rbf}
SKLEARN_KERNEL_GAMMA=${36:-1.0}
SKLEARN_KERNEL_NCOMPONENTS=${37:-100}

if [[ "$EXPERIMENT_TYPE" = "importance-weighted-erm" ]] || [[ "$EXPERIMENT_TYPE" = "importance-weighted-uniform" ]]; then
    DATASET_CREATION_FOLDER="postprocessed/importance-weight"
else
    DATASET_CREATION_FOLDER="created"
fi

EVALUATE_BEST_RESULT_AGAIN_ETA_MULT=0.1
if [[ "$PUBLISH_CM" = "true" ]]; then
    CM_TO_PUBLISH=${DATASET_ID_LOWER}-${SIZE}-${EXPERIMENT_TYPE}-${OPT_BUDGET}-${MIXTURE_SELECTION_STRATEGY}-${DATE}
else
    CM_TO_PUBLISH=""
fi
RECORD_TEST_ERROR=True
#RECORD_TEST_ERROR=""
RAND_ALPHNUM=$(openssl rand -hex 12)
EXP_GROUPING="exp-run-${DATE}"
UNIQUE_ID=$(echo $RAND_ALPHNUM | cut -c -$((63 - $(echo $EXP_GROUPING | wc -c) + 1)) )
JOB_NAME="${EXP_GROUPING}-${UNIQUE_ID}"
USE_ALT_LOSS_FN="True"

MMD_RBF_GAMMA="1.0"
MMD_RBF_NCOMPONENTS="100"
MMD_REPR_SET_SIZE="800"

cat run_experiment.yaml |\
    sed "s/<GCLOUD_PROJECT>/${GCLOUD_PROJECT}/g" |\
    sed "s|<GCLOUD_DATASET_BUCKET>|${GCLOUD_DATASET_BUCKET}|g" |\
    sed "s/<TAG>/${TAG}/g" |\
    sed "s/<DATASET_ID>/${DATASET_ID}/g" |\
    sed "s/<DATASET_ID_LOWER>/${DATASET_ID_LOWER}/g" |\
    sed "s/<DATASET_FILENAME>/${DATASET_FILENAME}/g" |\
    sed "s|<DATASET_CREATION_FOLDER>|${DATASET_CREATION_FOLDER}|g" |\
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
    sed "s/<INNER_LAYER_SIZE>/${INNER_LAYER_SIZE}/g" |\
    sed "s/<NUM_REPEATS>/${NUM_REPEATS}/g" |\
    sed "s/<BATCH_SIZE>/${BATCH_SIZE}/g" |\
    sed "s/<MODEL_MODE>/${MODEL_MODE}/g" |\
    sed "s/<SKLEARN_LOSS>/${SKLEARN_LOSS}/g" |\
    sed "s/<SKLEARN_LOSS_PENALTY>/${SKLEARN_LOSS_PENALTY}/g" |\
    sed "s/<SKLEARN_LEARNING_RATE>/${SKLEARN_LEARNING_RATE}/g" |\
    sed "s/<SKLEARN_LEARNING_RATE_ALPHA>/${SKLEARN_LEARNING_RATE_ALPHA}/g" |\
    sed "s/<SKLEARN_KERNEL>/${SKLEARN_KERNEL}/g" |\
    sed "s/<SKLEARN_KERNEL_GAMMA>/${SKLEARN_KERNEL_GAMMA}/g" |\
    sed "s/<SKLEARN_KERNEL_NCOMPONENTS>/${SKLEARN_KERNEL_NCOMPONENTS}/g" |\
    sed "s/<NU>/${NU}/g" |\
    sed "s/<RHO>/${RHO}/g" |\
    sed "s/<ETA>/${ETA}/g" |\
    sed "s/<ETA_DECAY_STEP>/${ETA_DECAY_STEP}/g" |\
    sed "s/<ETA_DECAY_MULT>/${ETA_DECAY_MULT}/g" |\
    sed "s/<RETURN_BEST_DEEPEST_NODE>/${RETURN_BEST_DEEPEST_NODE}/g" |\
    sed "s/<MIXTURE_SELECTION_STRATEGY>/${MIXTURE_SELECTION_STRATEGY}/g" |\
    sed "s/<CUSTOM_MIXTURE>/${CUSTOM_MIXTURE}/g" |\
    sed "s/<COLUMNS_TO_CENSOR>/${COLUMNS_TO_CENSOR}/g" |\
    sed "s/<ACTUAL_MIXTURES_AND_BUDGETS_CM>/${ACTUAL_MIXTURES_AND_BUDGETS_CM}/g" |\
    sed "s/<CM_TO_PUBLISH>/${CM_TO_PUBLISH}/g" |\
    sed "s/<RECORD_TEST_ERROR>/${RECORD_TEST_ERROR}/g" |\
    sed "s/<SIZE>/${SIZE}/g" |\
    sed "s/<EXP_GROUPING>/${EXP_GROUPING}/g" |\
    sed "s/<UNIQUE_ID>/${UNIQUE_ID}/g" |\
    sed "s/<JOB_NAME>/${JOB_NAME}/g" |\
    sed "s/<GROUP_WITH>/${GROUP_WITH}/g" |\
    sed "s/<TREE_SEARCH_OBJECTIVE>/${TREE_SEARCH_OBJECTIVE}/g" |\
    sed "s/<EVALUATE_BEST_RESULT_AGAIN>/${EVALUATE_BEST_RESULT_AGAIN}/g" |\
    sed "s/<EVALUATE_BEST_RESULT_AGAIN_ETA_MULT>/${EVALUATE_BEST_RESULT_AGAIN_ETA_MULT}/g" |\
    sed "s/<USE_ALT_LOSS_FN>/${USE_ALT_LOSS_FN}/g" |\
    sed "s/<MMD_RBF_GAMMA>/${MMD_RBF_GAMMA}/g" |\
    sed "s/<MMD_RBF_NCOMPONENTS>/${MMD_RBF_NCOMPONENTS}/g" |\
    sed "s/<MMD_REPR_SET_SIZE>/${MMD_REPR_SET_SIZE}/g" |\
    tee /dev/tty |\
    kubectl apply -f -
