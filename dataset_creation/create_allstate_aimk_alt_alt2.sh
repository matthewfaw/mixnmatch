#!/bin/bash
set -e

DOWNLOAD_TRANSFORMED=True
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
VAL_TRAIN_VAL_TEST="FL:0.4934,0.0016,0.005,0.5|CT:0.5,0.075,0.425,0.0|OH:0.0225,0.0075,0.0225,0.9475"
SIZE=aimk-newfeats-alt-alt2-fixerr

./create.sh "$DOWNLOAD_TRANSFORMED" "$DATASET_ID" "$IS_CATEGORICAL" "$SPLIT_KEY" "$TARGET" "$COLS_TO_DROP" "$VAL_TRAIN_VAL_TEST" "$SIZE"
