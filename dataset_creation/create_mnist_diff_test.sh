#!/bin/bash
set -e

DOWNLOAD_TRANSFORMED="True"
DATASET_ID=mnist
# Note: must pass empty string in order to have
# is-categorical evaluate to false,
# since in python only the empty string is falsy
IS_CATEGORICAL="True"
SPLIT_KEY="environment"
TARGET="smaller_than_five"
COLS_TO_DROP=""

VAL_TRAIN_VAL_TEST="0.1:1,0,0,0|0.2:1,0,0,0|0.7:1,0,0,0|0.5:0,0,0.7,0.3|0.9:0,0.3,0,0.7"
SIZE=mnist-diff-test
TRANSFORMED_FILE="transformed_train:0.1,0.2,0.7|val:0.9|test:0.5.csv"

./create.sh "$DOWNLOAD_TRANSFORMED" "$DATASET_ID" "$IS_CATEGORICAL" "$SPLIT_KEY" "$TARGET" "$COLS_TO_DROP" "$VAL_TRAIN_VAL_TEST" "$SIZE" "$TRANSFORMED_FILE"
