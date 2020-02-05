#!/bin/bash
set -e

SRC_TYPE=kaggle
SRC="amazon-employee-access-challenge"
KAGGLE_API_TYPE="competitions"
DATASET_ID=amazon
MODE=""
DATASET_NAME=train.csv

./transform.sh "$SRC_TYPE" "$SRC" "$KAGGLE_API_TYPE" "$DATASET_ID" "$MODE" "$DATASET_NAME"
