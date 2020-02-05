#!/bin/bash
set -e

SRC_TYPE=kaggle
KAGGLE_SRC=allstate-purchase-prediction-challenge
KAGGLE_API_TYPE=competitions
DATASET_ID=allstate
MODE=""
DATASET_NAME=train.csv

./transform.sh "$SRC_TYPE" "$KAGGLE_SRC" "$KAGGLE_API_TYPE" "$DATASET_ID" "$MODE" "$DATASET_NAME"
