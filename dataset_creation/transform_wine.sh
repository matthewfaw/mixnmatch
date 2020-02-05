#!/bin/bash
set -e

SRC_TYPE=kaggle
KAGGLE_SRC=dbahri/wine-ratings
KAGGLE_API_TYPE=datasets
DATASET_ID=wine
MODE="collapse-countries"
DATASET_NAME=train.csv,validation.csv,test.csv

./transform.sh "$SRC_TYPE" "$KAGGLE_SRC" "$KAGGLE_API_TYPE" "$DATASET_ID" "$MODE" "$DATASET_NAME"
