#!/bin/bash
set -e

SRC_TYPE=kaggle
KAGGLE_SRC=dbahri/wine-ratings
KAGGLE_API_TYPE=datasets
DATASET_ID=wine
MODE=""
DATASET_NAME=train.csv,validation.csv,test.csv
TRANSFORMED_CSV="transformed_ohe_countries.csv"

./transform.sh "$SRC_TYPE" "$KAGGLE_SRC" "$KAGGLE_API_TYPE" "$DATASET_ID" "$MODE" "$DATASET_NAME" "$TRANSFORMED_CSV"
