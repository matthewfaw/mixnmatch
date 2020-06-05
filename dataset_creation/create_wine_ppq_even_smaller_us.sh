#!/bin/bash
set -e

DOWNLOAD_TRANSFORMED=True
DATASET_ID=wine
# Note: must pass empty string in order to have
# is-categorical evaluate to false,
# since in python only the empty string is falsy
IS_CATEGORICAL="True"
SPLIT_KEY=country
TARGET="price_quartile"
COLS_TO_DROP="Other,price"
# Value counts
#{'US': 54265,
# 'Spain': 6573,
# 'France': 17776,
# 'Austria': 2799,
# 'Australia': 2294,
# 'Italy': 16914,
# 'Argentina': 3756,
# 'Portugal': 4875,
# 'South Africa': 1293,
# 'Israel': 489,
# 'Germany': 2120,
# 'Greece': 461,
# 'Chile': 4416,
# 'New Zealand': 1378,
# 'Uruguay': 109,
# 'Romania': 120,
# 'Canada': 254,
# 'Hungary': 145,
# 'Turkey': 90,
# 'Bulgaria': 141}
VAL_TRAIN_VAL_TEST="US:0.15,0.0,0.0,0.85|France:1.0,0.0,0.0,0.0|Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|Chile:0.0,0.05,0.95,0.0"
SIZE=new-country-smallerus-ppq

./create.sh "$DOWNLOAD_TRANSFORMED" "$DATASET_ID" "$IS_CATEGORICAL" "$SPLIT_KEY" "$TARGET" "$COLS_TO_DROP" "$VAL_TRAIN_VAL_TEST" "$SIZE"
