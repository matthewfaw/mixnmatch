#!/bin/bash
set -e

DOWNLOAD_TRANSFORMED=True
DATASET_ID=amazon
# Note: must pass empty string in order to have
# is-categorical evaluate to false,
# since in python only the empty string is falsy
IS_CATEGORICAL="True"
SPLIT_KEY="ROLE_DEPTNAME"
TARGET="ACTION"
COLS_TO_DROP=""
#  ID    | #0s  | #1s
# 117878 | 1064 | 71
# 117941 | 700  | 63
# 117945 | 570  | 89
# 117920 | 541  | 56
#-------------------
# 120663 | 306  | 29


VAL_TRAIN_VAL_TEST="117878:1,0,0,0|117941:1,0,0,0|117945:1,0,0,0|117920:1,0,0,0|120663:0,0.3,0.7,0"
SIZE=amazon

./create.sh "$DOWNLOAD_TRANSFORMED" "$DATASET_ID" "$IS_CATEGORICAL" "$SPLIT_KEY" "$TARGET" "$COLS_TO_DROP" "$VAL_TRAIN_VAL_TEST" "$SIZE"
