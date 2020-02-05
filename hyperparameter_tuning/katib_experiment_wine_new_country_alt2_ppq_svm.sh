#!/bin/bash
set -e

ID=wine
OPT_GOAL=-1000
FILE="wine_country_price_quartile_US:1.0,0.0,0.0,0.0|France:1.0,0.0,0.0,0.0|Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|Chile:0.0,0.05,0.95,0.0_latest.p"
BUDGET_MIN=30000
INNER_LAYER_MULT=2.0
SIZE=wine-nc-alt2-ppq-svm

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE} "svm"
