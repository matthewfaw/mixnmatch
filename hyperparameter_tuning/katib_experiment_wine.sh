#!/bin/bash
set -e

ID=wine
OPT_GOAL=0.0
FILE="wine_country_price_US:0.01,0.495,0.495|Italy:0.99,0.005,0.005|Portugal:0.99,0.005,0.005|Spain:0.99,0.005,0.005_latest.p"
BUDGET_MIN=5000
INNER_LAYER_MULT=1.0 # Not used
SIZE=regular

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
