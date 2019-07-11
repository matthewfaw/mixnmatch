#!/bin/bash
set -e

ID=wine
OPT_GOAL=0.0
FILE="wine_country_price_Italy:1.0,0.0,0.0|Spain:1.0,0.0,0.0|France:1.0,0.0,0.0|US:0.0,0.5,0.5_latest.p"
BUDGET_MIN=15000
INNER_LAYER_MULT=1.0 # Not used
SIZE=new-country

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
