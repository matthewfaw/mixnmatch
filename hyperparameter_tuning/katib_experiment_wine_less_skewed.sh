#!/bin/bash
set -e

ID=wine
OPT_GOAL=0.0
FILE="wine_country_price_US:0.4,0.3,0.3|Italy:0.2,0.4,0.4|Portugal:0.8,0.1,0.1|France:0.1,0.45,0.45|New Zealand:0.4,0.3,0.3.p"
BUDGET_MIN=5000
INNER_LAYER_MULT=1.0 # Not used
SIZE=less-skewed

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
