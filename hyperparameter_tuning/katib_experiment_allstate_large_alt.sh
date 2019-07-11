#!/bin/bash
set -e

ID=allstate
OPT_GOAL=-1000
FILE="allstate_state_F_final_NY:0.9,0.05,0.05|MO:0.6,0.2,0.2|OK:0.2,0.4,0.4|FL:0.2,0.4,0.4|KS:0.4,0.3,0.3_latest.p"
BUDGET_MIN=15000
INNER_LAYER_MULT=2.0
SIZE=large-alt

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
