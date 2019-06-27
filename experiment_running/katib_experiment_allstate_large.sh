#!/bin/bash
set -e

ID=allstate
OPT_GOAL=-1000
FILE="allstate_state_F_final_NY:0.01,0.495,0.495|MO:0.6,0.2,0.2|OK:0.2,0.4,0.4|FL:0.9,0.05,0.05|KS:0.4,0.3,0.3|CT:0.01,0.495,0.495_latest.p"
BUDGET_MIN=5000
INNER_LAYER_MULT=2.0
SIZE=large

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
