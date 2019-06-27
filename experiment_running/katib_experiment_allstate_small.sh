#!/bin/bash
set -e

ID=allstate
OPT_GOAL=-1000
FILE="allstate_state_F_final_NY:0.7,0.15,0.15|CT:0.1,0.45,0.45|FL:0.3,0.35,0.35_latest.p"
BUDGET_MIN=5000
INNER_LAYER_MULT=2.0
SIZE=small

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
