#!/bin/bash
set -e

ID=allstate
OPT_GOAL=-1000
FILE="allstate_state_G_final_FL:1.0,0.0,0.0,0.0|CT:0.7,0.075,0.225,0.0|OH:0.0,0.0,0.0225,0.9775_latest.p"
BUDGET_MIN=30000
INNER_LAYER_MULT=2.0
SIZE=aimk-extra

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE}
