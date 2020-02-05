#!/bin/bash
set -e

ID=allstate
OPT_GOAL=-1000
FILE="allstate_state_G_final_FL:0.4934,0.0016,0.005,0.5|CT:0.5,0.075,0.425,0.0|OH:0.0225,0.0075,0.0225,0.9475_latest.p"
BUDGET_MIN=30000
INNER_LAYER_MULT=2.0
SIZE=aimk-alt2-newfeats-svm

./katib_experiment.sh ${ID} ${OPT_GOAL} ${FILE} ${BUDGET_MIN} ${INNER_LAYER_MULT} ${SIZE} "svm"
