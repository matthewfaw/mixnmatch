#!/bin/bash

ID=allstate
FILE="allstate_state_G_final_FL:0.4934,0.0016,0.005,0.5|CT:0.5,0.075,0.425,0.0|OH:0.0225,0.0075,0.0225,0.9475_latest.p"
POSTPROCESS="importance-weight"
MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS="5000"
SIZE=aimk-alt-newfeats-alt2
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

./postprocess.sh "${ID}" "${FILE}" "${POSTPROCESS}" "${MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS}" "${SIZE}" "${DATE}"
