#!/bin/bash

ID=mnist
FILE="mnist_environment_smaller_than_five_0.1:1,0,0,0|0.2:1,0,0,0|0.7:1,0,0,0|0.6:0,0.3,0.7,0_latest-sparse-extensions.p"
POSTPROCESS="importance-weight"
MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS="60000"
SIZE=mnist-slight-shift
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

./postprocess.sh "${ID}" "${FILE}" "${POSTPROCESS}" "${MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS}" "${SIZE}" "${DATE}"
