#!/bin/bash

ID=wine
FILE="wine_country_price_US:1.0,0.0,0.0,0.0|France:1.0,0.0,0.0,0.0|Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|Chile:0.0,0.05,0.95,0.0|Australia:0.0,0.05,0.95,0.0_latest.p"
POSTPROCESS="importance-weight"
MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS="5000"
SIZE=new-country-alt11-npq
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

./postprocess.sh "${ID}" "${FILE}" "${POSTPROCESS}" "${MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS}" "${SIZE}" "${DATE}"