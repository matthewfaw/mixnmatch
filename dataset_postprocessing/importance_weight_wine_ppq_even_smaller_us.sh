#!/bin/bash

ID=wine
FILE="wine_country_price_quartile_US:0.15,0.0,0.0,0.85|France:1.0,0.0,0.0,0.0|Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|Chile:0.0,0.05,0.95,0.0_latest.p"
POSTPROCESS="importance-weight"
MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS="5000"
SIZE=new-country-small-us-ppq
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"

./postprocess.sh "${ID}" "${FILE}" "${POSTPROCESS}" "${MAX_LOGISTIC_REGRESSION_OPT_ITERATIONS}" "${SIZE}" "${DATE}"