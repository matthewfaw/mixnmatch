#!/bin/bash

ID=wine
FILE="wine_country_price_US:1.0,0.0,0.0,0.0|France:1.0,0.0,0.0,0.0|Italy:1.0,0.0,0.0,0.0|Spain:1.0,0.0,0.0,0.0|Argentina:0.0,0.01,0.99,0.0_latest.p"
OPT_MULT=500
OPT_MULT_CONST_STD=1
OPT_MULT_CONST_DOUBLE=1
OPT_HEIGHT_CAP=inf
INNER_LAYER_MULT=1.0 # Not used
BATCH=25
BUDGET_MIN=1000
BUDGET_MAX=60001
BUDGET_STEP=5000
BUDGET_MIN_HALF=$((BUDGET_MIN / 2))
BUDGET_MAX_HALF=$(( (($BUDGET_MAX - 1) / 2) + 1))
BUDGET_STEP_HALF=$((BUDGET_STEP / 2))
N_REPEATS=10
MEM_REQ=1
MEM_LIMIT=1.5
NU=50
RHO=0.7656
ETA=0.002
RETURN_DEEPEST="True"
SIZE=new-country-alt6
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"
TREE_EXP_TYPE=constant
COLS_TO_CENSOR=None
ACTUAL_MIXTURES_AND_BUDGETS_CM="${ID}-${SIZE}-tree-${TREE_EXP_TYPE}-coordinate-halving-${DATE}"
GROUP_WITH=""

./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "delaunay-partitioning" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN_HALF} ${BUDGET_MAX_HALF} ${BUDGET_STEP_HALF} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "delaunay-partitioning" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm false ${SIZE} ${DATE} "${GROUP_WITH}" "True"
./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "coordinate-halving" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN_HALF} ${BUDGET_MAX_HALF} ${BUDGET_STEP_HALF} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "coordinate-halving" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm false ${SIZE} ${DATE} "${GROUP_WITH}" "True"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "tree-results" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "custom" "1,0,0,0" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "custom" "0,1,0,0" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "custom" "0,0,1,0" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "custom" "0,0,0,1" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "uniform" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "uniform" ${COLS_TO_CENSOR} "" ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "validation" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "validation" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"

