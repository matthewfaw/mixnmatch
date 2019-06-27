#!/bin/bash

ID=wine
FILE="wine_country_price_US:0.4,0.3,0.3|Italy:0.2,0.4,0.4|Portugal:0.8,0.1,0.1|France:0.1,0.45,0.45|New Zealand:0.4,0.3,0.3_latest.p"
OPT_MULT=109
OPT_MULT_CONST_STD=1
OPT_MULT_CONST_DOUBLE=2
OPT_HEIGHT_CAP=10
INNER_LAYER_MULT=1.0 # Not used
BATCH=26
BUDGET_MIN=1000
BUDGET_MAX=30001
BUDGET_STEP=5000
N_REPEATS=3
MEM_REQ=2
MEM_LIMIT=2.5
NU=50.7983
RHO=0.6022
ETA=0.0005
RETURN_DEEPEST=""
SIZE=less-skewed
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"
TREE_EXP_TYPE=linear
COLS_TO_CENSOR=None
ACTUAL_MIXTURES_AND_BUDGETS_CM="${ID}-${SIZE}-tree-${TREE_EXP_TYPE}-delaunay-partitioning-${DATE}"
GROUP_WITH=""

./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "delaunay-partitioning" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "coordinate-halving" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "tree-results" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "constant-mixture" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "alpha-star" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "uniform" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "uniform" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} "${FILE}" "uniform" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "uniform" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
