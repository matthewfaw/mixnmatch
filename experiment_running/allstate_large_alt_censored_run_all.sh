#!/bin/bash

ID=allstate
FILE="allstate_state_F_final_NY:0.9,0.05,0.05|MO:0.6,0.2,0.2|OK:0.2,0.4,0.4|FL:0.2,0.4,0.4|KS:0.4,0.3,0.3_latest.p"
OPT_MULT=949
OPT_MULT_CONST_STD=1
OPT_MULT_CONST_DOUBLE=2
OPT_HEIGHT_CAP=6
INNER_LAYER_MULT=2.0
BATCH=22
BUDGET_MIN=1000
BUDGET_MAX=30001
BUDGET_STEP=5000
N_REPEATS=10
MEM_REQ=2
MEM_LIMIT=2.5
NU=18.1489
RHO=0.6992
ETA=0.0081
RETURN_DEEPEST=True
SIZE=large-alt-censored
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"
TREE_EXP_TYPE=constant
COLS_TO_CENSOR="homeowner,car_age,risk_factor,age_oldest,age_youngest,married_couple,car_value_a,car_value_b,car_value_c,car_value_d,car_value_e,car_value_f,car_value_g,car_value_h,car_value_i,"
ACTUAL_MIXTURES_AND_BUDGETS_CM="${ID}-${SIZE}-tree-${TREE_EXP_TYPE}-coordinate-halving-${DATE}"
GROUP_WITH=""

./experiment.sh ${ID} ${FILE} tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "delaunay-partitioning" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "coordinate-halving" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "constant-mixture" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "tree-results" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "alpha-star" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "uniform" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "uniform" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"

