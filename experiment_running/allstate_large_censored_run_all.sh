#!/bin/bash

ID=allstate
FILE="allstate_state_F_final_NY:0.01,0.495,0.495|MO:0.6,0.2,0.2|OK:0.2,0.4,0.4|FL:0.9,0.05,0.05|KS:0.4,0.3,0.3|CT:0.01,0.495,0.495_latest.p"
OPT_MULT=881
OPT_MULT_CONST_STD=1
OPT_MULT_CONST_DOUBLE=2
OPT_HEIGHT_CAP=8
INNER_LAYER_MULT=2.0
BATCH=22
BUDGET_MIN=1000
BUDGET_MAX=30001
BUDGET_STEP=5000
N_REPEATS=10
MEM_REQ=2
MEM_LIMIT=2.5
NU=43.7639
RHO=0.8209
ETA=0.0066
RETURN_DEEPEST=True
SIZE=large-censored
DATE="day-$(date '+%Y-%m-%d')-time-$(date '+%H-%M-%S')"
TREE_EXP_TYPE=height
COLS_TO_CENSOR="A_changes,B_changes,C_changes,C_previous,D_changes,E_changes,F_changes,G_changes,age_oldest,age_youngest,car_age,car_value_a,car_value_b,car_value_c,car_value_d,car_value_e,car_value_f,car_value_g,car_value_h,cost_avg,duration_previous,group_size,homeowner,location,married_couple,num_days,num_shopping_pts,risk_factor,time_range_sec"
ACTUAL_MIXTURES_AND_BUDGETS_CM="${ID}-${SIZE}-tree-${TREE_EXP_TYPE}-coordinate-halving-${DATE}"
GROUP_WITH=""

./experiment.sh ${ID} ${FILE} tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "delaunay-partitioning" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} tree ${TREE_EXP_TYPE} ${OPT_MULT} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "coordinate-halving" "" ${COLS_TO_CENSOR} dummy-mixtures-and-budgets-cm true ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "constant-mixture" constant ${OPT_MULT_CONST_STD} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "tree-results" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "constant-mixture" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "alpha-star" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
./experiment.sh ${ID} ${FILE} "uniform" constant ${OPT_MULT_CONST_DOUBLE} ${OPT_HEIGHT_CAP} ${BUDGET_MIN} ${BUDGET_MAX} ${BUDGET_STEP} ${N_REPEATS} ${MEM_REQ} ${MEM_LIMIT} ${INNER_LAYER_MULT} ${BATCH} ${NU} ${RHO} ${ETA} "${RETURN_DEEPEST}" "uniform" "" ${COLS_TO_CENSOR} ${ACTUAL_MIXTURES_AND_BUDGETS_CM} false ${SIZE} ${DATE} "${GROUP_WITH}"
