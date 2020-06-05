import numpy as np

from configuration.experiment_configuration import ExperimentConfiguration
from configuration.tree_search_configuration import TreeSearchConfiguration


class ExperimentResultRecorder:
    def __init__(self,
                 experiment_config: ExperimentConfiguration):
        self.experiment_config = experiment_config
        self.recording_times = self.experiment_config.recording_times

        self.results = RecorderResult()
        self.next_recording_idx = 0
        self.next_recording_time = self.recording_times[self.next_recording_idx]

    def record(self, model, node, prev_cost, curr_cost, force_record=False):
        if prev_cost < self.next_recording_time <= curr_cost or force_record:
            val_mf_fn_result = node.mf_fn.get_validation_error(model)
            test_mf_fn_result = node.mf_fn.get_test_error(model)
            if self.experiment_config.record_test_error:
                mf_fn_results = test_mf_fn_result
            else:
                mf_fn_results = val_mf_fn_result

            self.results.append(val=max([fn_res.error.item() for fn_res in mf_fn_results]),
                                val_mf_fn_result=val_mf_fn_result,
                                test_mf_fn_result=test_mf_fn_result,
                                mixture=node.mixture,
                                l1_dist=np.linalg.norm(np.array(node.mixture)
                                                       - np.array(self.experiment_config.alpha_star), ord=1),
                                cost=curr_cost)

            # Update indices
            self.next_recording_idx += 1
            if len(self.recording_times) > self.next_recording_idx:
                self.next_recording_time = self.recording_times[self.next_recording_idx]
            else:
                self.next_recording_time = np.inf


class RecorderResult:
    def __init__(self):
        self.vals = []
        self.val_mf_fn_results = []
        self.test_mf_fn_results = []
        self.mixtures = []
        self.l1_dists = []
        self.costs = []

    def append(self,
               val,
               val_mf_fn_result,
               test_mf_fn_result,
               mixture,
               l1_dist,
               cost):
        print("At iter {}, ")
        self.vals.append(val)
        self.val_mf_fn_results.append(val_mf_fn_result)
        self.test_mf_fn_results.append(test_mf_fn_result)
        self.mixtures.append(mixture)
        self.l1_dists.append(l1_dist)
        self.costs.append(cost)


class RepeatedExperimentResults:
    def __init__(self):
        self.mf_fn_results_all = []
        self.execution_times_avg = []
        self.execution_times_std = []
        self.actual_costs_all = []
        self.best_sols_all = []
        self.final_mixtures_all = []
        self.l1_dists_all = []
        self.test_errors_avg = []
        self.test_errors_std = []
        self.recorders_all = []

    def append(self, vals, mf_fn_results, execution_times, best_sols, final_mixtures, l1_dists, total_costs, recorders, test_errors=None):
        val_avg = np.average(vals)
        print("best_sol_val_avg={}".format(val_avg))
        val_std = np.std(vals)
        print("best_sol_val_std={}".format(val_std))
        self.mf_fn_results_all.append(mf_fn_results)
        avg_execution_time = np.average(execution_times)
        self.execution_times_avg.append(avg_execution_time)
        print("execution_time_avg={}".format(avg_execution_time))
        self.execution_times_std.append(np.std(execution_times))
        actual_cost_avg = np.average(total_costs)
        actual_cost_std = np.std(total_costs)
        print("actual_cost_avg={}".format(actual_cost_avg))
        print("actual_cost_std={}".format(actual_cost_std))
        self.actual_costs_all.append(total_costs)
        self.best_sols_all.append(best_sols)
        self.final_mixtures_all.append(final_mixtures)
        self.l1_dists_all.append(l1_dists)
        print("l1_dist_avg={}".format(np.average(l1_dists)))
        print("l1_dist_std={}".format(np.std(l1_dists)))
        self.recorders_all.append(recorders)
        if mf_fn_results is not None and len(mf_fn_results) > 0 and mf_fn_results[0][0].auc_roc_ovo is not None:
            print("auc_roc_ovo_avg={}".format(np.average([min([mf_fn_res.auc_roc_ovo for mf_fn_res in mf_fn_res_list]) for mf_fn_res_list in mf_fn_results])))
            print("auc_roc_ovr_avg={}".format(np.average([min([mf_fn_res.auc_roc_ovr for mf_fn_res in mf_fn_res_list]) for mf_fn_res_list in mf_fn_results])))
            print("auc_roc_ovr_std={}".format(np.std([min([mf_fn_res.auc_roc_ovr for mf_fn_res in mf_fn_res_list]) for mf_fn_res_list in mf_fn_results])))
        if test_errors is not None and len(test_errors) > 0:
            avg_test_err = np.average(test_errors)
            std_test_err = np.std(test_errors)
            print("test_error_avg={}".format(avg_test_err))
            print("test_error_std={}".format(std_test_err))
            self.test_errors_avg.append(avg_test_err)
            self.test_errors_std.append(std_test_err)
