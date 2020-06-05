from configuration.dataset_configuration import DatasetConfiguration
from configuration.experiment_configuration import ExperimentConfiguration
from configuration.mmd_configuration import MMDConfiguration
from configuration.model_configuration import ModelConfiguration
from configuration.optimization_configuration import OptimizationConfiguration
from configuration.tree_search_configuration import TreeSearchConfiguration
from mf_tree.eval_fns_torch import MFFunction, RBFKernelFn
from mf_tree.experiment_result_recorder import ExperimentResultRecorder, RepeatedExperimentResults
from mf_tree.node_torch import MFNode
from mf_tree.tree_evaluator import DOOTreeEvaluator, ExtendedRunsDOOTreeEvaluator

import numpy as np


class ExperimentRunner:
    def __init__(self,
                 dataset_config: DatasetConfiguration,
                 experiment_config: ExperimentConfiguration,
                 model_config: ModelConfiguration,
                 optimization_config: OptimizationConfiguration,
                 mmd_config: MMDConfiguration,
                 tree_search_config: TreeSearchConfiguration,
                 output_dir,
                 output_file):
        self.dataset_config = dataset_config
        self.experiment_config = experiment_config
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.mmd_config = mmd_config
        self.tree_search_config = tree_search_config
        self.output_dir = output_dir
        self.output_file = output_file

        self.mf_fn = None
        self.root = None
        self.tree_eval = None
        self.best_sol = None

    def _run_once(self, exp_budget, opt_budget, partitioning_strategy, starting_point):
        self.mf_fn = MFFunction(validation_fn=self.model_config.validation_fn,
                                test_fn=self.model_config.test_fn,
                                loss_fn=self.model_config.loss_fn,
                                data_loader_factory=self.dataset_config.data_loader_factory,
                                individual_data_loader_factory=self.dataset_config.individual_data_loader_factory,
                                num_training_datasources=self.experiment_config.alpha_dim,
                                tree_search_validation_datasource=self.tree_search_config.tree_search_validation_datasource,
                                individual_source_baselines=self.tree_search_config.individual_source_baselines,
                                validation_dataset=self.dataset_config.D_expensive,
                                validation_batch_size=self.optimization_config.batch_size,
                                test_dataset=self.dataset_config.test_dataset,
                                test_batch_size=self.optimization_config.batch_size,
                                optimization_strategy=self.optimization_config.opt_strategy)
        self.root = MFNode(simplex_pts=self.experiment_config.initial_simplex,
                           starting_point=starting_point,
                           mf_fn=self.mf_fn,
                           tree_search_objective=self.tree_search_config.tree_search_objective,
                           tree_search_objective_operation=self.tree_search_config.tree_search_objective_operation,
                           tree_search_validation_datasource=self.tree_search_config.tree_search_validation_datasource,
                           nu=self.tree_search_config.nu,
                           rho=self.tree_search_config.rho,
                           partitioning_strategy=partitioning_strategy,
                           opt_budget_fn=self.optimization_config.opt_budget_class(budget=opt_budget,
                                                                                   multiplier=self.optimization_config.optimization_budget_multiplier,
                                                                                   height_cap=self.optimization_config.optimization_budget_height_cap))

        # Use the same framework, but only evaluate the root
        recorder = ExperimentResultRecorder(experiment_config=self.experiment_config)
        if self.tree_search_config.evaluate_best_result_again:
            self.tree_eval = ExtendedRunsDOOTreeEvaluator(root=self.root,
                                                          budget=exp_budget,
                                                          recorder=recorder,
                                                          return_best_deepest_node=self.tree_search_config.return_best_deepest_node,
                                                          eta_mult=self.tree_search_config.evaluate_best_result_again_eta_mult)
        else:
            self.tree_eval = DOOTreeEvaluator(root=self.root,
                                              budget=exp_budget,
                                              recorder=recorder,
                                              return_best_deepest_node=self.tree_search_config.return_best_deepest_node)
        self.best_sol, execution_time = self.tree_eval.evaluate(debug=True)
        if self.tree_eval.total_cost > recorder.recording_times[-1]:
            recorder.record(model=self.best_sol.final_model,
                            node=self.best_sol,
                            prev_cost=self.tree_eval.total_cost,
                            curr_cost=self.tree_eval.total_cost,
                            force_record=True)
        print("Best solution:\n", self.best_sol, "\n")
        print(self.experiment_config.alpha_star)
        print(self.best_sol.mixture)
        return self.best_sol, self.tree_eval.total_cost, execution_time, recorder

    def run(self) -> RepeatedExperimentResults:
        repeated_experiment_res = RepeatedExperimentResults()
        experiment_budget = self.experiment_config.budget_max

        vals = []
        mf_fn_results = []
        execution_times = []
        final_mixtures = []
        l1_dists = []
        best_sols = []
        test_errors = []
        total_costs = []
        recorders = []

        for rep in range(self.experiment_config.num_repeats):
            if self.experiment_config.alt_budgets_to_use is not None:
                optimization_budget = self.experiment_config.alt_budgets_to_use[rep]
            else:
                optimization_budget = self.optimization_config.optimization_budget
            if self.tree_search_config.actual_best_sols is not None:
                starting_point = self.tree_search_config.actual_best_sols[rep].final_model
            else:
                starting_point = self.model_config.model
            print("Running for exp budget:", experiment_budget,
                  "and opt budget:", optimization_budget,
                  "at repeat:", rep)
            if type(self.experiment_config.partitioning_strategy) is list:
                partitioning_strategy = self.experiment_config.partitioning_strategy[rep]
            else:
                partitioning_strategy = self.experiment_config.partitioning_strategy
            best_sol, total_cost, execution_time, recorder = self._run_once(opt_budget=optimization_budget,
                                                                            exp_budget=experiment_budget,
                                                                            partitioning_strategy=partitioning_strategy,
                                                                            starting_point=starting_point)
            if self.experiment_config.record_test_error:
                mf_fn_result = best_sol.get_test_error()
            else:
                mf_fn_result = best_sol.validation_mf_fn_results
            # val = mf_fn_result.error.item()
            val = max([mf_fn_res.error.item() for mf_fn_res in mf_fn_result])
            print("best_sol_val_iter_{}={}".format(rep, val))
            vals.append(val)
            mf_fn_results.append(mf_fn_result)
            execution_times.append(execution_time.total_seconds())
            best_sols.append(best_sol)
            final_mixtures.append(best_sol.mixture)
            l1_dist = np.linalg.norm(np.array(best_sol.mixture) - np.array(self.experiment_config.alpha_star), ord=1)
            l1_dists.append(l1_dist)
            print("l1_dist_iter_{}={}".format(rep,l1_dist))
            total_costs.append(total_cost)
            print("total_cost_iter_{}={}".format(rep, total_cost))
            recorders.append(recorder)
        repeated_experiment_res.append(vals=vals,
                                       mf_fn_results=mf_fn_results,
                                       execution_times=execution_times,
                                       best_sols=best_sols,
                                       final_mixtures=final_mixtures,
                                       l1_dists=l1_dists,
                                       total_costs=total_costs,
                                       recorders=recorders,
                                       test_errors=test_errors)
        return repeated_experiment_res
