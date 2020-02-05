from mf_tree.eval_fns_torch import MFFunction, RBFKernelFn
from mf_tree.node_torch import MFNode
from mf_tree.tree_evaluator import DOOTreeEvaluator, ExtendedRunsDOOTreeEvaluator
from datasets.pandas_dataset import PandasData, PandasDataset
from datasets.torch_dataset import TorchData, TorchDataset
from datasets.data_loaders import DataLoaderFactory, MMDDataLoaderFactory
from mf_tree.optimization_strategies import SGDTorchOptimizationStrategy, SGDSklearnOptimizationStrategy
from mf_tree.optimization_budgets import ConstantBudgetFn, SqrtBudgetFn, LinearBudgetFn, HeightDependentBudgetFn, ConstUntilHeightBudgetFn
from mf_tree.step_schedules import SimpleTorchStepSchedule
from mf_tree.simplex_partitioning_strategies import DelaunayPartitioningStrategy, ConstantPartitioningStrategy, CoordinateHalvingPartitioningStrategy
from datasets.censor import DataCensorer

import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime as dt
import dill as pickle


class CommonSettings:
    def __init__(self,
                 validation_dataset,
                 test_dataset,
                 initial_simplex,
                 alpha_star,
                 test_mixture):
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset
        self.initial_simplex = initial_simplex
        self.alpha_star = alpha_star
        self.test_mixture = test_mixture


class ExperimentSettings:
    def __init__(self,
                 experiment_type,
                 tree_search_objective,
                 experiment_budgets,
                 return_best_deepest_node,
                 sample_with_replacement,
                 optimization_budget_type,
                 optimization_budgets,
                 optimization_budget_multiplier,
                 optimization_budget_height_cap,
                 loss_fn,
                 validation_fn,
                 test_fn,
                 starting_point,
                 use_tree_search_budgets,
                 n_repeats,
                 nu,
                 rho,
                 eta,
                 eta_decay_step,
                 eta_decay_mult,
                 batch_size,
                 plot_fmt,
                 mixture_selection_strategy,
                 custom_mixture,
                 evaluate_best_result_again,
                 evaluate_best_result_again_eta_mult,
                 actual_budgets,
                 actual_mixtures,
                 actual_best_sols,
                 record_test_error,
                 kernel,
                 model_mode):
        self.experiment_type=experiment_type
        self.tree_search_objective=tree_search_objective
        self.experiment_budgets=experiment_budgets
        self.return_best_deepest_node=return_best_deepest_node
        self.sample_with_replacement=sample_with_replacement
        self.optimization_budget_type=optimization_budget_type
        self.optimization_budgets=optimization_budgets
        self.optimization_budget_multiplier=optimization_budget_multiplier
        self.optimization_budget_height_cap=optimization_budget_height_cap
        self.loss_fn = loss_fn
        self.validation_fn = validation_fn
        self.test_fn = test_fn
        self.starting_point = starting_point
        self.use_tree_search_budgets=use_tree_search_budgets
        self.n_repeats=n_repeats
        self.nu=nu
        self.rho=rho
        self.eta=eta
        self.eta_decay_step=eta_decay_step
        self.eta_decay_mult=eta_decay_mult
        self.batch_size=batch_size
        self.plot_fmt = plot_fmt
        self.mixture_selection_strategy = mixture_selection_strategy
        self.custom_mixture = custom_mixture
        self.evaluate_best_result_again = evaluate_best_result_again
        self.evaluate_best_result_again_eta_mult = evaluate_best_result_again_eta_mult
        self.actual_budgets = actual_budgets
        self.actual_mixtures = actual_mixtures
        self.actual_best_sols = actual_best_sols
        self.record_test_error = record_test_error
        self.kernel = kernel
        self.model_mode = model_mode

    def __repr__(self):
        return "Experiment type:%s\n" \
               "Tree search objective:%s\n" \
               "Experiment budgets:%s\n" \
               "Return best deepest node:%s\n" \
               "Sample with replacement:%s\n" \
               "Optimization budget type:%s\n" \
               "Optimization budgets:%s\n" \
               "Optimization budget multiplier:%s\n" \
               "Optimization budget height cap:%s\n" \
               "Loss fn:%s\n" \
               "Validation fn:%s\n" \
               "Test fn:%s\n" \
               "Starting point model:%s\n" \
               "n_repeats:%s\n" \
               "nu:%s\n" \
               "rho:%s\n" \
               "eta:%s\n" \
               "eta_decay_step:%s\n" \
               "eta_decay_mult:%s\n" \
               "batch size:%s\n" \
               "plot_fmt:%s\n" \
               "mixture selection strategy:%s\n" \
               "custom mixture:%s\n" \
               "evaluate best result again:%s\n" \
               "actual budgets:%s\n" \
               "actual mixtures:%s\n" \
               "record test error:%s\n" \
               "kernel:%s\n" \
               "model mode:%s\n" % (self.experiment_type,
                                    self.tree_search_objective,
                                    self.experiment_budgets,
                                    self.return_best_deepest_node,
                                    self.sample_with_replacement,
                                    self.optimization_budget_type,
                                    self.optimization_budgets,
                                    self.optimization_budget_multiplier,
                                    self.optimization_budget_height_cap,
                                    self.loss_fn,
                                    self.validation_fn,
                                    self.test_fn,
                                    self.starting_point,
                                    self.n_repeats,
                                    self.nu,
                                    self.rho,
                                    self.eta,
                                    self.eta_decay_step,
                                    self.eta_decay_mult,
                                    self.batch_size,
                                    self.plot_fmt,
                                    self.mixture_selection_strategy,
                                    self.custom_mixture,
                                    self.evaluate_best_result_again,
                                    self.actual_budgets,
                                    self.actual_mixtures,
                                    self.record_test_error,
                                    self.kernel,
                                    self.model_mode)


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

    def append(self, vals, mf_fn_results, execution_times, best_sols, final_mixtures, l1_dists, total_costs, test_errors=None):
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
        if mf_fn_results is not None and len(mf_fn_results) > 0 and mf_fn_results[0].auc_roc_ovo is not None:
            print("auc_roc_ovo_avg={}".format(np.average([mf_fn_res.auc_roc_ovo for mf_fn_res in mf_fn_results])))
            print("auc_roc_ovo_std={}".format(np.std([mf_fn_res.auc_roc_ovo for mf_fn_res in mf_fn_results])))
            print("auc_roc_ovr_avg={}".format(np.average([mf_fn_res.auc_roc_ovr for mf_fn_res in mf_fn_results])))
            print("auc_roc_ovr_std={}".format(np.std([mf_fn_res.auc_roc_ovr for mf_fn_res in mf_fn_results])))
        if test_errors is not None and len(test_errors) > 0:
            avg_test_err = np.average(test_errors)
            std_test_err = np.std(test_errors)
            print("test_error_avg={}".format(avg_test_err))
            print("test_error_std={}".format(std_test_err))
            self.test_errors_avg.append(avg_test_err)
            self.test_errors_std.append(std_test_err)


class DefaultExperimentConfigurer:
    def __init__(self, data, cols_to_censor, record_test_error, train_on_validation):
        self.censorer = DataCensorer(cols_to_censor=cols_to_censor)
        self.data = self.censorer.censor(data)
        self.record_test_error = record_test_error
        print("Train size: %s\nValidate size: %s\nTest size: %s" % (len(self.data.train),
                                                                    len(self.data.validate),
                                                                    len(self.data.test)))
        print("Mixture order: %s"%self.data.vals_to_split)
        print("Train mixture: %s"%self.data.train_mixture)
        print("Validation mixture: %s"%self.data.validate_mixture)
        print("Test mixture order: %s"%self.data.vals_to_split)
        print("Test mixture: %s"%self.data.test_mixture)

        self.num_vals_for_product = self.data.get_num_labels()
        print("Num vals for product:", self.num_vals_for_product)

        if isinstance(self.data, PandasData):
            self.D_expensive = PandasDataset(self.data.validate,
                                             self.data.key_to_split_on,
                                             self.data.is_categorical)
            self.test_dataset = PandasDataset(self.data.test,
                                              self.data.key_to_split_on,
                                              self.data.is_categorical)
        elif isinstance(self.data, TorchData):
            self.D_expensive = TorchDataset(self.data.validate,
                                            self.data.key_to_split_on)
            self.test_dataset = TorchDataset(self.data.test,
                                            self.data.key_to_split_on)
        self.validate_mixture = self.data.validate_mixture
        print("validate mixture:", self.validate_mixture)
        self.test_mixture = self.data.test_mixture
        print("test mixture:", self.test_mixture)
        val_dim_eq_train_dim = len(self.validate_mixture) == len(self.data.train_mixture)
        print("Determined the validate mixture and train mixture have {} dim and that we {} training on validation set, so we can{} determine alpha star.".format(
            "the same" if val_dim_eq_train_dim else "different",
            "are" if train_on_validation else "aren't",
            "" if val_dim_eq_train_dim or train_on_validation else "not"))
        if val_dim_eq_train_dim or train_on_validation:
            if self.record_test_error:
                print("Determined that alpha star should be determined from test dataset")
                self.alpha_star = self.test_mixture
            else:
                print("Determined that alpha star should be determined from validate dataset")
                self.alpha_star = self.validate_mixture
        else:
            print("Setting alpha star to be uniform over train sets")
            self.alpha_star = np.ones_like(self.data.train_mixture) / len(self.data.train_mixture)

        print("alpha-star:", self.alpha_star)
        self.alpha_dim = len(self.alpha_star)
        print("Determined alpha dim to be:", self.alpha_dim)
        self.common_settings = None

        initial_simplex = np.identity(self.alpha_dim)

        self.sample_dim = self.D_expensive.dim

        self.common_settings = CommonSettings(validation_dataset=self.D_expensive,
                                              test_dataset=self.test_dataset,
                                              initial_simplex=initial_simplex,
                                              alpha_star=self.alpha_star,
                                              test_mixture=self.test_mixture)

    def configure(self,
                  experiment_settings: ExperimentSettings):
        # Set dataset factories
        if "uniform" == experiment_settings.experiment_type or\
                "constant-mixture" in experiment_settings.experiment_type or\
                "validation" in experiment_settings.experiment_type or\
                "tree" == experiment_settings.experiment_type:
            data_loader_factory = DataLoaderFactory(df_or_dataset=self.data.train if experiment_settings.experiment_type != "validation" else self.data.validate,
                                                    num_vals_for_product=self.num_vals_for_product,
                                                    key_to_split_on=self.data.key_to_split_on,
                                                    vals_to_split=self.data.vals_to_split,
                                                    with_replacement=experiment_settings.sample_with_replacement,
                                                    batch_size=experiment_settings.batch_size,
                                                    is_categorical=self.data.is_categorical)
        elif "mmd" == experiment_settings.experiment_type:
            kernel_gamma = 1
            rbf_kernel = RBFKernelFn(gamma=kernel_gamma)
            data_loader_factory = MMDDataLoaderFactory(df_or_dataset=self.data.train,
                                                       num_vals_for_product=self.num_vals_for_product,
                                                       validation_data=self.D_expensive,
                                                       kernel_fn=rbf_kernel,
                                                       key_to_split_on=self.data.key_to_split_on,
                                                       vals_to_split=self.data.vals_to_split,
                                                       with_replacement=experiment_settings.sample_with_replacement,
                                                       batch_size=experiment_settings.batch_size,
                                                       is_categorical=self.data.is_categorical)
        else:
            print("Invalid experiment type:", experiment_settings.experiment_type)
            assert False
        # Set budget functions
        if "constuntil" == experiment_settings.optimization_budget_type:
            opt_budget_class = ConstUntilHeightBudgetFn
        elif "linear" == experiment_settings.optimization_budget_type:
            opt_budget_class = LinearBudgetFn
        elif "sqrt" == experiment_settings.optimization_budget_type:
            opt_budget_class = SqrtBudgetFn
        elif "height" == experiment_settings.optimization_budget_type:
            opt_budget_class = HeightDependentBudgetFn
        elif "constant" == experiment_settings.optimization_budget_type:
            opt_budget_class = ConstantBudgetFn
        else:
            print("Invalid optimization budget type:", experiment_settings.optimization_budget_type)
            assert False

        alt_budgets_to_use = experiment_settings.actual_budgets


        # Set starting points and step sizes
        if experiment_settings.mixture_selection_strategy == "tree-results":
            alt_starting_point_nodes = experiment_settings.actual_best_sols
        else:
            alt_starting_point_nodes = None

        # Set partitioning strategies
        if experiment_settings.mixture_selection_strategy == "delaunay-partitioning":
            partitioning_strategy = DelaunayPartitioningStrategy(dim=self.alpha_dim)
        elif experiment_settings.mixture_selection_strategy == "coordinate-halving":
            partitioning_strategy = CoordinateHalvingPartitioningStrategy(dim=self.alpha_dim)
        elif experiment_settings.mixture_selection_strategy == "alpha-star":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                 simplex_point=self.alpha_star)
        elif experiment_settings.mixture_selection_strategy == "validation":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                 simplex_point=self.validate_mixture)
        elif experiment_settings.mixture_selection_strategy == "tree-results":
            partitioning_strategy = []
            for mixtures in experiment_settings.actual_mixtures:
                partitioning_strategy.append([ConstantPartitioningStrategy(dim=self.alpha_dim, simplex_point=mix) for mix in mixtures])
        elif experiment_settings.mixture_selection_strategy == "uniform":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim)
        elif experiment_settings.mixture_selection_strategy == "custom":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim, simplex_point=experiment_settings.custom_mixture)
        else:
            print("Invalid mixture selection strategy:", experiment_settings.mixture_selection_strategy)
            assert False

        # Set optimization strategy
        step_schedule = SimpleTorchStepSchedule(experiment_settings.loss_fn,
                                                c=0.5,
                                                nu=experiment_settings.nu,
                                                rho=experiment_settings.rho,
                                                lr=experiment_settings.eta)
        if experiment_settings.model_mode == "torch":
            opt_strategy = SGDTorchOptimizationStrategy(step_schedule=step_schedule,
                                                        decay_step=experiment_settings.eta_decay_step,
                                                        decay_mult=experiment_settings.eta_decay_mult)
        elif experiment_settings.model_mode == "sklearn":
            opt_strategy = SGDSklearnOptimizationStrategy(step_schedule=step_schedule)
        else:
            print("Invalid model mode {}".format(experiment_settings.model_mode))
            assert False

        return ExperimentRunner(data_loader_factory=data_loader_factory,
                                opt_budget_class=opt_budget_class,
                                partitioning_strategy=partitioning_strategy,
                                opt_strategy=opt_strategy,
                                common_settings=self.common_settings,
                                experiment_settings=experiment_settings,
                                alt_budgets_to_use=alt_budgets_to_use,
                                alt_starting_point_nodes=alt_starting_point_nodes,
                                validation_batch_size=experiment_settings.batch_size,
                                test_batch_size=experiment_settings.batch_size)


class ExperimentRunner:
    def __init__(self,
                 data_loader_factory,
                 opt_budget_class,
                 partitioning_strategy,
                 opt_strategy,
                 common_settings: CommonSettings,
                 experiment_settings: ExperimentSettings,
                 alt_budgets_to_use,
                 alt_starting_point_nodes,
                 validation_batch_size,
                 test_batch_size):
        self.data_loader_factory = data_loader_factory
        self.opt_budget_class = opt_budget_class
        self.partitioning_strategy = partitioning_strategy
        self.opt_strategy = opt_strategy
        self.common_settings = common_settings
        self.experiment_settings = experiment_settings
        self.alt_budgets_to_use = alt_budgets_to_use
        self.alt_starting_point_nodes = alt_starting_point_nodes
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        self.mf_fn = None
        self.root = None
        self.tree_eval = None
        self.best_sol = None

    def _run_once(self, exp_budget, opt_budget, partitioning_strategy, starting_point):
        self.mf_fn = MFFunction(validation_fn=self.experiment_settings.validation_fn,
                                test_fn=self.experiment_settings.test_fn,
                                loss_fn=self.experiment_settings.loss_fn,
                                data_loader_factory=self.data_loader_factory,
                                validation_dataset=self.common_settings.validation_dataset,
                                validation_batch_size=self.validation_batch_size,
                                test_dataset=self.common_settings.test_dataset,
                                test_batch_size=self.test_batch_size,
                                optimization_strategy=self.opt_strategy)
        self.root = MFNode(simplex_pts=self.common_settings.initial_simplex,
                           starting_point=starting_point,
                           mf_fn=self.mf_fn,
                           tree_search_objective=self.experiment_settings.tree_search_objective,
                           nu=self.experiment_settings.nu,
                           rho=self.experiment_settings.rho,
                           partitioning_strategy=partitioning_strategy,
                           opt_budget_fn=self.opt_budget_class(budget=opt_budget,
                                                               multiplier=self.experiment_settings.optimization_budget_multiplier,
                                                               height_cap=self.experiment_settings.optimization_budget_height_cap))

        # Use the same framework, but only evaluate the root
        if self.experiment_settings.evaluate_best_result_again:
            self.tree_eval = ExtendedRunsDOOTreeEvaluator(root=self.root,
                                                          budget=exp_budget,
                                                          return_best_deepest_node=self.experiment_settings.return_best_deepest_node,
                                                          eta_mult=self.experiment_settings.evaluate_best_result_again_eta_mult)
        else:
            self.tree_eval = DOOTreeEvaluator(root=self.root,
                                              budget=exp_budget,
                                              return_best_deepest_node=self.experiment_settings.return_best_deepest_node)
        self.best_sol, execution_time = self.tree_eval.evaluate(debug=True)
        print("Best solution:\n", self.best_sol, "\n")
        # if plot:
        #     self.best_sol.plot(self.common_settings.alpha_star)
        print(self.common_settings.alpha_star)
        print(self.best_sol.mixture)
        return self.best_sol, self.tree_eval.total_cost, execution_time

    def run(self, record_final_mixtures) -> RepeatedExperimentResults:
        repeated_experiment_res = RepeatedExperimentResults()
        for budget_id in range(len(self.experiment_settings.optimization_budgets)):
            experiment_budget = self.experiment_settings.experiment_budgets[budget_id]
            vals = []
            mf_fn_results = []
            execution_times = []
            final_mixtures = []
            l1_dists = []
            best_sols = []
            test_errors = []
            total_costs = []
            for rep in range(self.experiment_settings.n_repeats):
                if self.alt_budgets_to_use is not None:
                    optimization_budget = self.alt_budgets_to_use[budget_id][rep]
                else:
                    optimization_budget = self.experiment_settings.optimization_budgets[budget_id]
                if self.alt_starting_point_nodes is not None:
                    starting_point = self.alt_starting_point_nodes[budget_id][rep].final_model
                else:
                    starting_point = self.experiment_settings.starting_point
                print("Running for exp budget:", experiment_budget,
                      "and opt budget:", optimization_budget,
                      "at repeat:", rep)
                if type(self.partitioning_strategy) is list:
                    partitioning_strategy = self.partitioning_strategy[budget_id][rep]
                else:
                    partitioning_strategy = self.partitioning_strategy
                best_sol, total_cost, execution_time = self._run_once(opt_budget=optimization_budget,
                                                                      exp_budget=experiment_budget,
                                                                      partitioning_strategy=partitioning_strategy,
                                                                      starting_point=starting_point)
                if self.experiment_settings.record_test_error:
                    mf_fn_result = best_sol.get_test_error()
                else:
                    mf_fn_result = best_sol.validation_mf_fn_res
                val = mf_fn_result.error.item()
                print("best_sol_val_iter_{}={}".format(rep, val))
                vals.append(val)
                mf_fn_results.append(mf_fn_result)
                execution_times.append(execution_time.total_seconds())
                best_sols.append(best_sol)
                final_mixtures.append(best_sol.mixture)
                l1_dist = np.linalg.norm(np.array(best_sol.mixture) - np.array(self.common_settings.alpha_star), ord=1)
                l1_dists.append(l1_dist)
                print("l1_dist_iter_{}={}".format(rep,l1_dist))
                total_costs.append(total_cost)
                print("total_cost_iter_{}={}".format(rep, total_cost))
                # if self.record_test_error:
                #     test_error = best_sol.get_test_error()
                #     print("test_error={}".format(test_error))
                #     test_errors.append(test_error)
            repeated_experiment_res.append(vals=vals,
                                           mf_fn_results=mf_fn_results,
                                           execution_times=execution_times,
                                           best_sols=best_sols,
                                           final_mixtures=final_mixtures,
                                           l1_dists=l1_dists,
                                           total_costs=total_costs,
                                           test_errors=test_errors)
        return repeated_experiment_res


class ExperimentManager:
    def __init__(self, experiment_settings_list,
                 experiment_configurer: DefaultExperimentConfigurer,
                 output_dir,
                 output_filename):
        self.experiment_settings_list = experiment_settings_list
        self.experiment_configurer = experiment_configurer
        self.experiment_file_prefix = output_dir + "/" + output_filename
        self.output_filename = output_filename
        self.dump_file = self.experiment_file_prefix + ".p"
        self.plot_file = self.experiment_file_prefix + ".png"
        self.results = []

    def run(self):
        for exper_setting in self.experiment_settings_list:
            print("Experiment settings:")
            print(exper_setting)
            start = dt.now()
            print("Starting at:", start)
            if "tree" in exper_setting.experiment_type:
                runner = self.experiment_configurer.configure(exper_setting)
                repeated_experiment_results = runner.run(record_final_mixtures=True)
                with open(self.experiment_file_prefix + "_ACTUAL_MIXTURES_AND_BUDGETS.p", 'wb') as f:
                    pickle.dump({"mixtures": repeated_experiment_results.final_mixtures_all,
                                 "budgets": repeated_experiment_results.actual_costs_all,
                                 "best_sols": repeated_experiment_results.best_sols_all}, f)
            else:
                runner = self.experiment_configurer.configure(exper_setting)
                repeated_experiment_results = runner.run(record_final_mixtures=False)
            self.results.append(repeated_experiment_results)
            # with open(self.experiment_file_prefix + "PRE_" + exper_setting.experiment_type + ".p", 'wb') as f:
            #     pickle.dump({"result": res, "experiment_setting": exper_setting, "alpha_star": self.experiment_configurer.alpha_star}, f)
            end = dt.now()
            print("Ending at: %s with duration: %s" % (end, end - start))

    # def plot(self):
    #     for idx, res in enumerate(self.results):
    #         exper_setting = self.experiment_settings_list[idx]
    #         avg_costs = np.average(res.actual_costs_all, axis=1)
    #         std_costs = np.std(res.actual_costs_all, axis=1)
    #         plt.errorbar(avg_costs, res.vals_avg, xerr=std_costs, yerr=res.vals_std, fmt=exper_setting.plot_fmt, label=exper_setting.experiment_type)
    #     plt.xlabel("SGD Iteration budget")
    #     plt.ylabel("Validation error")
    #     plt.title("Size of validation dataset: %s, alpha star: %s" %
    #               (len(self.experiment_configurer.data.validate),
    #                self.experiment_configurer.alpha_star))
    #     plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #     plt.savefig(self.plot_file, bbox_inches='tight')
