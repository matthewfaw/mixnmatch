from mf_tree.eval_fns_torch import MFFunction, RBFKernelFn
from mf_tree.node_torch import MFNode
from mf_tree.tree_evaluator import DOOTreeEvaluator
from datasets.pandas_dataset import PandasData, PandasDataset
from datasets.torch_dataset import TorchData, TorchDataset
from datasets.data_loaders import DataLoaderFactory, MMDDataLoaderFactory
from mf_tree.optimization_strategies import SGDTorchOptimizationStrategy
from mf_tree.optimization_budgets import ConstantBudgetFn, LinearBudgetFn, HeightDependentBudgetFn, ConstUntilHeightBudgetFn
from mf_tree.step_schedules import SimpleTorchStepSchedule
from mf_tree.simplex_partitioning_strategies import DelaunayPartitioningStrategy, ConstantPartitioningStrategy, CoordinateHalvingPartitioningStrategy
from datasets.censor import DataCensorer

import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime as dt
import pickle


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
                 batch_size,
                 plot_fmt,
                 mixture_selection_strategy,
                 actual_budgets,
                 actual_mixtures,
                 record_test_error):
        self.experiment_type=experiment_type
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
        self.batch_size=batch_size
        self.plot_fmt = plot_fmt
        self.mixture_selection_strategy = mixture_selection_strategy
        self.actual_budgets = actual_budgets
        self.actual_mixtures = actual_mixtures
        self.record_test_error = record_test_error

    def __repr__(self):
        return "Experiment type:%s\n" \
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
               "batch size:%s\n" \
               "plot_fmt:%s\n" \
               "mixture selection strategy:%s\n" \
               "actual budgets:%s\n" \
               "actual mixtures:%s\n" \
               "record test error:%s\n" % (self.experiment_type,
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
                                           self.batch_size,
                                           self.plot_fmt,
                                           self.mixture_selection_strategy,
                                           self.actual_budgets,
                                           self.actual_mixtures,
                                           self.record_test_error)


class RepeatedExperimentResults:
    def __init__(self):
        self.vals_all = []
        self.vals_avg = []
        self.vals_std = []
        self.execution_times_avg = []
        self.execution_times_std = []
        self.actual_costs_all = []
        self.best_sols_all = []
        self.final_mixtures_all = []
        self.l1_dists_all = []
        self.test_errors_avg = []
        self.test_errors_std = []

    def append(self, vals, execution_times, best_sols, final_mixtures, l1_dists, total_costs, test_errors=None):
        self.vals_all.append(vals)
        val_avg = np.average(vals)
        self.vals_avg.append(val_avg)
        print("best_sol_val_avg={}".format(val_avg))
        val_std = np.std(vals)
        print("best_sol_val_std={}".format(val_std))
        self.vals_std.append(val_std)
        avg_execution_time = np.average(execution_times)
        self.execution_times_avg.append(avg_execution_time)
        print("execution_time_avg={}".format(avg_execution_time))
        self.execution_times_std.append(np.std(execution_times))
        actual_cost_avg = np.average(total_costs)
        actual_cost_std = np.std(total_costs)
        print("actual_cost_avg={}".format(actual_cost_avg))
        print("actual_cost_std={}".format(actual_cost_avg))
        self.actual_costs_all.append(total_costs)
        self.best_sols_all.append(best_sols)
        self.final_mixtures_all.append(final_mixtures)
        self.l1_dists_all.append(l1_dists)
        print("l1_dist_avg={}".format(np.average(l1_dists)))
        print("l1_dist_std={}".format(np.std(l1_dists)))
        if test_errors is not None and len(test_errors) > 0:
            avg_test_err = np.average(test_errors)
            std_test_err = np.std(test_errors)
            print("test_error_avg={}".format(avg_test_err))
            print("test_error_std={}".format(std_test_err))
            self.test_errors_avg.append(avg_test_err)
            self.test_errors_std.append(std_test_err)


class DefaultExperimentConfigurer:
    def __init__(self, data, cols_to_censor, record_test_error):
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
        if len(self.validate_mixture) == len(self.data.train_mixture):
            print("Determined that the validate mixture and train mixture have the same dim, so we can determine alpha star.")
            if self.record_test_error:
                print("Determined that alpha star should be determined from test dataset")
                self.alpha_star = self.test_mixture
            else:
                print("Determined that alpha star should be determined from validate dataset")
                self.alpha_star = self.validate_mixture
        else:
            print("The dimensions of the train and validate mixtures were determined to be different. Thus, cannot determine alpha star. Setting alpha star to uniform over training set:")
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
                "tree" == experiment_settings.experiment_type:
            data_loader_factory = DataLoaderFactory(df_or_dataset=self.data.train,
                                                    key_to_split_on=self.data.key_to_split_on,
                                                    vals_to_split=self.data.vals_to_split,
                                                    with_replacement=experiment_settings.sample_with_replacement,
                                                    batch_size=experiment_settings.batch_size,
                                                    is_categorical=self.data.is_categorical)
        elif "mmd" == experiment_settings.experiment_type:
            kernel_gamma = 1
            rbf_kernel = RBFKernelFn(gamma=kernel_gamma)
            data_loader_factory = MMDDataLoaderFactory(df_or_dataset=self.data.train,
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
        elif "height" == experiment_settings.optimization_budget_type:
            opt_budget_class = HeightDependentBudgetFn
        elif "constant" == experiment_settings.optimization_budget_type:
            opt_budget_class = ConstantBudgetFn
        else:
            print("Invalid optimization budget type:", experiment_settings.optimization_budget_type)
            assert False

        alt_budgets_to_use = experiment_settings.actual_budgets


        # Set partitioning strategies
        if experiment_settings.mixture_selection_strategy == "delaunay-partitioning":
            partitioning_strategy = DelaunayPartitioningStrategy(dim=self.alpha_dim)
        elif experiment_settings.mixture_selection_strategy == "coordinate-halving":
            partitioning_strategy = CoordinateHalvingPartitioningStrategy(dim=self.alpha_dim)
        elif experiment_settings.mixture_selection_strategy == "alpha-star":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                 simplex_point=self.alpha_star)
        elif experiment_settings.mixture_selection_strategy == "tree-results":
            partitioning_strategy = []
            for mixtures in experiment_settings.actual_mixtures:
                partitioning_strategy.append([ConstantPartitioningStrategy(dim=self.alpha_dim, simplex_point=mix) for mix in mixtures])
        elif experiment_settings.mixture_selection_strategy == "uniform":
            partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim)
        else:
            print("Invalid mixture selection strategy:", experiment_settings.mixture_selection_strategy)
            assert False

        # Set optimization strategy
        step_schedule = SimpleTorchStepSchedule(experiment_settings.loss_fn,
                                                c=0.5,
                                                nu=experiment_settings.nu,
                                                rho=experiment_settings.rho,
                                                lr=experiment_settings.eta)
        opt_strategy = SGDTorchOptimizationStrategy(step_schedule=step_schedule)

        return ExperimentRunner(data_loader_factory=data_loader_factory,
                                opt_budget_class=opt_budget_class,
                                partitioning_strategy=partitioning_strategy,
                                opt_strategy=opt_strategy,
                                common_settings=self.common_settings,
                                experiment_settings=experiment_settings,
                                alt_budgets_to_use=alt_budgets_to_use,
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
                 validation_batch_size,
                 test_batch_size):
        self.data_loader_factory = data_loader_factory
        self.opt_budget_class = opt_budget_class
        self.partitioning_strategy = partitioning_strategy
        self.opt_strategy = opt_strategy
        self.common_settings = common_settings
        self.experiment_settings = experiment_settings
        self.alt_budgets_to_use = alt_budgets_to_use
        self.validation_batch_size = validation_batch_size
        self.test_batch_size = test_batch_size

        self.mf_fn = None
        self.root = None
        self.tree_eval = None
        self.best_sol = None

    def _run_once(self, exp_budget, opt_budget, partitioning_strategy):
        self.mf_fn = MFFunction(validation_fn=self.experiment_settings.validation_fn,
                                test_fn=self.experiment_settings.test_fn,
                                loss_fn=self.experiment_settings.loss_fn,
                                data_loader_factory=self.data_loader_factory,
                                validation_dataset=self.common_settings.validation_dataset,
                                validation_batch_size=self.validation_batch_size,
                                test_dataset=self.common_settings.test_dataset,
                                test_batch_size=self.test_batch_size,
                                optimization_strategy=self.opt_strategy,
                                use_test_error=self.experiment_settings.record_test_error)
        self.root = MFNode(simplex_pts=self.common_settings.initial_simplex,
                           starting_point=self.experiment_settings.starting_point,
                           mf_fn=self.mf_fn,
                           nu=self.experiment_settings.nu,
                           rho=self.experiment_settings.rho,
                           partitioning_strategy=partitioning_strategy,
                           opt_budget_fn=self.opt_budget_class(budget=opt_budget,
                                                               multiplier=self.experiment_settings.optimization_budget_multiplier,
                                                               height_cap=self.experiment_settings.optimization_budget_height_cap))

        # Use the same framework, but only evaluate the root
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
                print("Running for exp budget:", experiment_budget,
                      "and opt budget:", optimization_budget,
                      "at repeat:", rep)
                if type(self.partitioning_strategy) is list:
                    partitioning_strategy = self.partitioning_strategy[budget_id][rep]
                else:
                    partitioning_strategy = self.partitioning_strategy
                best_sol, total_cost, execution_time = self._run_once(opt_budget=optimization_budget,
                                                                      exp_budget=experiment_budget,
                                                                      partitioning_strategy=partitioning_strategy)
                val = best_sol.value.item()
                print("best_sol_val_iter_{}={}".format(rep, val))
                vals.append(val)
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
                                 "budgets": repeated_experiment_results.actual_costs_all}, f)
            else:
                runner = self.experiment_configurer.configure(exper_setting)
                repeated_experiment_results = runner.run(record_final_mixtures=False)
            self.results.append(repeated_experiment_results)
            # with open(self.experiment_file_prefix + "PRE_" + exper_setting.experiment_type + ".p", 'wb') as f:
            #     pickle.dump({"result": res, "experiment_setting": exper_setting, "alpha_star": self.experiment_configurer.alpha_star}, f)
            end = dt.now()
            print("Ending at: %s with duration: %s" % (end, end - start))

    def plot(self):
        for idx, res in enumerate(self.results):
            exper_setting = self.experiment_settings_list[idx]
            avg_costs = np.average(res.actual_costs_all, axis=1)
            std_costs = np.std(res.actual_costs_all, axis=1)
            plt.errorbar(avg_costs, res.vals_avg, xerr=std_costs, yerr=res.vals_std, fmt=exper_setting.plot_fmt, label=exper_setting.experiment_type)
        plt.xlabel("SGD Iteration budget")
        plt.ylabel("Validation error")
        plt.title("Size of validation dataset: %s, alpha star: %s" %
                  (len(self.experiment_configurer.data.validate),
                   self.experiment_configurer.alpha_star))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(self.plot_file, bbox_inches='tight')
