from datetime import datetime as dt

from configuration.dataset_configuration import DatasetConfiguration
from configuration.experiment_configuration import ExperimentConfiguration
from configuration.mmd_configuration import MMDConfiguration
from configuration.model_configuration import ModelConfiguration
from configuration.optimization_configuration import OptimizationConfiguration
from configuration.tree_search_configuration import TreeSearchConfiguration


def setup_output_file(dataset_config: DatasetConfiguration,
                      experiment_config: ExperimentConfiguration,
                      model_config: ModelConfiguration,
                      optimization_config: OptimizationConfiguration,
                      mmd_config: MMDConfiguration,
                      tree_search_config: TreeSearchConfiguration,
                      tag):
    curr_time = dt.now().strftime('day-%Y-%m-%d-time-%H-%M-%S')
    print("Determined current time to be:", curr_time)
    custom_mixture_str = ",".join([str(mixture_val) for mixture_val in experiment_config.custom_mixture]) if experiment_config.custom_mixture is not None else ""
    output_filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        experiment_config.experiment_type,
        optimization_config.optimization_budget,
        optimization_config.optimization_budget_multiplier,
        optimization_config.optimization_budget_height_cap,
        experiment_config.budget_min,
        experiment_config.budget_max,
        experiment_config.budget_step,
        experiment_config.num_repeats,
        optimization_config.batch_size,
        tree_search_config.nu,
        tree_search_config.rho,
        optimization_config.eta,
        optimization_config.eta_decay_step,
        optimization_config.eta_decay_mult,
        tree_search_config.return_best_deepest_node,
        experiment_config.mixture_selection_strategy,
        custom_mixture_str,
        tree_search_config.evaluate_best_result_again,
        experiment_config.record_test_error,
        curr_time,
        tag)
    print("Using output filename:", output_filename)

    return output_filename
