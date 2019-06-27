import argparse, pickle, sys
from datetime import datetime as dt
import numpy as np
import torch.nn as nn
from mf_tree.eval_fns_torch import Net, MOEWNet, ExpMSE

from mf_tree.experiment_runner import DefaultExperimentConfigurer, ExperimentSettings, ExperimentManager


def process(args):
    with open(args.dataset_path, 'rb') as f:
        data = pickle.load(f)

    if args.actual_budgets_and_mixtures_path:
        print("Using actual budgets and mixtures path:", args.actual_budgets_and_mixtures_path)
        with open(args.actual_budgets_and_mixtures_path, 'rb') as bm:
            budget_mixture_map = pickle.load(bm)
            actual_budgets = budget_mixture_map["budgets"]
            actual_mixtures = budget_mixture_map["mixtures"]
    else:
        actual_budgets = None
        actual_mixtures = None

    cols_to_censor = args.columns_to_censor.split(',') if args.columns_to_censor is not None else []
    experiment_config = DefaultExperimentConfigurer(data=data,
                                                    cols_to_censor=cols_to_censor,
                                                    record_test_error=args.record_test_error)
    budgets = range(args.budget_min, args.budget_max, args.budget_step)
    if args.experiment_type == "tree":
        experiment_budgets = budgets
        optimization_budgets = 1 * np.ones_like(budgets)
    else:
        experiment_budgets = np.zeros_like(budgets)
        optimization_budgets = budgets

    if args.dataset_id in ["allstate", "MNIST"]:
        loss_fn = nn.CrossEntropyLoss()
        validation_fn = nn.CrossEntropyLoss()
        test_fn = nn.CrossEntropyLoss()
        model = Net(input_dim=experiment_config.sample_dim,
                    inner_dim_mult=args.inner_layer_mult,
                    output_dim=experiment_config.num_vals_for_product)
    elif args.dataset_id in ["wine"]:
        loss_fn = ExpMSE()
        validation_fn= ExpMSE()
        test_fn= ExpMSE()
        model = MOEWNet(experiment_config.sample_dim)
    else:
        print("Unsupported loss function/model for dataset id", args.dataset_id,"Cannot continue")
        assert False

    curr_time = dt.now().strftime('day-%Y-%m-%d-time-%H-%M-%S')
    print("Determined current time to be:", curr_time)
    output_filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.experiment_type,
                                                                                        args.optimization_budget,
                                                                                        args.optimization_budget_multiplier,
                                                                                        args.optimization_budget_height_cap,
                                                                                        args.inner_layer_mult,
                                                                                        args.budget_min,
                                                                                        args.budget_max,
                                                                                        args.budget_step,
                                                                                        args.num_repeats,
                                                                                        args.batch_size,
                                                                                        args.nu,
                                                                                        args.rho,
                                                                                        args.eta,
                                                                                        args.return_best_deepest_node,
                                                                                        args.mixture_selection_strategy,
                                                                                        args.record_test_error,
                                                                                        curr_time,
                                                                                        args.tag)
    print("Using output filename:", output_filename)

    exper_setting = ExperimentSettings(experiment_type=args.experiment_type,
                                       experiment_budgets=experiment_budgets,
                                       return_best_deepest_node=args.return_best_deepest_node,
                                       sample_with_replacement=args.sample_with_replacement,
                                       optimization_budget_type=args.optimization_budget,
                                       optimization_budgets=optimization_budgets,
                                       optimization_budget_multiplier=args.optimization_budget_multiplier,
                                       optimization_budget_height_cap=args.optimization_budget_height_cap,
                                       loss_fn=loss_fn,
                                       validation_fn=validation_fn,
                                       test_fn=test_fn,
                                       starting_point=model,
                                       n_repeats=args.num_repeats,
                                       use_tree_search_budgets=False,
                                       nu=args.nu,
                                       rho=args.rho,
                                       eta=args.eta,
                                       batch_size=args.batch_size,
                                       plot_fmt='rs-',
                                       mixture_selection_strategy=args.mixture_selection_strategy,
                                       actual_budgets=actual_budgets,
                                       actual_mixtures=actual_mixtures,
                                       record_test_error=args.record_test_error)

    exper_manager = ExperimentManager(experiment_settings_list=[exper_setting],
                                      experiment_configurer=experiment_config,
                                      output_dir=args.output_dir,
                                      output_filename=output_filename)

    exper_manager.run()
    print("dumping experiment results to", exper_manager.dump_file)
    filename = exper_manager.dump_file
    with open(filename, "wb") as f:
        pickle.dump(exper_manager, f)
    exper_manager.plot()


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to pickled _Data object to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate','MNIST','wine'], help="the dataset identifier to process")
    parser.add_argument("--actual-budgets-and-mixtures-path", type=str, default=None, help="The path to the pickled list of mixtures to use.  If used, the budgets override the budget min/max/step, and the mixtures in this file are used if the mixture-selection-strategy is tree-results")
    parser.add_argument("--experiment-type", type=str, required=True, choices=['tree', 'uniform', 'constant-mixture', 'mmd'], help="The type of experiment to run")
    parser.add_argument("--optimization-budget", type=str, required=True, choices=['constuntil', 'linear', 'height', 'constant'], help="The budget function to use at each node")
    parser.add_argument("--optimization-budget-multiplier", type=int, default=1, help="The constant to multiply each optimization budget by")
    parser.add_argument("--optimization-budget-height-cap", type=float, default=np.inf, help="The max height for which opt budget is height-dependent")
    parser.add_argument("--budget-min", type=int, required=True, help="The minimum budget to use")
    parser.add_argument("--budget-max", type=int, required=True, help="The maximum budget to use")
    parser.add_argument("--budget-step", type=int, required=True, help="The interval length between budgets")
    parser.add_argument("--num-repeats", type=int, required=True, help="The number of times to repeat each experiment")
    parser.add_argument("--batch-size", type=int, required=True, help="The batch size to use for computing stochastic gradients")
    parser.add_argument("--nu", type=float, required=True, help="Nu")
    parser.add_argument("--rho", type=float, required=True, help="Rho")
    parser.add_argument("--eta", type=float, required=True, help="The step size")
    parser.add_argument("--return-best-deepest-node", type=bool, required=True, help="Indicates whether best nodes only at the deepest height (True) or any height (False) should be considered")
    parser.add_argument("--sample-with-replacement", type=bool, default=True, help="Indicates whether best nodes only at the deepest height (True) or any height (False) should be considered")
    parser.add_argument("--mixture-selection-strategy", type=str, required=True, choices=["delaunay-partitioning", "coordinate-halving", "tree-results", "alpha-star", "uniform"], help="The mixture selection strategy to use.")
    parser.add_argument("--columns-to-censor", type=str, default=None, help="The columns to remove from the dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--tag", type=str, default="missingtag", help="The image tag used in running this experiment")
    parser.add_argument("--record-test-error", type=bool, default=False, help="Determines whether or not test error should be recorded. This toggles whether the validation or test dataset is used.")
    parser.add_argument("--inner-layer-mult", type=float, default=2., help="Determines the number of inner layers of the neural network (unless the dataset id is wine, in which case the network is not configurable).  Num inner layers will be int(inner_layer_mult * input_dim)")
    # parser.add_argument("--output-filename", type=str, required=True, help="The output filename")

    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path", "experiment_running/dataset.p",
    #     "--dataset-id", "allstate",
    #     "--experiment-type", "tree",
    #     "--optimization-budget", "constant",
    #     "--optimization-budget-multiplier", "100",
    #     "--budget-min", "1000",
    #     "--budget-max", "3000",
    #     "--budget-step", "1000",
    #     "--num-repeats", "2",
    #     "--batch-size", "50",
    #     "--nu", "80",
    #     "--rho", "0.9",
    #     "--eta", "0.0001",
    #     "--return-best-deepest-node", "True",
    #     "--output-dir", "derp",
    #     "--mixture-selection-strategy", "coordinate-halving",
    #     "--columns-to-censor", "",
    #     "--optimization-budget-height-cap", "inf"
    # ])
    main()
