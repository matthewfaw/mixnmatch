import argparse, sys
import dill as pickle
from datetime import datetime as dt
import numpy as np
import torch.nn as nn
from mf_tree.eval_fns_torch import Net, MOEWNet, ExpMSE, ExpL1
from mf_tree.eval_fns_torch import TorchLikeSGDClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import hinge_loss, log_loss

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
            actual_best_sols = budget_mixture_map["best_sols"]
    else:
        actual_budgets = None
        actual_mixtures = None
        actual_best_sols = None

    cols_to_censor = args.columns_to_censor.split(',') if args.columns_to_censor is not None else []
    experiment_config = DefaultExperimentConfigurer(data=data,
                                                    cols_to_censor=cols_to_censor,
                                                    record_test_error=args.record_test_error,
                                                    train_on_validation=args.mixture_selection_strategy == "validation")
    budgets = range(args.budget_min, args.budget_max, args.budget_step)
    if args.experiment_type == "tree":
        experiment_budgets = budgets
        optimization_budgets = 1 * np.ones_like(budgets)
    else:
        experiment_budgets = np.zeros_like(budgets)
        optimization_budgets = budgets

    if args.sklearn_kernel == "rbf":
        ker = RBFSampler(gamma=args.sklearn_kernel_gamma, n_components=args.sklearn_kernel_ncomponents)
        kernel = lambda x: ker.fit_transform(x)
    elif args.sklearn_kernel == "":
        kernel = lambda x: x
    else:
        print("Invalid kernel {}".format(args.sklearn_kernel))
        assert False

    if args.dataset_id in ["allstate"] or data.is_categorical:
        if args.model_mode == "torch":
            loss_fn = nn.CrossEntropyLoss()
            val_f = nn.CrossEntropyLoss()
            model = Net(input_dim=experiment_config.sample_dim,
                        inner_dim_mult=args.inner_layer_mult,
                        inner_layer_size=args.inner_layer_size,
                        num_hidden_layers=args.num_hidden_layers,
                        output_dim=experiment_config.num_vals_for_product)
        elif args.model_mode == "sklearn":
            loss_fn = None # Not needed
            if args.sklearn_loss == "hinge":
                validation_fn = lambda mod, _, x, y: hinge_loss(y, mod(x), labels=range(data.get_num_labels()))
                test_fn = validation_fn
            elif args.sklearn_loss == "log":
                validation_fn = lambda mod, _, x, y: log_loss(y, mod(x),
                                                              labels=range(data.get_num_labels()))
                test_fn = validation_fn
            else:
                print("Unsupported sklearn loss {}. Cannot continue.".format(args.sklearn_loss))
                assert False
            model = TorchLikeSGDClassifier(loss=args.sklearn_loss,
                                           penalty=args.sklearn_loss_penalty,
                                           warm_start=True,
                                           eta0=args.eta,
                                           alpha=args.sklearn_learning_rate_alpha,
                                           learning_rate=args.sklearn_learning_rate,
                                           kernel=kernel,
                                           num_classes=data.get_num_labels())
        else:
            print("Invalid model_mode {}. Cannot continue".format(args.model_mode))
            assert False
    elif args.dataset_id in ["wine"]:
        if args.use_alt_loss_fn:
            loss_fn = ExpL1()
            val_f= ExpL1()
        else:
            loss_fn = ExpMSE()
            val_f = ExpMSE()
        model = MOEWNet(experiment_config.sample_dim)
    else:
        print("Unsupported loss function/model for dataset id", args.dataset_id,"Cannot continue")
        assert False
    if args.model_mode == "torch":
        validation_fn = lambda _, preds, __, y: val_f(preds, y)
        test_fn = validation_fn

    if args.custom_mixture and args.mixture_selection_strategy == "custom":
        custom_mixture = [float(el) for el in args.custom_mixture.split(',')]
        print("Using custom mixture:", custom_mixture)
    else:
        custom_mixture = None

    curr_time = dt.now().strftime('day-%Y-%m-%d-time-%H-%M-%S')
    print("Determined current time to be:", curr_time)
    output_filename = "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}".format(args.experiment_type,
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
                                                                                                 args.eta_decay_step,
                                                                                                 args.eta_decay_mult,
                                                                                                 args.return_best_deepest_node,
                                                                                                 args.mixture_selection_strategy,
                                                                                                 args.custom_mixture,
                                                                                                 args.evaluate_best_result_again,
                                                                                                 args.record_test_error,
                                                                                                 curr_time,
                                                                                                 args.tag)
    print("Using output filename:", output_filename)

    exper_setting = ExperimentSettings(experiment_type=args.experiment_type,
                                       tree_search_objective=args.tree_search_objective,
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
                                       eta_decay_step=args.eta_decay_step,
                                       eta_decay_mult=args.eta_decay_mult,
                                       batch_size=args.batch_size,
                                       plot_fmt='rs-',
                                       mixture_selection_strategy=args.mixture_selection_strategy,
                                       custom_mixture=custom_mixture,
                                       evaluate_best_result_again=args.evaluate_best_result_again,
                                       evaluate_best_result_again_eta_mult=args.evaluate_best_result_again_eta_mult,
                                       actual_budgets=actual_budgets,
                                       actual_mixtures=actual_mixtures,
                                       actual_best_sols=actual_best_sols,
                                       record_test_error=args.record_test_error,
                                       kernel=kernel,
                                       model_mode=args.model_mode)

    exper_manager = ExperimentManager(experiment_settings_list=[exper_setting],
                                      experiment_configurer=experiment_config,
                                      output_dir=args.output_dir,
                                      output_filename=output_filename)

    exper_manager.run()
    print("dumping experiment results to", exper_manager.dump_file)
    filename = exper_manager.dump_file
    with open(filename, "wb") as f:
        pickle.dump(exper_manager, f)
    # exper_manager.plot()


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to pickled _Data object to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate','wine','amazon'], help="the dataset identifier to process")
    parser.add_argument("--actual-budgets-and-mixtures-path", type=str, default=None, help="The path to the pickled list of mixtures to use.  If used, the budgets override the budget min/max/step, and the mixtures in this file are used if the mixture-selection-strategy is tree-results")
    parser.add_argument("--experiment-type", type=str, required=True, choices=['tree', 'uniform', 'constant-mixture', 'mmd', 'validation'], help="The type of experiment to run")
    parser.add_argument("--tree-search-objective", type=str, required=False, default="error", choices=['error', 'auc_roc_ovo', 'min_precision', 'min_recall'], help="The tree search node value")
    parser.add_argument("--optimization-budget", type=str, required=True, choices=['constuntil', 'linear', 'sqrt', 'height', 'constant'], help="The budget function to use at each node")
    parser.add_argument("--optimization-budget-multiplier", type=int, default=1, help="The constant to multiply each optimization budget by")
    parser.add_argument("--optimization-budget-height-cap", type=float, default=np.inf, help="The max height for which opt budget is height-dependent")
    parser.add_argument("--budget-min", type=int, required=True, help="The minimum budget to use")
    parser.add_argument("--budget-max", type=int, required=True, help="The maximum budget to use")
    parser.add_argument("--budget-step", type=int, required=True, help="The interval length between budgets")
    parser.add_argument("--num-repeats", type=int, required=True, help="The number of times to repeat each experiment")
    parser.add_argument("--batch-size", type=int, required=True, help="The batch size to use for computing stochastic gradients")
    parser.add_argument("--model-mode", type=str, required=False, default="torch", choices=["torch", "sklearn"], help="The model framework to use")
    parser.add_argument("--sklearn-loss", type=str, required=False, default="hinge", choices=["hinge","log"], help="The loss function to use in sklearn SGDClassifier.")
    parser.add_argument("--sklearn-loss-penalty", type=str, required=False, default="l2", help="The loss function penalty to use in sklearn SGDClassifier.")
    parser.add_argument("--sklearn-learning-rate", type=str, required=False, default="optimal", choices=["optimal","constant","invscaling"], help="The learning rate option to use in sklearn SGDClassifier.")
    parser.add_argument("--sklearn-learning-rate-alpha", type=float, required=False, default=0.0001, help="The learning rate alpha option to use in sklearn SGDClassifier.")
    parser.add_argument("--sklearn-kernel", type=str, required=False, default="rbf", choices=["rbf",""], help="The kernel approximation class to use.")
    parser.add_argument("--sklearn-kernel-gamma", type=float, required=False, default=1., help="The kernel parameter.")
    parser.add_argument("--sklearn-kernel-ncomponents", type=int, required=False, default=100, help="The kernel num components.")
    parser.add_argument("--nu", type=float, required=True, help="Nu")
    parser.add_argument("--rho", type=float, required=True, help="Rho")
    parser.add_argument("--eta", type=float, required=True, help="The step size")
    parser.add_argument("--eta-decay-step", type=int, required=False, default=0, help="The number of steps between each step size decrease. If 0, then this setting is not used")
    parser.add_argument("--eta-decay-mult", type=float, required=False, default=1., help="The amount to multiply eta by after each eta-decay-step steps. If eta-decay-step is 0, then this setting is not used")
    parser.add_argument("--return-best-deepest-node", type=bool, required=True, help="Indicates whether best nodes only at the deepest height (True) or any height (False) should be considered")
    parser.add_argument("--sample-with-replacement", type=bool, default=True, help="Indicates whether best nodes only at the deepest height (True) or any height (False) should be considered")
    parser.add_argument("--mixture-selection-strategy", type=str, required=True, choices=["delaunay-partitioning", "coordinate-halving", "tree-results", "alpha-star", "validation", "uniform", "custom"], help="The mixture selection strategy to use.")
    parser.add_argument("--custom-mixture", type=str, required=False, default="", help="The (comma separated) mixture to use. Only used when mixture-selection-strategy is set to 'custom'")
    parser.add_argument("--columns-to-censor", type=str, default=None, help="The columns to remove from the dataset.")
    parser.add_argument("--output-dir", type=str, required=True, help="The directory in which output files will be placed")
    parser.add_argument("--tag", type=str, default="missingtag", help="The image tag used in running this experiment")
    parser.add_argument("--record-test-error", type=bool, default=False, help="Determines whether or not test error should be recorded. This toggles whether the validation or test dataset is used.")
    parser.add_argument("--inner-layer-mult", type=float, default=2., help="Determines the number of inner layers of the neural network (unless the dataset id is wine, in which case the network is not configurable).  Num inner layers will be int(inner_layer_mult * input_dim)")
    parser.add_argument("--num-hidden-layers", type=int, required=False, default=1, help="Number of nn layers")
    parser.add_argument("--inner-layer-size", type=int, required=False, default=-1, help="Number nodes in each inner layer of nn. When -1, use inner-layer-mult option instead")
    parser.add_argument("--evaluate-best-result-again", type=bool, default=False, help="If set to True, will evaluate the best node returned by tree search with the same total budget spent so far. Thus, the budget used will be doubled that requested")
    parser.add_argument("--evaluate-best-result-again-eta-mult", type=float, default=1., help="Amount to scale eta by during eval-best-result-again")
    parser.add_argument("--use-alt-loss-fn", type=bool, default=False, help="A flag to toggle whether default or alt loss function is used")
    # parser.add_argument("--output-filename", type=str, required=True, help="The output filename")

    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     "--dataset-path", "experiment_running/dataset.p",
    #     "--dataset-id", "amazon",
    #     "--experiment-type", "tree",
    #     "--tree-search-objective", "auc_roc_ovo",
    #     "--optimization-budget", "constant",
    #     "--optimization-budget-multiplier", "500",
    #     "--budget-min", "1000",
    #     "--budget-max", "3000",
    #     "--budget-step", "1000",
    #     "--num-repeats", "2",
    #     "--batch-size", "50",
    #     "--nu", "43.7639",
    #     "--rho", "0.8209",
    #     "--eta", "0.0066",
    #     "--return-best-deepest-node", "True",
    #     "--model-mode", "torch",
    #     "--num-hidden-layers", "3",
    #     "--inner-layer-size", "128",
    #     "--sklearn-loss", "hinge",
    #     "--sklearn-kernel", "rbf",
    #     "--sklearn-kernel-gamma", "0.00001",
    #     "--sklearn-kernel-ncomponents", "10",
    #     "--output-dir", "derp",
    #     "--mixture-selection-strategy", "delaunay-partitioning",
    #     "--columns-to-censor", "None",
    #     "--optimization-budget-height-cap", "8"
    # ])
    main()
