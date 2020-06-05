from configuration.dataset_configuration import DatasetConfiguration
from configuration.experiment_configuration import ExperimentConfiguration
from configuration.mmd_configuration import MMDConfiguration
from configuration.optimization_configuration import OptimizationConfiguration
from configuration.sklearn_configuration import SklearnConfiguration
from configuration.torch_configuration import TorchConfiguration
from configuration.tree_search_configuration import TreeSearchConfiguration
from mf_tree.experiment_configurer import DefaultExperimentConfigurer
from mf_tree.experiment_manager import ExperimentManager

import argparse, sys
import numpy as np


def process(args):
    dataset_config = DatasetConfiguration(dataset_path=args.dataset_path,
                                          columns_to_censor=args.columns_to_censor,
                                          experiment_type=args.experiment_type)
    experiment_config = ExperimentConfiguration(experiment_type=args.experiment_type,
                                                record_test_error=args.record_test_error,
                                                evaluate_best_result_again=args.evaluate_best_result_again,
                                                num_repeats=args.num_repeats,
                                                mixture_selection_strategy=args.mixture_selection_strategy,
                                                custom_mixture=args.custom_mixture,
                                                budget_min=args.budget_min,
                                                budget_max=args.budget_max,
                                                budget_step=args.budget_step,
                                                actual_budgets_and_mixtures_path=args.actual_budgets_and_mixtures_path)
    if args.model_mode == "sklearn":
        model_config = SklearnConfiguration(sklearn_kernel=args.sklearn_kernel,
                                            sklearn_kernel_gamma=args.sklearn_kernel_gamma,
                                            sklearn_kernel_ncomponents=args.sklearn_kernel_ncomponents,
                                            sklearn_loss=args.sklearn_loss,
                                            sklearn_loss_penalty=args.sklearn_loss_penalty,
                                            sklearn_learning_rate_alpha=args.sklearn_learning_rate_alpha,
                                            sklearn_learning_rate=args.sklearn_learning_rate)
    elif args.model_mode == "torch":
        model_config = TorchConfiguration(inner_layer_mult=args.inner_layer_mult,
                                          inner_layer_size=args.inner_layer_size,
                                          num_hidden_layers=args.num_hidden_layers,
                                          use_alt_loss_fn=args.use_alt_loss_fn,
                                          pretrained_model_path=args.pretrained_model_path,
                                          freeze_layer=args.torch_fine_tune_layer)
    else:
        print("{} is not a valid model mode. Cannot proceed".format(args.model_mode))
        assert False
    optimization_config = OptimizationConfiguration(experiment_type=args.experiment_type,
                                                    model_mode=args.model_mode,
                                                    budget_max=args.budget_max,
                                                    optimizer_class=args.torch_optimizer_class,
                                                    eta=args.eta,
                                                    eta_decay_step=args.eta_decay_step,
                                                    eta_decay_mult=args.eta_decay_mult,
                                                    batch_size=args.batch_size,
                                                    sample_with_replacement=args.sample_with_replacement,
                                                    optimization_budget_type=args.optimization_budget,
                                                    optimization_budget_multiplier=args.optimization_budget_multiplier,
                                                    optimization_budget_height_cap=args.optimization_budget_height_cap,
                                                    optimization_iht_k=args.optimization_iht_k,
                                                    optimization_iht_period=args.optimization_iht_period)
    if args.experiment_type == "mmd":
        mmd_config = MMDConfiguration(mmd_rbf_gamma=args.mmd_rbf_gamma,
                                      mmd_rbf_ncomponents=args.mmd_rbf_ncomponents,
                                      mmd_representative_set_size=args.mmd_representative_set_size)
    else:
        mmd_config = None

    tree_search_config = TreeSearchConfiguration(nu=args.nu,
                                                 rho=args.rho,
                                                 tree_search_objective=args.tree_search_objective,
                                                 tree_search_objective_operation=args.tree_search_objective_operation,
                                                 tree_search_validation_datasource=args.tree_search_validation_datasource,
                                                 return_best_deepest_node=args.return_best_deepest_node,
                                                 evaluate_best_result_again=args.evaluate_best_result_again,
                                                 evaluate_best_result_again_eta_mult=args.evaluate_best_result_again_eta_mult,
                                                 actual_best_sols_path=args.actual_budgets_and_mixtures_path,
                                                 individual_source_baseline_path=args.individual_source_baseline_path)

    configurer = DefaultExperimentConfigurer(dataset_config=dataset_config,
                                             experiment_config=experiment_config,
                                             model_config=model_config,
                                             optimization_config=optimization_config,
                                             mmd_config=mmd_config,
                                             tree_search_config=tree_search_config,
                                             output_dir=args.output_dir,
                                             tag=args.tag)

    exper_manager = ExperimentManager(experiment_configurer=configurer)

    exper_manager.run()
    exper_manager.dump_results()


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment")
    parser.add_argument("--dataset-path", type=str, required=True, help="path to pickled _Data object to process")
    parser.add_argument("--dataset-id", type=str, required=True, choices=['allstate', 'wine', 'amazon', 'mnist'], help="the dataset identifier to process")
    parser.add_argument("--actual-budgets-and-mixtures-path", type=str, default=None, help="The path to the pickled list of mixtures to use.  If used, the budgets override the budget min/max/step, and the mixtures in this file are used if the mixture-selection-strategy is tree-results")
    parser.add_argument("--individual-source-baseline-path", type=str, default=None, help="The path to the pickled map of per-source baseline models.  If used with max-k-train option set for tree-search-validation-datasource, max scores will be adjusted using these per-environment baselines to measure relative model quality")
    parser.add_argument("--pretrained-model-path", type=str, default=None, help="The path to the pickled map of the pretrained model. Will be used in place of a fresh model")
    parser.add_argument("--experiment-type", type=str, required=True, choices=['tree', 'uniform','importance-weighted-uniform', 'importance-weighted-erm', 'constant-mixture', 'mmd', 'validation'], help="The type of experiment to run")
    parser.add_argument("--tree-search-objective", type=str, required=False, default="error", choices=['error', 'auc_roc_ovo', 'min_precision', 'min_recall'], help="The tree search node value")
    parser.add_argument("--tree-search-objective-operation", type=str, required=False, default="max", choices=["max","max-pairwise-difference"], help="How to combine multiple objective values -- to be used with max-k-train validation-datasource")
    parser.add_argument("--tree-search-validation-datasource", type=str, required=False, default="validation", choices=['validation','max-k-train'], help="The data to compute the tree search node value")
    parser.add_argument("--optimization-budget", type=str, required=True, choices=['constuntil', 'linear', 'sqrt', 'height', 'constant'], help="The budget function to use at each node")
    parser.add_argument("--optimization-budget-multiplier", type=int, default=1, help="The constant to multiply each optimization budget by")
    parser.add_argument("--optimization-budget-height-cap", type=float, default=np.inf, help="The max height for which opt budget is height-dependent")
    parser.add_argument("--optimization-iht-k", type=int, default=-1, help="The number of entries to hard threshold. -1 == disabled")
    parser.add_argument("--optimization-iht-period", type=int, default=1, help="Optimization rounds between each HT operation. 1 == always threshold")
    parser.add_argument("--torch-fine-tune-layer", type=str, default="", choices=["", "last", "first"], help="When nonempty, PyTorch will update weights only in specified layer. Intended to be used when a pretrained-model-path is provided")
    parser.add_argument("--torch-optimizer-class", type=str, required=False, default="sgd", choices=['sgd','adam'], help="The optimizer used when training pytorch models")
    parser.add_argument("--budget-min", type=int, required=True, help="The minimum budget to use")
    parser.add_argument("--budget-max", type=int, required=True, help="The maximum budget to use")
    parser.add_argument("--budget-step", type=int, required=True, help="The interval length between budgets")
    parser.add_argument("--num-repeats", type=int, required=True, help="The number of times to repeat each experiment")
    parser.add_argument("--batch-size", type=int, required=True, help="The batch size to use for computing stochastic gradients")
    parser.add_argument("--model-mode", type=str, required=False, default="torch", choices=["torch", "sklearn"], help="The model framework to use")
    parser.add_argument("--sklearn-loss", type=str, required=False, default="hinge", choices=["hinge","log","squared_loss"], help="The loss function to use in sklearn SGDClassifier/Regressor.")
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
    parser.add_argument("--mixture-selection-strategy", type=str, required=True, choices=["delaunay-partitioning", "coordinate-halving", "tree-results", "alpha-star", "validation", "uniform", "custom", "all-individual-sources"], help="The mixture selection strategy to use.")
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
    parser.add_argument("--mmd-rbf-gamma", type=float, default=1.0, help="The gamma parameter for the RBFSampler for MMD")
    parser.add_argument("--mmd-rbf-ncomponents", type=int, default=100, help="The n_components parameter for the RBFSampler for MMD")
    parser.add_argument("--mmd-representative-set-size", type=int, default=100, help="The desired size of the representative set constructed by MMD")

    args = parser.parse_args()
    print(args)
    process(args)


if __name__ == "__main__":
    # sys.argv.extend([
    #     # "--dataset-path", "./derp/mnist_environment_smaller_than_five_0.1:1,0,0,0|0.2:1,0,0,0|0.3:1,0,0,0|0.9:0,0.3,0.7,0_latest.p",
    #     # "--dataset-path", "./derp/mnist_environment_smaller_than_five_0.1:1,0,0,0|0.2:1,0,0,0|0.5:1,0,0,0|0.9:0,0.3,0.7,0_latest.p",
    #     "--dataset-path", "./derp/mnist_environment_smaller_than_five_0.1:1,0,0,0|0.2:1,0,0,0|0.7:1,0,0,0|0.6:0,0.3,0.7,0_latest-sparse-extensions_iw.p",
    #     "--dataset-id", "mnist",
    #     "--tag", "latest",
    #     "--actual-budgets-and-mixtures-path", "",
    #     # "--individual-source-baseline-path", "./derp/constant-mixture_200000_1_inf_1000_200000_5000_none_100_5.0_0.8505_0.002_0_1.0_true_all-individual-sources_none_false_false_day-2020-05-25-time-14-15-38_latest_individual_src_baseline.p",
    #     # "--individual-source-baseline-path", "./derp/constant-mixture_200000_1_inf_1000_200000_5000_none_100_5.0_0.8505_0.002_0_1.0_true_all-individual-sources_none_false_false_day-2020-05-27-time-10-56-36_latest_individual_src_baseline.p",
    #     # "--individual-source-baseline-path", "",
    #     # "--pretrained-model-path", "",
    #     # "--pretrained-model-path", "./derp/uniform_200000_1_inf_1000_200000_5000_1_100_5.0_0.8505_0.002_0_1.0_true_delaunay-partitioning_none_false_false_day-2020-05-25-time-23-13-26_latest_pretrained_uniform_baseline.p",
    #     # "--pretrained-model-path", "./derp/uniform_200000_1_inf_1000_200000_5000_1_100_5.0_0.8505_0.002_0_1.0_true_delaunay-partitioning_none_false_false_day-2020-05-26-time-19-04-21_latest_pretrained_uniform_baseline.p",
    #     # "--pretrained-model-path", "./derp/uniform_200000_1_inf_1000_200000_5000_1_100_5.0_0.8505_0.002_0_1.0_true_delaunay-partitioning_none_false_false_day-2020-05-27-time-10-43-38_latest_pretrained_uniform_baseline.p",
    #     # "--torch-fine-tune-layer", "",
    #     # "--torch-fine-tune-layer", "first",
    #     # "--torch-optimizer-class", "adam",
    #     "--torch-optimizer-class", "sgd",
    #     # "--experiment-type", "tree",
    #     "--experiment-type", "importance-weighted-erm",
    #     # "--experiment-type", "uniform",
    #     # "--experiment-type", "constant-mixture",
    #     "--tree-search-objective", "auc_roc_ovo",
    #     # "--tree-search-validation-datasource", "max-k-train",
    #     # "--tree-search-validation-datasource", "validation",
    #     # "--tree-search-objective-operation", "max-pairwise-difference",
    #     "--tree-search-objective-operation", "max",
    #     "--optimization-budget", "constant",
    #     # "--optimization-budget-multiplier", "500",
    #     # "--optimization-budget-multiplier", "2000",
    #     # "--optimization-budget-multiplier", "10000",
    #     "--optimization-budget-multiplier", "1",
    #     "--optimization-budget-height-cap", "inf",
    #     # "--optimization-iht-k", "70",
    #     # "--optimization-iht-k", "-1",
    #     # "--optimization-iht-period", "10",
    #     "--budget-min", "1000",
    #     "--budget-max", "100001",
    #     "--budget-step", "5000",
    #     # "--num-repeats", "10",
    #     "--num-repeats", "1",
    #     # "--batch-size", "100",
    #     "--batch-size", "100",
    #     "--model-mode", "torch",
    #     "--sklearn-loss", "log",
    #     "--sklearn-loss-penalty", "l2",
    #     "--sklearn-learning-rate", "optimal",
    #     "--sklearn-learning-rate-alpha", "0.0001",
    #     "--sklearn-kernel", "",
    #     "--sklearn-kernel-gamma", "1.0",
    #     "--sklearn-kernel-ncomponents", "100",
    #     "--nu", "0.5",
    #     "--rho", "0.8505",
    #     "--eta", "0.002",
    #     # "--eta", "0.0005",
    #     "--eta-decay-step", "0",
    #     "--eta-decay-mult", "1",
    #     "--return-best-deepest-node", "true",
    #     # "--mixture-selection-strategy", "delaunay-partitioning",
    #     # "--mixture-selection-strategy", "coordinate-halving",
    #     "--mixture-selection-strategy", "uniform",
    #     # "--mixture-selection-strategy", "all-individual-sources",
    #     "--custom-mixture", "",
    #     "--columns-to-censor", "digit_num,train_val_test_split",
    #     "--output-dir", "./derp",
    #     # "--record-test-error", "true",
    #     "--record-test-error", "",
    #     # "--inner-layer-mult", "2.0",
    #     "--inner-layer-mult", "-1",
    #     # "--num-hidden-layers", "3",
    #     "--inner-layer-size", "390",
    #     "--evaluate-best-result-again", "",
    #     "--evaluate-best-result-again-eta-mult", "0.1",
    #     "--use-alt-loss-fn", "",
    #     "--mmd-rbf-gamma", "1.0",
    #     "--mmd-rbf-ncomponents", "20",
    #     "--mmd-representative-set-size", "400",
    # ])
    main()
