import dill as pickle
import numpy as np

from datasets.pandas_dataset import PandasData
from mf_tree.simplex_partitioning_strategies import DelaunayPartitioningStrategy, CoordinateHalvingPartitioningStrategy, \
    ConstantPartitioningStrategy


class ExperimentConfiguration:
    def __init__(self,
                 experiment_type,
                 record_test_error,
                 evaluate_best_result_again,
                 num_repeats,
                 mixture_selection_strategy,
                 custom_mixture,
                 budget_min,
                 budget_max,
                 budget_step,
                 actual_budgets_and_mixtures_path):
        self.experiment_type = experiment_type
        self.record_test_error = record_test_error
        self.train_on_validation = mixture_selection_strategy == "validation"

        self.experiment_budgets = range(budget_min, budget_max, budget_step)
        if evaluate_best_result_again:
            recording_budget_step = 2 * budget_step
            recording_budget_min = budget_min
            recording_budget_max = 2 * (budget_max - 1) + 1
            self.recording_times = range(recording_budget_min, recording_budget_max, recording_budget_step)
        else:
            self.recording_times = self.experiment_budgets
        self.budget_min = budget_min
        self.budget_max = budget_max - 1
        self.budget_step = budget_step
        if actual_budgets_and_mixtures_path:
            print("Using actual budgets path:", actual_budgets_and_mixtures_path)
            with open(actual_budgets_and_mixtures_path, 'rb') as bm:
                budget_mixture_map = pickle.load(bm)
                self.actual_budgets = budget_mixture_map["budgets"]
                self.actual_mixtures = budget_mixture_map["mixtures"]
        else:
            self.actual_budgets = None
            self.actual_mixtures = None
        self.alt_budgets_to_use = self.actual_budgets

        self.mixture_selection_strategy = mixture_selection_strategy
        if custom_mixture and mixture_selection_strategy == "custom":
            self.custom_mixture = [float(el) for el in custom_mixture.split(',')]
            print("Using custom mixture:", self.custom_mixture)
        else:
            self.custom_mixture = None
        if mixture_selection_strategy == "all-individual-sources":
            # Set num repeats once we know the mixture dimension -- see configure below
            self.num_repeats = None
        else:
            self.num_repeats = num_repeats
        self.partitioning_strategy = None
        self.initial_simplex = None
        self.alpha_star = None
        self.alpha_dim = None

    def configure(self, data: PandasData):
        self.alpha_star = data.get_alpha_star()
        self.alpha_dim = len(self.alpha_star)
        self.initial_simplex = np.identity(self.alpha_dim)
        self.test_mixture = data.get_test_mixture()
        if self.mixture_selection_strategy == "delaunay-partitioning":
            self.partitioning_strategy = DelaunayPartitioningStrategy(dim=self.alpha_dim)
        elif self.mixture_selection_strategy == "coordinate-halving":
            self.partitioning_strategy = CoordinateHalvingPartitioningStrategy(dim=self.alpha_dim)
        elif self.mixture_selection_strategy == "alpha-star":
            self.partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                      simplex_point=self.alpha_star)
        elif self.mixture_selection_strategy == "validation":
            self.partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                      simplex_point=data.get_validate_mixture())
        elif self.mixture_selection_strategy == "tree-results":
            self.partitioning_strategy = [ConstantPartitioningStrategy(dim=self.alpha_dim, simplex_point=mixture) for
                                          mixture in self.actual_mixtures]
        elif self.mixture_selection_strategy == "uniform":
            self.partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim)
        elif self.mixture_selection_strategy == "custom":
            self.partitioning_strategy = ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                      simplex_point=self.custom_mixture)
        elif self.mixture_selection_strategy == "all-individual-sources":
            self.partitioning_strategy = [ConstantPartitioningStrategy(dim=self.alpha_dim,
                                                                       simplex_point=np.eye(1, self.alpha_dim, src_idx)[0])
                                          for src_idx in range(self.alpha_dim)]
            self.num_repeats = self.alpha_dim
        else:
            print("Invalid mixture selection strategy:", self.mixture_selection_strategy)
            assert False
