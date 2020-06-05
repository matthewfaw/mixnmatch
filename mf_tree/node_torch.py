from functools import total_ordering
import numpy as np
from matplotlib import pyplot as plt
import os
from copy import deepcopy
from datetime import datetime as dt

from mf_tree.eval_fns_torch import MFFunction
from functools import partial

from mf_tree.experiment_result_recorder import ExperimentResultRecorder


@total_ordering
class MFNode:
    def __init__(self,
                 simplex_pts,
                 starting_point,
                 mf_fn: MFFunction,
                 tree_search_objective,
                 tree_search_objective_operation,
                 tree_search_validation_datasource,
                 partitioning_strategy,
                 nu,
                 rho,
                 opt_budget_fn,
                 parent=None):
        self.children = []
        self.eval_number = -1
        self.simplex_pts = simplex_pts
        self.starting_point = deepcopy(starting_point)
        self.mf_fn = mf_fn
        self.tree_search_objective = tree_search_objective
        self.tree_search_validation_datasource = tree_search_validation_datasource
        self.partitioning_strategy = partitioning_strategy
        self.parent = parent
        self.mixture = partitioning_strategy.get_centroid(simplex_pts)
        self.height = parent.height + 1 if parent is not None else 0
        self.nu = nu
        self.rho = rho
        self.opt_budget_fn = opt_budget_fn
        self.opt_budget = -1
        self.execution_time = -1
        self.validation_mf_fn_results = None

        self.tree_search_objective_operation_name = tree_search_objective_operation
        if self.tree_search_objective_operation_name == "max":
            self.tree_search_objective_operation = lambda l: max(l)
        elif self.tree_search_objective_operation_name == "max-pairwise-difference":
            self.tree_search_objective_operation = lambda l: max(l) - min(l)
        else:
            print("{} is not a valid tree search objective operation. Cannot continue".format(self.tree_search_objective_operation_name))
            assert False

        self.value = np.inf
        self.final_model = None
        # print("Creating node at height", self.height,
        #       "beta=", self.starting_point,
        #       "mixture=", self.mixture)
        # if parent is not None:
        #     print("Actual rho^%d:" % (self.height), np.linalg.norm(self.mixture - parent.mixture, ord=1) ** self.height)

    def __eq__(self, other):
        return self.get_value_estimate() == other.get_value_estimate() and self.height == other.height

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        myest = self.get_value_estimate()
        otherest = other.get_value_estimate()
        return myest < otherest

    def __repr__(self):
        return "eval_number:" + str(self.eval_number) + os.linesep + \
               "mixture alpha:" + str(self.mixture) + os.linesep + \
               "height:" + str(self.height) + os.linesep + \
               "opt_budget:" + str(self.opt_budget) + os.linesep + \
               "opt_budget_fn:" + str(self.opt_budget_fn) + os.linesep + \
               "partitioning_strategy:" + str(self.partitioning_strategy) + os.linesep + \
               "value estimate:" + str(self.get_value_estimate())

    def get_value_estimate(self):
        return self.value - self.nu * (self.rho ** self.height)
        # return self.value

    def get_worst_case_estimate(self):
        return self.value + self.nu * (self.rho ** self.height)

    def evaluate(self, eval_number, num_remaining_iterations, starting_cost, recorder: ExperimentResultRecorder):
        self.opt_budget = min(self.opt_budget_fn(self.height, self.eval_number), num_remaining_iterations)
        if self.opt_budget > 0:
            return self._evaluate(eval_number=eval_number,
                                  starting_point=self.starting_point,
                                  opt_budget=self.opt_budget,
                                  starting_cost=starting_cost,
                                  recorder=recorder)
        else:
            return None

    def evaluate_with_final(self, opt_budget, starting_cost, recorder, eta_mult):
        return self._evaluate(eval_number=self.eval_number + 1,
                              starting_point=self.final_model,
                              opt_budget=opt_budget,
                              starting_cost=starting_cost,
                              recorder=recorder,
                              eta_mult=eta_mult)

    def _evaluate(self, eval_number, starting_point, opt_budget, starting_cost, recorder: ExperimentResultRecorder, eta_mult=1.):
        start = dt.now()
        self.eval_number = eval_number
        self.validation_mf_fn_results, self.final_model = self.mf_fn.fn(starting_point=starting_point,
                                                                        mixture=self.mixture,
                                                                        opt_budget=opt_budget,
                                                                        starting_cost=starting_cost,
                                                                        record_fn=partial(recorder.record, node=self),
                                                                        eta_mult=eta_mult)
        if self.tree_search_objective == "error":
            self.value = self.tree_search_objective_operation([fn_res.error for fn_res in self.validation_mf_fn_results])
        elif self.tree_search_objective == "auc_roc_ovo":
            self.value = self.tree_search_objective_operation([-fn_res.auc_roc_ovo for fn_res in self.validation_mf_fn_results])
        elif self.tree_search_objective == "min_precision":
            self.value = self.tree_search_objective_operation([-min(fn_res.precision) for fn_res in self.validation_mf_fn_results])
        elif self.tree_search_objective == "min_recall":
            self.value = self.tree_search_objective_operation([-min(fn_res.recall) for fn_res in self.validation_mf_fn_results])
        else:
            print("ERROR: Tree search objective {} is invalid. Cannot proceed".format(self.tree_search_objective))
            assert False
        end = dt.now()
        self.execution_time = end - start
        return self.value

    def get_test_error(self):
        return self.mf_fn.get_test_error(self.final_model)

    def get_cost(self):
        return self.opt_budget

    def split(self):
        children_simplex_points = self.partitioning_strategy.partition(self.simplex_pts)
        children = []
        for child_simplex_points in children_simplex_points:
            child = MFNode(simplex_pts=child_simplex_points,
                           starting_point=self.final_model,
                           mf_fn=self.mf_fn,
                           tree_search_objective=self.tree_search_objective,
                           tree_search_objective_operation=self.tree_search_objective_operation_name,
                           tree_search_validation_datasource=self.tree_search_validation_datasource,
                           partitioning_strategy=self.partitioning_strategy,
                           nu=self.nu,
                           rho=self.rho,
                           opt_budget_fn=self.opt_budget_fn,
                           parent=self)
            children.append(child)
        self.children = children
        return children

    def plot(self, expected_mixture):
        curr = self
        validation_vals = []
        errors = []
        mixtures = np.array([])
        while curr is not None:
            print("Node at height", curr.height, "with mixture", curr.mixture, "has value", curr.get_value_estimate())
            validation_vals.append(curr.get_value_estimate())
            errors.append(np.linalg.norm(curr.mixture - expected_mixture, ord=1))
            mixtures = np.vstack([curr.mixture, mixtures]) if mixtures.size else curr.mixture
            curr = curr.parent
        plt.plot(range(len(validation_vals)), list(reversed(validation_vals)))
        plt.xlabel('Node depth')
        plt.ylabel('Value estimate')
        plt.title('Validation error over solution path')
        plt.show()
        plt.plot(range(len(errors)), list(reversed(errors)))
        plt.xlabel('Node depth')
        plt.ylabel('$\ell_1$ distance to target mixture' + str(expected_mixture))
        plt.show()
        if len(self.mixture) == 2:
            plt.plot(mixtures[:, 0], mixtures[:, 1], 'bs-')
            plt.xlabel(r'$\alpha_1$')
            plt.ylabel(r'$\alpha_2$')
            plt.title(r'Points sampled from the $\alpha$ simplex')

        plt.tight_layout()
        plt.show()

