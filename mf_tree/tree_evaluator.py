from queue import PriorityQueue
import os, sys, math
from datetime import datetime as dt

from mf_tree.experiment_result_recorder import ExperimentResultRecorder
from mf_tree.node_torch import MFNode


class TreeEvaluator:
    def __init__(self, root: MFNode, budget, recorder: ExperimentResultRecorder, return_best_deepest_node):
        self.root = root
        self.budget = budget
        self.recorder = recorder
        self.leaves = PriorityQueue()
        self.total_cost = 0
        self.return_best_deepest_node = return_best_deepest_node
        self.deepest_node_height = 0
        self.execution_time = -1


class DOOTreeEvaluator(TreeEvaluator):
    def disable_printing(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def enable_printing(self):
        sys.stdout.close()
        sys.stdout = self._original_stdout

    def _evaluate(self) -> MFNode:
        eval_number = 1
        self.root.evaluate(eval_number=eval_number,
                           num_remaining_iterations=self.get_remaining_iterations(),
                           starting_cost=self.total_cost,
                           recorder=self.recorder)
        self.leaves.put(self.root)

        self.total_cost = self.root.get_cost()
        print("eval_root_cost={}".format(self.total_cost))
        print("remaining_budget={}".format(self.get_remaining_iterations()))
        while self.total_cost < self.budget:
            node = self.leaves.get()
            # print("Selected node with value estimate:", node.get_value_estimate())
            for child in node.split():
                self.deepest_node_height = max(self.deepest_node_height, child.height)
                eval_number += 1
                child.evaluate(eval_number=eval_number,
                               num_remaining_iterations=self.get_remaining_iterations(),
                               starting_cost=self.total_cost,
                               recorder=self.recorder)
                self.total_cost += child.get_cost()
                print("eval_child_cost={}".format(child.get_cost()))
                print("remaining_budget={}".format(self.get_remaining_iterations()))
                if child.get_cost() > 0:
                    self.leaves.put(child)
                else:
                    print("Warning: child was not added to leaves because we ran out of budget.")
        print("Out of money -- exiting")
        best_node = self._get_best_node()
        print("best_node_height={}".format(best_node.height))
        print("best_node_value={}".format(best_node.value))
        return best_node

    def get_remaining_iterations(self):
        return self.budget - self.total_cost

    def _get_best_node(self):
        best_node = None
        for node in self.leaves.queue:
            satisfies_height_constraint = (not self.return_best_deepest_node) or node.height == self.deepest_node_height
            best_so_far = best_node is None or best_node.get_worst_case_estimate() > node.get_worst_case_estimate()

            if satisfies_height_constraint and best_so_far:
                best_node = node
        return best_node

    def evaluate(self, debug=False):
        if not debug:
            # Blocking all print calls
            self.disable_printing()
        try:
            start = dt.now()
            best_node = self._evaluate()
            end = dt.now()
            self.execution_time = end - start
            print("doo_tree_eval_execution_time={}".format(self.execution_time.total_seconds()))
            return best_node, self.execution_time
        finally:
            if not debug:
                self.enable_printing()


class ExtendedRunsDOOTreeEvaluator(DOOTreeEvaluator):
    def __init__(self, root, budget, recorder: ExperimentResultRecorder, return_best_deepest_node, eta_mult):
        super().__init__(root=root, budget=budget, recorder=recorder, return_best_deepest_node=return_best_deepest_node)
        self.eta_mult = eta_mult

    def _evaluate(self):
        best_node = super()._evaluate()
        print("Evaluating best node with budget {}".format(self.total_cost))
        best_node.evaluate_with_final(opt_budget=self.total_cost,
                                      starting_cost=self.total_cost,
                                      recorder=self.recorder,
                                      eta_mult=self.eta_mult)
        self.total_cost *= 2
        print("final_best_node_value={}".format(best_node.value))
        print("final_total_cost={}".format(self.total_cost))
        return best_node
