from queue import PriorityQueue
import os, sys, math
from datetime import datetime as dt


class TreeEvaluator:
    def __init__(self, root, budget, return_best_deepest_node):
        self.root = root
        self.budget = budget
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

    def _evaluate(self):
        start = dt.now()
        eval_number = 1
        self.root.evaluate(eval_number)
        self.leaves.put(self.root)

        self.total_cost = self.root.get_cost()
        print("eval_root_cost={}".format(self.total_cost))
        print("remaining_budget={}".format(self.budget - self.total_cost))
        while self.total_cost < self.budget:
            node = self.leaves.get()
            # print("Selected node with value estimate:", node.get_value_estimate())
            for child in node.split():
                self.deepest_node_height = max(self.deepest_node_height, child.height)
                eval_number += 1
                child.evaluate(eval_number)
                self.total_cost += child.get_cost()
                print("eval_child_cost={}".format(child.get_cost()))
                print("remaining_budget={}".format(self.budget - self.total_cost))
                self.leaves.put(child)
        print("Out of money -- exiting")
        end = dt.now()
        self.execution_time = end - start
        print("doo_tree_eval_execution_time={}".format(self.execution_time.total_seconds()))
        best_node = self._get_best_node()
        print("best_node_height={}".format(best_node.height))
        print("best_node_value={}".format(best_node.value))
        return best_node, self.execution_time

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
            return self._evaluate()
        finally:
            if not debug:
                self.enable_printing()
