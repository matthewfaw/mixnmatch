import dill as pickle


class TreeSearchConfiguration:
    def __init__(self,
                 nu,
                 rho,
                 tree_search_objective,
                 tree_search_objective_operation,
                 tree_search_validation_datasource,
                 return_best_deepest_node,
                 evaluate_best_result_again,
                 evaluate_best_result_again_eta_mult,
                 actual_best_sols_path,
                 individual_source_baseline_path):
        self.nu=nu
        self.rho=rho
        self.tree_search_objective=tree_search_objective
        self.tree_search_objective_operation=tree_search_objective_operation
        self.tree_search_validation_datasource=tree_search_validation_datasource
        self.return_best_deepest_node=return_best_deepest_node
        self.evaluate_best_result_again=evaluate_best_result_again
        self.evaluate_best_result_again_eta_mult=evaluate_best_result_again_eta_mult
        if actual_best_sols_path:
            print("Using actual budgets and mixtures path:", actual_best_sols_path)
            with open(actual_best_sols_path, 'rb') as bm:
                budget_mixture_map = pickle.load(bm)
                self.actual_best_sols = budget_mixture_map["best_sols"]
        else:
            self.actual_best_sols = None
        if individual_source_baseline_path:
            print("Using individual source baseline path:", individual_source_baseline_path)
            with open(individual_source_baseline_path, 'rb') as isbm:
                individual_source_baseline_map = pickle.load(isbm)
                self.individual_source_baselines = individual_source_baseline_map["best_sols"][0]
        else:
            self.individual_source_baselines = None
