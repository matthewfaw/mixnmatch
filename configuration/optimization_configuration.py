from mf_tree.optimization_budgets import ConstUntilHeightBudgetFn, LinearBudgetFn, SqrtBudgetFn, \
    HeightDependentBudgetFn, ConstantBudgetFn
from mf_tree.optimization_strategies import SGDTorchOptimizationStrategy, SGDSklearnOptimizationStrategy
from mf_tree.step_schedules import SimpleTorchStepSchedule


class OptimizationConfiguration:
    def __init__(self,
                 experiment_type,
                 model_mode,
                 budget_max,
                 optimizer_class,
                 eta,
                 eta_decay_step,
                 eta_decay_mult,
                 batch_size,
                 sample_with_replacement,
                 optimization_budget_type,
                 optimization_budget_multiplier,
                 optimization_budget_height_cap,
                 optimization_iht_k,
                 optimization_iht_period):
        if experiment_type == "tree":
            self.optimization_budget = 1
        else:
            self.optimization_budget = budget_max - 1

        if "constuntil" == optimization_budget_type:
            self.opt_budget_class = ConstUntilHeightBudgetFn
        elif "linear" == optimization_budget_type:
            self.opt_budget_class = LinearBudgetFn
        elif "sqrt" == optimization_budget_type:
            self.opt_budget_class = SqrtBudgetFn
        elif "height" == optimization_budget_type:
            self.opt_budget_class = HeightDependentBudgetFn
        elif "constant" == optimization_budget_type:
            self.opt_budget_class = ConstantBudgetFn
        else:
            print("Invalid optimization budget type:", optimization_budget_type)
            assert False

        self.eta = eta
        self.eta_decay_step = eta_decay_step
        self.eta_decay_mult = eta_decay_mult
        step_schedule = SimpleTorchStepSchedule(lr=self.eta)
        if model_mode == "torch":
            self.opt_strategy = SGDTorchOptimizationStrategy(step_schedule=step_schedule,
                                                             optimizer_class=optimizer_class,
                                                             decay_step=self.eta_decay_step,
                                                             decay_mult=self.eta_decay_mult,
                                                             iht_k=optimization_iht_k,
                                                             iht_period=optimization_iht_period)
        elif model_mode == "sklearn":
            self.opt_strategy = SGDSklearnOptimizationStrategy(step_schedule=step_schedule,
                                                               iht_k=optimization_iht_k,
                                                               iht_period=optimization_iht_period)
        else:
            print("Invalid model mode {}".format(model_mode))
            assert False

        self.batch_size=batch_size
        self.sample_with_replacement=sample_with_replacement
        self.optimization_budget_multiplier=optimization_budget_multiplier
        self.optimization_budget_height_cap=optimization_budget_height_cap
