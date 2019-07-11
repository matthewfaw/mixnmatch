import numpy as np

class Budget:
    def __init__(self, budget, multiplier, height_cap):
        self.budget = budget
        self.multiplier = multiplier
        self.height_cap = height_cap


class ConstantBudgetFn(Budget):
    def __call__(self, height, eval_number):
        return self.multiplier * self.budget

class SqrtBudgetFn(Budget):
    def __call__(self, height, eval_number):
        return int(self.multiplier * self.budget * np.sqrt(height + 1))

class LinearBudgetFn(Budget):
    def __call__(self, height, eval_number):
        return self.multiplier * self.budget * (height + 1)


class HeightDependentBudgetFn(Budget):
    def __call__(self, height, eval_number):
        return self.multiplier * (self.budget ** min(height, self.height_cap))


class ConstUntilHeightBudgetFn(Budget):
    def __call__(self, height, eval_number):
        if height <= self.height_cap:
            return self.multiplier
        else:
            return self.multiplier * (self.budget ** height)
