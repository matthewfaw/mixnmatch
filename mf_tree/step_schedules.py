import numpy as np


class StepSchedule:
    def __init__(self, loss_fn, c, rho, nu):
        assert c <= 6. / np.pi ** 2
        # assert rho > 0 and rho < 1
        assert 1 > rho > 0
        self.loss_fn = loss_fn
        self.c = c
        self.rho = rho
        self.nu = nu

    def get_num_steps(self, node_eval_number):
        # Do nothing
        return 0

    def get_step_size(self, t):
        # Do nothing
        return 0


class DuchiStepSchedule(StepSchedule):
    def __init__(self, loss_fn, c, rho, nu):
        super().__init__(loss_fn, c, rho, nu)

    def get_num_steps(self, node_eval_number):
        return int(64 * self.loss_fn.lips ** 2 * (1 + 4 * np.sqrt(np.log(node_eval_number ** 2 / self.c))) ** 2 \
                   / (self.loss_fn.str_conv ** 3 * self.rho ** 4))

    def get_step_size(self, t):
        return np.sqrt(self.loss_fn.str_conv) / (self.loss_fn.lips * np.sqrt(t + 1))


class BottouStepSchedule(StepSchedule):
    def __init__(self, loss_fn, k, gamma, c, rho, nu):
        super().__init__(loss_fn, c, rho, nu)
        assert k > 1. / (2 * loss_fn.str_conv)
        assert gamma > 0
        self.k = k
        self.gamma = gamma

    def get_num_steps(self, node_eval_number):
        '''
        This is the step size schedule we can derive from our modification to
        Bottou Theorem 4.7, assuming that we're in the regime where the noise of
        the stochastic gradient does not dominate the distance from the origin.
        We assume here also that we can replace the lipschitz constant on f
        by \beta * D_1
        Additionally, we approximate t - log(t) approx = t.
        '''
        return int(8 * self.loss_fn.smooth / (self.loss_fn.str_conv * self.rho ** 2) * \
                   ((self.gamma + 1.) / 2 + self.loss_fn.sigma * np.pi * np.sqrt(
                       np.log(node_eval_number ** 2 / self.c) / 3)))

    def get_step_size(self, t):
        return self.k / (self.gamma + t)


class SimpleTorchStepSchedule(StepSchedule):
    def __init__(self, loss_fn, c, rho, nu, lr):
        super().__init__(loss_fn, c, rho, nu)
        self.lr = lr

    def get_step_size(self, t):
        return self.lr

