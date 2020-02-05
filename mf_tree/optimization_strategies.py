import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.kernel_approximation import RBFSampler


class OptimizationStrategy:
    def __init__(self,
                 step_schedule):
        self.step_schedule = step_schedule

    def get_lr(self):
        return self.step_schedule.lr


class SGDOptimizationStrategy(OptimizationStrategy):
    def get_num_steps(self, node_eval_number):
        return int(self.step_schedule.get_num_steps(node_eval_number))

    def optimize(self, fn_to_opt, starting_point, samples, node_eval_number):
        T = self.step_schedule.get_num_steps(node_eval_number)

        traj = [starting_point]
        for t in range(T):
            beta_t = traj[t]
            beta_tplus = beta_t - self.step_schedule.get_step_size(t) * fn_to_opt.grad(beta_t, samples[t])
            traj.append(beta_tplus)
        return traj


class SGDTorchOptimizationStrategy(OptimizationStrategy):

    def __init__(self, step_schedule, decay_step=0, decay_mult=1):
        super().__init__(step_schedule)
        self.decay_step = decay_step
        self.decay_mult = decay_mult

    def optimize(self, fn_to_opt, starting_point, data_loader, eta_mult):
        optimizer = optim.SGD(starting_point.parameters(), lr=eta_mult * self.get_lr())
        if self.decay_step > 0:
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.decay_step, gamma=self.decay_mult)
        else:
            scheduler = None

        model = starting_point

        for i_batch, (sample_batch, label_batch) in enumerate(data_loader):
            # Flatten the tensor
            sample_batch_view = sample_batch.view(sample_batch.shape[0], -1)
            optimizer.zero_grad()
            preds = model(sample_batch_view)
            loss = fn_to_opt(preds, label_batch)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        return model


class SGDSklearnOptimizationStrategy(OptimizationStrategy):
    def __init__(self, step_schedule):
        super().__init__(step_schedule)

    def optimize(self, fn_to_opt, starting_point, data_loader, eta_mult):

        model = starting_point

        for i_batch, (sample_batch, label_batch) in enumerate(data_loader):
            # Flatten the tensor
            sample_batch_view = sample_batch.view(sample_batch.shape[0], -1)
            model.partial_fit(X=sample_batch_view, y=label_batch)
        return model
