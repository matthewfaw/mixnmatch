import torch
import torch.optim as optim
import numpy as np


class OptimizationStrategy:
    def __init__(self,
                 step_schedule,
                 iht_k,
                 iht_period):
        self.step_schedule = step_schedule
        self.iht_k = iht_k
        self.iht_period = iht_period

    def get_lr(self):
        return self.step_schedule.lr


class IHTClipper(object):
    def __init__(self, top_k):
        self.top_k = top_k

    def __call__(self, module):
        if self.top_k > 0 and hasattr(module, 'weight') and module.weight.requires_grad:
            w = module.weight.data
            col_norm = w.norm(p=2, dim=0)
            _, idx_of_min_norm_columns = torch.topk(col_norm, k=len(col_norm)-self.top_k, largest=False, sorted=False)
            # w[:, idx_of_min_norm_columns] = 0.
            w[:,49:]=0
            # topk, idx = torch.topk(torch.abs(w), self.top_k)
            # mask = torch.scatter(torch.zeros_like(w), dim=1, index=idx, value=1)
            # w *= mask


class SGDTorchOptimizationStrategy(OptimizationStrategy):

    def __init__(self, step_schedule, optimizer_class, iht_k, iht_period, decay_step=0, decay_mult=1):
        super().__init__(step_schedule=step_schedule, iht_k=iht_k, iht_period=iht_period)
        self.decay_step = decay_step
        self.decay_mult = decay_mult
        if optimizer_class == "sgd":
            self.optimizer_class = optim.SGD
        elif optimizer_class == "adam":
            self.optimizer_class = optim.Adam
        else:
            print("{} is not a recognized optimizer class. Cannot continue.".format(optimizer_class))
            assert False
        self.iht_clipper = IHTClipper(self.iht_k)

    def optimize(self, fn_to_opt, starting_point, data_loader, starting_cost, record_fn, eta_mult):
        optimizer = self.optimizer_class(starting_point.parameters(), lr=eta_mult * self.get_lr())
        if self.decay_step > 0:
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.decay_step, gamma=self.decay_mult)
        else:
            scheduler = None

        model = starting_point
        prev_cost = starting_cost

        for i_batch, (sample_batch, sample_weight_batch, label_batch) in enumerate(data_loader):
            # Flatten the tensor
            sample_batch_view = sample_batch.view(sample_batch.shape[0], -1)
            optimizer.zero_grad()
            preds = model(sample_batch_view)
            loss = fn_to_opt(preds, label_batch)
            loss = loss * sample_weight_batch
            loss.mean().backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if self.iht_k > 0 and (i_batch+1) % self.iht_period == 0:
                model.apply(self.iht_clipper)
                # model._modules['_main'][0].apply(self.iht_clipper)

            curr_cost = prev_cost + len(sample_batch)
            record_fn(model=model, prev_cost=prev_cost, curr_cost=curr_cost)
            prev_cost = curr_cost

        return model


class SGDSklearnOptimizationStrategy(OptimizationStrategy):
    def __init__(self, step_schedule, iht_k, iht_period,):
        super().__init__(step_schedule=step_schedule, iht_k=iht_k, iht_period=iht_period)

    def optimize(self, fn_to_opt, starting_point, data_loader, starting_cost, record_fn, eta_mult):
        model = starting_point
        prev_cost = starting_cost

        for i_batch, (sample_batch, sample_weight_batch, label_batch) in enumerate(data_loader):
            # Flatten the tensor
            sample_batch_view = sample_batch.view(sample_batch.shape[0], -1)
            model.partial_fit(X=sample_batch_view, y=label_batch, sample_weight=sample_weight_batch)

            curr_cost = prev_cost + len(sample_batch)
            model = self._hard_threshold(model, i_batch)
            record_fn(model=model, prev_cost=prev_cost, curr_cost=curr_cost)
            prev_cost = curr_cost
        return model

    def _hard_threshold(self, model, i_batch):
        if self.iht_k > 0 and (i_batch+1) % self.iht_period == 0:
            bottom_idx = np.argpartition(np.abs(model.classifier.coef_[0]), -self.iht_k)[:-self.iht_k]
            model.classifier.coef_[0, bottom_idx] = 0.
        return model
