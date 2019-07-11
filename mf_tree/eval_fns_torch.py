import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Net(nn.Module):

    def __init__(self, input_dim, inner_dim_mult, output_dim):
        super().__init__()
        # an affine operation: y = Wx + b
        num_inner_layers = int(inner_dim_mult * input_dim)
        print("num_inner_layers={}".format(num_inner_layers))
        self.fc1 = nn.Linear(input_dim, num_inner_layers)
        self.fc2 = nn.Linear(num_inner_layers, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MOEWNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.fc3 = nn.Linear(int(input_dim/4), 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x


class ExpMSE(nn.MSELoss):

    def forward(self, input, target):
        in_view = input.view_as(target)
        return super().forward(torch.exp(in_view), torch.exp(target))


class MFFunction:
    def __init__(self,
                 validation_fn,
                 test_fn,
                 loss_fn,
                 data_loader_factory,
                 validation_dataset,
                 validation_batch_size,
                 test_dataset,
                 test_batch_size,
                 optimization_strategy,
                 use_test_error):
        self.validation_fn = validation_fn
        self.test_fn = test_fn
        self.loss_fn = loss_fn
        self.data_loader_factory = data_loader_factory
        self.validation_dataset = validation_dataset
        self.validation_batch_size = validation_batch_size
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.optimization_strategy = optimization_strategy
        self.use_test_error = use_test_error

        self.validation_dl = DataLoader(self.validation_dataset,
                                        batch_size=self.validation_batch_size,
                                        shuffle=False)
        self.test_dl = DataLoader(self.test_dataset,
                                  batch_size=self.test_batch_size,
                                  shuffle=False)

    def get_validation_error(self, model):
        return self._get_error(model, self.validation_dataset, self.validation_dl, self.validation_fn)

    def get_test_error(self, model):
        return self._get_error(model, self.test_dataset, self.test_dl, self.test_fn)

    def _get_error(self, model, dataset, data_loader, eval_fn):
        err = 0
        len_dataset = len(dataset)
        with torch.no_grad():
            for sample, label in data_loader:
                len_batch = len(label)
                sample_view = sample.view(sample.shape[0], -1)
                preds = model(sample_view)
                # Reweight error
                err += eval_fn(preds, label) * (len_batch / len_dataset)
        return err

    def fn(self, starting_point, mixture, opt_budget, eta_mult):
        dl = self.data_loader_factory.get_data_loader(mixture, opt_budget)
        # Optimize the loss function
        ending_model = self.optimization_strategy \
            .optimize(self.loss_fn,
                      starting_point,
                      dl,
                      eta_mult)

        if self.use_test_error:
            err = self.get_test_error(ending_model)
        else:
            err = self.get_validation_error(ending_model)
        return err, ending_model


class RBFKernelFn:
    def __init__(self, gamma):
        assert gamma > 0
        self.gamma = gamma

    def __call__(self, x, y):
        return np.exp(- self.gamma * np.linalg.norm(x - y, ord=2)**2)
