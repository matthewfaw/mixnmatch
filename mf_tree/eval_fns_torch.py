import torch
from torch.utils.data import DataLoader
from datasets.data_loaders import DataLoaderFactory
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier


class Net(nn.Module):

    def __init__(self, input_dim, inner_dim_mult, inner_layer_size, num_hidden_layers, output_dim):
        super().__init__()
        # an affine operation: y = Wx + b
        num_inner_nodes = int(inner_dim_mult * input_dim) if inner_layer_size == -1 else inner_layer_size
        print("num_inner_layers={}".format(num_inner_nodes))
        self.fc1 = nn.Linear(input_dim, num_inner_nodes)
        self.fcinner = [nn.Linear(num_inner_nodes, num_inner_nodes) for _ in range(num_hidden_layers - 1)]
        self.fc2 = nn.Linear(num_inner_nodes, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for fc in self.fcinner:
            x = F.relu(fc(x))
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


class ExpL1(nn.L1Loss):

    def forward(self, input, target):
        in_view = input.view_as(target)
        return super().forward(torch.exp(in_view), torch.exp(target))


class MFFunctionResult:
    def __init__(self,
                 error,
                 precision,
                 recall,
                 f1,
                 support,
                 auc_roc_ovo,
                 auc_roc_ovr):
        self.error = error
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.support = support
        self.auc_roc_ovo = auc_roc_ovo
        self.auc_roc_ovr = auc_roc_ovr


class MFFunction:
    def __init__(self,
                 validation_fn,
                 test_fn,
                 loss_fn,
                 data_loader_factory: DataLoaderFactory,
                 validation_dataset,
                 validation_batch_size,
                 test_dataset,
                 test_batch_size,
                 optimization_strategy):
        self.validation_fn = validation_fn
        self.test_fn = test_fn
        self.loss_fn = loss_fn
        self.data_loader_factory = data_loader_factory
        self.validation_dataset = validation_dataset
        self.validation_batch_size = validation_batch_size
        self.test_dataset = test_dataset
        self.test_batch_size = test_batch_size
        self.optimization_strategy = optimization_strategy

        self.validation_dl = DataLoader(self.validation_dataset,
                                        batch_size=self.validation_batch_size,
                                        shuffle=False)
        self.test_dl = DataLoader(self.test_dataset,
                                  batch_size=self.test_batch_size,
                                  shuffle=False)

    def get_validation_error(self, model) -> MFFunctionResult:
        return self._get_error(model, self.validation_dataset, self.validation_dl, self.validation_fn)

    def get_test_error(self, model) -> MFFunctionResult:
        return self._get_error(model, self.test_dataset, self.test_dl, self.test_fn)

    def _get_error(self, model, dataset, data_loader, eval_fn) -> MFFunctionResult:
        err = 0
        len_dataset = len(dataset)
        labels_all = torch.zeros(len(dataset), dtype=torch.long if dataset.is_categorical else torch.float)
        preds_all = torch.zeros(len(dataset), self.data_loader_factory.num_vals_for_product)
        samples_all = torch.zeros(len(dataset), dataset.dim)
        curr_idx = 0

        with torch.no_grad():
            for sample, label in data_loader:
                len_batch = len(label)
                sample_view = sample.view(sample.shape[0], -1)
                preds = model(sample_view)
                samples_all[curr_idx:curr_idx + len_batch, :] = deepcopy(sample_view)
                # Reweight error
                # err += eval_fn(model, preds, sample_view, label) * (len_batch / len_dataset)

                if dataset.is_categorical:
                    labels_all[curr_idx:curr_idx + len_batch] = deepcopy(label)
                    # _, preds_all[curr_idx:curr_idx + len_batch] = preds.max(1)
                    preds_all[curr_idx:curr_idx + len_batch, :] = torch.Tensor(preds)
                    curr_idx += len_batch
        err = eval_fn(model, preds_all, samples_all, labels_all)
        prec, recall, f1, supp = None, None, None, None
        auc_roc_ovo = None
        auc_roc_ovr = None
        if dataset.is_categorical:
            _, pred_class_all = preds_all.max(1)
            prec, recall, f1, supp = prfs(labels_all, pred_class_all)
            if len(labels_all.unique()) == 2:
                auc_roc_ovo = roc_auc_score(y_true=labels_all,
                                            y_score=preds_all[:, 1])
                auc_roc_ovr = auc_roc_ovo
            else:
                auc_roc_ovo = roc_auc_score(y_true=label_binarize(labels_all,
                                                       classes=range(self.data_loader_factory.num_vals_for_product)),
                                        y_score=preds_all,
                                        average='macro',
                                        multi_class='ovo')
                auc_roc_ovr = roc_auc_score(y_true=label_binarize(labels_all,
                                                                  classes=range(self.data_loader_factory.num_vals_for_product)),
                                            y_score=preds_all,
                                            average='macro',
                                            multi_class='ovr')

        result = MFFunctionResult(error=err, precision=prec, recall=recall, f1=f1, support=supp, auc_roc_ovo=auc_roc_ovo, auc_roc_ovr=auc_roc_ovr)
        return result

    def fn(self, starting_point, mixture, opt_budget, eta_mult):
        dl = self.data_loader_factory.get_data_loader(mixture, opt_budget)
        # Optimize the loss function
        ending_model = self.optimization_strategy \
            .optimize(self.loss_fn,
                      starting_point,
                      dl,
                      eta_mult)

        validation_fn_result = self.get_validation_error(ending_model)
        return validation_fn_result, ending_model


class RBFKernelFn:
    def __init__(self, gamma):
        assert gamma > 0
        self.gamma = gamma

    def __call__(self, x, y):
        return np.exp(- self.gamma * np.linalg.norm(x - y, ord=2)**2)


class TorchLikeSGDClassifier:
    def __init__(self, loss, penalty, warm_start, eta0, alpha, learning_rate, kernel, num_classes):
        self.loss = loss
        self.classifier = SGDClassifier(loss=loss, penalty=penalty, warm_start=warm_start, eta0=eta0, alpha=alpha, learning_rate=learning_rate)
        self.kernel = kernel
        self.classes = range(num_classes)

    def __call__(self, X):
        if self.loss == "hinge":
            return self.classifier.decision_function(self.kernel(X))
        elif self.loss == "log":
            return self.classifier.predict_proba(self.kernel(X))

    def partial_fit(self, X, y):
        self.classifier.partial_fit(X=self.kernel(X), y=y, classes=self.classes)
