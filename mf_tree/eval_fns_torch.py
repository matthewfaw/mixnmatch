import torch
from torch.utils.data import DataLoader
from datasets.data_loaders import DataLoaderFactory, IndividualTrainingSourceDataLoaderFactory
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import SGDClassifier, SGDRegressor


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


class IRMNet(nn.Module):
    def __init__(self, input_dim, inner_layer_size):
        super().__init__()
        lin1 = nn.Linear(int(input_dim), inner_layer_size)
        lin2 = nn.Linear(inner_layer_size, inner_layer_size)
        lin3 = nn.Linear(inner_layer_size, 1)
        for lin in [lin1, lin2, lin3]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)

    def freeze_all_but_last_layer(self):
        for param in self.named_parameters():
            if "_main.4." not in param[0]:
                param[1].requires_grad = False

    def freeze_all_but_first_layer(self):
        for param in self.named_parameters():
            if "_main.0." not in param[0]:
                param[1].requires_grad = False

    def forward(self, x):
        return self._main(x)


class AmazonNet(Net):
    def __init__(self, input_dim, inner_dim_mult, inner_layer_size, num_hidden_layers, output_dim):
        super().__init__(input_dim=input_dim,
                         inner_dim_mult=inner_dim_mult,
                         inner_layer_size=inner_layer_size,
                         num_hidden_layers=num_hidden_layers,
                         output_dim=output_dim)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        for layer in self.fcinner:
            nn.init.xavier_uniform(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)


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


class CustomBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    # A wrapper for BCEWithLogitsLoss to make inputs more compatible with this codebase
    # In particular, the target will be a 1d tensor, but the input will be n x 1, so need to
    # ensure the target is of the right size. Also, this loss only works when the (categorical) target
    # is a float, not a long, so convert it here too
    def forward(self, input, target):
        return super().forward(input, target[:, None].float())


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

    def minus(self, other: 'MFFunctionResult') -> 'MFFunctionResult':
        return MFFunctionResult(error=self.error-other.error,
                                precision=self.precision-other.precision,
                                recall=self.precision-other.precision,
                                f1=self.f1-other.f1,
                                support=self.support,
                                auc_roc_ovo=self.auc_roc_ovo-other.auc_roc_ovo,
                                auc_roc_ovr=self.auc_roc_ovr-other.auc_roc_ovr)


class MFFunction:
    def __init__(self,
                 validation_fn,
                 test_fn,
                 loss_fn,
                 data_loader_factory: DataLoaderFactory,
                 individual_data_loader_factory: IndividualTrainingSourceDataLoaderFactory,
                 num_training_datasources,
                 tree_search_validation_datasource,
                 individual_source_baselines,
                 validation_dataset,
                 validation_batch_size,
                 test_dataset,
                 test_batch_size,
                 optimization_strategy):
        self.validation_fn = validation_fn
        self.test_fn = test_fn
        self.loss_fn = loss_fn
        self.data_loader_factory = data_loader_factory
        self.individual_data_loader_factory = individual_data_loader_factory
        self.num_training_datasources = num_training_datasources
        self.tree_search_validation_datasource = tree_search_validation_datasource
        self.individual_source_baselines = individual_source_baselines
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

    def get_max_k_training_error(self, model) -> [MFFunctionResult]:
        results = []
        for train_src_idx in range(self.num_training_datasources):
            dl_for_src = self.individual_data_loader_factory.get_data_loader(train_src_idx,
                                                                             self.num_training_datasources)
            dataset_for_src = dl_for_src.dataset
            error_for_src = self._get_error(model=model,
                                            dataset=dataset_for_src,
                                            data_loader=dl_for_src,
                                            eval_fn=self.validation_fn)
            if self.individual_source_baselines is not None:
                individual_final_model_for_src = self.individual_source_baselines[train_src_idx].final_model
                error_for_individual_model_on_src = self._get_error(model=individual_final_model_for_src,
                                                                    dataset=dataset_for_src,
                                                                    data_loader=dl_for_src,
                                                                    eval_fn=self.validation_fn)
                results.append(error_for_src.minus(error_for_individual_model_on_src))
            else:
                results.append(error_for_src)
        return results

    def get_validation_error(self, model) -> [MFFunctionResult]:
        return [self._get_error(model, self.validation_dataset, self.validation_dl, self.validation_fn)]

    def get_test_error(self, model) -> [MFFunctionResult]:
        return [self._get_error(model, self.test_dataset, self.test_dl, self.test_fn)]

    def _get_error(self, model, dataset, data_loader, eval_fn) -> MFFunctionResult:
        err = 0
        len_dataset = len(dataset)
        labels_all = torch.zeros(len(dataset), dtype=torch.long if dataset.is_categorical else torch.float)
        # preds_all = torch.zeros(len(dataset), self.data_loader_factory.num_vals_for_product)
        preds_all = None
        samples_all = torch.zeros(len(dataset), dataset.dim)
        curr_idx = 0

        with torch.no_grad():
            for sample, _, label in data_loader:
                len_batch = len(label)
                sample_view = sample.view(sample.shape[0], -1)
                preds = model(sample_view)
                samples_all[curr_idx:curr_idx + len_batch, :] = deepcopy(sample_view)
                # Reweight error
                # err += eval_fn(model, preds, sample_view, label) * (len_batch / len_dataset)

                # if dataset.is_categorical:
                labels_all[curr_idx:curr_idx + len_batch] = deepcopy(label)
                # _, preds_all[curr_idx:curr_idx + len_batch] = preds.max(1)
                if preds_all is None:
                    preds_all = torch.zeros(len(dataset), preds.shape[1])
                preds_all[curr_idx:curr_idx + len_batch, :] = torch.Tensor(preds)
                curr_idx += len_batch
        err = eval_fn(model, preds_all, samples_all, labels_all)
        prec, recall, f1, supp = None, None, None, None
        auc_roc_ovo = None
        auc_roc_ovr = None
        if dataset.is_categorical:
            if preds_all.shape[1] > 1:
                _, pred_class_all = preds_all.max(1)
            else:
                pred_class_all = (preds_all[:,0] > 0).float()
            prec, recall, f1, supp = prfs(labels_all, pred_class_all)
            if self.data_loader_factory.num_vals_for_product == 2:
                if preds_all.shape[1] > 1:
                    auc_roc_ovo = roc_auc_score(y_true=labels_all,
                                                y_score=preds_all[:, 1])
                else:
                    auc_roc_ovo = roc_auc_score(y_true=labels_all,
                                                y_score=torch.sigmoid(preds_all[:, 0]))
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

    def fn(self, starting_point, mixture, opt_budget, starting_cost, record_fn, eta_mult):
        dl = self.data_loader_factory.get_data_loader(mixture, opt_budget)
        # Optimize the loss function
        ending_model = self.optimization_strategy \
            .optimize(fn_to_opt=self.loss_fn,
                      starting_point=starting_point,
                      data_loader=dl,
                      starting_cost=starting_cost,
                      record_fn=record_fn,
                      eta_mult=eta_mult)

        if self.tree_search_validation_datasource == "max-k-train":
            validation_fn_result = self.get_max_k_training_error(ending_model)
        elif self.tree_search_validation_datasource == "validation":
            validation_fn_result = self.get_validation_error(ending_model)
        else:
            print("{} is not a valid tree search validation datasource. Cannot continue.")
            assert False
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

    def partial_fit(self, X, y, sample_weight):
        self.classifier.partial_fit(X=self.kernel(X), y=y, sample_weight=sample_weight, classes=self.classes)


class TorchLikeSGDRegressor:
    def __init__(self, loss, penalty, warm_start, eta0, alpha, learning_rate, kernel):
        self.loss = loss
        self.classifier = SGDRegressor(loss=loss, penalty=penalty, warm_start=warm_start, eta0=eta0, alpha=alpha, learning_rate=learning_rate)
        self.kernel = kernel

    def __call__(self, X):
        if self.loss == "squared_loss":
            return self.classifier.predict(self.kernel(X))

    def partial_fit(self, X, y, sample_weight):
        self.classifier.partial_fit(X=self.kernel(X), y=y, sample_weight=sample_weight)
