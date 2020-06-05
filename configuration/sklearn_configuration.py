from sklearn.kernel_approximation import RBFSampler
from sklearn.metrics import hinge_loss, log_loss, mean_squared_error

from configuration.model_configuration import ModelConfiguration
from datasets.pandas_dataset import PandasData
from mf_tree.eval_fns_torch import TorchLikeSGDClassifier, TorchLikeSGDRegressor


class SklearnConfiguration(ModelConfiguration):
    def __init__(self,
                 sklearn_kernel,
                 sklearn_kernel_gamma,
                 sklearn_kernel_ncomponents,
                 sklearn_loss,
                 sklearn_loss_penalty,
                 sklearn_learning_rate_alpha,
                 sklearn_learning_rate):
        super().__init__("sklearn")

        if sklearn_kernel == "rbf":
            ker = RBFSampler(gamma=sklearn_kernel_gamma,
                             n_components=sklearn_kernel_ncomponents)
            self.kernel = lambda x: ker.fit_transform(x)
        elif sklearn_kernel == "":
            self.kernel = lambda x: x
        else:
            print("Invalid kernel {}".format(sklearn_kernel))
            assert False

        self.sklearn_loss = sklearn_loss
        self.sklearn_loss_penalty = sklearn_loss_penalty
        self.sklearn_learning_rate_alpha = sklearn_learning_rate_alpha
        self.sklearn_learning_rate = sklearn_learning_rate

    def configure(self, data: PandasData, eta):
        self.loss_fn = None  # Not needed
        if self.sklearn_loss == "hinge":
            self.validation_fn = lambda mod, _, x, y: hinge_loss(y, mod(x), labels=range(data.get_num_labels()))
            self.test_fn = self.validation_fn
        elif self.sklearn_loss == "log":
            self.validation_fn = lambda mod, _, x, y: log_loss(y, mod(x),
                                                          labels=range(data.get_num_labels()))
            self.test_fn = self.validation_fn
        elif self.sklearn_loss == "squared_loss":
            self.validation_fn = lambda mod, _, x, y: mean_squared_error(y, mod(x))
            self.test_fn = self.validation_fn
        else:
            print("Unsupported sklearn loss {}. Cannot continue.".format(self.sklearn_loss))
            assert False
        if data.dataset_id in ["allstate"] or data.is_categorical:
            self.model = TorchLikeSGDClassifier(loss=self.sklearn_loss,
                                           penalty=self.sklearn_loss_penalty,
                                           warm_start=True,
                                           eta0=eta,
                                           alpha=self.sklearn_learning_rate_alpha,
                                           learning_rate=self.sklearn_learning_rate,
                                           kernel=self.kernel,
                                           num_classes=data.get_num_labels())
        elif data.dataset_id in ["wine"]:
            self.model = TorchLikeSGDRegressor(loss=self.sklearn_loss,
                                          penalty=self.sklearn_loss_penalty,
                                          warm_start=True,
                                          eta0=eta,
                                          alpha=self.sklearn_learning_rate_alpha,
                                          learning_rate=self.sklearn_learning_rate,
                                          kernel=self.kernel)
        else:
            print("Invalid dataset_id {}. Cannot continue".format(data.dataset_id))
            assert False
