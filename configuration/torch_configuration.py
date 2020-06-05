from configuration.model_configuration import ModelConfiguration
from datasets.pandas_dataset import PandasData
import dill as pickle
import torch.nn as nn

from mf_tree.eval_fns_torch import Net, IRMNet, AmazonNet, ExpL1, ExpMSE, MOEWNet, CustomBCEWithLogitsLoss


class TorchConfiguration(ModelConfiguration):
    def __init__(self,
                 inner_layer_mult,
                 inner_layer_size,
                 num_hidden_layers,
                 use_alt_loss_fn,
                 pretrained_model_path,
                 freeze_layer):
        super().__init__("torch")

        self.inner_layer_mult = inner_layer_mult
        self.inner_layer_size = inner_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.use_alt_loss_fn = use_alt_loss_fn
        self.freeze_layer = freeze_layer
        self.pretrained_model_path = pretrained_model_path
        self.model = None

    def configure(self, data: PandasData, eta):
        if data.dataset_id in ["mnist"] and data.get_num_labels() == 2:
            self.loss_fn = CustomBCEWithLogitsLoss(reduction='none')
            val_f = CustomBCEWithLogitsLoss()
            self.model = IRMNet(input_dim=data.get_dim(),
                                inner_layer_size=self.inner_layer_size)
        elif data.dataset_id in ["allstate"] or data.is_categorical:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            val_f = nn.CrossEntropyLoss()
            model_class = AmazonNet if data.dataset_id in ["amazon"] else Net
            self.model = model_class(input_dim=data.get_dim(),
                                     inner_dim_mult=self.inner_layer_mult,
                                     inner_layer_size=self.inner_layer_size,
                                     num_hidden_layers=self.num_hidden_layers,
                                     output_dim=data.get_num_labels())
        elif data.dataset_id in ["wine"]:
            if self.use_alt_loss_fn:
                self.loss_fn = ExpL1(reduction='none')
                val_f = ExpL1()
            else:
                self.loss_fn = ExpMSE(reduction='none')
                val_f = ExpMSE()
            self.model = MOEWNet(data.get_dim())
        else:
            print("Invalid dataset_id {}. Cannot continue".format(data.dataset_id))
            assert False
        self.validation_fn = lambda _, preds, __, y: val_f(preds, y)
        self.test_fn = self.validation_fn

        if self.pretrained_model_path:
            print("Using pretrained model path:", self.pretrained_model_path)
            with open(self.pretrained_model_path, 'rb') as ptmm:
                pretrained_model_map = pickle.load(ptmm)
                print("Determined a pretrained model should be used -- overriding previous model setting.")
                self.model = pretrained_model_map["best_sols"][0][0].final_model
        if self.freeze_layer == "last":
            print("Freezing all but final layer of model. Weights will only be updated in this final layer")
            self.model.freeze_all_but_last_layer()
        elif self.freeze_layer == "first":
            print("Freezing all but first layer of model. Weights will only be updated in this first layer")
            self.model.freeze_all_but_first_layer()
