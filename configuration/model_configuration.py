
from datasets.pandas_dataset import PandasData


class ModelConfiguration:
    def __init__(self, model_mode):
        self.model_mode = model_mode

        self.loss_fn = None
        self.validation_fn = None
        self.test_fn = None
        self.model = None

    def configure(self, data: PandasData, eta):
        # do nothing
        return True
