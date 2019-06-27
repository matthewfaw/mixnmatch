from copy import deepcopy
from datasets.pandas_dataset import PandasData
from datasets.torch_dataset import TorchData


class DataCensorer:
    def __init__(self,
                 cols_to_censor=None):
        self.cols_to_censor = cols_to_censor if cols_to_censor is not None else {}

    def _censor_df(self, df):
        filtered_cols = [col for col in df.columns if col not in self.cols_to_censor]
        return df[filtered_cols]

    def censor(self, data):
        if isinstance(data, TorchData):
            print("Censoring TorchData not supported at this time. Returning the TorchData unmodified")
            return data
        elif isinstance(data, PandasData):
            copy = deepcopy(data)

            print("Censoring columns:",self.cols_to_censor)
            copy.train = self._censor_df(copy.train)
            copy.validate = self._censor_df(copy.validate)
            copy.test = self._censor_df(copy.test)

            return copy
        else:
            print("Unrecongnized data type", type(data), "Cannot proceed")
            assert False

