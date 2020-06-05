import dill as pickle

from configuration.mmd_configuration import MMDConfiguration
from configuration.optimization_configuration import OptimizationConfiguration
from datasets.censor import DataCensorer
from datasets.data_loaders import IndividualTrainingSourceDataLoaderFactory, DataLoaderFactory, \
    MixtureIgnoringDataLoaderFactory, MMDDataLoaderFactory
from datasets.pandas_dataset import PandasData, PandasDataset


class DatasetConfiguration:
    def __init__(self,
                 dataset_path,
                 columns_to_censor,
                 experiment_type):
        self.dataset_path = dataset_path
        self.columns_to_censor = columns_to_censor
        self.experiment_type = experiment_type

        with open(dataset_path, 'rb') as f:
            self.data = pickle.load(f)
        self.cols_to_censor = columns_to_censor.split(
            ',') if columns_to_censor is not None else []
        self.censorer = DataCensorer(cols_to_censor=self.cols_to_censor)
        self.data = self.censorer.censor(self.data)
        print("Train size: %s\nValidate size: %s\nTest size: %s" % (len(self.data.train),
                                                                    len(self.data.validate),
                                                                    len(self.data.test)))
        print("Mixture order: %s" % self.data.vals_to_split)
        print("Train mixture: %s" % self.data.get_train_mixture())
        print("Validation mixture: %s" % self.data.get_validate_mixture())
        print("Test mixture order: %s" % self.data.vals_to_split)
        print("Test mixture: %s" % self.data.get_test_mixture())

        print("Num vals for product:", self.data.get_num_labels())
        iw_column = self.data.importance_weight_column_name

        if isinstance(self.data, PandasData):
            self.D_expensive = PandasDataset(self.data.validate,
                                             self.data.key_to_split_on,
                                             self.data.is_categorical,
                                             iw_column)
            self.test_dataset = PandasDataset(self.data.test,
                                              self.data.key_to_split_on,
                                              self.data.is_categorical,
                                              iw_column)
        else:
            print("data object of type", type(self.data), "is not PandasDataset or TorchDataset. Cannot proceed.")
            assert False

        self.data_loader_factory = None
        self.individual_data_loader_factory = None

    def configure(self,
                  optimization_config: OptimizationConfiguration,
                  mmd_config: MMDConfiguration):
        if "uniform" == self.experiment_type or \
                "importance-weighted-uniform" == self.experiment_type or \
                "constant-mixture" in self.experiment_type or \
                "validation" in self.experiment_type or \
                "tree" == self.experiment_type:
            self.data_loader_factory = DataLoaderFactory(
                df_or_dataset=self.data.train if self.experiment_type != "validation" else self.data.validate,
                num_vals_for_product=self.data.get_num_labels(),
                key_to_split_on=self.data.key_to_split_on,
                vals_to_split=self.data.vals_to_split,
                with_replacement=optimization_config.sample_with_replacement,
                batch_size=optimization_config.batch_size,
                is_categorical=self.data.is_categorical,
                importance_weight_column_name=self.data.importance_weight_column_name)
        elif "importance-weighted-erm" == self.experiment_type:
            self.data_loader_factory = MixtureIgnoringDataLoaderFactory(
                df_or_dataset=self.data.train if self.experiment_type != "validation" else self.data.validate,
                num_vals_for_product=self.data.get_num_labels(),
                key_to_split_on=self.data.key_to_split_on,
                vals_to_split=self.data.vals_to_split,
                with_replacement=optimization_config.sample_with_replacement,
                batch_size=optimization_config.batch_size,
                is_categorical=self.data.is_categorical,
                importance_weight_column_name=self.data.importance_weight_column_name)
        elif "mmd" == self.experiment_type:
            self.data_loader_factory = MMDDataLoaderFactory(df_or_dataset=self.data.train,
                                                            num_vals_for_product=self.data.get_num_labels(),
                                                            validation_data=self.data.validate,
                                                            rbf_gamma=mmd_config.mmd_rbf_gamma,
                                                            rbf_ncomponents=mmd_config.mmd_rbf_ncomponents,
                                                            representative_set_size=mmd_config.mmd_representative_set_size,
                                                            key_to_split_on=self.data.key_to_split_on,
                                                            vals_to_split=self.data.vals_to_split,
                                                            product_key_to_keep=self.data.product_key_to_keep,
                                                            with_replacement=optimization_config.sample_with_replacement,
                                                            batch_size=optimization_config.batch_size,
                                                            is_categorical=self.data.is_categorical,
                                                            importance_weight_column_name=self.data.importance_weight_column_name)
        else:
            print("Invalid experiment type:", self.experiment_type)
            assert False
        # TODO: Can probably just create this *only* when validation is calculated as max of k training data sources
        self.individual_data_loader_factory = IndividualTrainingSourceDataLoaderFactory(df_or_dataset=self.data.train,
                                                                                        num_vals_for_product=self.data.get_num_labels(),
                                                                                        key_to_split_on=self.data.key_to_split_on,
                                                                                        vals_to_split=self.data.vals_to_split,
                                                                                        with_replacement=optimization_config.sample_with_replacement,
                                                                                        batch_size=optimization_config.batch_size,
                                                                                        is_categorical=self.data.is_categorical,
                                                                                        importance_weight_column_name=self.data.importance_weight_column_name)

