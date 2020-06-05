import torch
from pandas import DataFrame
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from datasets.pandas_dataset import SampledPandasDatasetFactory, MMDPandasDatasetFactory, MixtureIgnoringSampledPandasDatasetFactory


class DataLoaderFactory:
    def __init__(self,
                 df_or_dataset,
                 num_vals_for_product,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 batch_size,
                 is_categorical,
                 importance_weight_column_name):
        self.df_or_dataset = df_or_dataset
        self.num_vals_for_product = num_vals_for_product
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = vals_to_split
        self.with_replacement = with_replacement
        self.batch_size = batch_size
        self.is_categorical = is_categorical
        self.importance_weight_column_name = importance_weight_column_name
        self.dataset_factory = self._get_dataset_factory()
        self.targets = self._get_targets()

    def _get_dataset_factory(self):
        if isinstance(self.df_or_dataset, DataFrame):
            return SampledPandasDatasetFactory(df=self.df_or_dataset,
                                               key_to_split_on=self.key_to_split_on,
                                               vals_to_split=self.vals_to_split,
                                               with_replacement=self.with_replacement,
                                               is_categorical=self.is_categorical,
                                               importance_weight_column_name=self.importance_weight_column_name)
        else:
            return None

    def _get_targets(self):
        if isinstance(self.df_or_dataset, Dataset):
            with torch.no_grad():
                dl = DataLoader(self.df_or_dataset,
                                batch_size=1,
                                shuffle=False)
                print("Beginning to create the target tensor")
                targets = torch.zeros(len(self.df_or_dataset)).long()
                for idx, (_, curr_target) in enumerate(dl):
                    targets[idx] = curr_target
                print("Finished creating the target tensor")
                return targets
        else:
            return None

    def get_data_loader(self, mixture, opt_budget):
        if isinstance(self.df_or_dataset, DataFrame):
            sampled_dataset = self.dataset_factory.get_dataset(mixture, opt_budget)
            return DataLoader(sampled_dataset,
                              batch_size=self.batch_size,
                              shuffle=True)

        else:
            print("Cannot support creating dataloader for dataset of type",type(self.df_or_dataset))
            assert False


class IndividualTrainingSourceDataLoaderFactory(DataLoaderFactory):
    def get_data_loader(self, mixture_idx, num_training_sources):
        if isinstance(self.df_or_dataset, DataFrame):
            mixture = np.eye(1, num_training_sources, mixture_idx)[0]
            sampled_dataset = self.dataset_factory.get_dataset(mixture=mixture, n_samples=-1)
            return DataLoader(sampled_dataset,
                              batch_size=self.batch_size,
                              shuffle=True)

        else:
            print("Cannot support creating dataloader for dataset of type",type(self.df_or_dataset))
            assert False


class MixtureIgnoringDataLoaderFactory(DataLoaderFactory):
    def _get_dataset_factory(self):
        if isinstance(self.df_or_dataset, DataFrame):
            return MixtureIgnoringSampledPandasDatasetFactory(df=self.df_or_dataset,
                                                              key_to_split_on=self.key_to_split_on,
                                                              vals_to_split=self.vals_to_split,
                                                              with_replacement=self.with_replacement,
                                                              is_categorical=self.is_categorical,
                                                              importance_weight_column_name=self.importance_weight_column_name)
        else:
            return None


class MMDDataLoaderFactory(DataLoaderFactory):
    def __init__(self,
                 df_or_dataset,
                 num_vals_for_product,
                 validation_data,
                 rbf_gamma,
                 rbf_ncomponents,
                 representative_set_size,
                 key_to_split_on,
                 vals_to_split,
                 product_key_to_keep,
                 with_replacement,
                 batch_size,
                 is_categorical,
                 importance_weight_column_name):
        assert isinstance(validation_data, DataFrame)
        self.validation_df = validation_data
        self.rbf_gamma = rbf_gamma
        self.rbf_ncomponents = rbf_ncomponents
        self.representative_set_size = representative_set_size
        self.product_key_to_keep = product_key_to_keep
        super().__init__(
            df_or_dataset=df_or_dataset,
            num_vals_for_product=num_vals_for_product,
            key_to_split_on=key_to_split_on,
            vals_to_split=vals_to_split,
            with_replacement=with_replacement,
            batch_size=batch_size,
            is_categorical=is_categorical,
            importance_weight_column_name=importance_weight_column_name)

    def _get_dataset_factory(self):
        if isinstance(self.df_or_dataset, DataFrame):
            return MMDPandasDatasetFactory(df=self.df_or_dataset,
                                           validation_df=self.validation_df,
                                           rbf_gamma=self.rbf_gamma,
                                           rbf_ncomponents=self.rbf_ncomponents,
                                           representative_set_size=self.representative_set_size,
                                           key_to_split_on=self.key_to_split_on,
                                           vals_to_split=self.vals_to_split,
                                           product_key_to_keep=self.product_key_to_keep,
                                           with_replacement=self.with_replacement,
                                           is_categorical=self.is_categorical,
                                           importance_weight_column_name=self.importance_weight_column_name)
        elif isinstance(self.df_or_dataset, Dataset):
            #TODO: implement MMD in PyTorch
            print("MMD with PyTorch dataset not yet supported!")
            assert False
        else:
            return None
