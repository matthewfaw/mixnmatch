import torch
from pandas import DataFrame
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from datasets.pandas_dataset import SampledPandasDatasetFactory, MMDPandasDatasetFactory, PandasDataset


class DataLoaderFactory:
    def __init__(self,
                 df_or_dataset,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 batch_size,
                 is_categorical):
        self.df_or_dataset = df_or_dataset
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = vals_to_split
        self.with_replacement = with_replacement
        self.batch_size = batch_size
        self.is_categorical = is_categorical
        self.dataset_factory = self._get_dataset_factory()
        self.targets = self._get_targets()

    def _get_dataset_factory(self):
        if isinstance(self.df_or_dataset, DataFrame):
            return SampledPandasDatasetFactory(df=self.df_or_dataset,
                                               key_to_split_on=self.key_to_split_on,
                                               vals_to_split=self.vals_to_split,
                                               with_replacement=self.with_replacement,
                                               is_categorical=self.is_categorical)
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
        if isinstance(self.df_or_dataset, Dataset):
            mix_tensor = torch.Tensor(mixture)

            sample_weights = mix_tensor[self.targets]
            sampler = WeightedRandomSampler(weights=sample_weights,
                                            num_samples=self.batch_size,
                                            replacement=self.with_replacement)

            return DataLoader(self.df_or_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              sampler=sampler)

        elif isinstance(self.df_or_dataset, DataFrame):
            sampled_dataset = self.dataset_factory.get_dataset(mixture, opt_budget)
            return DataLoader(sampled_dataset,
                              batch_size=self.batch_size,
                              shuffle=True)

        else:
            print("Cannot support creating dataloader for dataset of type",type(self.df_or_dataset))
            assert False

class MMDDataLoaderFactory(DataLoaderFactory):
    def __init__(self,
                 df_or_dataset,
                 validation_data,
                 kernel_fn,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 batch_size,
                 is_categorical):
        assert isinstance(validation_data, PandasDataset)
        self.validation_df = validation_data.df
        self.kernel_fn = kernel_fn
        super().__init__(
            df_or_dataset=df_or_dataset,
            key_to_split_on=key_to_split_on,
            vals_to_split=vals_to_split,
            with_replacement=with_replacement,
            batch_size=batch_size,
            is_categorical=is_categorical)

    def _get_dataset_factory(self):
        if isinstance(self.df_or_dataset, DataFrame):
            return MMDPandasDatasetFactory(df=self.df_or_dataset,
                                           validation_df=self.validation_df,
                                           kernel_fn=self.kernel_fn,
                                           key_to_split_on=self.key_to_split_on,
                                           vals_to_split=self.vals_to_split,
                                           with_replacement=self.with_replacement,
                                           is_categorical=self.is_categorical)
        elif isinstance(self.df_or_dataset, Dataset):
            #TODO: implement MMD in PyTorch
            print("MMD with PyTorch dataset not yet supported!")
            assert False
        else:
            return None
