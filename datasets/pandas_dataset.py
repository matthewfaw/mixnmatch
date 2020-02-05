import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.kernel_approximation import RBFSampler
from copy import deepcopy


class PandasData:
    def __init__(self,
                 csv_file,
                 product_key_to_keep,
                 is_categorical,
                 key_to_split_on,
                 vals_to_split,
                 col_to_filter,
                 vals_to_keep_in_filtered_col,
                 breakdown,
                 cols_to_drop):
        self.df = self._load_df(csv_file, key_to_split_on, product_key_to_keep)
        self.df[key_to_split_on] = self.df[key_to_split_on].apply(str)
        self.orig_df = deepcopy(self.df)
        self.cols_to_drop = cols_to_drop
        print("Dropping columns %s" % self.cols_to_drop)
        self.df.drop(columns=[col for col in self.df.columns
                              for bad in self.cols_to_drop if bad == col],
                     inplace=True)
        # Rearrange columns so that the label is last
        reordered_col_list = sorted(list(set(self.df.columns) - {product_key_to_keep}))
        reordered_col_list.append(product_key_to_keep)
        self.df = self.df[reordered_col_list]
        print("Reordered columns:")
        print(self.df.columns)
        self.product_key_to_keep = product_key_to_keep
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = vals_to_split

        # self.df = self.df.loc[self.df[self.key_to_split_on].isin(self.vals_to_split)]
        self.df = self.df.loc[self.df[self.key_to_split_on].isin(breakdown.keys())]
        self.df.fillna(self.df.mean(), inplace=True)

        self.col_to_filter = col_to_filter
        self.vals_to_keep_in_filtered_col = vals_to_keep_in_filtered_col
        if self.col_to_filter in self.df.columns:
            print("Filtering column: {} to contain only values: {}".format(self.col_to_filter, self.vals_to_keep_in_filtered_col))
            self.df = self.df.loc[self.df[self.col_to_filter].isin(self.vals_to_keep_in_filtered_col)]
        else:
            print("Determined the col_to_filter: {} to not be in the df columns, so not filtering a column.".format(self.col_to_filter))

        self.is_categorical = is_categorical
        if is_categorical:
            print("Determined the dataset to be categorical, so mapping original vals to index vals")
            self.unique_label_vals = sorted(self.df[product_key_to_keep].unique())
            print("Mapping", self.unique_label_vals,"in",product_key_to_keep, "col to the corresponding index value")
            self.df[product_key_to_keep] = self.df[product_key_to_keep].apply(lambda val: self.unique_label_vals.index(val))
        else:
            print("Determined the dataset to be categorical, so skipping target val mapping step")

        self.train = None
        self.validate = None
        self.test = None
        for key, info in breakdown.items():
            df_for_key = self.df.loc[self.df[key_to_split_on] == key]
            if info['setting'] == "percents":
                orig_train_size = info["train"]
                drop_size = info["drop"]
                val_test_size = 1. - orig_train_size - drop_size

                train_size = orig_train_size / (orig_train_size + val_test_size) if orig_train_size != 0. else 0.
                validate_size = info["validate"] / val_test_size if val_test_size != 0. else 0.
                test_size = 1. - validate_size if val_test_size != 0. else 0.

                # Discard drop proportion
                if drop_size == 0.0:
                    curr_data, _ = df_for_key, None
                elif drop_size == 1.0:
                    curr_data, _ = None, df_for_key
                else:
                    curr_data, _ = train_test_split(df_for_key, train_size=(1.-drop_size))

                # Obtain train set and not train set
                if train_size == 0.0:
                    train, not_train = None, curr_data
                elif train_size == 1.0:
                    train, not_train = curr_data, None
                else:
                    train, not_train = train_test_split(curr_data, train_size=train_size)

                # Obtain valiate and test set
                if validate_size == 0.0:
                    validate, test = None, not_train
                elif test_size == 0.0:
                    validate, test = not_train, None
                else:
                    validate, test = train_test_split(not_train, train_size=validate_size)

                # Add these results to the rest
                if train is not None:
                    self.train = pd.concat((self.train, train))
                if validate is not None:
                    self.validate = pd.concat((self.validate, validate))
                if test is not None:
                    self.test = pd.concat((self.test, test))

        self.train_mixture = self.train[key_to_split_on].value_counts(normalize=True)[vals_to_split].dropna().to_numpy()
        print("Train mixture:",self.train_mixture)
        self.validate_mixture = self.validate[key_to_split_on].value_counts(normalize=True)[vals_to_split].dropna().to_numpy()
        print("Validate mixture:", self.validate_mixture)
        self.test_mixture = self.test[key_to_split_on].value_counts(normalize=True)[vals_to_split].dropna().to_numpy()
        print("Test mixture:", self.test_mixture)

    def _load_df(self, csv_file, key_to_split_on, product_key_to_keep):
        return pd.read_csv(csv_file)

    def get_num_labels(self):
        return len(self.df[self.product_key_to_keep].unique())


class SparseOHEPandasData(PandasData):
    def _load_df(self, csv_file, key_to_split_on, product_key_to_keep):
        df = pd.read_csv(csv_file, dtype=np.str)
        print("OHE columns")
        # df = pd.get_dummies(df, sparse=True, columns=df.columns.difference({key_to_split_on, product_key_to_keep}))
        df = pd.get_dummies(df, columns=df.columns.difference({key_to_split_on, product_key_to_keep}))
        print("Combine rare features")
        # rare_col = 'rare_feats_is_true'
        # df[rare_col] = 0
        cols_to_drop = []
        # for col in df.columns.difference({key_to_split_on, product_key_to_keep, rare_col}):
        for col in df.columns.difference({key_to_split_on, product_key_to_keep}):
            if df[col].value_counts()[1] < 50:
                # df[rare_col] |= df[col]
                cols_to_drop.append(col)
        print("Dropping {} rare cols".format(len(cols_to_drop)))
        df.drop(columns=cols_to_drop, inplace=True)
        print("Finished dropping rare cols")
        return df


class PandasDataset(Dataset):
    def __init__(self, df, key_to_split_on, is_categorical):
        self.df = df.loc[:, df.columns != key_to_split_on]
        self.dim = self.df.shape[1] - 1
        self.key_to_split_on = key_to_split_on
        self.is_categorical = is_categorical

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy()
        x = torch.from_numpy(row[:-1]).float()
        if self.is_categorical:
            y = torch.tensor(row[-1]).long()
        else:
            y = torch.tensor(row[-1]).float()
        return x, y


class SampledPandasDataset(Dataset):
    def __init__(self,
                 df,
                 mixture,
                 n_samples,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 is_categorical):
        self.mixture = mixture
        self.is_categorical = is_categorical

        dist_samples = np.random.multinomial(n=n_samples, pvals=mixture)
        self.data = None

        for dist_id, num_for_dist in enumerate(dist_samples):
            new_data = df.loc[df[key_to_split_on] == vals_to_split[dist_id]]
            new_data = new_data.loc[:, new_data.columns != key_to_split_on]
            new_data = new_data.sample(num_for_dist, replace=with_replacement)
            new_data = new_data.to_numpy()
            self.data = np.vstack([self.data, new_data]) if self.data is not None else new_data

        self.dim = self.data.shape[1] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x = torch.from_numpy(row[:-1]).float()
        if self.is_categorical:
            y = torch.tensor(row[-1]).long()
        else:
            y = torch.tensor(row[-1]).float()
        return x, y


class SampledPandasDatasetFactory:
    def __init__(self, df, key_to_split_on, vals_to_split, with_replacement, is_categorical):
        self.df = deepcopy(df)
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = [val for val in vals_to_split if val in self.df[self.key_to_split_on].unique()]
        self.with_replacement = with_replacement
        self.is_categorical = is_categorical

    def get_dataset(self, mixture, n_samples):
        return SampledPandasDataset(self.df,
                                    mixture,
                                    n_samples,
                                    self.key_to_split_on,
                                    self.vals_to_split,
                                    with_replacement=self.with_replacement,
                                    is_categorical=self.is_categorical)


class MMDPandasDataset(Dataset):
    def __init__(self,
                 df,
                 validation_df,
                 kernel_fn,
                 n_samples,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 is_categorical):
        self.df = df
        self.validation_df = validation_df
        self.kernel_fn = kernel_fn
        self.n_samples = n_samples
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = vals_to_split
        self.is_categorical = is_categorical

        self.data = None
        self.prev_objective = 0

        datasets = {}
        for val in vals_to_split:
            to_add = self.df.loc[self.df[key_to_split_on] == val]
            to_add = to_add.loc[:, to_add.columns != self.key_to_split_on]
            to_add = to_add.sample(min(n_samples, len(to_add)), replace=with_replacement)
            print("Adding dataset for %s: %s with shape: %s" %(self.key_to_split_on, val, to_add.shape))
            datasets[val] = to_add.reset_index(drop=True)

        sampler = RBFSampler()
        phi_V = sampler.fit_transform(self.validation_df)
        len_v = len(self.validation_df)

        phi_Ss = []
        for idx, val in enumerate(vals_to_split):
            phi_Ss.append(sampler.fit_transform(datasets[val]))

        for i in range(n_samples):
            proposed_entry_idx = - np.ones(len(vals_to_split)).astype(int)
            max_objectives = - np.inf * np.ones(len(vals_to_split))

            # Each state proposes an addition to the dataset
            for idx, val in enumerate(vals_to_split):
                state_dataset = datasets[val]
                len_s = len(state_dataset)
                if len_s == 0:
                    print("Exhausted samples from", val, "skipping this state")
                    continue
                # phi_S = sampler.fit_transform(state_dataset)
                phi_S = phi_Ss[idx]

                objectives = 2./(len_v * len_s) * (phi_S @ phi_V.T @ np.ones((len(self.validation_df),))) - 1./(len_s**2) * np.diag(phi_S @ phi_S.T)
                max_idx = np.argmax(objectives)
                proposed_entry_idx[idx] = max_idx
                max_objectives[idx] = objectives[max_idx]
            # Pick the addition with the max objective
            max_idx = np.argmax(max_objectives)

            dataset_to_mod = datasets[vals_to_split[max_idx]]
            entry_to_mod = dataset_to_mod.iloc[[proposed_entry_idx[max_idx]]]
            # Add this to our running dataset
            self.data = pd.concat((self.data, entry_to_mod)) if self.data is not None else entry_to_mod
            # Remove it from source
            dataset_to_mod.drop([entry_to_mod.iloc[0].name], inplace=True)

            phi_S_to_mod = phi_Ss[max_idx]
            phi_Ss[max_idx] = np.delete(phi_S_to_mod, (proposed_entry_idx[max_idx]), axis=0)
            # phi_Ss[max_idx][proposed_entry_idx[max_idx], :] = -np.inf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx].to_numpy()
        x = torch.from_numpy(row[:-1]).float()
        if self.is_categorical:
            y = torch.tensor(row[-1]).long()
        else:
            y = torch.tensor(row[-1]).float()
        return x, y


class MMDPandasDatasetFactory:
    def __init__(self,
                 df,
                 validation_df,
                 kernel_fn,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 is_categorical):
        self.df = deepcopy(df)
        self.validation_df = validation_df
        self.kernel_fn = kernel_fn
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = vals_to_split
        self.with_replacement = with_replacement
        self.is_categorical = is_categorical

    def get_dataset(self, mixture, n_samples):
        return MMDPandasDataset(self.df,
                                self.validation_df,
                                self.kernel_fn,
                                n_samples,
                                self.key_to_split_on,
                                self.vals_to_split,
                                with_replacement=self.with_replacement,
                                is_categorical=self.is_categorical)
