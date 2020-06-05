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
                 dataset_id,
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

        self.dataset_id = dataset_id

        if "train_val_test_split" in self.df.columns:
            print("Determined that the train/val/test split is specified in the dataframe. Overriding the breakdown.")
            self.train_val_test_split = self.df["train_val_test_split"]
        else:
            self.train_val_test_split = None

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
        if self.train_val_test_split is not None:
            print("Setting the train/val/test split from the df settings")
            self.train = self.df.loc[self.df['train_val_test_split'] == 1]
            self.validate = self.df.loc[self.df['train_val_test_split'] == 2]
            self.test = self.df.loc[self.df['train_val_test_split'] == 3]
            print("Now, dropping this column")
            self.df.drop(columns=['train_val_test_split'], inplace=True)
            self.train.drop(columns=['train_val_test_split'], inplace=True)
            self.validate.drop(columns=['train_val_test_split'], inplace=True)
            self.test.drop(columns=['train_val_test_split'], inplace=True)
        else:
            for key, info in breakdown.items():
                df_for_key = self.df.loc[self.df[self.key_to_split_on] == key]
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

        print("Train mixture:", self.get_train_mixture())
        print("Validate mixture:", self.get_validate_mixture())
        print("Test mixture:", self.get_test_mixture())
        print("alpha-star:", self.get_alpha_star())

        self.importance_weight_column_name = None

    def _load_df(self, csv_file, key_to_split_on, product_key_to_keep):
        return pd.read_csv(csv_file)

    def get_num_labels(self):
        return len(self.df[self.product_key_to_keep].unique()) if self.is_categorical else 1

    def get_dim(self):
        return len(self.train.columns.difference({self.key_to_split_on,
                                                  self.product_key_to_keep,
                                                  self.importance_weight_column_name}))

    def _get_mixture(self, df):
        return df[self.key_to_split_on].value_counts(normalize=True)[self.vals_to_split].dropna().to_numpy()

    def get_train_mixture(self):
        return self._get_mixture(self.train)

    def get_validate_mixture(self):
        return self._get_mixture(self.validate)

    def get_test_mixture(self):
        return self._get_mixture(self.test)

    def get_alpha_star(self):
        train_mixture = self.get_train_mixture()
        validate_mixture = self.get_validate_mixture()
        test_mixture = self.get_test_mixture()

        if len(test_mixture) == len(train_mixture):
            print("Determined that alpha star should be determined from test dataset")
            return test_mixture
        else:
            print("Setting alpha star to be uniform over train sets")
            return np.ones_like(train_mixture) / len(train_mixture)

    def get_alpha_dim(self):
        return len(self.get_alpha_star())


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
    def __init__(self, df, key_to_split_on, is_categorical, importance_weight_column_name):
        self.df = df.loc[:, df.columns != key_to_split_on]
        self.importance_weight_column_name = importance_weight_column_name
        if self.importance_weight_column_name is None:
            self.importance_weights = pd.Series(np.ones(len(self.df)))
        else:
            self.importance_weights = self.df[self.importance_weight_column_name]
            self.df = self.df.loc[:, self.df.columns != self.importance_weight_column_name]
        self.dim = self.df.shape[1] - 1
        self.key_to_split_on = key_to_split_on
        self.is_categorical = is_categorical

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_numpy()
        x = torch.from_numpy(row[:-1]).float()
        w = torch.tensor(self.importance_weights.iloc[idx]).float()
        if self.is_categorical:
            y = torch.tensor(row[-1]).long()
        else:
            y = torch.tensor(row[-1]).float()
        return x, w, y


class SampledPandasDataset(Dataset):
    def __init__(self,
                 df,
                 mixture,
                 n_samples,
                 key_to_split_on,
                 vals_to_split,
                 with_replacement,
                 is_categorical,
                 importance_weight_column_name):
        self.mixture = mixture
        self.is_categorical = is_categorical
        self.importance_weight_column_name = importance_weight_column_name

        if n_samples == -1:
            dist_samples = (-np.ones_like(mixture) * mixture).astype(np.int)
        else:
            dist_samples = np.random.multinomial(n=n_samples, pvals=mixture)
        self.data = None
        self.importance_weights = None

        for dist_id, num_for_dist in enumerate(dist_samples):
            new_data = df.loc[df[key_to_split_on] == vals_to_split[dist_id]]
            new_data = new_data.loc[:, new_data.columns != key_to_split_on]
            if num_for_dist >= 0:
                new_data = new_data.sample(num_for_dist, replace=with_replacement)
            if self.importance_weight_column_name is None:
                new_iw = pd.Series(np.ones(len(new_data)))
            else:
                new_iw = new_data[self.importance_weight_column_name]
                new_data = new_data.loc[:, new_data.columns != self.importance_weight_column_name]
            new_data = new_data.to_numpy()
            new_iw = new_iw.to_numpy()
            self.data = np.vstack([self.data, new_data]) if self.data is not None else new_data
            self.importance_weights = np.append(self.importance_weights, new_iw) if self.importance_weights is not None else new_iw

        self.dim = self.data.shape[1] - 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        x = torch.from_numpy(row[:-1]).float()
        w = torch.tensor(self.importance_weights[idx]).float()
        if self.is_categorical:
            y = torch.tensor(row[-1]).long()
        else:
            y = torch.tensor(row[-1]).float()
        return x, w, y


class DatasetFactory:
    def __init__(self, df, key_to_split_on, vals_to_split, with_replacement, is_categorical, importance_weight_column_name):
        self.df = deepcopy(df)
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = [val for val in vals_to_split if val in self.df[self.key_to_split_on].unique()]
        self.with_replacement = with_replacement
        self.is_categorical = is_categorical
        self.importance_weight_column_name = importance_weight_column_name

    def get_dataset(self, mixture, n_samples):
        # do nothing
        assert False


class SampledPandasDatasetFactory(DatasetFactory):
    def get_dataset(self, mixture, n_samples):
        return SampledPandasDataset(self.df,
                                    mixture,
                                    n_samples,
                                    self.key_to_split_on,
                                    self.vals_to_split,
                                    with_replacement=self.with_replacement,
                                    is_categorical=self.is_categorical,
                                    importance_weight_column_name=self.importance_weight_column_name)


class MixtureIgnoringSampledPandasDatasetFactory(DatasetFactory):
    def get_dataset(self, mixture, n_samples):
        df = self.df.sample(n=n_samples, replace=self.with_replacement) if n_samples >= 0 else self.df
        return PandasDataset(df=df,
                             key_to_split_on=self.key_to_split_on,
                             is_categorical=self.is_categorical,
                             importance_weight_column_name=self.importance_weight_column_name)


class MMDPandasDatasetFactory(MixtureIgnoringSampledPandasDatasetFactory):
    def __init__(self,
                 df,
                 validation_df,
                 rbf_gamma,
                 rbf_ncomponents,
                 representative_set_size,
                 key_to_split_on,
                 vals_to_split,
                 product_key_to_keep,
                 with_replacement,
                 is_categorical,
                 importance_weight_column_name):
        super().__init__(df=df,
                         key_to_split_on=key_to_split_on,
                         vals_to_split=vals_to_split,
                         with_replacement=with_replacement,
                         is_categorical=is_categorical,
                         importance_weight_column_name=importance_weight_column_name)
        self.validation_df = deepcopy(validation_df)
        self.product_key_to_keep = product_key_to_keep

        self.gamma = rbf_gamma
        self.n_components = rbf_ncomponents
        self.representative_set_size = representative_set_size
        rbf_kernel = RBFSampler(gamma=self.gamma, n_components=self.n_components)

        # Get only the features of the datasets
        cols_to_keep = set(self.df.columns) - {self.importance_weight_column_name, self.product_key_to_keep}
        tr_dataset_features = pd.get_dummies(self.df[cols_to_keep], columns=[self.key_to_split_on])
        val_dataset_features = pd.get_dummies(self.validation_df[cols_to_keep], columns=[self.key_to_split_on])

        # Compute all feature maps using RBF Sampler
        phi_train = rbf_kernel.fit_transform(tr_dataset_features)
        phi_validation = rbf_kernel.fit_transform(val_dataset_features)

        # Pre-computations
        T1 = phi_train @ phi_validation.T @ np.ones(len(self.validation_df))
        T2 = np.array([phi_train[i, :].T @ phi_train[i, :] for i in range(len(self.df))])

        # Greedily select indices for dataset
        best_indices = []
        for i in range(1, self.representative_set_size+1):
            phi_S = phi_train[best_indices, :]

            T3 = phi_train @ phi_S.T @ np.ones(i-1) if len(phi_S) > 0 else np.zeros(len(self.df))
            objectives = 2./(len(self.validation_df)*i)*T1 - 1./(i**2)*(T2 + 2*T3)
            objectives[best_indices] = -np.inf
            best_indices.append(np.argmax(objectives))

        # Set our dataset as the selected indices
        self.df = self.df.iloc[sorted(best_indices)].reset_index(drop=True)
