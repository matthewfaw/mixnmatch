import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, random_split
import torchvision
import numpy as np


class TorchData:
    def __init__(self,
                 dataset_path,
                 dataset_id,
                 is_categorical,
                 key_to_split_on,
                 vals_to_split,
                 breakdown):
        self.is_categorical = is_categorical
        if is_categorical:
            print("Determined the dataset to be categorical")
        else:
            print("Determined the dataset to not be categorical")
        self.dataset_class = eval("torchvision.datasets.{}".format(dataset_id))
        self.train, self.validate, self.test = None, None, None
        self.key_to_split_on = key_to_split_on
        self.vals_to_split = [int(v) for v in vals_to_split]
        self.train_mixture, self.validate_mixture, self.test_mixture = np.zeros(len(self.vals_to_split)),\
                                                                       np.zeros(len(self.vals_to_split)),\
                                                                       np.zeros(len(self.vals_to_split))
        for key, info in breakdown.items():
            curr_label = int(key)
            transformed_label = self.vals_to_split.index(curr_label)
            print("Using transformed label", transformed_label, "in place of label",curr_label)
            src_dataset = self.dataset_class(root=dataset_path,
                                             train=True,
                                             download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor()
                                             ]))
            idx_to_keep = src_dataset.targets == curr_label
            src_dataset.data = src_dataset.data[idx_to_keep]
            src_dataset.targets = src_dataset.targets[idx_to_keep]
            src_dataset.targets[:] = transformed_label

            if info['setting'] == "percents":
                mult = len(src_dataset)
            else:
                mult = 1
            tr, val, test, _ = random_split(src_dataset,
                                         mult * \
                                         torch.Tensor([info['train'],
                                                       info['validate'],
                                                       info['test'],
                                                       info['drop']]).int())
            self.train_mixture[transformed_label] = len(tr)
            self.validate_mixture[transformed_label] = len(val)
            self.test_mixture[transformed_label] = len(test)
            self.train = ConcatDataset((self.train, tr)) if self.train is not None else tr
            self.validate = ConcatDataset((self.validate, val)) if self.validate is not None else val
            self.test = ConcatDataset((self.test, test)) if self.test is not None else test
        self.train_mixture /= self.train_mixture.sum()
        self.validate_mixture /= self.validate_mixture.sum()
        self.test_mixture /= self.test_mixture.sum()

    def get_num_labels(self):
        return len(self.vals_to_split)


class TorchDataset(Dataset):
    def __init__(self, dataset, key_to_split_on):
        self.dataset = dataset
        dl = DataLoader(self.dataset, batch_size=1)
        features, label = next(iter(dl))
        self.dim = list(features[0].view(-1).shape)[0]
        self.key_to_split_on = key_to_split_on

    def __len__(self):
        return self.dataset.__len__()

    def __getitem__(self, item):
        return self.dataset.__getitem__(item)
