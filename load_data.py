import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


def load_data():

    assert os.path.exists('data/train.npz'), 'Train data not found!'
    assert os.path.exists('data/test.npz'), 'Test data not found!'

    train = np.load('data/train.npz', allow_pickle=True)['train']
    test = np.load('data/test.npz', allow_pickle=True)['test']

    train = train[:100]
    test = test[:100]

    train_data = train[:, 0]
    train_labels = train[:, 1]

    return train_data, train_labels, test


def remove_nans(data):
    """
    Takes data in the form (# Num Samples, # Num Lidar Points, 3)

    Points which are not useful are encoded as -1 for x, y, and z.
    """

    clean_data = []

    for sample in data:

        # Get the point index from which the data is useless
        useable = np.all(sample != -1, axis=1)

        # Remove the points which are not useful
        clean_data.append(sample[useable])
        
    return clean_data


class LidarDataset(Dataset):

    def __init__(self, data, labels=None, data_transforms=None, label_transforms=None):

        self.data = data
        self.labels = labels
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, idx):
            
        sample = self.data[idx]

        if self.data_transforms:
            sample = self.data_transforms(sample)            

        if self.labels is not None:
            labels = self.labels[idx]
            if self.label_transforms is not None:
                labels = self.label_transforms(labels)
            return sample, labels
        else:
            return sample


class LidarModule(pl.LightningDataModule):

    def __init__(self, params, data_transforms=None, label_transforms=None, debug=False):
        super().__init__()
        self.params = params
        self.data_transforms = data_transforms
        self.label_transforms = label_transforms
        self.debug = debug

    def setup(self, stage=None):

        full_data, full_labels, test = load_data()
        if self.debug:
            full_data = full_data[:10]
            full_labels = full_labels[:10]
            test = test[:10]
        full_data = remove_nans(full_data)

        full_dataset = LidarDataset(full_data, full_labels, self.data_transforms, self.label_transforms)
        train_size = int((1 - self.params['ratio_validation']) * len(full_dataset))
        validation_size = len(full_dataset) - train_size

        self.train_dataset, self.validation_dataset = random_split(full_dataset, [train_size, validation_size])
        self.test_dataset = LidarDataset(test, data_transforms=self.data_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.params['batch_size'], 
                            shuffle=True, num_workers=self.params['num_workers'], 
                            collate_fn=collate_fn, persistent_workers=self.params['num_workers']>0)
    
    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.params['batch_size'],
                            shuffle=False, num_workers=self.params['num_workers'],
                            collate_fn=collate_fn, persistent_workers=self.params['num_workers']>0)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.params['batch_size'],
                            shuffle=False, num_workers=self.params['num_workers'],
                            collate_fn=collate_fn, persistent_workers=self.params['num_workers']>0)


def collate_fn(batch):
    import torch

    if len(batch[0]) == 2:
        data, labels = zip(*batch)
        labels = torch.tensor(labels)

    else:
        data = batch

    num_points = [sample.shape[0] for sample in data]
    max_points = max(num_points)
    paddings = [max_points - num_point for num_point in num_points]
    padded_data = [np.pad(sample, ((0, padding), (0, 0)), 'constant', constant_values=0) for sample, padding in zip(data, paddings)]
    padded_data = np.stack(padded_data)

    padded_data = torch.tensor(padded_data).float().transpose(1, 2)

    if len(batch[0]) == 2:
        return padded_data, labels
    else:
        return padded_data