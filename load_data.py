import numpy as np
import os
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

def load_data():

    os.makedirs('data', exist_ok=True)

    if not os.path.exists('data/train.npz') or not os.path.exists('data/test.npz'):
        from aicrowd.dataset.download import download_dataset
        print("Downloading dataset from AIcrowd API")
        print("This might not work if no login token is provided")
        print("Consider downloading the dataset manually.")
        download_dataset('lidar-car-detection', 'data/', 1, [])

    train = np.load('data/train.npz', allow_pickle=True)['train']
    test = np.load('data/test.npz', allow_pickle=True)['test']

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
        full_data = np.stack(full_data)
        test = np.stack(test)
        coords_max = np.max(np.max(full_data, axis=0, keepdims=True), axis=1, keepdims=True)
        coords_min = np.min(np.min(full_data, axis=0, keepdims=True), axis=1, keepdims=True)
        
        # Scale the Lidar points to be between 0 and 1
        full_data = (full_data - coords_min) / (coords_max - coords_min)
        test = (test - coords_min) / (coords_max - coords_min)
        
        if self.debug:
            full_data = full_data[:10, :1000, :]
            full_labels = full_labels[:10]
            test = test[:10, :1000, :]
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
    
    def predict_dataloader(self):
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
