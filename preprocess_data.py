
import numpy as np
from load_data import remove_nans
import os

def preprocess_data(train_data, train_labels, test_data, dir='data'):
    """ Preprocess the data and saves it in the data folder.

    Steps:
        - Remove the points where the coordinates are (-1, -1, -1)
        - Remove the points where the elevation is not between -1.6 and 0.0
        - Scale the data to be between 0 and 1

    Args:
        train_data (np.ndarray): training set of shape (n_samples, n_points, n_features)
        train_labels (np.ndarray): training labels of shape (n_samples, 1)
        test_data (np.ndarray): test set of shape (n_samples, n_points, n_features)
        dir (str): directory to save the data in
    """

    # Remove the lines where the data is only -1
    train_data = remove_nans(train_data)
    test_data = remove_nans(test_data)

    # Only keep data with elevation between -1.6 and 0.0
    train_data = [sample[sample[:, 2] < -0.0] for sample in train_data]
    test_data = [sample[sample[:, 2] < -0.0] for sample in test_data]
    train_data = [sample[sample[:, 2] > -1.6] for sample in train_data]
    test_data = [sample[sample[:, 2] > -1.6] for sample in test_data]

    # Save the data, train_data is a list
    np.savez_compressed(os.path.join(dir, 'processed_train_data.npz'), *train_data)
    np.save(os.path.join(dir, 'processed_train_labels.npy'), train_labels)
    np.savez_compressed(os.path.join(dir, 'processed_test_data.npz'), *test_data)

    return train_data, train_labels, test_data
