import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold


def invert_nonzero_elements(tensor):
    """
    Inverts the non-zero elements of a NumPy array.

    Parameters:
    tensor (numpy.ndarray): Input NumPy array.

    Returns:
    numpy.ndarray: The processed NumPy array with in-place modifications.
    """
    # Invert non-zero elements
    tensor[tensor != 0] = 1 / tensor[tensor != 0]
    return tensor


def StratifiedKFold_generator(dataset, n_splits):
    """
    Generates Stratified K-Fold indices for a given dataset.

    Args:
        dataset (list): A list of dataset objects, where each object contains 'y' (labels) and 'id'.
        n_splits (int): Number of folds for the K-Fold split.

    Returns:
        skf_generator (generator): Yields train/val and test indices for each fold split.
    """
    skf = StratifiedKFold(n_splits=n_splits)
    merged_labels = torch.cat([data.y for data in dataset], dim=0)
    skf_generator = skf.split(np.zeros((len(dataset), 1)),
                              merged_labels,
                              groups=[data.id.item() for data in dataset])
    return skf_generator


def train_val_test_split(dataset, n_splits, fold=0):
    """
    Splits the dataset into training, validation, and test sets using nested Stratified K-Fold cross-validation.

    Args:
        dataset (list): A list of dataset objects to be split.
        n_splits (int): Number of splits (folds) for the outer and inner cross-validation.
        fold (int): Specifies which fold (split) to use for the current outer/inner split. Defaults to 0.

    Returns:
        X_train_in (Tensor): Training dataset after the inner split.
        X_val_in (Tensor): Validation dataset after the inner split.
        test_dataset (Tensor): Test dataset after the outer split.
    """

    # Outer Stratified K-Fold split: Split dataset into train+val and test sets
    skf_outer_generator = StratifiedKFold_generator(dataset, n_splits=n_splits)
    for i, (train_val_index, test_index) in enumerate(skf_outer_generator):
        if i == fold:
            train_val_dataset = dataset[torch.tensor(train_val_index).long()]
            test_dataset = dataset[torch.tensor(test_index).long()]
            break

    # Inner Stratified K-Fold split: Further split train+val set into train and validation sets
    skf_inner_generator = StratifiedKFold_generator(train_val_dataset, n_splits=n_splits)
    for j, (inner_train_index, inner_val_index) in enumerate(skf_inner_generator):
        if j == fold:
            X_train_in = train_val_dataset[torch.tensor(inner_train_index).long()]
            X_val_in = train_val_dataset[torch.tensor(inner_val_index).long()]
            break

    return X_train_in, X_val_in, test_dataset

