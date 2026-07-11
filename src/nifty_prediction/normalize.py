import numpy as np

def compute_norm_stats(train_array):
    mean = np.mean(train_array)
    std = np.std(train_array)
    return mean, std


def apply_norm(array, mean, std):
    array = array.astype(float)
    array = (array - mean) / std
    return array


def undo_norm(array, mean, std):
    return array * std + mean
