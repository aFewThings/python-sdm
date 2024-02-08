import random
import numpy as np


def random_rotation(arr):
    # randomly rotate the array
    rd = random.randint(0, 3)
    if rd > 0:
        arr = np.rot90(arr, k=rd, axes=(1, 2))
    return arr


def permutation(arr):
    # randomly permute the array on axis 1 and 2
    sh = arr.shape
    t = arr.reshape((sh[0], sh[1]*sh[2]))
    t = np.transpose(t)
    np.random.shuffle(t)
    t = np.transpose(t)
    return t.reshape((sh[0], sh[1], sh[2]))


def normalize(arr):
    # normalize each 2D arrays of the 3D array on axis 0
    arr_res = np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        if arr[i].std() != 0:
            arr_res[i] = ((arr[i] - arr[i].mean()) / arr[i].std())
        else:
            arr_res[i] = arr[i] - arr[i].mean()
    return arr_res


def constant_patch(arr):
    # repeat the center value for each 2D arrays of the 3D array on axis 0
    sh = arr.shape
    return np.stack([np.full((sh[1], sh[2]), arr[i, int(sh[1]/2.0)-1, int(sh[2]/2.0)-1]) for i in range(sh[0])], axis=0)


def mean_patch(arr):
    # repeat the mean value for each 2D arrays of the 3D array on axis 0
    sh = arr.shape
    return np.stack([np.full((sh[1], sh[2]), np.mean(arr[i])) for i in range(sh[0])], axis=0)
