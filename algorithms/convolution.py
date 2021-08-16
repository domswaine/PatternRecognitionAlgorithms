import numpy as np
from math import ceil
from scipy.signal import correlate2d


def pad(arr, padding):
    y_dim, x_dim = arr.shape
    out = np.zeros((y_dim + 2 * padding, x_dim + 2 * padding))
    out[padding:y_dim + padding, padding:x_dim + padding] = arr
    return out


def dilate(arr, dilation):
    y_dim, x_dim = arr.shape
    out = np.zeros((dilation * (y_dim - 1) + 1, dilation * (x_dim - 1) + 1))
    for ix, iy in np.ndindex(arr.shape):
        out[dilation*ix, dilation*iy] = arr[ix, iy]
    return out


def after_stride(arr, stride):
    y_dim, x_dim = arr.shape
    new_y_dim = ceil(y_dim/stride)
    new_x_dim = ceil(x_dim / stride)
    out = np.zeros((new_y_dim, new_x_dim))
    for y in range(new_y_dim):
        for x in range(new_x_dim):
            out[y, x] = arr[stride * y, stride * x]
    return out


def convolve(arr, kernel, *, padding=0, stride=1, dilation=1):
    arr = pad(np.array(arr), padding)
    kernel = dilate(kernel, dilation)
    return after_stride(correlate2d(arr, kernel, 'valid'), stride)
