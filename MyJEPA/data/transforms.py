"""ECG signal transformations."""

import numpy as np
from scipy import signal


def interpolate_NaNs_(x):
    """
    Interpolate NaN values in-place.
    
    Args:
        x: Array of shape (channel_size, num_channels)
    
    Returns:
        x: Array with NaNs interpolated
    """
    nan_mask = np.isnan(x)
    for index, contains_nans in enumerate(nan_mask.any(axis=0)):
        if contains_nans:
            mask = nan_mask[:, index]
            x[mask, index] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                x[~mask, index])
    return x


def normalize_(x, mean_std=None, eps=0):
    """
    Normalize signal in-place.
    
    Args:
        x: Array of shape (channel_size, num_channels)
        mean_std: Tuple of (mean, std) arrays, or None for per-channel normalization
        eps: Small value to prevent division by zero
    
    Returns:
        x: Normalized array
    """
    if mean_std is None:
        mean = x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
    else:
        mean, std = mean_std
    x -= mean
    x /= std + eps
    return x


def highpass_filter(x, fs):
    """
    Apply highpass filter to remove baseline wander.
    
    Args:
        x: Array of shape (channel_size, num_channels)
        fs: Sampling frequency
    
    Returns:
        x: Filtered array
    """
    dtype = x.dtype
    [b, a] = signal.butter(4, 0.5, btype='highpass', fs=fs)
    x = signal.filtfilt(b, a, x, axis=0)
    x = x.astype(dtype)
    return x


def resample(x, channel_size):
    """
    Resample signal to target length.
    
    Args:
        x: Array of shape (channel_size, num_channels)
        channel_size: Target length
    
    Returns:
        x: Resampled array
    """
    dtype = x.dtype
    x = signal.resample(x, channel_size, axis=0)
    x = x.astype(dtype)
    return x


def random_crop(x, size):
    """
    Randomly crop signal to target size.
    
    Args:
        x: Array of shape (channel_size, num_channels)
        size: Target crop size
    
    Returns:
        x: Cropped array
    """
    start = np.random.randint(len(x) - size + 1)
    x = x[start:start + size]
    return x

