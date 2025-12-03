"""
ECG-specific augmentations for multi-view generation in LeJEPA.

Each ECG sample is transformed into multiple augmented views.
The model learns invariance to these augmentations.
"""

import numpy as np
import torch


class RandomCrop:
    """Randomly crop a contiguous segment from the ECG signal."""
    
    def __init__(self, crop_size: int, scale: tuple = (0.5, 1.0)):
        self.crop_size = crop_size
        self.scale = scale
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: (channel_size, num_channels) or (num_channels, channel_size)
        if x.shape[0] < x.shape[1]:  # channels first
            channel_size = x.shape[1]
            channels_first = True
        else:
            channel_size = x.shape[0]
            channels_first = False
        
        scale = np.random.uniform(*self.scale)
        crop_len = int(channel_size * scale)
        crop_len = max(crop_len, self.crop_size)
        
        if crop_len >= channel_size:
            start = 0
        else:
            start = np.random.randint(0, channel_size - crop_len + 1)
        
        if channels_first:
            cropped = x[:, start:start + crop_len]
        else:
            cropped = x[start:start + crop_len, :]
        
        return cropped


class AmplitudeScale:
    """Randomly scale the amplitude of the ECG signal."""
    
    def __init__(self, scale_range: tuple = (0.8, 1.2)):
        self.scale_range = scale_range
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(*self.scale_range)
        return x * scale


class GaussianNoise:
    """Add Gaussian noise to the ECG signal."""
    
    def __init__(self, std: float = 0.05):
        self.std = std
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*x.shape).astype(x.dtype) * self.std
        return x + noise


class TimeShift:
    """Randomly shift the signal in time (circular shift)."""
    
    def __init__(self, max_shift_ratio: float = 0.1):
        self.max_shift_ratio = max_shift_ratio
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < x.shape[1]:  # channels first
            channel_size = x.shape[1]
            axis = 1
        else:
            channel_size = x.shape[0]
            axis = 0
        
        max_shift = int(channel_size * self.max_shift_ratio)
        shift = np.random.randint(-max_shift, max_shift + 1)
        return np.roll(x, shift, axis=axis)


class BaselineWander:
    """Simulate baseline wander artifact using low-frequency sine waves."""
    
    def __init__(self, amplitude: float = 0.1, freq_range: tuple = (0.1, 0.5)):
        self.amplitude = amplitude
        self.freq_range = freq_range
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < x.shape[1]:  # channels first
            channel_size = x.shape[1]
            channels_first = True
        else:
            channel_size = x.shape[0]
            channels_first = False
        
        freq = np.random.uniform(*self.freq_range)
        t = np.linspace(0, 2 * np.pi * freq * channel_size / 500, channel_size)
        phase = np.random.uniform(0, 2 * np.pi)
        wander = (self.amplitude * np.sin(t + phase)).astype(x.dtype)
        
        if channels_first:
            return x + wander[None, :]
        else:
            return x + wander[:, None]


class ChannelDropout:
    """Randomly zero out some ECG leads."""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] < x.shape[1]:  # channels first
            num_channels = x.shape[0]
            axis = 0
        else:
            num_channels = x.shape[1]
            axis = 1
        
        x = x.copy()
        mask = np.random.random(num_channels) < self.p
        if axis == 0:
            x[mask, :] = 0
        else:
            x[:, mask] = 0
        return x


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: list):
        self.transforms = transforms
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


class RandomApply:
    """Apply a transform with a given probability."""
    
    def __init__(self, transform, p: float = 0.5):
        self.transform = transform
        self.p = p
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.transform(x)
        return x


class MultiViewTransform:
    """Generate multiple augmented views of an ECG signal."""
    
    def __init__(
        self,
        num_views: int = 4,
        crop_size: int = 5000,
        crop_scale: tuple = (0.5, 1.0),
        amplitude_scale: tuple = (0.8, 1.2),
        noise_std: float = 0.05,
        baseline_wander_amp: float = 0.05,
        time_shift_ratio: float = 0.05,
        channel_dropout_p: float = 0.0
    ):
        self.num_views = num_views
        self.crop_size = crop_size
        
        self.transform = Compose([
            RandomCrop(crop_size=crop_size, scale=crop_scale),
            RandomApply(AmplitudeScale(scale_range=amplitude_scale), p=0.8),
            RandomApply(GaussianNoise(std=noise_std), p=0.5),
            RandomApply(BaselineWander(amplitude=baseline_wander_amp), p=0.3),
            RandomApply(TimeShift(max_shift_ratio=time_shift_ratio), p=0.3),
        ])
        
        if channel_dropout_p > 0:
            self.transform = Compose([
                self.transform,
                RandomApply(ChannelDropout(p=channel_dropout_p), p=0.2)
            ])
    
    def __call__(self, x: np.ndarray) -> torch.Tensor:
        """
        Args:
            x: ECG signal of shape (channel_size, num_channels)
        
        Returns:
            views: Tensor of shape (num_views, num_channels, crop_size)
        """
        views = []
        for _ in range(self.num_views):
            view = self.transform(x.copy())
            # Ensure correct size
            if view.shape[0] > view.shape[1]:  # (channel_size, num_channels)
                view = view.T  # -> (num_channels, channel_size)
            # Pad or crop to exact size
            if view.shape[1] < self.crop_size:
                pad = np.zeros((view.shape[0], self.crop_size), dtype=view.dtype)
                pad[:, :view.shape[1]] = view
                view = pad
            elif view.shape[1] > self.crop_size:
                start = np.random.randint(0, view.shape[1] - self.crop_size + 1)
                view = view[:, start:start + self.crop_size]
            views.append(view)
        
        views = np.stack(views, axis=0)  # (num_views, num_channels, crop_size)
        return torch.from_numpy(views).float()


class MultiViewCollator:
    """Collate function that generates multi-view batches for LeJEPA."""
    
    def __init__(
        self,
        num_views: int = 4,
        crop_size: int = 5000,
        crop_scale: tuple = (0.5, 1.0),
        amplitude_scale: tuple = (0.8, 1.2),
        noise_std: float = 0.05
    ):
        self.multi_view = MultiViewTransform(
            num_views=num_views,
            crop_size=crop_size,
            crop_scale=crop_scale,
            amplitude_scale=amplitude_scale,
            noise_std=noise_std
        )
    
    def __call__(self, batch: list) -> torch.Tensor:
        """
        Args:
            batch: List of ECG tensors, each of shape (num_channels, channel_size)
        
        Returns:
            views: Tensor of shape (batch_size, num_views, num_channels, crop_size)
        """
        views = []
        for x in batch:
            if isinstance(x, torch.Tensor):
                x = x.numpy()
            if x.shape[0] < x.shape[1]:  # channels first
                x = x.T  # -> (channel_size, num_channels)
            view = self.multi_view(x)
            views.append(view)
        
        return torch.stack(views, dim=0)  # (B, V, C, T)

