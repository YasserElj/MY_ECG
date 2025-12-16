"""Masking strategies for JEPA-style pretraining."""

import math
import numpy as np
import torch
import torch.utils.data


class MaskCollator:
    """
    ECG-JEPA masking strategy: mask 75-85% of patches, keep 15-25% as context.
    
    Uses block-wise masking to create contiguous masked regions.
    """

    def __init__(self, patch_size, min_block_size, min_keep_ratio, max_keep_ratio):
        """
        Args:
            patch_size: Size of each patch in samples
            min_block_size: Minimum size of masked blocks
            min_keep_ratio: Minimum ratio of patches to keep as context (e.g., 0.15)
            max_keep_ratio: Maximum ratio of patches to keep as context (e.g., 0.25)
        """
        self.patch_size = patch_size
        self.min_block_size = min_block_size
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

    def __call__(self, batch):
        """
        Collate batch and generate masks.
        
        Args:
            batch: List of ECG tensors
        
        Returns:
            batch: Collated tensor (B, C, T)
            mask_encoder: Context patch indices (B, K_ctx)
            mask_predictor: Target patch indices (B, K_tgt)
        """
        batch = torch.utils.data.default_collate(batch)
        batch_size, _, channel_size = batch.size()
        num_patches = channel_size // self.patch_size

        keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
        num_keep = math.ceil(keep_ratio * num_patches)

        mask_encoder, mask_predictor = [], []
        for _ in range(batch_size):
            mask = self._sample_mask(num_keep, num_patches)
            mask_encoder.append(mask.nonzero().squeeze())
            mask_predictor.append((1 - mask).nonzero().squeeze())

        mask_encoder = torch.utils.data.default_collate(mask_encoder)
        mask_predictor = torch.utils.data.default_collate(mask_predictor)
        return batch, mask_encoder, mask_predictor

    def _sample_mask(self, num_keep, num_patches):
        """
        Sample a block-wise mask.
        
        Args:
            num_keep: Number of patches to keep as context
            num_patches: Total number of patches
        
        Returns:
            mask: Binary mask tensor (1 = keep, 0 = mask)
        """
        patch_intervals = [(0, num_patches)]
        num_mask = num_patches - num_keep
        total_mask_size = 0

        while total_mask_size < num_mask:
            interval_sizes = np.diff(patch_intervals).flatten()
            if interval_sizes.sum() == 0:
                break
            index = np.random.choice(len(patch_intervals), p=interval_sizes / interval_sizes.sum())
            start, end = patch_intervals.pop(index)
            interval_size = end - start
            max_block_size = num_mask - total_mask_size

            if max_block_size >= self.min_block_size:
                block_size = np.random.randint(self.min_block_size, max_block_size + 1)
            else:
                block_size = max_block_size

            if interval_size <= block_size:
                total_mask_size += interval_size
            else:
                if max_block_size >= self.min_block_size:
                    split = np.random.randint(start, end - block_size + 1)
                else:
                    attach_choices = []
                    if start > 0:
                        attach_choices.append(start)
                    if end < num_patches:
                        attach_choices.append(end - block_size)
                    split = np.random.choice(attach_choices) if attach_choices else start

                patch_intervals.append((start, split))
                patch_intervals.append((split + block_size, end))
                total_mask_size += block_size

        mask = torch.zeros(num_patches)
        for start, end in patch_intervals:
            mask[start:end] = 1.
        return mask

