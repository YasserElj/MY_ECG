import math
import numpy as np
import torch
import torch.utils.data


class MaskCollator:
    """Original ECG-JEPA: mask 75-85% of patches, keep 15-25% as context."""

    def __init__(self, patch_size, min_block_size, min_keep_ratio, max_keep_ratio):
        self.patch_size = patch_size
        self.min_block_size = min_block_size
        self.min_keep_ratio = min_keep_ratio
        self.max_keep_ratio = max_keep_ratio

    def __call__(self, batch):
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


class IJEPAMaskCollator:
    """
    I-JEPA strategy: one contiguous context block, multiple target blocks.
    Context is 85-95% contiguous, targets are 15-20% each.
    Targets may fall outside context (harder prediction).

    All target blocks use the SAME length within a batch to allow tensor operations.
    """

    def __init__(self, patch_size, context_scale=(0.85, 0.95), pred_scale=(0.15, 0.20),
                 num_pred_blocks=4, min_keep=10):
        self.patch_size = patch_size
        self.context_scale = context_scale
        self.pred_scale = pred_scale
        self.num_pred_blocks = num_pred_blocks
        self.min_keep = min_keep

    def __call__(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch_size, _, channel_size = batch.size()
        num_patches = channel_size // self.patch_size

        # Sample context length
        ctx_scale = np.random.uniform(*self.context_scale)
        ctx_len = max(self.min_keep, int(num_patches * ctx_scale))

        # Sample ONE target length for ALL blocks (ensures uniform tensor shape)
        t_scale = np.random.uniform(*self.pred_scale)
        target_len = max(1, int(num_patches * t_scale))

        all_context = []
        all_target_blocks = [[] for _ in range(self.num_pred_blocks)]

        for _ in range(batch_size):
            # Sample context start position
            max_ctx_start = max(0, num_patches - ctx_len)
            ctx_start = np.random.randint(0, max_ctx_start + 1)
            context_indices = list(range(ctx_start, min(ctx_start + ctx_len, num_patches)))

            # Sample target blocks (all same length)
            target_blocks = []
            all_target_set = set()
            for _ in range(self.num_pred_blocks):
                max_start = max(0, num_patches - target_len)
                start = np.random.randint(0, max_start + 1)
                block = list(range(start, min(start + target_len, num_patches)))
                target_blocks.append(block)
                all_target_set.update(block)

            # Remove target overlap from context
            context_set = set(context_indices) - all_target_set
            context_final = sorted(context_set)

            # Ensure minimum context
            if len(context_final) < self.min_keep:
                available = sorted(set(range(num_patches)) - all_target_set)
                if len(available) >= self.min_keep:
                    context_final = available[:self.min_keep]
                else:
                    context_final = list(range(self.min_keep))

            all_context.append(context_final)
            for i, block in enumerate(target_blocks):
                all_target_blocks[i].append(block)

        # Build context tensor - pad to max length
        max_ctx_len = max(len(c) for c in all_context)
        context_tensor = torch.zeros(batch_size, max_ctx_len, dtype=torch.long)
        for i, ctx in enumerate(all_context):
            pad_len = max_ctx_len - len(ctx)
            padded = ctx + [ctx[-1]] * pad_len if ctx else [0] * max_ctx_len
            context_tensor[i] = torch.tensor(padded[:max_ctx_len], dtype=torch.long)

        # Build target tensors - all blocks have same length
        target_tensors = []
        for block_idx in range(self.num_pred_blocks):
            block_tensor = torch.zeros(batch_size, target_len, dtype=torch.long)
            for i, block in enumerate(all_target_blocks[block_idx]):
                # Pad if needed (shouldn't happen but safety)
                if len(block) < target_len:
                    block = block + [block[-1]] * (target_len - len(block))
                block_tensor[i] = torch.tensor(block[:target_len], dtype=torch.long)
            target_tensors.append(block_tensor)

        return batch, [context_tensor], target_tensors


class IJEPAMaskCollatorFlat:
    """I-JEPA with flattened output for compatibility with original JEPA model."""

    def __init__(self, patch_size, context_scale=(0.85, 0.95), pred_scale=(0.15, 0.20),
                 num_pred_blocks=4, min_keep=10):
        self.inner = IJEPAMaskCollator(patch_size, context_scale, pred_scale,
                                        num_pred_blocks, min_keep)

    def __call__(self, batch):
        batch, masks_context, masks_targets = self.inner(batch)
        mask_encoder = masks_context[0]
        mask_predictor = torch.cat(masks_targets, dim=1)
        return batch, mask_encoder, mask_predictor
