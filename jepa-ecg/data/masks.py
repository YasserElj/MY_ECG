import math
import numpy as np
import torch
import torch.utils.data
from scipy.signal import find_peaks

class MaskCollator:
    def __init__(self, patch_size, min_block_size=None, min_keep_ratio=None, max_keep_ratio=None):
        self.patch_size = patch_size

    def __call__(self, batch):
        # batch is a list of tensors, stack them: (B, 12, 5000)
        batch = torch.stack(batch) 
        batch_size, num_channels, channel_size = batch.size()
        
        # Calculate number of patches
        # Note: If you switched to overlapping tokenizer (stride=25), this remains correct.
        # If stride != patch_size, change this to: channel_size // stride
        num_patches = channel_size // self.patch_size 
        
        mask_encoder = []
        mask_predictor = []

        # We will modify the batch IN-PLACE for Lead Masking
        # But we create a clone to avoid side effects if data is reused
        masked_batch = batch.clone()

        for i in range(batch_size):
            # --- 1. SPATIAL MASKING (Lead Masking) ---
            # This MUST happen on the raw signal, before embedding!
            # 20% chance to mask 1-3 random leads
            if np.random.rand() < 0.20:
                num_leads_to_mask = np.random.randint(1, 4) # Mask 1, 2, or 3 leads
                leads_to_mask = np.random.choice(range(12), size=num_leads_to_mask, replace=False)
                
                # Set the raw signal of those leads to 0.0
                masked_batch[i, leads_to_mask, :] = 0.0

            # --- 2. TEMPORAL MASKING (Beat Masking) ---
            # This generates the mask indices for the Transformer
            mask = torch.zeros(num_patches) # 0 = Visible, 1 = Masked
            
            # Use Lead II (index 1) to find R-peaks
            # If Lead II was masked above, we might miss peaks, so fallback to Lead I or V5
            lead_idx = 1
            if masked_batch[i, 1, :].sum() == 0: lead_idx = 0 # Fallback
            
            lead_signal = masked_batch[i, lead_idx, :].numpy()
            
            # Fast peak detection
            peaks, _ = find_peaks(lead_signal, distance=150, prominence=0.5)
            
            # Mask 25% of beats (Time Masking)
            for peak in peaks:
                if np.random.rand() < 0.25:
                    center_patch = peak // self.patch_size
                    start = max(0, center_patch - 2)
                    end = min(num_patches, center_patch + 5)
                    mask[start:end] = 1.0

            # --- 3. RANDOM FILL ---
            # Ensure we reach ~75% masking ratio
            current_mask_ratio = mask.sum() / num_patches
            if current_mask_ratio < 0.6:
                remaining_needed = int((0.6 - current_mask_ratio) * num_patches)
                if remaining_needed > 0:
                    available_indices = np.where(mask == 0)[0]
                    if len(available_indices) > remaining_needed:
                        random_indices = np.random.choice(available_indices, size=remaining_needed, replace=False)
                        mask[random_indices] = 1.0

            # Create encoder/predictor views for TIME masking
            mask_encoder.append((1 - mask).nonzero().squeeze())
            mask_predictor.append(mask.nonzero().squeeze())

        mask_encoder = torch.utils.data.default_collate(mask_encoder)
        mask_predictor = torch.utils.data.default_collate(mask_predictor)
        
        # Return the PHYSICALLY MODIFIED batch (for lead masking) 
        # and the INDICES (for beat masking)
        return masked_batch, mask_encoder, mask_predictor