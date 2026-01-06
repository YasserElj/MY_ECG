"""
Utility functions for factorized (spatio-temporal) ECG processing.

These functions handle masking and indexing operations that respect
the 2D structure of ECG signals (Leads × Time).
"""

import torch


def apply_mask_factorized(x, mask_indices, num_leads):
    """
    Applies a time-step mask to a factorized (Leads*Time) sequence.
    
    This function takes a mask meant for time (e.g., "Mask Time Step 5") 
    and applies it to *all* leads (Mask Lead I at Step 5, Lead II at Step 5, etc.).
    
    Args:
        x: (B, L*T, D) - The flattened factorized sequence
        mask_indices: (B, K) - Indices of time steps to keep/mask (0..T-1)
        num_leads: int (L) - Number of leads
    
    Returns:
        x_masked: (B, L*K, D) - The gathered tokens for all leads at mask_indices
    """
    B, total_tokens, D = x.size()
    # mask_indices contains time indices [0, T). 
    # We need to convert these to flat indices [0, L*T).
    # Flat Index = (Lead_Index * T) + Time_Index
    
    T = total_tokens // num_leads
    K = mask_indices.size(1)
    
    # x is currently (B, L*T, D). Let's view as (B, L, T, D)
    x = x.reshape(B, num_leads, T, D)
    
    # Expand mask to all leads: (B, 1, K, 1) -> (B, L, K, D)
    mask_expanded = mask_indices.reshape(B, 1, K, 1).expand(-1, num_leads, -1, D)
    
    # Gather: (B, L, K, D)
    x_gathered = torch.gather(x, 2, mask_expanded)
    
    # Flatten back: (B, L*K, D)
    return x_gathered.flatten(1, 2)


