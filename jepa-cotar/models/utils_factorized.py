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


def apply_mask_2d(x, mask_indices_2d, num_leads, num_time):
    """
    Apply 2D mask to factorized sequence using (lead, time) index pairs.
    
    This function gathers specific (lead, time) positions from the flattened
    factorized sequence. Unlike apply_mask_factorized which applies the same
    time mask to all leads, this allows different masks per lead.
    
    Args:
        x: (B, L*T, D) - The flattened factorized sequence
        mask_indices_2d: (B, K, 2) - Indices as (lead_idx, time_idx) pairs
        num_leads: int (L) - Number of leads
        num_time: int (T) - Number of time patches
    
    Returns:
        x_masked: (B, K, D) - The gathered tokens at specified (lead, time) positions
    """
    B, total_tokens, D = x.size()
    K = mask_indices_2d.size(1)
    
    # Convert 2D indices (lead, time) to flat indices
    # Flat index = lead_idx * num_time + time_idx
    lead_idx = mask_indices_2d[:, :, 0]  # (B, K)
    time_idx = mask_indices_2d[:, :, 1]  # (B, K)
    flat_indices = lead_idx * num_time + time_idx  # (B, K)
    
    # Expand for gathering: (B, K, D)
    flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, -1, D)
    
    # Gather tokens at specified positions
    return torch.gather(x, 1, flat_indices_expanded)  # (B, K, D)


def apply_mask_2d_with_lengths(x, mask_indices_2d, mask_lengths, num_leads, num_time):
    """
    Apply 2D mask with variable lengths per sample.
    
    When different samples have different numbers of masked positions,
    the mask tensor is padded. This function handles the variable lengths.
    
    Args:
        x: (B, L*T, D) - The flattened factorized sequence
        mask_indices_2d: (B, K_max, 2) - Padded indices as (lead_idx, time_idx) pairs
        mask_lengths: (B,) - Actual number of valid indices per sample
        num_leads: int (L) - Number of leads
        num_time: int (T) - Number of time patches
    
    Returns:
        List of (K_i, D) tensors, one per sample (variable length)
    """
    B, total_tokens, D = x.size()
    
    results = []
    for b in range(B):
        K = mask_lengths[b].item()
        indices = mask_indices_2d[b, :K]  # (K, 2)
        lead_idx = indices[:, 0]  # (K,)
        time_idx = indices[:, 1]  # (K,)
        
        # Convert to flat indices
        flat_indices = lead_idx * num_time + time_idx  # (K,)
        
        # Gather: (K, D)
        flat_indices_expanded = flat_indices.unsqueeze(-1).expand(-1, D)
        gathered = torch.gather(x[b], 0, flat_indices_expanded)
        results.append(gathered)
    
    return results


