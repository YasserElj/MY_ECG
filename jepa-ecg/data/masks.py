import math

import numpy as np
import torch
import torch.utils.data


class MaskCollator:
  def __init__(self, patch_size, min_block_size, min_keep_ratio, max_keep_ratio):
    self.patch_size = patch_size
    self.min_block_size = min_block_size
    self.min_keep_ratio = min_keep_ratio
    self.max_keep_ratio = max_keep_ratio

  def __call__(self, batch):
    batch = torch.utils.data.default_collate(batch)
    batch_size, num_channels, channel_size = batch.size()
    assert channel_size % self.patch_size == 0
    num_patches = channel_size // self.patch_size
    keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    mask_encoder, mask_predictor = [], []
    for _ in range(batch_size):
      mask = self.sample_mask(num_keep, num_patches)
      mask_encoder.append(mask.nonzero().squeeze())  # patches to keep
      mask_predictor.append((1 - mask).nonzero().squeeze())  # patches to mask
    mask_encoder = torch.utils.data.default_collate(mask_encoder)
    mask_predictor = torch.utils.data.default_collate(mask_predictor)
    return batch, mask_encoder, mask_predictor

  def sample_mask(self, num_keep, num_patches):  # number of patches to keep (i.e., to not mask)
    # intervals that represent unmasked patches in the mask
    patch_intervals = [(0, num_patches)]
    num_mask = num_patches - num_keep
    total_mask_size = 0  # total number of all masked patches
    while total_mask_size < num_mask:
      interval_sizes = np.diff(patch_intervals).flatten()
      # select a random interval for masking
      index = np.random.choice(len(patch_intervals), p=interval_sizes / interval_sizes.sum())
      start, end = patch_intervals.pop(index)
      interval_size = end - start
      # select a number of consecutive patches to mask, i.e., create a block
      max_block_size = num_mask - total_mask_size
      if max_block_size >= self.min_block_size:
        block_size = np.random.randint(self.min_block_size, max_block_size + 1)
      else:
        block_size = max_block_size
      if interval_size <= block_size:
        # mask entire interval because it is so small, i.e., attach this block to another block
        total_mask_size += interval_size
      else:
        if max_block_size >= self.min_block_size:
          split = np.random.randint(start, end - block_size + 1)  # randomly position this block
        else:
          # this remaining block is too small to be on its own, so attach it to another block
          attach_choices = []
          if start > 0:
            attach_choices.append(start)
          if end < num_patches:
            attach_choices.append(end - block_size)
          split = np.random.choice(attach_choices)
        # split the interval and make place for this new block
        patch_intervals.append((start, split))
        patch_intervals.append((split + block_size, end))
        total_mask_size += block_size
    total_remaining_patches = np.diff(patch_intervals).sum()
    assert total_mask_size + total_remaining_patches == num_patches
    # create the binary mask from blocks and remaining patch intervals
    mask = torch.zeros(num_patches)
    for start, end in patch_intervals:
      mask[start:end] = 1.
    return mask


class MaskCollator2D:
  """
  2D Block masking for factorized ECG.
  Masks rectangular blocks in (lead, time) space.
  
  Returns masks as (lead_idx, time_idx) pairs instead of just time indices.
  """
  
  # Standard 12-lead ECG order
  LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
  
  def __init__(self, 
               patch_size,
               num_leads=12,
               min_keep_ratio=0.15,
               max_keep_ratio=0.25,
               min_block_leads=2,
               max_block_leads=6,
               min_block_time=10,
               max_block_time=40):
    self.patch_size = patch_size
    self.num_leads = num_leads
    self.min_keep_ratio = min_keep_ratio
    self.max_keep_ratio = max_keep_ratio
    self.min_block_leads = min_block_leads
    self.max_block_leads = max_block_leads
    self.min_block_time = min_block_time
    self.max_block_time = max_block_time

  def __call__(self, batch):
    batch = torch.utils.data.default_collate(batch)
    batch_size, num_channels, channel_size = batch.size()
    assert channel_size % self.patch_size == 0
    num_time_patches = channel_size // self.patch_size
    
    mask_encoder_list, mask_predictor_list = [], []
    
    for _ in range(batch_size):
      # Create 2D mask grid: (num_leads, num_time_patches)
      # 1 = keep (context), 0 = mask (target)
      mask_2d = torch.ones(self.num_leads, num_time_patches)
      
      # Sample random 2D blocks to mask
      keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
      target_mask_ratio = 1.0 - keep_ratio
      current_mask_ratio = 0.0
      
      attempts = 0
      while current_mask_ratio < target_mask_ratio and attempts < 50:
        # Random block dimensions
        block_leads = np.random.randint(self.min_block_leads, 
                                        min(self.max_block_leads + 1, self.num_leads + 1))
        block_time = np.random.randint(self.min_block_time, 
                                       min(self.max_block_time + 1, num_time_patches + 1))
        
        # Random block position
        lead_start = np.random.randint(0, self.num_leads - block_leads + 1)
        time_start = np.random.randint(0, num_time_patches - block_time + 1)
        
        # Mask this block (set to 0)
        mask_2d[lead_start:lead_start+block_leads, 
                time_start:time_start+block_time] = 0
        
        current_mask_ratio = 1.0 - mask_2d.mean().item()
        attempts += 1
      
      # Convert 2D mask to (lead, time) index pairs
      # mask_encoder: indices of KEPT (lead, time) pairs - where mask_2d == 1
      # mask_predictor: indices of MASKED (lead, time) pairs - where mask_2d == 0
      kept_indices = mask_2d.nonzero(as_tuple=False)       # (N_kept, 2)
      masked_indices = (1 - mask_2d).nonzero(as_tuple=False)  # (N_masked, 2)
      
      mask_encoder_list.append(kept_indices)
      mask_predictor_list.append(masked_indices)
    
    # Pad to same length (different samples may have different mask sizes)
    max_kept = max(m.size(0) for m in mask_encoder_list)
    max_masked = max(m.size(0) for m in mask_predictor_list)
    
    # Create padded tensors
    mask_encoder = torch.zeros(batch_size, max_kept, 2, dtype=torch.long)
    mask_predictor = torch.zeros(batch_size, max_masked, 2, dtype=torch.long)
    mask_encoder_lengths = torch.zeros(batch_size, dtype=torch.long)
    mask_predictor_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (enc, pred) in enumerate(zip(mask_encoder_list, mask_predictor_list)):
      mask_encoder[i, :enc.size(0)] = enc
      mask_predictor[i, :pred.size(0)] = pred
      mask_encoder_lengths[i] = enc.size(0)
      mask_predictor_lengths[i] = pred.size(0)
    
    return batch, (mask_encoder, mask_encoder_lengths), (mask_predictor, mask_predictor_lengths)


class MaskCollator2DLeadGroup:
  """
  2D Block masking with clinically meaningful lead groups.
  
  Lead Groups (based on cardiac anatomy):
  - Inferior: II, III, aVF (right coronary artery territory)
  - Lateral: I, aVL, V5, V6 (circumflex artery territory)
  - Anterior: V1, V2, V3, V4 (LAD territory)
  - Septal: V1, V2
  - High Lateral: I, aVL
  - Low Lateral: V5, V6
  
  This is more interpretable and clinically motivated than random blocks.
  """
  
  LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
  
  # Lead indices for each clinical group
  LEAD_GROUPS = {
    'inferior': [1, 2, 5],       # II, III, aVF
    'lateral': [0, 4, 10, 11],   # I, aVL, V5, V6
    'anterior': [6, 7, 8, 9],    # V1, V2, V3, V4
    'septal': [6, 7],            # V1, V2
    'high_lateral': [0, 4],      # I, aVL
    'low_lateral': [10, 11],     # V5, V6
    'precordial': [6, 7, 8, 9, 10, 11],  # V1-V6
    'limb': [0, 1, 2, 3, 4, 5],  # I, II, III, aVR, aVL, aVF
  }
  
  def __init__(self,
               patch_size,
               num_leads=12,
               min_keep_ratio=0.15,
               max_keep_ratio=0.25,
               min_block_time=15,
               max_block_time=50):
    self.patch_size = patch_size
    self.num_leads = num_leads
    self.min_keep_ratio = min_keep_ratio
    self.max_keep_ratio = max_keep_ratio
    self.min_block_time = min_block_time
    self.max_block_time = max_block_time
    self.lead_group_names = list(self.LEAD_GROUPS.keys())

  def __call__(self, batch):
    batch = torch.utils.data.default_collate(batch)
    batch_size, num_channels, channel_size = batch.size()
    assert channel_size % self.patch_size == 0
    num_time_patches = channel_size // self.patch_size
    
    mask_encoder_list, mask_predictor_list = [], []
    
    for _ in range(batch_size):
      # Create 2D mask grid: (num_leads, num_time_patches)
      mask_2d = torch.ones(self.num_leads, num_time_patches)
      
      keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
      target_mask_ratio = 1.0 - keep_ratio
      current_mask_ratio = 0.0
      
      attempts = 0
      while current_mask_ratio < target_mask_ratio and attempts < 20:
        # Pick a random lead group
        group_name = np.random.choice(self.lead_group_names)
        lead_indices = self.LEAD_GROUPS[group_name]
        
        # Pick a random time block
        block_time = np.random.randint(self.min_block_time, 
                                       min(self.max_block_time + 1, num_time_patches + 1))
        time_start = np.random.randint(0, num_time_patches - block_time + 1)
        
        # Mask this lead group at this time range
        for lead_idx in lead_indices:
          mask_2d[lead_idx, time_start:time_start+block_time] = 0
        
        current_mask_ratio = 1.0 - mask_2d.mean().item()
        attempts += 1
      
      # Convert to index pairs
      kept_indices = mask_2d.nonzero(as_tuple=False)
      masked_indices = (1 - mask_2d).nonzero(as_tuple=False)
      
      mask_encoder_list.append(kept_indices)
      mask_predictor_list.append(masked_indices)
    
    # Pad to same length
    max_kept = max(m.size(0) for m in mask_encoder_list)
    max_masked = max(m.size(0) for m in mask_predictor_list)
    
    mask_encoder = torch.zeros(batch_size, max_kept, 2, dtype=torch.long)
    mask_predictor = torch.zeros(batch_size, max_masked, 2, dtype=torch.long)
    mask_encoder_lengths = torch.zeros(batch_size, dtype=torch.long)
    mask_predictor_lengths = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (enc, pred) in enumerate(zip(mask_encoder_list, mask_predictor_list)):
      mask_encoder[i, :enc.size(0)] = enc
      mask_predictor[i, :pred.size(0)] = pred
      mask_encoder_lengths[i] = enc.size(0)
      mask_predictor_lengths[i] = pred.size(0)
    
    return batch, (mask_encoder, mask_encoder_lengths), (mask_predictor, mask_predictor_lengths)


if __name__ == '__main__':  # visualize masks
  import matplotlib.patches as patches
  import matplotlib.pyplot as plt

  def draw_mask(mask, gap):
    N = len(mask)
    fig, ax = plt.subplots()
    fig.set_size_inches(N + (N - 1) * gap, 1)
    for i, unmasked in enumerate(mask):
      x = i * (1 + gap)  # patch position
      color = 'white' if unmasked else 'black'
      patch = patches.Rectangle((x, 0), 1, 1, edgecolor='black', linewidth=1, facecolor=color)
      ax.add_patch(patch)
    ax.set_aspect('equal')
    ax.set_xlim(0, N + (N - 1) * gap)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.show()
  num_patches = 36
  collate = MaskCollator(
    patch_size=1,
    min_block_size=3,
    min_keep_ratio=0.2,
    max_keep_ratio=0.5)
  for _ in range(10):
    keep_ratio = np.random.uniform(collate.min_keep_ratio, collate.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    mask = collate.sample_mask(num_keep, num_patches)
    draw_mask(mask, gap=0.1)