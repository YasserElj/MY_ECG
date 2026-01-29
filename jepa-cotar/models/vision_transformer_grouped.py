"""
Grouped Vision Transformer for ECG

Groups 3 leads together at the patch level, reducing tokens from 2400 to 800.
This forces early lead fusion and is more efficient for training.
"""

import math

import torch
from torch import nn

import configs
from models.modules import GroupedPatchEmbedding, Block, CoTARBlock
from models.utils import apply_mask
from models.utils_factorized import apply_mask_2d


def get_2d_pos_embed(dim, num_groups, num_patches):
  """
  Generate 2D positional embeddings for grouped patches.
  
  Uses separate embeddings for group position and time position,
  then combines them.
  
  Args:
    dim: embedding dimension
    num_groups: number of lead groups (4)
    num_patches: number of time patches (200)
  
  Returns:
    (1, num_groups * num_patches, dim) positional embeddings
  """
  # Group positional embeddings (learnable-style sinusoidal)
  group_embed = torch.zeros(num_groups, dim // 2)
  position = torch.arange(num_groups).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim // 2, 2) * (-math.log(10000.0) / (dim // 2)))
  group_embed[:, 0::2] = torch.sin(position * div_term)
  group_embed[:, 1::2] = torch.cos(position * div_term)
  
  # Time positional embeddings
  time_embed = torch.zeros(num_patches, dim // 2)
  position = torch.arange(num_patches).unsqueeze(1)
  div_term = torch.exp(torch.arange(0, dim // 2, 2) * (-math.log(10000.0) / (dim // 2)))
  time_embed[:, 0::2] = torch.sin(position * div_term)
  time_embed[:, 1::2] = torch.cos(position * div_term)
  
  # Combine: (num_groups, num_patches, dim)
  # Each position (g, t) gets [group_embed[g], time_embed[t]]
  pos_embed = torch.zeros(num_groups, num_patches, dim)
  for g in range(num_groups):
    for t in range(num_patches):
      pos_embed[g, t, :dim//2] = group_embed[g]
      pos_embed[g, t, dim//2:] = time_embed[t]
  
  # Flatten to (1, G*T, dim)
  pos_embed = pos_embed.reshape(1, num_groups * num_patches, dim)
  
  return pos_embed


class GroupedViT(nn.Module):
  """
  Vision Transformer with grouped lead patches.
  
  Instead of processing 12 leads independently and getting 2400 tokens,
  this groups 3 leads together to get 800 tokens (4 groups × 200 time patches).
  
  This is more efficient and forces early cross-lead feature learning.
  """
  
  def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.keep_registers = keep_registers
    
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    self.num_patches = num_patches
    self.num_groups = 4  # Fixed: 4 groups of 3 leads each
    
    # Grouped patch embedding (12 leads -> 4 groups)
    self.patch_embed = GroupedPatchEmbedding(
      dim=config.dim,
      patch_size=config.patch_size,
      leads_per_group=3,
      bias=config.bias)
    
    # 2D positional embeddings (group + time)
    self.register_buffer(
      'pos_embed',
      get_2d_pos_embed(
        dim=config.dim,
        num_groups=self.num_groups,
        num_patches=num_patches),
      persistent=False)
    
    # Optional register tokens
    if config.num_registers > 0:
      self.registers = nn.Parameter(torch.empty(1, config.num_registers, config.dim))
      nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
    
    # Transformer blocks - conditionally use CoTAR or standard attention
    attention_type = getattr(config, 'attention_type', 'attention')
    use_cotar = attention_type == 'cotar'
    
    if use_cotar:
      self.blocks = nn.ModuleList([
        CoTARBlock(
          dim=config.dim,
          mlp_ratio=config.mlp_ratio,
          bias=config.bias,
          dropout=config.dropout,
          eps=config.norm_eps,
          layer_scale_eps=config.layer_scale_eps)
        for _ in range(config.depth)
      ])
    else:
      self.blocks = nn.ModuleList([
        Block(
          dim=config.dim,
          num_heads=config.num_heads,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          bias=config.bias,
          dropout=config.dropout,
          attn_dropout=config.attn_dropout,
          eps=config.norm_eps,
          layer_scale_eps=config.layer_scale_eps,
          use_sdp_kernel=use_sdp_kernel)
        for _ in range(config.depth)
      ])
    
    self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)
    
    # Initialize weights
    self._init_weights()
  
  def _init_weights(self):
    for name, module in self.named_modules():
      if isinstance(module, (nn.Linear, nn.Conv1d)):
        # Output projections get reduced init std for better training stability
        if name.endswith('mlp.fc2') or name.endswith('attn.proj') or name.endswith('cotar.lin4'):
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * self.config.depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
  
  def forward(self, x, mask=None):
    """
    Args:
      x: (B, 12, signal_length) - raw 12-lead ECG
      mask: (B, K) - indices of tokens to keep (optional)
    
    Returns:
      (B, N, D) - encoded representations
      where N = num_groups * num_patches = 800 (or K if masked)
    """
    # Embed: (B, 800, D)
    x = self.patch_embed(x)
    B, N, D = x.size()
    
    # Add positional embeddings
    x = x + self.pos_embed[:, :N]
    
    # Apply masking if provided
    if mask is not None:
      # Detect 2D mask (B, K, 2) vs 1D mask (B, K)
      if mask.dim() == 3 and mask.size(-1) == 2:
        # 2D mask: (group_idx, time_idx) pairs
        num_groups = 4
        num_time = self.config.num_patches
        x = apply_mask_2d(x, mask, num_groups, num_time)
      else:
        # 1D mask: time indices only
        x = apply_mask(x, mask)
    
    # Add register tokens
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    
    # Transformer blocks
    for block in self.blocks:
      x = block(x)
    
    # Final norm
    x = self.norm(x)
    
    # Remove registers if not keeping them
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    
    return x
  
  @property
  def total_tokens(self):
    """Total number of tokens (without registers)."""
    return self.num_groups * self.num_patches


