"""
Predictor for Grouped ViT architecture.

Operates on 4 groups × 200 time patches = 800 tokens.
Uses 2D positional embeddings (group + time).
"""

import math

import torch
from torch import nn

import configs
from models.modules import Block, CoTARBlock
from models.utils import apply_mask
from models.vision_transformer_grouped import get_2d_pos_embed


class GroupedPredictor(nn.Module):
  """
  Predictor for Grouped ViT JEPA.
  
  Takes encoded context tokens and predicts masked tokens.
  Works with 4 groups × 200 time patches = 800 tokens.
  """
  
  def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    
    assert config.channel_size % config.patch_size == 0
    num_patches = config.channel_size // config.patch_size
    num_groups = 4  # Fixed for grouped architecture
    
    self.num_patches = num_patches
    self.num_groups = num_groups
    
    # Project encoder output to predictor dim
    self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
    
    # Learnable mask token
    self.mask_token = nn.Parameter(torch.zeros(1, 1, config.pred_dim))
    
    # 2D positional embeddings (group + time)
    self.register_buffer(
      'pos_embed',
      get_2d_pos_embed(
        dim=config.pred_dim,
        num_groups=num_groups,
        num_patches=num_patches),
      persistent=False)
    
    # Transformer blocks - conditionally use CoTAR or standard attention
    attention_type = getattr(config, 'attention_type', 'attention')
    use_cotar = attention_type == 'cotar'
    
    if use_cotar:
      self.blocks = nn.ModuleList([
        CoTARBlock(
          dim=config.pred_dim,
          mlp_ratio=config.mlp_ratio,
          bias=config.bias,
          dropout=config.dropout,
          eps=config.norm_eps)
        for _ in range(config.pred_depth)
      ])
    else:
      self.blocks = nn.ModuleList([
        Block(
          dim=config.pred_dim,
          num_heads=config.pred_num_heads,
          mlp_ratio=config.mlp_ratio,
          qkv_bias=config.qkv_bias,
          bias=config.bias,
          dropout=config.dropout,
          attn_dropout=config.attn_dropout,
          eps=config.norm_eps,
          use_sdp_kernel=use_sdp_kernel)
        for _ in range(config.pred_depth)
      ])
    
    self.norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
    self.proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)
    
    # Initialize weights
    self._init_weights()
  
  def _init_weights(self):
    for name, module in self.named_modules():
      if isinstance(module, nn.Linear):
        # Output projections get reduced init std for better training stability
        if name.endswith('mlp.fc2') or name.endswith('attn.proj') or name.endswith('cotar.lin4'):
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * self.config.pred_depth))
        else:
          nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
      elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
    nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)
  
  def forward(self, x, mask_encoder, mask_predictor):
    """
    Args:
      x: (B, K_enc, D) - encoded context tokens from GroupedViT
      mask_encoder: (B, K_enc) - indices of kept tokens (time indices for 1D, or (B, K, 2) for 2D)
      mask_predictor: (B, K_pred) - indices of masked tokens (time indices for 1D, or (B, K, 2) for 2D)
    
    Returns:
      (B, K_pred, D) - predictions for masked tokens
    """
    B = x.shape[0]
    
    # Check if 1D or 2D masking
    # Note: DataLoader multiprocessing can convert tuples to lists
    is_2d_mask = isinstance(mask_predictor, (tuple, list)) or (mask_predictor.dim() == 3)
    
    if is_2d_mask:
      # 2D masking: mask_predictor is (indices, lengths) tuple/list or (B, K, 2) tensor
      if isinstance(mask_predictor, (tuple, list)):
        mask_pred_indices, mask_pred_lengths = mask_predictor
        mask_enc_indices, mask_enc_lengths = mask_encoder
      else:
        # Direct tensor (B, K, 2)
        mask_pred_indices = mask_predictor
        mask_enc_indices = mask_encoder
      
      K_pred = mask_pred_indices.shape[1]
      
      # Get positional embeddings for encoder and predictor positions
      pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, G*T, pred_dim)
      
      # Convert 2D indices (group, time) to flat indices
      enc_flat = mask_enc_indices[:, :, 0] * self.num_patches + mask_enc_indices[:, :, 1]  # (B, K_enc)
      pred_flat = mask_pred_indices[:, :, 0] * self.num_patches + mask_pred_indices[:, :, 1]  # (B, K_pred)
      
      # Gather positional embeddings
      pos_encoder = torch.gather(pos_embed, 1, enc_flat.unsqueeze(-1).expand(-1, -1, self.config.pred_dim))
      pos_predictor = torch.gather(pos_embed, 1, pred_flat.unsqueeze(-1).expand(-1, -1, self.config.pred_dim))
      
    else:
      # 1D masking: same time mask applied to all groups
      # mask_predictor is (B, K) time indices
      K_pred = mask_predictor.shape[1]
      
      pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, G*T, pred_dim)
      pos_encoder = apply_mask(pos_embed, mask_encoder)
      pos_predictor = apply_mask(pos_embed, mask_predictor)
    
    # Embed encoder output
    x = self.embed(x)  # (B, K_enc, pred_dim)
    x = x + pos_encoder
    
    # Create mask tokens with positional embeddings
    mask_token = self.mask_token.repeat(B, K_pred, 1)  # (B, K_pred, pred_dim)
    mask_token = mask_token + pos_predictor
    
    # Concatenate context and mask tokens
    x = torch.cat([x, mask_token], dim=1)  # (B, K_enc + K_pred, pred_dim)
    
    # Apply transformer blocks
    for block in self.blocks:
      x = block(x)
    
    x = self.norm(x)
    
    # Extract and project mask token predictions
    mask_token = x[:, -K_pred:]  # (B, K_pred, pred_dim)
    mask_token = self.proj(mask_token)  # (B, K_pred, D)
    
    return mask_token


