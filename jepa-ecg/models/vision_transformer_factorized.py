"""
Factorized Vision Transformer for ECG Signals.

This module implements a Vision Transformer that treats ECG signals as a
spatio-temporal surface (Time × Leads), using factorized attention to separately
process temporal patterns (within each lead) and spatial relationships (between leads).
"""

import math

import torch
from torch import nn

import configs
from models.modules_factorized import FactorizedBlock
from models.utils import get_1d_pos_embed


class FactorizedViT(nn.Module):
    """
    Factorized Vision Transformer for ECG signals.
    
    Key differences from standard ViT:
    1. Lead-Independent Embedding: Each lead is embedded separately
    2. Factorized Attention: Temporal and Spatial attention are separated
    3. Spatio-Temporal Positional Embeddings: Separate embeddings for time and leads
    
    This architecture is motivated by the physiological structure of ECG signals,
    where leads represent distinct spatial viewpoints of the heart's electrical activity.
    """
    
    def __init__(self, config: configs.pretrain.Config, keep_registers=False, use_sdp_kernel=True):
        super().__init__()
        self.config = config
        self.keep_registers = keep_registers
        self.num_leads = len(config.channels)
        
        assert config.channel_size % config.patch_size == 0
        self.num_time_patches = config.channel_size // config.patch_size
        
        # 1. Lead-Independent Embedding
        # In-channels = 1 (Shared weights across leads)
        self.patch_embed = nn.Conv1d(
            1, config.dim, 
            kernel_size=config.patch_size, stride=config.patch_size, bias=config.bias
        )
        
        # 2. Positional Embeddings
        # Time (Sinusoidal, fixed)
        self.register_buffer('pos_embed_time', get_1d_pos_embed(config.dim, self.num_time_patches), persistent=False)
        # Space (Learnable, per-lead)
        self.pos_embed_space = nn.Parameter(torch.zeros(1, self.num_leads, 1, config.dim))
        nn.init.trunc_normal_(self.pos_embed_space, std=0.02)
        
        # Factorized Blocks
        self.blocks = nn.ModuleList([
            FactorizedBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                num_leads=self.num_leads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
                bias=config.bias,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                eps=config.norm_eps,
                layer_scale_eps=config.layer_scale_eps,
                use_sdp_kernel=use_sdp_kernel
            )
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.dim, eps=config.norm_eps, bias=config.bias)
        
        # Init weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: 
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask=None):
        """
        Forward pass through factorized ViT.
        
        Args:
            x: Input tensor of shape (B, L, S)
               B = batch size
               L = number of leads (typically 12)
               S = signal length (typically 5000)
            mask: Optional mask tensor of shape (B, K) with indices of patches to keep
                  If None, all patches are kept
        
        Returns:
            Output tensor of shape (B, N, D) where N is the number of kept patches
            (flattened from L × T structure)
        """
        # Input x: (B, 12, 5000)
        B, L, S = x.shape
        
        # --- Embedding ---
        # Merge Batch and Lead: (B*L, 1, 5000)
        x = x.reshape(B * L, 1, S)
        x = self.patch_embed(x)  # -> (B*L, Dim, TimePatches)
        x = x.transpose(1, 2)     # -> (B*L, TimePatches, Dim)
        
        # Reshape to (B, L, T, D)
        # FIX: Use reshape() to safe-guard against non-contiguous memory from transpose
        x = x.reshape(B, L, self.num_time_patches, -1)
        
        # --- Add Positional Embeddings ---
        # Time (broadcast across leads)
        x = x + self.pos_embed_time.unsqueeze(1) 
        # Space (broadcast across time)
        x = x + self.pos_embed_space
        
        # --- Masking ---
        if mask is not None:
            # mask: (B, T_kept) -> (B, 1, T_kept, 1) -> expanded to (B, L, T_kept, D)
            mask_indices = mask.reshape(B, 1, -1, 1).expand(-1, L, -1, x.size(-1))
            # Gather the kept time steps
            x = torch.gather(x, 2, mask_indices)
        
        # --- Blocks ---
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Output: Flatten to (B, N_tokens, D) for the predictor
        # Predictor expects standard sequence
        x = x.flatten(1, 2) 
        return x

