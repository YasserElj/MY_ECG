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
               S = signal length (can vary, e.g., 5000 for full or 1250 for cropped)
            mask: Optional mask tensor:
                  - Shape (B, K) for 1D masking: time indices to keep (same for all leads)
                  - Shape (B, K, 2) for 2D masking: (lead_idx, time_idx) pairs to keep
                  If None, all patches are kept
        
        Returns:
            Output tensor of shape (B, N, D) where N is the number of kept patches
            (flattened from L × T structure)
        """
        # Input x: (B, 12, S) where S can vary
        B, L, S = x.shape
        
        # --- Embedding ---
        # Merge Batch and Lead: (B*L, 1, S)
        x = x.reshape(B * L, 1, S)
        x = self.patch_embed(x)  # -> (B*L, Dim, T_actual)
        x = x.transpose(1, 2)     # -> (B*L, T_actual, Dim)
        
        # Compute actual number of time patches from the embedding output
        T_actual = x.shape[1]
        D = x.shape[2]
        
        # Reshape to (B, L, T_actual, D)
        x = x.reshape(B, L, T_actual, D)
        
        # --- Add Positional Embeddings ---
        # Time (broadcast across leads) - interpolate if input size differs
        # pos_embed_time shape: (1, T_train, D)
        pos_embed_time = self.pos_embed_time
        if T_actual != self.num_time_patches:
            # Interpolate positional embeddings to match actual input size
            # (1, T, D) -> (1, D, T) for interpolate -> (1, D, T_actual) -> (1, T_actual, D)
            pos_embed_time = pos_embed_time.permute(0, 2, 1)  # (1, D, T_train)
            pos_embed_time = torch.nn.functional.interpolate(
                pos_embed_time, size=T_actual, mode='linear', align_corners=False)
            pos_embed_time = pos_embed_time.permute(0, 2, 1)  # (1, T_actual, D)
        # pos_embed_time: (1, T, D) -> unsqueeze to (1, 1, T, D) for broadcasting with x: (B, L, T, D)
        x = x + pos_embed_time.unsqueeze(1)
        # Space (broadcast across time)
        x = x + self.pos_embed_space
        
        # --- Masking ---
        if mask is not None:
            is_2d_mask = mask.dim() == 3  # (B, K, 2) for 2D masking
            
            if is_2d_mask:
                # 2D masking: mask contains (lead_idx, time_idx) pairs
                # Flatten x to (B, L*T, D) and gather specific positions
                x_flat = x.flatten(1, 2)  # (B, L*T, D)
                
                # Convert 2D indices to flat indices: flat = lead * T + time
                lead_indices = mask[:, :, 0]  # (B, K)
                time_indices = mask[:, :, 1]  # (B, K)
                flat_indices = lead_indices * T_actual + time_indices  # (B, K)
                
                # Expand for gathering
                flat_indices = flat_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, K, D)
                x = torch.gather(x_flat, 1, flat_indices)  # (B, K, D)
                
                # For 2D masking, we can't use standard factorized blocks
                # because we don't have a regular (L, T) grid anymore.
                # We'll process as a flat sequence through the blocks.
                # Reshape back to a pseudo-grid for block processing
                # Since positions are irregular, we'll use (B, K, D) directly
                # But factorized blocks expect (B, L, T, D), so we need to handle this.
                # For now, skip factorized attention and use flat attention.
                # This is a simplification - proper implementation would need
                # sparse attention or different block structure.
                
                # Process as flat sequence through blocks
                # Blocks expect (B, L, T, D), so we fake L=1, T=K
                x = x.unsqueeze(1)  # (B, 1, K, D)
                for block in self.blocks:
                    x = block(x)
                x = self.norm(x)
                x = x.squeeze(1)  # (B, K, D)
            else:
                # 1D masking: mask is (B, T_kept), same mask for all leads
                mask_indices = mask.reshape(B, 1, -1, 1).expand(-1, L, -1, x.size(-1))
                x = torch.gather(x, 2, mask_indices)
                
                # --- Blocks ---
                for block in self.blocks:
                    x = block(x)
                
                x = self.norm(x)
                # Flatten to (B, N_tokens, D)
                x = x.flatten(1, 2)
        else:
            # No masking - process all tokens
            for block in self.blocks:
                x = block(x)
            
            x = self.norm(x)
            x = x.flatten(1, 2)
        
        return x

