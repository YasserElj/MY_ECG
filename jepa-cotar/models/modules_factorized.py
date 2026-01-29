"""
Factorized Spatio-Temporal Attention Modules for ECG Signals.

This module implements factorized attention that separates temporal (time-series)
and spatial (lead relationships) processing, which is more aligned with the
physiological structure of ECG signals.
"""

import torch
from torch import nn

from models.modules import Attention, MLP, LayerScale


class FactorizedBlock(nn.Module):
    """
    Factorized Block that splits attention into Temporal and Spatial passes.
    
    This block processes ECG signals by:
    1. Temporal Attention: Each lead processes its own time-series independently
       (learning morphology patterns)
    2. Spatial Attention: At each time step, leads attend to each other
       (learning electrical axis/geometry)
    
    Input Shape: (Batch, Leads, Time, Dim)
    Output Shape: (Batch, Leads, Time, Dim)
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        num_leads=12,
        mlp_ratio=4.,
        qkv_bias=False,
        bias=True,
        dropout=0.,
        attn_dropout=0.,
        eps=1e-6,
        layer_scale_eps=0.,
        use_sdp_kernel=True,
    ):
        super().__init__()
        
        # --- Temporal Branch (Process each lead independently) ---
        # We KEEP use_sdp_kernel here because Sequence Length is large (~200)
        self.norm1_time = nn.LayerNorm(dim, eps=eps, bias=bias)
        self.attn_time = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, bias=bias,
            attn_dropout=attn_dropout, proj_dropout=dropout, use_sdp_kernel=use_sdp_kernel
        )
        self.ls_time = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
        
        # --- Spatial Branch (Process relationship between leads) ---
        # CRITICAL FIX: Force use_sdp_kernel=False here.
        # The effective batch size here is (Batch * Time), which can exceed 200k.
        # This crashes SDPA kernels. Since Seq Len is only 12, standard attention is fine.
        self.norm1_space = nn.LayerNorm(dim, eps=eps, bias=bias)
        self.attn_space = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, bias=bias,
            attn_dropout=attn_dropout, proj_dropout=dropout, use_sdp_kernel=False
        )
        self.ls_space = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
        
        # --- MLP (Shared) ---
        self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), dropout=dropout, bias=bias)
        self.ls_mlp = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
    
    def forward(self, x):
        """
        Forward pass through factorized attention.
        
        Args:
            x: Input tensor of shape (B, L, T, D)
               B = batch size
               L = number of leads
               T = number of time patches
               D = embedding dimension
        
        Returns:
            Output tensor of shape (B, L, T, D)
        """
        # x shape: (B, Leads, Time, Dim)
        B, L, T, D = x.shape
        
        # 1. Temporal Attention
        # Merge Batch and Leads: (B*L, T, D)
        # FIX: Use reshape() instead of view() to handle non-contiguous inputs from previous blocks
        x_time = x.reshape(B * L, T, D)
        x_time = x_time + self.ls_time(self.attn_time(self.norm1_time(x_time)))
        x = x_time.reshape(B, L, T, D)
        
        # 2. Spatial Attention
        # Permute to make Leads the sequence dimension: (B, T, L, D) -> (B*T, L, D)
        x_space = x.permute(0, 2, 1, 3).reshape(B * T, L, D)
        x_space = x_space + self.ls_space(self.attn_space(self.norm1_space(x_space)))
        # Reshape back: (B*T, L, D) -> (B, T, L, D) -> (B, L, T, D)
        x = x_space.reshape(B, T, L, D).permute(0, 2, 1, 3)
        
        # 3. MLP
        x = x + self.ls_mlp(self.mlp(self.norm2(x)))
        
        return x

