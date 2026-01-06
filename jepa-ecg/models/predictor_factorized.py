"""
Factorized Predictor for ECG JEPA.

This predictor mirrors the FactorizedViT encoder, using the same spatio-temporal
structure to predict masked patches across all leads.
"""

import math

import torch
from torch import nn

import configs
from models.modules_factorized import FactorizedBlock
from models.utils import get_1d_pos_embed
from models.utils_factorized import apply_mask_factorized


class FactorizedPredictor(nn.Module):
    """
    Factorized Predictor for ECG signals.
    
    This predictor uses the same spatio-temporal structure as FactorizedViT,
    processing data using FactorizedBlocks to maintain consistency with the encoder.
    """
    
    def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
        super().__init__()
        self.config = config
        self.num_leads = len(config.channels)
        num_time_patches = config.channel_size // config.patch_size

        self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.pred_dim))

        # --- Positional Embeddings (Time + Space) ---
        self.register_buffer(
            'pos_embed_time',
            get_1d_pos_embed(dim=config.pred_dim, num_patches=num_time_patches),
            persistent=False
        )
        self.pos_embed_space = nn.Parameter(torch.zeros(1, self.num_leads, 1, config.pred_dim))
        nn.init.trunc_normal_(self.pos_embed_space, std=0.02)

        # --- Blocks ---
        # We use FactorizedBlocks here too for efficiency and consistency
        self.blocks = nn.ModuleList([
            FactorizedBlock(
                dim=config.pred_dim,
                num_heads=config.pred_num_heads,
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
            for _ in range(config.pred_depth)
        ])
        
        self.norm = nn.LayerNorm(config.pred_dim, eps=config.norm_eps, bias=config.bias)
        self.proj = nn.Linear(config.pred_dim, config.dim, bias=config.bias)

        self.apply(self._init_weights)
        # Initialize mask token separately
        nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if m.weight.shape[0] == self.config.pred_dim and m.weight.shape[1] == self.config.dim:
                # Projection layer (pred_dim -> dim)
                nn.init.trunc_normal_(m.weight, mean=0., std=0.02 / math.sqrt(2 * self.config.pred_depth))
            else:
                nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            # FIX: Check if bias exists before zeroing
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x, mask_encoder, mask_predictor):
        """
        Forward pass through factorized predictor.
        
        Args:
            x: (B, L*T_kept, D) - Encoder output (flattened)
            mask_encoder: (B, T_kept) - Time indices seen by encoder
            mask_predictor: (B, K) - Time indices to predict
        
        Returns:
            z_pred: (B, L*K, D) - Predictions for all leads at masked time steps
        """
        B = x.shape[0]
        K = mask_predictor.shape[1]
        T_kept = mask_encoder.shape[1]
        
        # 1. Prepare Positional Embeddings
        # Combine Time + Space: (1, L, T, D)
        # pos_embed_time: (1, T, D) -> (1, 1, T, D)
        pos_embed = self.pos_embed_time.unsqueeze(1) + self.pos_embed_space
        # Flatten to (1, L*T, D) for easier helper usage
        pos_embed = pos_embed.flatten(1, 2).repeat(B, 1, 1)

        # 2. Get Pos Embeds for Encoder (kept) and Predictor (masked) parts
        # We use the helper to extract the correct Lead+Time positions
        pos_encoder = apply_mask_factorized(pos_embed, mask_encoder, self.num_leads)
        pos_predictor = apply_mask_factorized(pos_embed, mask_predictor, self.num_leads)

        # 3. Embed Context (Encoder output)
        x = self.embed(x)  # (B, L*T_kept, pred_dim)
        x = x + pos_encoder

        # 4. Prepare Mask Tokens (Targets)
        # We need L*K mask tokens
        mask_token = self.mask_token.repeat(B, self.num_leads * K, 1)
        mask_token = mask_token + pos_predictor

        # 5. Concatenate
        # x: (B, L*(T_kept + K), pred_dim)
        x = torch.cat([x, mask_token], dim=1)

        # 6. Apply Blocks
        # FactorizedBlock expects (B, L, T, D)
        # We assume T_kept + K = T (Total Time)
        total_time = T_kept + K
        x = x.reshape(B, self.num_leads, total_time, -1)
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        # 7. Extract Predictions
        # The mask tokens were appended at the end.
        # We need the last L*K tokens.
        x_flat = x.flatten(1, 2)
        num_preds = self.num_leads * K
        z_pred = x_flat[:, -num_preds:, :]
        
        z_pred = self.proj(z_pred)
        return z_pred

