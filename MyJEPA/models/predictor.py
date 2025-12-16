"""Predictor module for MyJEPA - adapted from ECG-JEPA."""

import math
import torch
from torch import nn

from models.modules import Block
from models.utils import get_1d_pos_embed, apply_mask


class Predictor(nn.Module):
    """
    Predictor for JEPA-style training.
    
    Takes context embeddings and predicts target embeddings at masked positions.
    Uses learnable mask tokens with positional embeddings.
    """
    
    def __init__(
        self,
        dim: int,
        pred_dim: int,
        pred_depth: int,
        pred_num_heads: int,
        num_patches: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        bias: bool = True,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        norm_eps: float = 1e-6,
        use_sdp_kernel: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.pred_dim = pred_dim
        self.depth = pred_depth
        
        # Embed context to predictor dimension
        self.embed = nn.Linear(dim, pred_dim, bias=bias)
        
        # Learnable mask token for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        
        # Position embeddings
        self.register_buffer(
            'pos_embed',
            get_1d_pos_embed(dim=pred_dim, num_patches=num_patches),
            persistent=False)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=pred_dim,
                num_heads=pred_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                bias=bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                eps=norm_eps,
                use_sdp_kernel=use_sdp_kernel)
            for _ in range(pred_depth)
        ])
        
        self.norm = nn.LayerNorm(pred_dim, eps=norm_eps, bias=bias)
        
        # Project back to encoder dimension
        self.proj = nn.Linear(pred_dim, dim, bias=bias)
        
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * self.depth))
                elif module.bias is not None:
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)
    
    def forward(self, x, mask_encoder, mask_predictor):
        """
        Forward pass.
        
        Args:
            x: Context embeddings from encoder, shape (B, K_ctx, dim)
            mask_encoder: Indices of context patches, shape (B, K_ctx)
            mask_predictor: Indices of target patches to predict, shape (B, K_tgt)
        
        Returns:
            Predicted embeddings for target positions, shape (B, K_tgt, dim)
        """
        B, K_tgt = mask_predictor.size()
        
        # Expand position embeddings for batch
        pos_embed = self.pos_embed.repeat(B, 1, 1)  # (B, N, pred_dim)
        
        # Get positional embeddings for context
        pos_encoder = apply_mask(pos_embed, mask_encoder)  # (B, K_ctx, pred_dim)
        
        # Embed context and add positional embeddings
        x = self.embed(x)  # (B, K_ctx, pred_dim)
        x = x + pos_encoder
        
        # Get positional embeddings for targets
        pos_predictor = apply_mask(pos_embed, mask_predictor)  # (B, K_tgt, pred_dim)
        
        # Create mask tokens with positional embeddings
        mask_token = self.mask_token.repeat(B, K_tgt, 1)  # (B, K_tgt, pred_dim)
        mask_token = mask_token + pos_predictor
        
        # Concatenate context and mask tokens
        x = torch.cat([x, mask_token], dim=1)  # (B, K_ctx + K_tgt, pred_dim)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Extract predictions for target positions (last K_tgt tokens)
        x = x[:, -K_tgt:]  # (B, K_tgt, pred_dim)
        
        # Project back to encoder dimension
        x = self.proj(x)  # (B, K_tgt, dim)
        
        return x

