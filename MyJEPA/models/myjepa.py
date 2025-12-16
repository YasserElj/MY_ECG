"""
MyJEPA: Single-Encoder JEPA with SIGReg Regularization

Key differences from ECG-JEPA:
- Single encoder (no EMA target encoder)
- SIGReg regularization instead of EMA for collapse prevention
- Same masking and prediction task

Architecture:
1. Encoder: VisionTransformer that encodes ECG patches
2. Predictor: Transformer that predicts target embeddings from context
3. Projector: MLP that projects embeddings for SIGReg
4. SIGReg: Regularizes embeddings to follow isotropic Gaussian distribution

Loss = pred_loss + λ * sigreg_loss
"""

from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

from models.vit import VisionTransformer
from models.predictor import Predictor
from models.modules import Projector
from models.utils import apply_mask


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularization.
    
    Constrains embeddings to follow an isotropic Gaussian distribution
    using characteristic function comparison.
    """
    
    def __init__(self, num_slices: int = 1024, num_points: int = 17, t_max: float = 3.0):
        super().__init__()
        self.num_slices = num_slices
        
        # Integration points for characteristic function
        t = torch.linspace(0, t_max, num_points, dtype=torch.float32)
        dt = t_max / (num_points - 1)
        weights = torch.full((num_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        phi = torch.exp(-t.square() / 2.0)
        
        self.register_buffer("t", t)
        self.register_buffer("phi", phi)
        self.register_buffer("weights", weights * phi)
    
    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Compute SIGReg loss.
        
        Args:
            proj: Projected embeddings of shape (B, proj_dim)
        
        Returns:
            loss: Scalar SIGReg loss
        """
        N, D = proj.shape
        
        # Random projection directions
        A = torch.randn(D, self.num_slices, device=proj.device, dtype=proj.dtype)
        A = A / A.norm(p=2, dim=0)
        
        # Project embeddings onto random directions
        x_proj = proj @ A  # (N, num_slices)
        
        # Compute empirical characteristic function
        x_t = x_proj.unsqueeze(-1) * self.t  # (N, num_slices, num_points)
        
        cos_mean = x_t.cos().mean(dim=0)  # (num_slices, num_points)
        sin_mean = x_t.sin().mean(dim=0)
        
        # Error from target Gaussian CF
        err = (cos_mean - self.phi).square() + sin_mean.square()
        
        # Weighted integration
        statistic = (err @ self.weights) * N
        
        return statistic.mean()


class MyJEPA(nn.Module):
    """
    MyJEPA: Single-Encoder JEPA with SIGReg.
    
    Combines ECG-JEPA's masking + prediction with LeJEPA's single-encoder + SIGReg.
    No EMA, no stop-gradient - SIGReg prevents collapse.
    """
    
    def __init__(
        self,
        # Encoder config
        dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        num_channels: int = 12,
        channel_size: int = 5000,
        patch_size: int = 25,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        bias: bool = True,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        norm_eps: float = 1e-6,
        layer_scale_eps: float = 0.,
        num_registers: int = 1,
        # Predictor config
        pred_dim: int = 192,
        pred_depth: int = 8,
        pred_num_heads: int = 6,
        # SIGReg config
        proj_hidden_dim: int = 2048,
        proj_dim: int = 128,
        num_slices: int = 1024,
        sigreg_lambda: float = 0.02,
        # Other
        use_sdp_kernel: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.sigreg_lambda = sigreg_lambda
        
        num_patches = channel_size // patch_size
        
        # Single encoder (no EMA target encoder)
        self.encoder = VisionTransformer(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            num_channels=num_channels,
            channel_size=channel_size,
            patch_size=patch_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            bias=bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_eps=norm_eps,
            layer_scale_eps=layer_scale_eps,
            num_registers=num_registers,
            keep_registers=False,
            use_sdp_kernel=use_sdp_kernel
        )
        
        # Predictor (predicts target embeddings from context)
        self.predictor = Predictor(
            dim=dim,
            pred_dim=pred_dim,
            pred_depth=pred_depth,
            pred_num_heads=pred_num_heads,
            num_patches=num_patches,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            bias=bias,
            dropout=dropout,
            attn_dropout=attn_dropout,
            norm_eps=norm_eps,
            use_sdp_kernel=use_sdp_kernel
        )
        
        # Projector and SIGReg for collapse prevention
        self.projector = Projector(
            in_dim=dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim
        )
        self.sigreg = SIGReg(num_slices=num_slices)
    
    def forward(self, x, mask_encoder, mask_predictor):
        """
        Forward pass.
        
        Args:
            x: Input ECG tensor, shape (B, C, T)
            mask_encoder: Context patch indices, shape (B, K_ctx)
            mask_predictor: Target patch indices to predict, shape (B, K_tgt)
        
        Returns:
            total_loss: Combined prediction + SIGReg loss
            pred_loss: Prediction loss (smooth L1)
            sigreg_loss: SIGReg regularization loss
        """
        # Context encoding (only visible patches)
        z_ctx = self.encoder(x, mask_encoder)  # (B, K_ctx, dim)
        
        # Full encoding (all patches) - NO stop_gradient, learns normally
        z_full = self.encoder(x)  # (B, N, dim)
        
        # Extract target embeddings
        z_tgt = apply_mask(z_full, mask_predictor)  # (B, K_tgt, dim)
        
        # Predict target embeddings from context
        z_pred = self.predictor(z_ctx, mask_encoder, mask_predictor)  # (B, K_tgt, dim)
        
        # Prediction loss (smooth L1, same as ECG-JEPA uses L1)
        pred_loss = F.smooth_l1_loss(z_pred, z_tgt)
        
        # SIGReg on global representation (mean pooled full encoding)
        z_global = z_full.mean(dim=1)  # (B, dim)
        z_proj = self.projector(z_global)  # (B, proj_dim)
        sigreg_loss = self.sigreg(z_proj)
        
        # Combined loss
        total_loss = pred_loss + self.sigreg_lambda * sigreg_loss
        
        return total_loss, pred_loss, sigreg_loss
    
    def get_embeddings(self, x):
        """
        Get embeddings for inference (no masking).
        
        Args:
            x: Input ECG tensor, shape (B, C, T)
        
        Returns:
            Global embeddings, shape (B, dim)
        """
        z = self.encoder(x)
        return z.mean(dim=1)
    
    def get_optimizer(self, lr: float = 1e-3, weight_decay: float = 1e-2, fused: bool = False):
        """
        Get AdamW optimizer with separate weight decay handling.
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay for decay params
            fused: Use fused AdamW (faster on CUDA)
        
        Returns:
            AdamW optimizer
        """
        decay_modules = (nn.Linear, nn.Conv1d)
        decay = set()
        
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
                    full_param_name = f'{module_name}.{param_name}' if module_name else param_name
                    decay.add(full_param_name)
        
        decay_params, non_decay_params = OrderedDict(), OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in decay:
                    decay_params[name] = param
                else:
                    non_decay_params[name] = param
        
        param_groups = [
            {'params': list(decay_params.values()),
             'weight_decay': weight_decay,
             'use_weight_decay': True},
            {'params': list(non_decay_params.values()),
             'weight_decay': 0.,
             'use_weight_decay': False}
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-6,
            weight_decay=0.,
            fused=fused
        )
        
        return optimizer

