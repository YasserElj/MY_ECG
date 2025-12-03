"""
LeJEPA model for ECG self-supervised learning.

Key differences from I-JEPA/ECG-JEPA:
- No teacher-student architecture (no EMA)
- No masking - uses multi-view augmentation instead
- Uses SIGReg loss for regularization
- Single encoder with projector
"""

import torch
from torch import nn

from models.vit import VisionTransformer
from models.modules import Projector


class LeJEPA(nn.Module):
    """
    LeJEPA model for ECG representation learning.
    
    Uses multi-view augmentation and SIGReg regularization:
    - Invariance loss: different views of same ECG should have similar embeddings
    - SIGReg loss: embeddings should follow isotropic Gaussian distribution
    """
    
    def __init__(
        self,
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
        proj_hidden_dim: int = 2048,
        proj_dim: int = 128,
        use_sdp_kernel: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.proj_dim = proj_dim
        
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
        
        self.projector = Projector(
            in_dim=dim,
            hidden_dim=proj_hidden_dim,
            out_dim=proj_dim
        )
    
    def forward(self, views: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-view input.
        
        Args:
            views: Tensor of shape (B, V, C, T) where
                   B = batch size
                   V = number of views
                   C = number of channels (12 for ECG)
                   T = time samples
        
        Returns:
            emb: Encoder embeddings of shape (B*V, dim)
            proj: Projected embeddings of shape (V, B, proj_dim)
        """
        B, V, C, T = views.shape
        
        # Flatten batch and views
        x = views.reshape(B * V, C, T)  # (B*V, C, T)
        
        # Encode
        tokens = self.encoder(x)  # (B*V, num_patches, dim)
        
        # Global average pooling over patches
        emb = tokens.mean(dim=1)  # (B*V, dim)
        
        # Project
        proj = self.projector(emb)  # (B*V, proj_dim)
        
        # Reshape for loss computation: (V, B, proj_dim)
        proj = proj.reshape(B, V, -1).permute(1, 0, 2)
        
        return emb, proj
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for a single batch (no views).
        
        Args:
            x: Tensor of shape (B, C, T)
        
        Returns:
            emb: Embeddings of shape (B, dim)
        """
        tokens = self.encoder(x)
        return tokens.mean(dim=1)
    
    def get_optimizer(self, lr: float = 5e-4, weight_decay: float = 5e-2, fused: bool = False):
        """Get AdamW optimizer with weight decay."""
        return torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            fused=fused
        )


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
            proj: Projected embeddings of shape (V, B, proj_dim) or (B, proj_dim)
        
        Returns:
            loss: Scalar SIGReg loss
        """
        if proj.dim() == 3:
            # (V, B, D) -> (V*B, D)
            proj = proj.reshape(-1, proj.size(-1))
        
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


def compute_lejepa_loss(
    proj: torch.Tensor,
    sigreg: SIGReg,
    lamb: float = 0.02
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute LeJEPA loss = λ * SIGReg + (1-λ) * Invariance.
    
    Args:
        proj: Projected embeddings of shape (V, B, proj_dim)
        sigreg: SIGReg module
        lamb: Weight for SIGReg loss (default 0.02)
    
    Returns:
        loss: Total LeJEPA loss
        inv_loss: Invariance loss
        sigreg_loss: SIGReg loss
    """
    # Invariance loss: different views should have similar embeddings
    # proj: (V, B, D)
    mean_proj = proj.mean(dim=0, keepdim=True)  # (1, B, D)
    inv_loss = (proj - mean_proj).square().mean()
    
    # SIGReg loss
    sigreg_loss = sigreg(proj)
    
    # Combined loss
    loss = lamb * sigreg_loss + (1 - lamb) * inv_loss
    
    return loss, inv_loss, sigreg_loss

