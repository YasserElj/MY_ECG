"""
I-JEPA Model for ECG.

This implements the proper I-JEPA training procedure where:
1. Multiple target blocks are predicted from a single context.
2. The predictor processes each block's mask tokens with the same context.
3. Loss is computed per-block and averaged.

Key difference from ECG-JEPA:
- ECG-JEPA: Flattens all targets into one sequence, predicts all at once.
- I-JEPA: Processes each target block separately (via batch expansion), 
          allowing the model to learn block-specific predictions.
"""

import copy
from collections import OrderedDict

import torch
from torch import nn
import torch.nn.functional as F

import configs
from models.predictor import Predictor
from models.utils import apply_mask, get_1d_pos_embed
from models.vision_transformer import VisionTransformer


def apply_masks_list(x, masks):
    """
    Apply a list of masks to x, concatenating results along batch dimension.
    
    Args:
        x: [B, N, D] tensor
        masks: list of M tensors, each [B, K_i] containing indices
        
    Returns:
        [B*M, max_K, D] tensor (padded to max_K if needed)
    """
    all_x = []
    for m in masks:
        # m: [B, K]
        mask_keep = m.unsqueeze(-1).expand(-1, -1, x.size(-1))  # [B, K, D]
        gathered = torch.gather(x, dim=1, index=mask_keep)  # [B, K, D]
        all_x.append(gathered)
    return torch.cat(all_x, dim=0)  # [B*M, K, D]


def repeat_interleave_batch(x, B, repeat):
    """
    Repeat x along batch dimension.
    
    Args:
        x: [B*M, ...] tensor
        B: original batch size
        repeat: number of times to repeat
        
    Returns:
        [B*repeat, ...] tensor
    """
    N = len(x) // B
    return x.repeat(repeat, *([1] * (len(x.shape) - 1)))


class IJEPAPredictor(nn.Module):
    """
    I-JEPA style predictor that handles multiple target blocks.
    
    The predictor:
    1. Takes context embeddings z (from encoder).
    2. For each target block, creates mask tokens with positional embeddings.
    3. Concatenates context + mask tokens and runs transformer.
    4. Returns predictions for all mask tokens.
    """
    def __init__(self, config: configs.pretrain.Config, use_sdp_kernel=True):
        super().__init__()
        self.config = config
        assert config.channel_size % config.patch_size == 0
        num_patches = config.channel_size // config.patch_size
        
        self.embed = nn.Linear(config.dim, config.pred_dim, bias=config.bias)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.pred_dim))
        self.register_buffer(
            'pos_embed',
            get_1d_pos_embed(dim=config.pred_dim, num_patches=num_patches),
            persistent=False)
        
        from models.modules import Block
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
        
        self._init_weights()
        
    def _init_weights(self):
        import math
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * self.config.pred_depth))
                elif module.bias is not None:
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.mask_token, mean=0., std=0.02)

    def forward(self, z, masks_context, masks_targets):
        """
        Args:
            z: [B*M, K_ctx, D] context embeddings (repeated for each target block)
            masks_context: list of M tensors [B, K_ctx] (context indices, same for all)
            masks_targets: list of M tensors [B, K_tgt_i] (target indices per block)
            
        Returns:
            predictions: [B*M, K_tgt, D] predicted embeddings for all target blocks
        """
        if not isinstance(masks_context, list):
            masks_context = [masks_context]
        if not isinstance(masks_targets, list):
            masks_targets = [masks_targets]
            
        B = z.size(0) // len(masks_targets)
        M = len(masks_targets)
        
        # Project context embeddings
        z = self.embed(z)  # [B*M, K_ctx, pred_dim]
        
        # Add positional embeddings to context
        pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, N, pred_dim]
        ctx_pos = apply_masks_list(pos_embed, masks_context)  # [B*M, K_ctx, pred_dim]
        z = z + ctx_pos
        
        _, N_ctx, D = z.shape
        
        # Create mask tokens for targets
        pos_embed = self.pos_embed.repeat(B, 1, 1)
        tgt_pos = apply_masks_list(pos_embed, masks_targets)  # [B*M, K_tgt, pred_dim]
        
        mask_tokens = self.mask_token.expand(tgt_pos.size(0), tgt_pos.size(1), -1)
        mask_tokens = mask_tokens + tgt_pos
        
        # Concatenate context and mask tokens
        x = torch.cat([z, mask_tokens], dim=1)  # [B*M, K_ctx + K_tgt, pred_dim]
        
        # Forward through transformer
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # Extract predictions for mask tokens
        x = x[:, N_ctx:]  # [B*M, K_tgt, pred_dim]
        x = self.proj(x)  # [B*M, K_tgt, D]
        
        return x


class IJEPA(nn.Module):
    """
    I-JEPA model for ECG signals.
    
    Key features:
    - Multi-block target prediction
    - Each target block is processed with the same context
    - Proper handling of overlapping targets
    """
    def __init__(self, config: configs.pretrain.Config, momentum_schedule, use_sdp_kernel=True):
        super().__init__()
        self.config = config
        self.momentum_schedule = momentum_schedule
        self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)
        self.predictor = IJEPAPredictor(config, use_sdp_kernel=use_sdp_kernel)
        self.target_encoder = copy.deepcopy(self.encoder)
        
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.compiler.disable()
    def update_momentum_encoder(self):
        m = next(self.momentum_schedule)
        for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            param_h.data = m * param_h.data + (1. - m) * param_z.data

    def forward(self, x, masks_context, masks_targets):
        """
        Args:
            x: [B, C, T] input ECG
            masks_context: list of M tensors [B, K_ctx] OR single tensor [B, K_ctx]
            masks_targets: list of M tensors [B, K_tgt_i] (one per target block)
            
        Returns:
            loss: scalar loss value
        """
        # Ensure masks are lists
        if not isinstance(masks_context, list):
            masks_context = [masks_context]
        if not isinstance(masks_targets, list):
            masks_targets = [masks_targets]
            
        B = x.size(0)
        M = len(masks_targets)
        
        with torch.no_grad():
            self.update_momentum_encoder()
            # Get full embeddings from target encoder
            h_full = self.target_encoder(x)  # [B, N, D]
            h_full = F.layer_norm(h_full, (h_full.size(-1),))  # Normalize features
            
            # Extract target embeddings for each block
            h_targets = apply_masks_list(h_full, masks_targets)  # [B*M, K_tgt, D]
        
        # Encode context
        z_context = self.encoder(x, masks_context[0])  # [B, K_ctx, D]
        
        # Repeat context for each target block
        z_context = z_context.repeat(M, 1, 1)  # [B*M, K_ctx, D]
        
        # Predict target embeddings
        z_pred = self.predictor(z_context, masks_context * M, masks_targets)  # [B*M, K_tgt, D]
        
        # Compute loss (smooth L1 as in original I-JEPA)
        loss = F.smooth_l1_loss(z_pred, h_targets)
        
        return loss

    def get_optimizer(self, fused=False):
        decay_modules = (nn.Linear, nn.Conv1d)
        decay = set()
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if isinstance(module, decay_modules) and param_name.endswith('weight') and param.requires_grad:
                    param_name = f'{module_name}.{param_name}' if module_name else param_name
                    decay.add(param_name)

        decay_params, non_decay_params = OrderedDict(), OrderedDict()
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name in decay:
                    decay_params[name] = param
                else:
                    non_decay_params[name] = param

        param_groups = [
            {'params': list(decay_params.values()),
             'weight_decay': self.config.weight_decay,
             'use_weight_decay': True},
            {'params': list(non_decay_params.values()),
             'weight_decay': 0.,
             'use_weight_decay': False}
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=self.config.opt_betas,
            eps=self.config.opt_eps,
            weight_decay=0.,
            fused=fused)

        return optimizer

