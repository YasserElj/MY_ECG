import math
import torch
from torch import nn

from models.modules import PatchEmbedding, Block
from models.utils import get_1d_pos_embed


class VisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        num_channels: int,
        channel_size: int,
        patch_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        bias: bool = True,
        dropout: float = 0.,
        attn_dropout: float = 0.,
        norm_eps: float = 1e-6,
        layer_scale_eps: float = 0.,
        num_registers: int = 0,
        keep_registers: bool = False,
        use_sdp_kernel: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.keep_registers = keep_registers
        self.num_registers = num_registers
        
        assert channel_size % patch_size == 0
        num_patches = channel_size // patch_size
        
        self.patch_embed = PatchEmbedding(
            dim=dim,
            in_channels=num_channels,
            patch_size=patch_size,
            bias=bias)
        
        self.register_buffer(
            'pos_embed',
            get_1d_pos_embed(dim=dim, num_patches=num_patches),
            persistent=False)
        
        if num_registers > 0:
            self.registers = nn.Parameter(torch.empty(1, num_registers, dim))
            nn.init.trunc_normal_(self.registers, mean=0., std=0.02)
        
        self.blocks = nn.ModuleList([
            Block(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                bias=bias,
                dropout=dropout,
                attn_dropout=attn_dropout,
                eps=norm_eps,
                layer_scale_eps=layer_scale_eps,
                use_sdp_kernel=use_sdp_kernel)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim, eps=norm_eps, bias=bias)
        
        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                if name.endswith('mlp.fc2') or name.endswith('attn.proj'):
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02 / math.sqrt(2 * len(self.blocks)))
                elif module.bias is not None:
                    nn.init.zeros_(module.bias)
                else:
                    nn.init.trunc_normal_(module.weight, mean=0., std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        B, N, D = x.size()
        x = x + self.pos_embed[:, :N]
        
        if self.num_registers > 0:
            registers = self.registers.repeat(B, 1, 1)
            x = torch.cat([registers, x], dim=1)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        if not self.keep_registers and self.num_registers > 0:
            x = x[:, self.num_registers:]
        
        return x

    def get_cls_token(self, x):
        """Get CLS token (mean of all patch tokens if no CLS token exists)."""
        x = self.forward(x)
        return x.mean(dim=1)

