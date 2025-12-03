import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
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
        self.norm1 = nn.LayerNorm(dim, eps=eps, bias=bias)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            bias=bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            use_sdp_kernel=use_sdp_kernel)
        self.attn_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
            bias=bias)
        self.mlp_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()

    def forward(self, x):
        x = x + self.attn_ls(self.attn(self.norm1(x)))
        x = x + self.mlp_ls(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        bias=True,
        attn_dropout=0.,
        proj_dropout=0.,
        use_sdp_kernel=True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.use_sdp_kernel = use_sdp_kernel
        assert dim % num_heads == 0
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        if self.use_sdp_kernel:
            self.attn_dropout = attn_dropout
        else:
            self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.proj_dropout = nn.Dropout(proj_dropout) if proj_dropout else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.use_sdp_kernel:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        return x


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        dropout=0.,
        bias=True
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout) if dropout else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        patch_size,
        bias=True
    ):
        super().__init__()
        self.proj = nn.Conv1d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias)

    def forward(self, x):
        x = self.proj(x).transpose(1, 2)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, eps: float = 0.1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(eps * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class Projector(nn.Module):
    """MLP projector for LeJEPA with BatchNorm."""
    def __init__(self, in_dim, hidden_dim=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)

