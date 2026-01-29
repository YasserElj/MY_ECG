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


class CoTAR(nn.Module):
  """
  Core Token Aggregation-Redistribution.
  
  Replaces O(N^2) attention with O(N) centralized communication.
  Based on the TeCh paper: "Decentralized Attention Fails Centralized Signals"
  
  Instead of pairwise token interactions (attention), CoTAR:
  1. Aggregates all tokens into a single "core token" (heart/brain state)
  2. Redistributes this global context back to each token
  
  This matches the centralized nature of physiological signals where
  a single source (heart/brain) drives all channels.
  """
  def __init__(self, dim, core_dim=None, dropout=0.):
    super().__init__()
    self.core_dim = core_dim or dim // 4
    
    # Aggregation pathway
    self.lin1 = nn.Linear(dim, dim)
    self.lin2 = nn.Linear(dim, self.core_dim)
    
    # Redistribution pathway
    self.lin3 = nn.Linear(dim + self.core_dim, dim)
    self.lin4 = nn.Linear(dim, dim)
    
    self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

  def forward(self, x):
    B, N, D = x.shape
    
    # === Aggregation: Create core token ===
    # Project to latent space
    core = F.gelu(self.lin1(x))  # (B, N, D)
    core = self.lin2(core)       # (B, N, core_dim)
    
    # Compute attention weights across tokens
    weight = F.softmax(core, dim=1)  # (B, N, core_dim)
    
    # Weighted sum to get global core token
    core = torch.sum(core * weight, dim=1, keepdim=True)  # (B, 1, core_dim)
    
    # Expand core token to all positions
    core = core.expand(-1, N, -1)  # (B, N, core_dim)
    
    # === Redistribution: Combine local + global ===
    # Concatenate original tokens with core context
    out = torch.cat([x, core], dim=-1)  # (B, N, D + core_dim)
    
    # Fuse information
    out = F.gelu(self.lin3(out))  # (B, N, D)
    out = self.lin4(out)          # (B, N, D)
    
    return self.dropout(out)


class CoTARBlock(nn.Module):
  """
  Transformer block using CoTAR instead of attention.
  
  Same structure as Block but with CoTAR replacing multi-head attention.
  """
  def __init__(
      self,
      dim,
      mlp_ratio=4.,
      bias=True,
      dropout=0.,
      eps=1e-6,
      layer_scale_eps=0.,
  ):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=eps, bias=bias)
    self.cotar = CoTAR(dim, core_dim=dim // 4, dropout=dropout)
    self.cotar_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()
    self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = MLP(
      in_features=dim,
      hidden_features=mlp_hidden_dim,
      dropout=dropout,
      bias=bias)
    self.mlp_ls = LayerScale(dim, eps=layer_scale_eps) if layer_scale_eps else nn.Identity()

  def forward(self, x):
    x = x + self.cotar_ls(self.cotar(self.norm1(x)))
    x = x + self.mlp_ls(self.mlp(self.norm2(x)))
    return x


class CrossAttentionBlock(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      mlp_ratio=4.,
      qkv_bias=False,
      bias=True,
      eps=1e-6,
      use_sdp_kernel=True,
  ):
    super().__init__()
    self.norm1 = nn.LayerNorm(dim, eps=eps, bias=bias)
    self.xattn = CrossAttention(
      dim,
      num_heads=num_heads,
      qkv_bias=qkv_bias,
      bias=bias,
      use_sdp_kernel=use_sdp_kernel)
    self.norm2 = nn.LayerNorm(dim, eps=eps, bias=bias)
    mlp_hidden_dim = int(dim * mlp_ratio)
    self.mlp = MLP(
      in_features=dim,
      hidden_features=mlp_hidden_dim,
      bias=bias)

  def forward(self, q, x):
    q = q + self.xattn(q, self.norm1(x))
    q = q + self.mlp(self.norm2(q))
    return q


class AttentivePooler(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      num_queries=1,
      depth=1,
      mlp_ratio=4.,
      qkv_bias=False,
      bias=True,
      complete_block=False,
      proj_dim=None,
      eps=1e-6,
      use_sdp_kernel=True
  ):
    super().__init__()
    self.query_token = nn.Parameter(torch.empty(1, num_queries, dim))
    nn.init.trunc_normal_(self.query_token, mean=0., std=0.02)
    self.blocks = None
    if complete_block:
      self.cross_attention_block = CrossAttentionBlock(
        dim=dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=qkv_bias,
        bias=bias,
        eps=eps,
        use_sdp_kernel=use_sdp_kernel)
      if depth > 1:
        self.blocks = nn.ModuleList([
          Block(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            bias=bias,
            eps=1e-6,
            use_sdp_kernel=use_sdp_kernel)
          for _ in range(depth - 1)])
    else:
      self.cross_attention_block = CrossAttention(
        dim=dim,
        num_heads=num_heads,
        proj_dim=proj_dim,
        qkv_bias=qkv_bias,
        bias=bias,
        use_sdp_kernel=use_sdp_kernel)

  def forward(self, x):
    q = self.query_token.repeat(len(x), 1, 1)
    q = self.cross_attention_block(q, x)
    if self.blocks is not None:
      for block in self.blocks:
        q = block(q)
    return q


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
    q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]
    if self.use_sdp_kernel:
      x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout)
    else:
      attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
      attn = attn.softmax(dim=-1)
      attn = self.attn_dropout(attn)
      x = (attn @ v)
    x = x.transpose(1, 2).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_dropout(x)
    return x


class CrossAttention(nn.Module):
  def __init__(
      self,
      dim,
      num_heads,
      proj_dim=None,
      qkv_bias=False,
      bias=True,
      use_sdp_kernel=True
  ):
    super().__init__()
    self.num_heads = num_heads
    assert dim % num_heads == 0
    head_dim = dim // num_heads
    proj_dim = proj_dim or dim
    self.scale = head_dim ** -0.5
    self.q = nn.Linear(dim, dim, bias=qkv_bias)
    self.kv = nn.Linear(dim, int(dim * 2), bias=qkv_bias)
    self.proj = nn.Linear(dim, proj_dim, bias=bias)
    self.use_sdp_kernel = use_sdp_kernel

  def forward(self, q, x):
    B, n, C = q.shape
    q = self.q(q).reshape(B, n, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
    B, N, C = x.shape
    kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    k, v = kv[0], kv[1]  # (batch_size, num_heads, seq_len, feature_dim_per_head)
    if self.use_sdp_kernel:
      q = F.scaled_dot_product_attention(q, k, v)
    else:
      xattn = (q @ k.transpose(-2, -1)) * self.scale
      xattn = xattn.softmax(dim=-1)  # (batch_size, num_heads, query_len, seq_len)
      self.xattn = xattn.detach()  # store attention weights for later inspection
      q = (xattn @ v)
    q = q.transpose(1, 2).reshape(B, n, C)
    q = self.proj(q)
    return q


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
  """https://arxiv.org/abs/2103.17239"""
  def __init__(self, dim: int, eps: float = 0.1):
    super().__init__()
    self.dim = dim
    self.eps = eps
    self.weight = nn.Parameter(eps * torch.ones(dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * self.weight


class GroupedPatchEmbedding(nn.Module):
  """
  Embed groups of 3 leads together into a single token per time patch.
  
  Instead of 12 leads × 200 patches = 2400 tokens,
  we get 4 groups × 200 patches = 800 tokens.
  
  Default groupings (clinically meaningful):
    Group 0: I, II, III      (Limb leads - Einthoven triangle)
    Group 1: aVR, aVL, aVF   (Augmented leads)
    Group 2: V1, V2, V3      (Septal/Anterior precordial)
    Group 3: V4, V5, V6      (Lateral precordial)
  """
  
  # Lead indices for standard 12-lead ECG ordering: I, II, III, aVR, aVL, aVF, V1-V6
  DEFAULT_LEAD_GROUPS = [
    [0, 1, 2],     # I, II, III (Limb leads)
    [3, 4, 5],     # aVR, aVL, aVF (Augmented leads)
    [6, 7, 8],     # V1, V2, V3 (Septal/Anterior)
    [9, 10, 11],   # V4, V5, V6 (Lateral precordial)
  ]
  
  GROUP_NAMES = ['Limb', 'Augmented', 'Septal', 'Lateral']
  
  def __init__(
      self,
      dim,
      patch_size,
      leads_per_group=3,
      lead_groups=None,
      bias=True
  ):
    super().__init__()
    self.patch_size = patch_size
    self.leads_per_group = leads_per_group
    self.lead_groups = lead_groups or self.DEFAULT_LEAD_GROUPS
    self.num_groups = len(self.lead_groups)
    
    # Conv1d processes `leads_per_group` leads at once
    self.proj = nn.Conv1d(
      leads_per_group,
      dim,
      kernel_size=patch_size,
      stride=patch_size,
      bias=bias)

  def forward(self, x):
    """
    Args:
      x: (B, 12, signal_length) - raw 12-lead ECG
    
    Returns:
      (B, num_groups * num_patches, D) - grouped patch embeddings
      e.g., (B, 4 * 200, 768) = (B, 800, 768)
    """
    B, num_leads, signal_length = x.shape
    
    # Gather leads for each group and process
    group_embeddings = []
    for lead_indices in self.lead_groups:
      # Extract leads for this group: (B, 3, signal_length)
      group_leads = x[:, lead_indices, :]
      
      # Apply patch embedding: (B, D, num_patches)
      embedded = self.proj(group_leads)
      
      # Transpose to (B, num_patches, D)
      embedded = embedded.transpose(1, 2)
      group_embeddings.append(embedded)
    
    # Stack groups: (B, num_groups, num_patches, D)
    x = torch.stack(group_embeddings, dim=1)
    
    # Flatten to (B, num_groups * num_patches, D)
    B, G, T, D = x.shape
    x = x.reshape(B, G * T, D)
    
    return x
  
  def forward_grouped(self, x):
    """
    Alternative forward that keeps group dimension separate.
    
    Returns:
      (B, num_groups, num_patches, D) - keeps group structure
    """
    B, num_leads, signal_length = x.shape
    
    group_embeddings = []
    for lead_indices in self.lead_groups:
      group_leads = x[:, lead_indices, :]
      embedded = self.proj(group_leads).transpose(1, 2)
      group_embeddings.append(embedded)
    
    # (B, num_groups, num_patches, D)
    return torch.stack(group_embeddings, dim=1)