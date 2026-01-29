from dataclasses import dataclass, field


@dataclass
class Config:
  # data
  sampling_frequency: int
  channels: tuple[str, ...]
  channel_size: int
  patch_size: int
  min_block_size: int
  min_keep_ratio: float
  max_keep_ratio: float
  datasets: dict[str, float]
  # model architecture
  dim: int
  depth: int
  num_heads: int
  pred_dim: int
  pred_depth: int
  pred_num_heads: int
  mlp_ratio: float
  qkv_bias: bool
  dropout: float
  attn_dropout: float
  num_registers: int
  bias: bool
  norm_eps: float
  layer_scale_eps: float
  # training
  steps: int
  batch_size: int
  encoder_momentum: float
  final_encoder_momentum: float
  learning_rate: float
  final_learning_rate: float
  learning_rate_warmup_steps: int
  weight_decay: float
  final_weight_decay: float
  opt_betas: tuple[float, float]
  opt_eps: float
  gradient_clip: float
  gradient_accumulation_steps: int
  checkpoint_interval: int
  structure: str = field(default='standard')  # 'standard', 'factorized', or 'grouped' - architecture type
  mask_type: str = field(default='1d')  # '1d', '2d_random', or '2d_lead_group' - masking strategy
  attention_type: str = field(default='attention')  # 'attention' or 'cotar' - token interaction mechanism

  @property
  def num_channels(self):
    return len(self.channels)
  
  @property
  def num_patches(self):
    return self.channel_size // self.patch_size
  
  @property
  def num_groups(self):
    """Number of lead groups for grouped architecture (always 4: limb, augmented, septal, lateral)."""
    return 4
  
  @property
  def total_tokens(self):
    """Total number of tokens depending on architecture."""
    if self.structure == 'grouped':
      return self.num_groups * self.num_patches  # 4 * 200 = 800
    else:
      return self.num_channels * self.num_patches  # 12 * 200 = 2400 (factorized) or just 200 (standard)
