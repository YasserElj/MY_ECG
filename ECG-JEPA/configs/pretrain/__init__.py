from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
  # data
  sampling_frequency: int
  channels: tuple[str, ...]
  channel_size: int
  patch_size: int
  # -- datasets (MUST come before default args) --
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

  # --- OPTIONAL / DEFAULT ARGUMENTS GO LAST ---
  
  # --- ECG-JEPA masking params ---
  min_block_size: Optional[int] = None
  min_keep_ratio: Optional[float] = None
  max_keep_ratio: Optional[float] = None
  
  # --- I-JEPA masking params ---
  masking_strategy: str = 'ecg-jepa' 
  context_scale: Optional[tuple[float, float]] = None
  n_pred_blocks: Optional[int] = None
  pred_scale: Optional[tuple[float, float]] = None
  min_keep: Optional[int] = None
  allow_overlap: bool = False

  @property
  def num_channels(self):
    return len(self.channels)
