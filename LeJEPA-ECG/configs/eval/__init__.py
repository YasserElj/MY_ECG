from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # data
    crop_duration: Optional[float]  # seconds
    crop_stride: Optional[float]  # seconds
    # model architecture
    num_classes: int
    use_register: bool
    attn_pooling: bool
    layer_scale_eps: float
    bias: bool
    dropout: float
    frozen: bool
    depth: int = 8  # for weight init scaling
    # training
    steps: int = 5000
    batch_size: int = 128
    learning_rate: float = 1e-3
    final_learning_rate: float = 1e-3
    learning_rate_warmup_steps: int = 0
    weight_decay: float = 0.1
    opt_betas: tuple[float, float] = (0.9, 0.999)
    gradient_clip: float = 1.0
    checkpoint_interval: int = 100
    early_stopping_patience: int = 5000

