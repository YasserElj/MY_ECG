import copy
from collections import OrderedDict

import torch
from torch import nn

import configs
from models.predictor import Predictor
from models.predictor_factorized import FactorizedPredictor
from models.utils import apply_mask
from models.utils_factorized import apply_mask_factorized, apply_mask_2d
from models.vision_transformer import VisionTransformer
from models.vision_transformer_factorized import FactorizedViT


class JEPA(nn.Module):
  def __init__(self, config: configs.pretrain.Config, momentum_schedule, use_sdp_kernel=True):
    super().__init__()
    self.config = config
    self.momentum_schedule = momentum_schedule
    
    # --- Architecture Selection ---
    is_factorized = getattr(config, 'structure', 'standard') == 'factorized'
    
    if is_factorized:
      self.encoder = FactorizedViT(config, keep_registers=False, use_sdp_kernel=use_sdp_kernel)
      self.predictor = FactorizedPredictor(config, use_sdp_kernel=use_sdp_kernel)
    else:
      self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)
      self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)
    
    self.target_encoder = copy.deepcopy(self.encoder)

    for param in self.target_encoder.parameters():
      param.requires_grad = False

  @torch.compiler.disable()
  def update_momentum_encoder(self):
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, mask_encoder, mask_predictor):
    is_factorized = getattr(self.config, 'structure', 'standard') == 'factorized'
    
    # Detect if using 2D masking (tuple format with 3D indices tensor)
    is_2d_mask = isinstance(mask_encoder, tuple) and mask_encoder[0].dim() == 3
    
    with torch.no_grad():
      self.update_momentum_encoder()
      # compute prediction targets (EMA encoder sees full input)
      h = self.target_encoder(x)
      
      if is_2d_mask:
        # 2D masking: mask_predictor is (indices, lengths) tuple
        mask_pred_indices, mask_pred_lengths = mask_predictor
        num_leads = len(self.config.channels)
        num_time = self.config.num_patches
        h = apply_mask_2d(h, mask_pred_indices, num_leads, num_time)
      elif is_factorized:
        # 1D masking with factorized model: same time mask for all leads
        num_leads = len(self.config.channels)
        h = apply_mask_factorized(h, mask_predictor, num_leads)
      else:
        # Standard 1D masking
        h = apply_mask(h, mask_predictor)
    
    # encode unmasked patches
    if is_2d_mask:
      mask_enc_indices, mask_enc_lengths = mask_encoder
      z = self.encoder(x, mask_enc_indices)
    else:
      z = self.encoder(x, mask_encoder)
    
    # predict masked patches
    z = self.predictor(z, mask_encoder, mask_predictor)
    
    loss = torch.mean(torch.abs(z - h))
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
