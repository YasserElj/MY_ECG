# JEPA-ECG Implementation Evidence

## Input Representation
- The `ViTS_ptbxl` pretraining config defines 12 leads, a 10 s crop (`channel_size=5000` at 500 Hz), and 1D patches of 25 samples, so every token holds all leads for that time window.

```3:9:ECG-JEPA/configs/pretrain/ViTS_ptbxl.yaml
channels: [I, II, III, AVR, AVL, AVF, V1, V2, V3, V4, V5, V6]
channel_size: 5000
patch_size: 25
min_block_size: 10
min_keep_ratio: 0.15
max_keep_ratio: 0.25
```

- Each training crop is transposed into `(num_channels, channel_size)` tensors before batching, keeping all 12 leads stacked along the channel axis.

```344:348:ECG-JEPA/pretrain.py
    x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()  # channels first
    x = torch.from_numpy(x).float()
    return x
```

- The dataloader batches these crops as `(batch, channels, samples)`, then samples keep/mask index lists (per time patch) that feed the encoders and predictor.

```17:28:ECG-JEPA/data/masks.py
    batch = torch.utils.data.default_collate(batch)
    batch_size, num_channels, channel_size = batch.size()
    assert channel_size % self.patch_size == 0
    num_patches = channel_size // self.patch_size
    keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    mask_encoder, mask_predictor = [], []
    for _ in range(batch_size):
      mask = self.sample_mask(num_keep, num_patches)
      mask_encoder.append(mask.nonzero().squeeze())
      mask_predictor.append((1 - mask).nonzero().squeeze())
```

- Tokens are produced by a 1D convolution with stride equal to the patch size, so every token aggregates all 12 leads across one 25-sample window.

```243:261:ECG-JEPA/models/modules.py
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
```

- After adding positional encodings, optional masking gathers the requested patch indices, so the student encoder receives only the visible tokens while the teacher processes the full sequence.

```63:77:ECG-JEPA/models/vision_transformer.py
  def forward(self, x, mask=None):
    x = self.patch_embed(x)
    B, N, D = x.size()
    x = x + self.pos_embed[:, :N]
    if mask is not None:
      x = apply_mask(x, mask)
    if self.config.num_registers > 0:
      registers = self.registers.repeat(B, 1, 1)
      x = torch.cat([registers, x], dim=1)
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    if not self.keep_registers and self.config.num_registers > 0:
      x = x[:, self.config.num_registers:]
    return x
```

```6:9:ECG-JEPA/models/utils.py
def apply_mask(x, mask):
  mask = mask.unsqueeze(-1).repeat(1, 1, x.size(-1))
  x = torch.gather(x, dim=1, index=mask)
  return x
```

## 1. Self-Supervised Families (What `z` Is)
### 1A. Joint-Embedding (JEA)
- The implementation maintains two encoders (`self.encoder` and `self.target_encoder`) whose outputs are compared in the joint embedding space—aligning with JEA ideas while relying on EMA tricks to avoid collapse.

```17:24:ECG-JEPA/models/JEPA.py
    self.encoder = VisionTransformer(config, use_sdp_kernel=use_sdp_kernel)
    self.predictor = Predictor(config, use_sdp_kernel=use_sdp_kernel)
    self.target_encoder = copy.deepcopy(self.encoder)

    for param in self.target_encoder.parameters():
      param.requires_grad = False
```

### 1B. Generative
- There is no decoder reconstructing raw waveform samples—the predictor only outputs embeddings that match the teacher space. The absence of any `ConvTranspose1d`/decoder stack confirms the generative branch is not implemented here.

```39:72:ECG-JEPA/models/predictor.py
    for block in self.blocks:
      x = block(x)
    x = self.norm(x)
    mask_token = x[:, -K:]
    mask_token = self.proj(mask_token)
    return mask_token
```

### 1C. Joint-Embedding Predictive (JEPA)
- The forward step encodes the full ECG with the teacher to define targets `h`, encodes only visible patches with the student to obtain `z`, predicts embeddings for masked positions, and minimizes L1 distance in representation space.

```25:42:ECG-JEPA/models/JEPA.py
  def update_momentum_encoder(self):
    m = next(self.momentum_schedule)
    for param_z, param_h in zip(self.encoder.parameters(), self.target_encoder.parameters()):
      param_h.data = m * param_h.data + (1. - m) * param_z.data

  def forward(self, x, mask_encoder, mask_predictor):
    with torch.no_grad():
      self.update_momentum_encoder()
      h = self.target_encoder(x)
      h = apply_mask(h, mask_predictor)
    z = self.encoder(x, mask_encoder)
    z = self.predictor(z, mask_encoder, mask_predictor)
    loss = torch.mean(torch.abs(z - h))
    return loss
```

## 2. JEPA-ECG Pipeline (Code Evidence)
- **Patch the ECG (1D):** `PatchEmbedding` performs a strided 1D convolution, yielding `num_patches = channel_size / patch_size` tokens per sequence (see above).
- **Mask 75–85 % of patches:** The config enforces `keep_ratio` between 0.15 and 0.25, and the collator samples contiguous blocks via `sample_mask`.

```20:66:ECG-JEPA/data/masks.py
    keep_ratio = np.random.uniform(self.min_keep_ratio, self.max_keep_ratio)
    num_keep = math.ceil(keep_ratio * num_patches)
    ...
  def sample_mask(self, num_keep, num_patches):
    patch_intervals = [(0, num_patches)]
    num_mask = num_patches - num_keep
    ...
      if interval_size <= block_size:
        total_mask_size += interval_size
      else:
        ...
        patch_intervals.append((start, split))
        patch_intervals.append((split + block_size, end))
        total_mask_size += block_size
    ...
    mask = torch.zeros(num_patches)
    for start, end in patch_intervals:
      mask[start:end] = 1.
    return mask
```

- **Encode context vs. target:** Teacher sees the full sequence (no mask), while the student gathers only indices from `mask_encoder`; both use the same ViT weights structure.
- **Mask tokens & predictor:** Learnable mask tokens are created per missing index, positional encodings are added, and the predictor outputs one embedding per masked patch.

```57:71:ECG-JEPA/models/predictor.py
    pos_embed = self.pos_embed.repeat(B, 1, 1)
    pos_encoder = apply_mask(pos_embed, mask_encoder)
    x = self.embed(x)
    x = x + pos_encoder
    pos_predictor = apply_mask(pos_embed, mask_predictor)
    mask_token = self.mask_token.repeat(B, K, 1)
    mask_token = mask_token + pos_predictor
    x = torch.cat([x, mask_token], dim=1)
    ...
    mask_token = x[:, -K:]
    mask_token = self.proj(mask_token)
```

- **L1 feature loss:** The JEPA objective stays in representation space (`torch.abs(z - h)`), never reconstructing raw ECG samples.

## 3. 1D vs. 2D Choices
- Tokens operate purely along the time axis (`Conv1d` stride and kernel apply only over samples), and masking removes identical time windows across all leads simultaneously. No 2D convolution or per-lead masking exists in the repository, confirming a 1D temporal design.

## 4. Classifier Heads
- Linear evaluation uses attention pooling on frozen features with a single query, matching the paper’s description.

```1:10:ECG-JEPA/configs/eval/linear.yaml
crop_duration: 2.5
crop_stride: 1.25
use_register: True
attn_pooling: True
...
frozen: True
```

```18:71:ECG-JEPA/models/vit_classifier.py
    if config.attn_pooling:
      self.attn_pool = AttentivePooler(
        dim=encoder.config.dim,
        num_heads=encoder.config.num_heads,
        ...,
        proj_dim=config.num_classes)
    else:
      ...
  def forward(self, x, encoded=False):
    if not encoded:
      x = self.encode(x)
    if self.config.attn_pooling:
      x = self.attn_pool(x).squeeze(dim=1)
    else:
      ...
      x = self.fc(x)
    return x
```

- Fine-tuning pipelines reuse the same encoder with attention pooling or register-token pooling variants.

## 5. Pre-, Aug-, and Post-Processing
- Pretraining datasets are interpolated, resampled to the common frequency, z-normalized, clipped, and remapped to the desired lead order.

```328:337:ECG-JEPA/pretrain.py
    transforms.interpolate_NaNs_(x)
    if self.resample_ratio != 1.0:
      ...
      x = transforms.resample(x, channel_size)
    transforms.normalize_(x, mean_std=(self.mean, self.std))
    x.clip(-5, 5, out=x)
    x = x[:, self.channel_order]
    return x
```

```18:48:ECG-JEPA/data/transforms.py
def normalize_(x, mean_std=None, eps=0):
  ...
  x -= mean
  x /= std + eps
  return x

def random_crop(x, size):
  start = np.random.randint(len(x) - size + 1)
  x = x[start:start + size]
  return x
```

- Evaluation pipelines reuse random crops for training and strided sliding windows (2.5 s duration, 1.25 s stride) at inference; logits are averaged across crops.

```365:389:ECG-JEPA/finetune.py
class TrainTransformECG:
  def __call__(self, x):
    if self.crop_size is not None:
      x = transforms.random_crop(x, self.crop_size)
    x = x.transpose()
    x = torch.from_numpy(x).float()

class EvalTransformECG:
  def __call__(self, x):
    if self.crop_size is not None:
      x = strided_crops(x, self.crop_size, self.crop_stride)
      x = np.swapaxes(x, 1, 2)
    else:
      x = x.transpose()
    x = torch.from_numpy(x).float()
    return x
```

- Losses and crop ensembling during evaluation follow the reported strategy: binary cross-entropy for multilabel targets (or cross-entropy for single-label tasks) with mean aggregation across temporal crops.

```248:279:ECG-JEPA/finetune.py
    with auto_mixed_precision:
      logits = model(x)
      if single_label:
        loss = F.cross_entropy(logits, y)
      else:
        loss = F.binary_cross_entropy_with_logits(logits, y)
    ...
      if eval_config.crop_duration is not None:
        batch_size, num_crops, num_channels, channel_size = x.size()
        x = x.reshape(-1, num_channels, channel_size)
      logits = model(x)
      if eval_config.crop_duration is not None:
        logits = logits.reshape(batch_size, num_crops, eval_config.num_classes)
        logits = logits.mean(dim=1)
      val_logits.append(logits.clone())
      val_targets.append(y.clone())
```

## 6. Why Use an EMA Teacher
- The teacher weights are updated by a linear-momentum schedule that ramps from 0.998 to 0.9995, providing slowly moving targets to prevent collapse.

```28:37:ECG-JEPA/configs/pretrain/ViTS_ptbxl.yaml
encoder_momentum: 0.998
final_encoder_momentum: 0.9995
```

```206:230:ECG-JEPA/pretrain.py
  momentum_schedule = linear_schedule(
    total_steps=config.steps,
    start_value=config.encoder_momentum,
    final_value=config.final_encoder_momentum,
    step=step)
  ...
  model = original_model = JEPA(
    config=config,
    momentum_schedule=momentum_schedule,
    use_sdp_kernel=using_cuda
  ).to(device)
```

```36:48:ECG-JEPA/utils/schedules.py
def linear_schedule(
  total_steps: int,
  start_value: float,
  final_value: float,
  step: int = 0
):
  while True:
    if step < total_steps:
      value = start_value + (final_value - start_value) * step / total_steps
    else:
      value = final_value
    yield value
    step += 1
```

## 7. Why the Teacher Sees All Patches
- In `JEPA.forward`, only the teacher call omits a mask argument, so it always processes every patch. `VisionTransformer.forward` applies masking only when `mask` is provided (student path).

```35:40:ECG-JEPA/models/JEPA.py
      h = self.target_encoder(x)
      h = apply_mask(h, mask_predictor)
    z = self.encoder(x, mask_encoder)
```

## 8. Expert Q → A Proof Points
- **Q1 – Representational loss:** The objective is `torch.abs(z - h)` (L1), confirming predictions happen in embedding space, not waveform space (`JEPA.forward`, above).
- **Q2 – Purpose of mask tokens:** `Predictor.forward` adds a learnable `mask_token` with positional encodings for every masked index before running Transformer blocks (see snippet under Section 2).
- **Q3 – Contiguous masking:** `MaskCollator.sample_mask` builds blocks by removing intervals and re-inserting contiguous spans (Section 2 code).
- **Q4 – Teacher never backpropagates:** Teacher parameters have `requires_grad=False` and are updated only by EMA (Section 1A code plus `update_momentum_encoder`).
- **Q5 – Evaluation process:** `EvalTransformECG` performs crop ensembling; logits are averaged per crop before computing metrics (Section 5 code and `finetune.py` lines 266–334 not shown here).
- **Q6 – 2.5 s crops, 1.25 s stride:** Defined in `linear.yaml` (Section 4 code).
- **Q7 – 1D tokens:** `PatchEmbedding` uses 1D convolutions; there is no 2D patching (Section 0).
- **Q8 – Single shared encoder:** No per-lead encoders exist; masking and tokenization operate jointly across channels.
- **Q9 – Handling class imbalance:** The classifier supports frozen linear evaluation with optional attention pooling; training uses BCE/CE losses in `finetune.py` (lines 248–255) and allows per-class threshold tuning during evaluation.
- **Q10 – First tuning knobs:** Config files expose `min_keep_ratio`, `pred_depth`, `pred_num_heads`, lead registers, etc., making mask ratio and predictor depth easy sweep targets.

## 9. Additional Clarification
- Downstream scripts load EMA (teacher) weights from the checkpoint into the evaluation encoder, matching the “slow teacher” hypothesis used during pretraining.

```104:108:ECG-JEPA/finetune.py
    if 'eval_config' in chkpt:
      model_state_dict = chkpt['model']
    else:
      model_state_dict = {'encoder.' + k.removeprefix('target_encoder.'): v
                          for k, v in chkpt['model'].items()
                          if k.startswith('target_encoder.')}
```

- The repository never reconstructs raw ECG samples and always matches latent features, reinforcing that `z` represents latent embeddings of masked patches rather than waveform values.


