# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch ViTNepaECG model for ECG signals."""

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from transformers.modeling_outputs import BaseModelOutput, ImageClassifierOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, logging, torch_int
from transformers.utils.generic import can_return_tuple, check_model_inputs

# Import shared components from original NEPA
from ..vit_nepa.modeling_vit_nepa import (
    prediction_loss,
    rotate_half,
    apply_rotary_pos_emb,
    eager_attention_forward,
    BaseModelOutputWithEmbedding,
    EmbeddedModelingOutput,
    ViTNepaSelfAttention,
    ViTNepaSelfOutput,
    ViTNepaAttention,
    ViTNepaLayerScale,
    ViTNepaIntermediate,
    ViTNepaOutput,
    ViTNepaDropPath,
    drop_path,
    ViTNepaLayer,
    ViTNepaEncoder,
    ViTNepaPreTrainedModel,
)

from .configuration_vit_nepa_ecg import ViTNepaECGConfig


logger = logging.get_logger(__name__)


class ViTNepaECGPatchEmbeddings(nn.Module):
    """
    This class turns ECG signal values of shape `(batch_size, num_channels, signal_length)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: ViTNepaECGConfig):
        super().__init__()
        signal_length = config.signal_length
        patch_size = config.patch_size
        num_channels = config.num_channels
        hidden_size = config.hidden_size

        num_patches = signal_length // patch_size
        self.signal_length = signal_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv1d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, signal_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, signal_length = signal_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                f"Make sure that the channel dimension of the signal values match with the one set in the configuration. "
                f"Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if signal_length != self.signal_length:
                raise ValueError(
                    f"Input signal length ({signal_length}) doesn't match model ({self.signal_length})."
                )
        
        # Conv1d: (B, C, L) -> (B, D, L//patch_size)
        # Transpose: (B, D, num_patches) -> (B, num_patches, D)
        embeddings = self.projection(signal_values).transpose(1, 2)
        return embeddings


def get_1d_patch_coordinates(
    num_patches: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Computes the 1D coordinates of patch centers, normalized to the range [-1, +1].

    Args:
        num_patches (int): Number of patches along the temporal axis.
        dtype (torch.dtype): The desired data type of the returned tensor.
        device (torch.device): Device on which the tensor is allocated.

    Returns:
        torch.Tensor: A tensor of shape (num_patches, 1), where each row contains the temporal
            coordinate of a patch center, normalized to [-1, +1].
    """
    # Patch centers: 0.5, 1.5, 2.5, ..., num_patches-0.5
    coords = torch.arange(0.5, num_patches, dtype=dtype, device=device)
    # Normalize to [0, 1]
    coords = coords / num_patches
    # Shift range [0, 1] to [-1, +1]
    coords = 2.0 * coords - 1.0
    # Add dimension for compatibility with 2D case: (num_patches, 1)
    coords = coords.unsqueeze(-1)
    return coords


def augment_1d_patch_coordinates(
    coords: torch.Tensor,
    shift: Optional[float] = None,
    jitter: Optional[float] = None,
    rescale: Optional[float] = None,
) -> torch.Tensor:
    """Augment 1D patch coordinates for training."""
    # Shift coords by adding a uniform value in [-shift, shift]
    if shift is not None:
        shift_val = torch.empty((1, 1), device=coords.device, dtype=coords.dtype)
        shift_val = shift_val.uniform_(-shift, shift)
        coords = coords + shift_val

    # Jitter coords by multiplying by a log-uniform value in [1/jitter, jitter]
    if jitter is not None:
        jitter_range = math.log(jitter)
        jitter_val = torch.empty((1, 1), device=coords.device, dtype=coords.dtype)
        jitter_val = jitter_val.uniform_(-jitter_range, jitter_range).exp()
        coords = coords * jitter_val

    # Rescale coords by multiplying by a log-uniform value in [1/rescale, rescale]
    if rescale is not None:
        rescale_range = math.log(rescale)
        rescale_val = torch.empty(1, device=coords.device, dtype=coords.dtype)
        rescale_val = rescale_val.uniform_(-rescale_range, rescale_range).exp()
        coords = coords * rescale_val

    return coords


class ViTNepaECGRopePositionEmbedding(nn.Module):
    """Rotary Position Embedding for 1D ECG signals."""

    inv_freq: torch.Tensor

    def __init__(self, config: ViTNepaECGConfig):
        super().__init__()

        self.config = config
        self.base = config.rope_theta
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_patches = config.signal_length // config.patch_size

        # For 1D RoPE: use head_dim / 2 frequencies (same total as 2D which uses head_dim/4 per coordinate × 2)
        # Original 2D: torch.arange(0, 1, 4 / head_dim) gives head_dim/4 values
        # For 1D: torch.arange(0, 1, 2 / head_dim) gives head_dim/2 values
        # Use same formula as original but with step 2/head_dim instead of 4/head_dim
        step = 2.0 / self.head_dim
        inv_freq = 1.0 / (self.base ** torch.arange(0, 1, step, dtype=torch.float32))  # (head_dim / 2,)
        # Check for NaN/Inf
        if torch.isnan(inv_freq).any() or torch.isinf(inv_freq).any():
            raise ValueError(f"NaN/Inf in inv_freq computation: head_dim={self.head_dim}, base={self.base}")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, signal_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, signal_length = signal_values.shape
        num_patches = signal_length // self.config.patch_size

        device = signal_values.device
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"

        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            # Compute 1D patch coordinates: (num_patches, 1)
            patch_coords = get_1d_patch_coordinates(
                num_patches, dtype=torch.float32, device=device
            )
            if self.training:
                patch_coords = augment_1d_patch_coordinates(
                    patch_coords,
                    shift=self.config.pos_embed_shift,
                    jitter=self.config.pos_embed_jitter,
                    rescale=self.config.pos_embed_rescale,
                )

            # For 1D RoPE:
            # patch_coords: (num_patches, 1)
            # inv_freq: (head_dim / 2,)
            # angles: (num_patches, 1) × (head_dim / 2,) -> (num_patches, head_dim / 2)
            # Then repeat to get (num_patches, head_dim)
            angles = 2 * math.pi * patch_coords[:, :, None] * self.inv_freq[None, None, :]  # (num_patches, 1, head_dim/2)
            angles = angles.squeeze(1)  # (num_patches, head_dim/2)
            
            # Check for NaN/Inf before repeating
            if torch.isnan(angles).any() or torch.isinf(angles).any():
                raise ValueError(f"NaN/Inf in angles computation: patch_coords_range=[{patch_coords.min():.6f}, {patch_coords.max():.6f}], inv_freq_range=[{self.inv_freq.min():.6f}, {self.inv_freq.max():.6f}]")
            
            angles = angles.repeat(1, 2)  # (num_patches, head_dim)

            cos = torch.cos(angles)
            sin = torch.sin(angles)
            
            # Final check
            if torch.isnan(cos).any() or torch.isnan(sin).any():
                raise ValueError(f"NaN in cos/sin after computation")

        dtype = signal_values.dtype
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


class ViTNepaECGEmbeddings(nn.Module):
    """
    Construct the CLS token and patch embeddings for ECG signals.
    Positional information is expected to be handled with RoPE inside attention.
    """

    def __init__(self, config: ViTNepaECGConfig, use_mask_token: bool = False):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTNepaECGPatchEmbeddings(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def forward(
        self,
        signal_values: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is not None:
            warnings.warn(
                "`position_ids` provided but ViTNepaECGEmbeddings does not use absolute position embeddings. "
                "Positional information should be added with RoPE in the attention layer."
            )

        batch_size, num_channels, signal_length = signal_values.shape
        embeddings = self.patch_embeddings(signal_values, interpolate_pos_encoding=False)
        embeddings_clean = embeddings

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings_clean = torch.cat((cls_tokens, embeddings_clean), dim=1)

        embeddings = self.dropout(embeddings)

        return embeddings, embeddings_clean


@auto_docstring
class ViTNepaECGModel(ViTNepaPreTrainedModel):
    def __init__(self, config: ViTNepaECGConfig, use_mask_token: bool = False):
        r"""
        use_mask_token (`bool`, *optional*, defaults to `False`):
            Whether to use a mask token for masked signal modeling.
        """
        super().__init__(config)
        self.config = config

        self.embeddings = ViTNepaECGEmbeddings(config, use_mask_token=use_mask_token)
        self.rope_embeddings = ViTNepaECGRopePositionEmbedding(config)
        self.encoder = ViTNepaEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()
    
    def _init_weights(self, module):
        """Initialize weights - override to handle Conv1d for ECG"""
        # First call parent to handle Conv2d, Linear, LayerNorm, etc.
        super()._init_weights(module)
        
        # Also handle Conv1d (used in ECG patch embeddings)
        if isinstance(module, nn.Conv1d):
            # Upcast to float32 for trunc_normal, then cast back
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), 
                mean=0.0, 
                std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()

    def get_input_embeddings(self) -> ViTNepaECGPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: dict[int, list[int]]):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        is_pretraining: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithEmbedding:
        r"""
        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, signal_length)`):
                ECG signal values. For compatibility with HuggingFace, we use `pixel_values` as the input name.
            position_ids (`torch.LongTensor`, *optional*):
                Ignored. Positional information is handled by RoPE.
            bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
                Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
            head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            interpolate_pos_encoding (`bool`, *optional*):
                Whether to interpolate positional encodings (not used for ECG).
            is_pretraining (`bool`, *optional*, defaults to `False`):
                Whether the model is being used for pretraining.
        """

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # Cast input to expected dtype
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_input, embedding_clean = self.embeddings(
            pixel_values,
            position_ids=position_ids,
            bool_masked_pos=bool_masked_pos,
            interpolate_pos_encoding=interpolate_pos_encoding
        )
        position_embeds = self.rope_embeddings(pixel_values)

        encoder_outputs: BaseModelOutput = self.encoder(
            embedding_input,
            head_mask=head_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeds
        )
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.layernorm(sequence_output)
        attentions = encoder_outputs.attentions

        hidden_states = encoder_outputs.hidden_states

        return BaseModelOutputWithEmbedding(
            last_hidden_state=sequence_output,
            input_embedding=embedding_clean,
            attentions=attentions,
            hidden_states=hidden_states
        )


class ViTNepaECGForPreTraining(ViTNepaPreTrainedModel):
    def __init__(self, config: ViTNepaECGConfig):
        super().__init__(config)

        self.vit_nepa = ViTNepaECGModel(config)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> EmbeddedModelingOutput:
        r"""
        Forward pass for pretraining. Computes prediction loss between input and output embeddings.
        """

        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
            position_ids=position_ids,
            head_mask=head_mask,
            output_attentions=output_attentions,
            interpolate_pos_encoding=interpolate_pos_encoding,
            is_pretraining=True,
            **kwargs,
        )

        sequence_input = outputs.input_embedding
        sequence_output = outputs.last_hidden_state

        # Debug: Check for NaN/Inf in embeddings before loss computation
        if torch.isnan(sequence_input).any() or torch.isinf(sequence_input).any():
            logger.warning(f"NaN/Inf detected in input_embedding: input_nan={torch.isnan(sequence_input).any()}, input_inf={torch.isinf(sequence_input).any()}")
        if torch.isnan(sequence_output).any() or torch.isinf(sequence_output).any():
            logger.warning(f"NaN/Inf detected in last_hidden_state: output_nan={torch.isnan(sequence_output).any()}, output_inf={torch.isinf(sequence_output).any()}")

        embedded_loss = prediction_loss(sequence_input, sequence_output)

        # Debug: Check loss value
        if torch.isnan(embedded_loss) or torch.isinf(embedded_loss):
            logger.error(f"NaN/Inf in loss! input_stats: mean={sequence_input.abs().mean():.6f}, std={sequence_input.std():.6f}, "
                        f"output_stats: mean={sequence_output.abs().mean():.6f}, std={sequence_output.std():.6f}")
        elif embedded_loss.item() == 0.0:
            logger.warning(f"Loss is exactly 0.0 - this may indicate constant embeddings. "
                          f"input_range: [{sequence_input.min():.6f}, {sequence_input.max():.6f}], "
                          f"output_range: [{sequence_output.min():.6f}, {sequence_output.max():.6f}]")

        return EmbeddedModelingOutput(
            loss=embedded_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@auto_docstring
class ViTNepaECGForImageClassification(ViTNepaPreTrainedModel):
    def __init__(self, config: ViTNepaECGConfig):
        super().__init__(config)
        self.add_pooling_layer = config.add_pooling_layer
        self.num_signal_tokens = config.signal_length // config.patch_size

        self.num_labels = config.num_labels
        self.vit_nepa = ViTNepaECGModel(config)
        self.pooler = lambda hidden_states: hidden_states.mean(dim=1) if config.add_pooling_layer else None
        self.fc_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) if config.add_pooling_layer else None

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> ImageClassifierOutput:
        r"""
        Args:
            pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, signal_length)`):
                ECG signal values.
            head_mask (`torch.Tensor`, *optional*):
                Mask to nullify selected heads of the self-attention modules.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
            labels (`torch.Tensor` of shape `(batch_size, num_labels)`, *optional*):
                Labels for computing the multi-label classification loss. Values should be in `[0, 1]`.
                Uses binary cross-entropy loss.
            interpolate_pos_encoding (`bool`, *optional*):
                Whether to interpolate positional encodings (not used for ECG).
        """

        outputs: BaseModelOutputWithEmbedding = self.vit_nepa(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            interpolate_pos_encoding=interpolate_pos_encoding,
            **kwargs,
        )

        sequence_output = outputs.last_hidden_state
        if self.add_pooling_layer:
            signal_tokens = sequence_output[:, -self.num_signal_tokens:, :]
            pooled_output = signal_tokens.mean(dim=1)
            pooled_output = self.fc_norm(pooled_output)
        else:
            pooled_output = sequence_output[:, 0, :]  # CLS token
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss = self.loss_function(labels, logits, self.config, **kwargs)

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "ViTNepaECGForImageClassification",
    "ViTNepaECGForPreTraining",
    "ViTNepaECGModel",
]

