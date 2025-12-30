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
"""ViTNepaECG model configuration"""

from typing import Optional

from transformers.utils import logging
from ..vit_nepa.configuration_vit_nepa import ViTNepaConfig


logger = logging.get_logger(__name__)


class ViTNepaECGConfig(ViTNepaConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTNepaECGModel`]. It is used to instantiate a ViTNepaECG
    model according to the specified arguments, defining the model architecture for ECG signals.

    Configuration objects inherit from [`ViTNepaConfig`] and can be used to control the model outputs. Read the
    documentation from [`ViTNepaConfig`] for more information.

    Args:
        signal_length (`int`, *optional*, defaults to 5000):
            The length of the input ECG signal in samples (at 500 Hz, 5000 samples = 10 seconds).
        patch_size (`int`, *optional*, defaults to 25):
            The size of each patch in samples. With signal_length=5000, this gives 200 patches.
        num_channels (`int`, *optional*, defaults to 12):
            The number of ECG leads (channels). Standard 12-lead ECG.
        **kwargs:
            Additional keyword arguments passed to [`ViTNepaConfig`].

    Example:

    ```python
    >>> from models.vit_nepa_ecg import ViTNepaECGConfig, ViTNepaECGModel

    >>> # Initializing a ViTNepaECG configuration
    >>> configuration = ViTNepaECGConfig(signal_length=5000, patch_size=25)

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = ViTNepaECGModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "vit_nepa_ecg"

    def __init__(
        self,
        signal_length: int = 5000,
        patch_size: int = 25,
        num_channels: int = 12,
        **kwargs,
    ):
        # For ECG, we don't use image_size, but we need to set it for compatibility
        # We'll use signal_length for the 1D case
        super().__init__(
            patch_size=patch_size,
            num_channels=num_channels,
            image_size=signal_length,  # For compatibility, but not used in 1D
            **kwargs,
        )
        self.signal_length = signal_length

