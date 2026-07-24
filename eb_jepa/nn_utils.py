"""Shared utilities for neural network initialization and common patterns."""

import torch.nn as nn


def init_module_weights(m, std: float = 0.02):
    """
    Initialize weights for common layer types using truncated normal distribution.

    This is a unified weight initialization function used across the codebase.
    Apply it via module.apply(init_module_weights) or as a method wrapper.

    Args:
        m: PyTorch module to initialize
        std: Standard deviation for truncated normal initialization (default: 0.02)
    """
    if isinstance(
        m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)
    ):
        nn.init.trunc_normal_(m.weight, std=std)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

