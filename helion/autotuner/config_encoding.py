from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from .config_fragment import Category

if TYPE_CHECKING:
    from .config_generation import ConfigGeneration
    from .config_generation import FlatConfig


class ConfigEncoder:
    """
    Encodes Helion configurations into numerical vectors for Gaussian Process models.

    Handles various config types:
    - Power-of-2 values: log2 encoding
    - Integers: direct encoding with normalization
    - Booleans: 0/1 encoding
    - Enums: one-hot encoding
    - Permutations: inversion count encoding
    """

    def __init__(self, config_gen: ConfigGeneration) -> None:
        """
        Initialize the encoder with a configuration generator.

        Args:
            config_gen: The configuration generator containing the flat spec.
        """
        self.config_gen = config_gen
        self.flat_spec = config_gen.flat_spec
        self._compute_encoding_metadata()

    def _compute_encoding_metadata(self) -> None:
        """Precompute metadata for encoding to determine output dimensionality."""
        self.encoded_dim = 0
        self.encoding_map: list[tuple[int, int, str]] = []  # (start_idx, end_idx, type)

        for spec in self.flat_spec:
            category = spec.category()
            start_idx = self.encoded_dim

            if category in {Category.BLOCK_SIZE, Category.NUM_WARPS, Category.NUM_STAGES}:
                # Single numerical value
                self.encoded_dim += 1
                self.encoding_map.append((start_idx, self.encoded_dim, "numerical"))
            elif hasattr(spec, "choices"):
                # Enum - one-hot encoding
                num_choices = len(spec.choices)  # type: ignore
                self.encoded_dim += num_choices
                self.encoding_map.append((start_idx, self.encoded_dim, "enum"))
            else:
                # Boolean or other single value
                self.encoded_dim += 1
                self.encoding_map.append((start_idx, self.encoded_dim, "numerical"))

    def encode(self, flat_config: FlatConfig) -> np.ndarray:
        """
        Convert a flat configuration to a numerical vector.

        Args:
            flat_config: The flat configuration values.

        Returns:
            A numpy array suitable for GP training.
        """
        encoded = np.zeros(self.encoded_dim, dtype=np.float64)
        flat_idx = 0

        for spec in self.flat_spec:
            value = flat_config[flat_idx]
            category = spec.category()
            enc_start, enc_end, enc_type = self.encoding_map[flat_idx]

            if enc_type == "numerical":
                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: use log2 encoding
                    if isinstance(value, (int, float)) and value > 0:
                        encoded[enc_start] = math.log2(float(value))
                    else:
                        encoded[enc_start] = 0.0
                elif category == Category.NUM_STAGES:
                    # Integer: direct encoding
                    encoded[enc_start] = float(value) if isinstance(value, (int, float)) else 0.0
                else:
                    # Boolean or other: 0/1
                    encoded[enc_start] = float(value) if isinstance(value, (bool, int, float)) else 0.0
            elif enc_type == "enum":
                # One-hot encoding
                if hasattr(spec, "choices"):
                    choices = spec.choices  # type: ignore
                    try:
                        choice_idx = choices.index(value)
                        encoded[enc_start + choice_idx] = 1.0
                    except (ValueError, IndexError):
                        # Default to first choice if value not found
                        encoded[enc_start] = 1.0

            flat_idx += 1

        return encoded

    def get_bounds(self) -> list[tuple[float, float]]:
        """
        Get bounds for each encoded dimension.

        Returns:
            List of (min, max) tuples for each dimension.
        """
        bounds: list[tuple[float, float]] = []
        flat_idx = 0

        for spec in self.flat_spec:
            category = spec.category()
            enc_start, enc_end, enc_type = self.encoding_map[flat_idx]

            if enc_type == "numerical":
                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: log2 bounds
                    min_val = math.log2(float(spec.min_size))  # type: ignore
                    max_val = math.log2(float(spec.max_size))  # type: ignore
                    bounds.append((min_val, max_val))
                elif category == Category.NUM_STAGES:
                    # Integer bounds
                    bounds.append((float(spec.min_size), float(spec.max_size)))  # type: ignore
                else:
                    # Boolean: 0 or 1
                    bounds.append((0.0, 1.0))
            elif enc_type == "enum":
                # One-hot: each dimension is 0 or 1
                num_choices = enc_end - enc_start
                bounds.extend([(0.0, 1.0)] * num_choices)

            flat_idx += 1

        return bounds
