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

            if category in {
                Category.BLOCK_SIZE,
                Category.NUM_WARPS,
            }:
                # Single numerical value
                self.encoded_dim += 1
                self.encoding_map.append((start_idx, self.encoded_dim, "numerical"))
            elif hasattr(spec, "choices"):
                # Enum - one-hot encoding
                num_choices = len(spec.choices)  # type: ignore[no-untyped-call]
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

        for flat_idx, spec in enumerate(self.flat_spec):
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
                else:
                    # Other numerical: direct encoding
                    encoded[enc_start] = (
                        float(value) if isinstance(value, (int, float)) else 0.0
                    )
            elif enc_type == "enum":
                # One-hot encoding
                if hasattr(spec, "choices"):
                    choices = spec.choices  # type: ignore[attr-defined]
                    try:
                        choice_idx = choices.index(value)
                        encoded[enc_start + choice_idx] = 1.0
                    except (ValueError, IndexError):
                        # Default to first choice if value not found
                        encoded[enc_start] = 1.0

        return encoded

    def get_bounds(self) -> list[tuple[float, float]]:
        """
        Get bounds for each encoded dimension.

        Returns:
            List of (min, max) tuples for each dimension.
        """
        bounds: list[tuple[float, float]] = []

        for flat_idx, spec in enumerate(self.flat_spec):
            category = spec.category()
            enc_start, enc_end, enc_type = self.encoding_map[flat_idx]

            if enc_type == "numerical":
                if category in {Category.BLOCK_SIZE, Category.NUM_WARPS}:
                    # Power-of-2: log2 bounds
                    min_val = math.log2(float(spec.low))  # type: ignore[attr-defined]
                    max_val = math.log2(float(spec.high))  # type: ignore[attr-defined]
                    bounds.append((min_val, max_val))
                else:
                    # Other numerical bounds
                    bounds.append(
                        (float(spec.low), float(spec.high))  # type: ignore[attr-defined]
                    )
            elif enc_type == "enum":
                # One-hot: each dimension is 0 or 1
                num_choices = enc_end - enc_start
                bounds.extend([(0.0, 1.0)] * num_choices)

        return bounds
