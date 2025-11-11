"""Fragment encoding/decoding strategies for machine learning based autotuners.

This module provides a clean abstraction for encoding different fragment types
into numerical tensors and decoding them back. Each fragment type has its own
encoder that knows how to analyze, encode, and decode itself.
"""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
import math

from .config_fragment import BooleanFragment
from .config_fragment import ConfigSpecFragment
from .config_fragment import EnumFragment
from .config_fragment import IntegerFragment
from .config_fragment import ListOf
from .config_fragment import PermutationFragment
from .config_fragment import PowerOfTwoFragment


class FragmentEncoder(ABC):
    """Base class for encoding/decoding fragment values."""

    def __init__(self, fragment: ConfigSpecFragment) -> None:
        self.fragment = fragment

    @abstractmethod
    def n_dims(self) -> int:
        """Return the number of dimensions this fragment uses in encoded space."""

    @abstractmethod
    def is_categorical(self) -> bool:
        """Return whether this fragment represents categorical data."""

    @abstractmethod
    def encode(self, value: object) -> list[float]:
        """Encode a value into a list of floats."""

    @abstractmethod
    def decode(self, encoded: list[float]) -> object:
        """Decode a list of floats back to the original value type."""


class CategoricalEncoder(FragmentEncoder):
    """Encoder for EnumFragment and BooleanFragment using one-hot encoding."""

    def __init__(
        self, fragment: EnumFragment | BooleanFragment, choices: list[object]
    ) -> None:
        super().__init__(fragment)
        self.choices = choices

    def n_dims(self) -> int:
        return len(self.choices)

    def is_categorical(self) -> bool:
        return True

    def encode(self, value: object) -> list[float]:
        idx = self.choices.index(value)
        return [1.0 if i == idx else 0.0 for i in range(len(self.choices))]

    def decode(self, encoded: list[float]) -> object:
        choice_idx = max(range(len(self.choices)), key=lambda i: encoded[i])
        return self.choices[choice_idx]


class PowerOfTwoEncoder(FragmentEncoder):
    """Encoder for PowerOfTwoFragment using log2 transformation."""

    def __init__(self, fragment: PowerOfTwoFragment) -> None:
        super().__init__(fragment)
        self.log_min = math.log2(fragment.low)
        self.log_max = math.log2(fragment.high)

    def n_dims(self) -> int:
        return 1

    def is_categorical(self) -> bool:
        return False

    def encode(self, value: int) -> list[float]:
        return [math.log2(value)]

    def decode(self, encoded: list[float]) -> int:
        log_val = encoded[0]
        power = int(round(log_val))
        power = max(int(self.log_min), min(power, int(self.log_max)))
        return 2**power


class IntegerEncoder(FragmentEncoder):
    """Encoder for IntegerFragment using raw values."""

    def __init__(self, fragment: IntegerFragment) -> None:
        super().__init__(fragment)
        self.min_val = fragment.low
        self.max_val = fragment.high

    def n_dims(self) -> int:
        return 1

    def is_categorical(self) -> bool:
        return False

    def encode(self, value: object) -> list[float]:
        return [float(value)]

    def decode(self, encoded: list[float]) -> int:
        value = int(round(encoded[0]))
        return max(self.min_val, min(value, self.max_val))


class PermutationEncoder(FragmentEncoder):
    """Encoder for PermutationFragment using one-hot encoding for each position."""

    def __init__(self, fragment: PermutationFragment) -> None:
        super().__init__(fragment)
        self.length = fragment.length

    def n_dims(self) -> int:
        return self.length * self.length

    def is_categorical(self) -> bool:
        return True

    def encode(self, value: list[int]) -> list[float]:
        encoded = []
        for pos in range(self.length):
            val = value[pos]
            for v in range(self.length):
                encoded.append(1.0 if v == val else 0.0)
        return encoded

    def decode(self, encoded: list[float]) -> list[int]:
        perm = []
        used = set()

        for pos in range(self.length):
            start_idx = pos * self.length
            one_hot = encoded[start_idx : start_idx + self.length]
            val = max(range(self.length), key=lambda i: one_hot[i])
            perm.append(val)
            used.add(val)

        # Fix invalid permutation (duplicates/missing values)
        if len(used) != self.length:
            available = [v for v in range(self.length) if v not in used]
            seen = set()
            fixed_perm = []
            for val in perm:
                if val in seen:
                    fixed_val = available.pop(0)
                    fixed_perm.append(fixed_val)
                else:
                    fixed_perm.append(val)
                    seen.add(val)
            return fixed_perm

        return perm


class ListOfEncoder(FragmentEncoder):
    """Encoder for ListOf fragments, delegates to inner encoder."""

    def __init__(self, fragment: ListOf, inner_encoder: FragmentEncoder) -> None:
        super().__init__(fragment)
        self.length = fragment.length
        self.inner_encoder = inner_encoder
        self.inner_dims = inner_encoder.n_dims()

    def n_dims(self) -> int:
        return self.length * self.inner_dims

    def is_categorical(self) -> bool:
        """Return True if the inner encoder is categorical."""
        return self.inner_encoder.is_categorical()

    def encode(self, value: list[object]) -> list[float]:
        encoded = []
        for v in value:
            encoded.extend(self.inner_encoder.encode(v))
        return encoded

    def decode(self, encoded: list[float]) -> list[object]:
        decoded = []
        for i in range(self.length):
            start_idx = i * self.inner_dims
            element_encoded = encoded[start_idx : start_idx + self.inner_dims]
            decoded.append(self.inner_encoder.decode(element_encoded))
        return decoded


def create_encoder(fragment: ConfigSpecFragment) -> FragmentEncoder:
    """Factory function to create the appropriate encoder for a fragment."""
    if isinstance(fragment, BooleanFragment):
        return CategoricalEncoder(fragment, [False, True])
    if isinstance(fragment, EnumFragment):
        return CategoricalEncoder(fragment, list(fragment.choices))
    if isinstance(fragment, PowerOfTwoFragment):
        return PowerOfTwoEncoder(fragment)
    if isinstance(fragment, IntegerFragment):
        return IntegerEncoder(fragment)
    if isinstance(fragment, PermutationFragment):
        return PermutationEncoder(fragment)
    if isinstance(fragment, ListOf):
        inner_encoder = create_encoder(fragment.inner)
        return ListOfEncoder(fragment, inner_encoder)
    raise ValueError(f"Unsupported fragment type: {type(fragment).__name__}")


class ConfigEncoder:
    """Encodes and decodes entire configurations using fragment encoders."""

    def __init__(self, flat_spec: list[ConfigSpecFragment]) -> None:
        """Initialize encoders for all fragments in the spec.

        Args:
            flat_spec: List of fragment specifications
        """
        self.encoders = [create_encoder(fragment) for fragment in flat_spec]
        self.total_dims = sum(encoder.n_dims() for encoder in self.encoders)

        # Build categorical dimension indices (absolute positions)
        self.cat_dims = []
        offset = 0
        for encoder in self.encoders:
            n_dims = encoder.n_dims()
            if encoder.is_categorical():
                # All dimensions of this encoder are categorical
                self.cat_dims.extend(range(offset, offset + n_dims))
            offset += n_dims

    def encode(self, flat_config: list[object]) -> list[float]:
        """Encode a flat configuration into a list of floats.

        Args:
            flat_config: List of configuration values

        Returns:
            List of encoded float values
        """
        encoded = []
        for value, encoder in zip(flat_config, self.encoders, strict=False):
            encoded.extend(encoder.encode(value))
        return encoded

    def decode(self, encoded: list[float]) -> list[object]:
        """Decode a list of floats back into a flat configuration.

        Args:
            encoded: List of encoded float values

        Returns:
            List of decoded configuration values
        """
        decoded = []
        idx = 0
        for encoder in self.encoders:
            n_dims = encoder.n_dims()
            fragment_encoded = encoded[idx : idx + n_dims]
            decoded.append(encoder.decode(fragment_encoded))
            idx += n_dims
        return decoded
