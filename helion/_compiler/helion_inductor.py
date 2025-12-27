"""Helion integration with PyTorch Inductor.

Importing this module registers the Helion kernel lowering with Inductor.
"""

from __future__ import annotations

# Import to register the @register_lowering decorator
from .fusion import inductor as _inductor  # noqa: F401
