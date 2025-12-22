"""Helion template kernel. Re-exports from fusion package."""
from .fusion import HelionTemplate, inject_closure_params, rename_function
HelionTemplateKernel = HelionTemplate  # Backwards compatibility
