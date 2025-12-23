"""Helion template kernel. Re-exports from fusion package."""
from .fusion import HelionTemplateBuffer, inject_closure_params, rename_function
HelionTemplateKernel = HelionTemplateBuffer  # Backwards compatibility
