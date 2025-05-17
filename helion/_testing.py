from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING

import torch

from .runtime.config import Config

if TYPE_CHECKING:
    from pathlib import Path
    import types

    from .runtime.kernel import Kernel


DEVICE = torch.device("cuda")


def import_path(filename: Path) -> types.ModuleType:
    module_name = f"{__name__}.{filename.stem}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, filename)
        assert spec is not None
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        sys.modules[module_name] = module
    return sys.modules[module_name]


def code_and_output(
    # pyre-ignore[11]
    fn: Kernel,
    args: tuple[object, ...],
    **kwargs: object,
) -> tuple[str, object]:
    if kwargs:
        config = Config(**kwargs)  # pyre-ignore[6]
    elif fn.configs:
        (config,) = fn.configs
    else:
        config = fn.bind(args).config_spec.default_config()
    code = fn.bind(args).to_triton_code(config)
    compiled_kernel = fn.bind(args).compile_config(config)
    try:
        result = compiled_kernel(*args)
    except Exception:
        sys.stderr.write(f"Failed to run kernel:\n{code}\n")
        raise
    return code, result
