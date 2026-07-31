"""Opt-in Helion KDA substitutions for SGLang A/B benchmarks."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_source_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


enable_decode = (
    os.environ.get("SGLANG_KDA_HELION_DECODE") == "1"
    or os.environ.get("SGLANG_KDA_HELION") == "1"
)
enable_prefill = os.environ.get("SGLANG_KDA_HELION_PREFILL") == "1"

if enable_decode or enable_prefill:
    from sglang.srt.layers.attention.linear.kernels import kda_triton

if enable_decode:
    decode_root = os.environ.get("HELION_KDA_DECODE_ROOT")
    if decode_root is None:
        from examples.linear.kda_packed_decode import (
            helion_fused_recurrent_kda_packed_decode,
        )
    else:
        decode_module = _load_source_module(
            "_helion_kda_decode_ab",
            Path(decode_root) / "examples/linear/kda_packed_decode.py",
        )
        helion_fused_recurrent_kda_packed_decode = (
            decode_module.helion_fused_recurrent_kda_packed_decode
        )

    _decode_called = False

    def _helion_packed_decode(*args: object, **kwargs: object) -> object:
        global _decode_called
        if not _decode_called:
            print("[helion-kda-ab] first packed decode call", flush=True)
            _decode_called = True
        return helion_fused_recurrent_kda_packed_decode(*args, **kwargs)

    kda_triton.fused_recurrent_kda_packed_decode = _helion_packed_decode
    print("[helion-kda-ab] installed packed decode substitution", flush=True)

if enable_prefill:
    prefill_root = os.environ.get("HELION_KDA_PREFILL_ROOT")
    if prefill_root is None:
        from examples.linear.kda_prefill import chunk_kda as helion_chunk_kda
    else:
        prefill_module = _load_source_module(
            "_helion_kda_prefill_ab",
            Path(prefill_root) / "examples/linear/kda_prefill.py",
        )
        helion_chunk_kda = prefill_module.chunk_kda
    newton_schulz = os.environ.get("HELION_KDA_PREFILL_NEWTON_SCHULZ") == "1"

    _prefill_called = False

    def _helion_chunk_kda(*args: object, **kwargs: object) -> object:
        global _prefill_called
        if not _prefill_called:
            inverse = "newton-schulz" if newton_schulz else "forward-substitution"
            print(f"[helion-kda-ab] first prefill call ({inverse})", flush=True)
            _prefill_called = True
        kwargs["newton_schulz"] = newton_schulz
        return helion_chunk_kda(*args, **kwargs)

    kda_triton.chunk_kda = _helion_chunk_kda
    print("[helion-kda-ab] installed prefill substitution", flush=True)
