"""Test script for Modal-based autotuner.

Run from Mac (no GPU needed):
    python test_modal_search.py

Requires:
    - modal package installed (pip install modal)
    - modal auth configured (modal token set)
    - torch installed (CPU-only is fine for the coordinator)
"""

from __future__ import annotations

import os
import sys

# Test 1: Import test
print("=" * 60)
print("Test 1: Import test")
print("=" * 60)

from helion.autotuner.modal_search import (
    ModalBenchmarkDispatcher,
    ModalSearch,
    _deserialize_args,
    _find_compiled_fn_name,
    _serialize_args,
    modal_autotune,
)

print("  All imports OK")

# Test 2: Args serialization roundtrip
print()
print("=" * 60)
print("Test 2: Args serialization roundtrip")
print("=" * 60)

import torch

x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)
args = [x, y]

serialized = _serialize_args(args)
print(f"  Serialized {len(serialized)} bytes")

deserialized = _deserialize_args(serialized, device="cpu")
assert len(deserialized) == 2
assert isinstance(deserialized[0], torch.Tensor)
assert isinstance(deserialized[1], torch.Tensor)
assert torch.equal(deserialized[0], x)
assert torch.equal(deserialized[1], y)
print("  Roundtrip OK: tensors match exactly")

# Test 3: Mixed args (tensor + scalar)
args_mixed = [x, 42, "hello", 3.14]
serialized_mixed = _serialize_args(args_mixed)
deserialized_mixed = _deserialize_args(serialized_mixed, device="cpu")
assert len(deserialized_mixed) == 4
assert torch.equal(deserialized_mixed[0], x)
assert deserialized_mixed[1] == 42
assert deserialized_mixed[2] == "hello"
assert deserialized_mixed[3] == 3.14
print("  Mixed args roundtrip OK")

# Test 4: _find_compiled_fn_name
print()
print("=" * 60)
print("Test 4: _find_compiled_fn_name")
print("=" * 60)

sample_triton_code = '''
import triton
import triton.language as tl

@triton.jit
def _triton_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _triton_add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
'''
fn_name = _find_compiled_fn_name(sample_triton_code)
assert fn_name == "add", f"Expected 'add', got '{fn_name}'"
print(f"  Found fn name: '{fn_name}' (correct)")

# Test with only triton kernel functions
triton_only_code = '''
@triton.jit
def _triton_kernel(x_ptr):
    pass
'''
fn_name2 = _find_compiled_fn_name(triton_only_code)
assert fn_name2 == "_triton_kernel"
print(f"  Fallback to triton fn: '{fn_name2}' (correct)")

# Test 5: ModalBenchmarkDispatcher creation
print()
print("=" * 60)
print("Test 5: ModalBenchmarkDispatcher creation")
print("=" * 60)

dispatcher = ModalBenchmarkDispatcher(gpu_type="H100", max_concurrent=50)
assert dispatcher.gpu_type == "H100"
assert dispatcher.max_concurrent == 50
assert dispatcher._app is None  # Lazy init
print("  Dispatcher created OK (lazy, no Modal app yet)")

# Test 6: Settings integration
print()
print("=" * 60)
print("Test 6: Settings integration")
print("=" * 60)

from helion.runtime.settings import Settings

settings = Settings(autotune_modal_gpu="A100", autotune_modal_max_concurrent=25)
assert settings.autotune_modal_gpu == "A100"
assert settings.autotune_modal_max_concurrent == 25
print(f"  autotune_modal_gpu: {settings.autotune_modal_gpu}")
print(f"  autotune_modal_max_concurrent: {settings.autotune_modal_max_concurrent}")

# Test defaults
settings_default = Settings()
assert settings_default.autotune_modal_gpu == "H100"
assert settings_default.autotune_modal_max_concurrent == 50
print(f"  Defaults: gpu={settings_default.autotune_modal_gpu}, concurrent={settings_default.autotune_modal_max_concurrent}")

# Test env var override
os.environ["HELION_AUTOTUNE_MODAL_GPU"] = "T4"
os.environ["HELION_AUTOTUNE_MODAL_MAX_CONCURRENT"] = "10"
settings_env = Settings()
assert settings_env.autotune_modal_gpu == "T4"
assert settings_env.autotune_modal_max_concurrent == 10
print(f"  Env override: gpu={settings_env.autotune_modal_gpu}, concurrent={settings_env.autotune_modal_max_concurrent}")
del os.environ["HELION_AUTOTUNE_MODAL_GPU"]
del os.environ["HELION_AUTOTUNE_MODAL_MAX_CONCURRENT"]

# Test 7: search_algorithms registry
print()
print("=" * 60)
print("Test 7: search_algorithms registry")
print("=" * 60)

from helion.autotuner import search_algorithms

assert "ModalSearch" in search_algorithms
assert search_algorithms["ModalSearch"] is ModalSearch
print(f"  Registry: {list(search_algorithms.keys())}")
print("  ModalSearch is registered correctly")

print()
print("=" * 60)
print("ALL OFFLINE TESTS PASSED")
print("=" * 60)
print()
print("To run the full Modal integration test:")
print("  1. Run: modal token set")
print("  2. Then on a machine with a GPU, run:")
print("     HELION_AUTOTUNER=ModalSearch python examples/add.py")
print()
print("Or use modal_autotune() for GPU-less Mac usage.")
