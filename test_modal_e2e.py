"""End-to-end test: dispatch a real triton kernel benchmark to Modal from Mac.

This test does NOT require a local GPU. It:
1. Creates a ModalBenchmarkDispatcher
2. Sends hand-crafted triton code + serialized args to a Modal H100 worker
3. Gets back benchmark timing results

Run: python test_modal_e2e.py
"""

from __future__ import annotations

import torch

from helion.autotuner.modal_search import (
    ModalBenchmarkDispatcher,
    _serialize_args,
)

# Hand-crafted triton add kernel (self-contained, no helion dependency)
TRITON_ADD_CODE = '''
import torch
import triton
import triton.language as tl

@triton.jit
def _add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)

def add_kernel(x, y):
    out = torch.empty_like(x)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=1024)
    return out
'''

# A second version with a different block size to test parallel dispatch
TRITON_ADD_CODE_V2 = TRITON_ADD_CODE.replace("BLOCK_SIZE=1024", "BLOCK_SIZE=512")

def main():
    print("=" * 60)
    print("Modal E2E Test: Dispatching triton benchmarks to H100")
    print("=" * 60)

    # Create test tensors on CPU and serialize them
    n = 1024 * 1024  # 1M elements
    x = torch.randn(n, dtype=torch.float32)
    y = torch.randn(n, dtype=torch.float32)
    args_bytes = _serialize_args([x, y])
    print(f"Serialized args: {len(args_bytes)} bytes ({n} float32 elements each)")

    # Create dispatcher
    dispatcher = ModalBenchmarkDispatcher(gpu_type="H100", max_concurrent=10)
    print("Dispatcher created, dispatching 2 configs to Modal...")
    print()

    # Dispatch both configs in parallel
    results = dispatcher.dispatch_batch(
        triton_codes=[TRITON_ADD_CODE, TRITON_ADD_CODE_V2],
        fn_names=["add_kernel", "add_kernel"],
        args_bytes=args_bytes,
    )

    # Print results
    for i, result in enumerate(results):
        status = result.get("status", "unknown")
        perf = result.get("perf", float("inf"))
        error = result.get("error")
        block_size = 1024 if i == 0 else 512
        print(f"Config {i+1} (BLOCK_SIZE={block_size}):")
        print(f"  status: {status}")
        if status == "ok":
            print(f"  perf:   {perf:.4f} ms")
        else:
            print(f"  error:  {error}")
        print()

    # Verify at least one succeeded
    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"Results: {ok_count}/{len(results)} configs benchmarked successfully")

    if ok_count > 0:
        best = min(results, key=lambda r: float(r.get("perf", float("inf"))))
        print(f"Best: {best.get('perf'):.4f} ms")
        print()
        print("SUCCESS: Modal dispatch working end-to-end!")
    else:
        print()
        print("FAILED: No configs succeeded on Modal")
        for r in results:
            if r.get("error"):
                print(f"  Error: {r['error']}")


if __name__ == "__main__":
    main()
