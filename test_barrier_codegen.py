"""
Minimal repro of Helion persistent_blocked + hl.barrier() code duplication issue.

This test defines a simple 3-phase kernel with 2 barriers, generates the Helion
Triton code, and compares its structure against a hand-written Triton equivalent.

The expected behavior is that each phase's code appears exactly ONCE in the
generated kernel. The actual behavior is that the full phase dispatch (if/elif
chain for ALL phases) is duplicated in every barrier segment, leading to ~Nx
code bloat for N phases.

Run: python test_barrier_codegen.py
"""

import io
import math
import os
import re
import sys
import textwrap

# Set before any Helion kernel is compiled so the output path is printed
os.environ["HELION_PRINT_OUTPUT_CODE"] = "1"

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers
from helion.runtime import default_launcher as _default_launcher

import helion
import helion.language as hl


# ==============================================================================
# Simple 3-phase Helion kernel with barriers
# ==============================================================================

@helion.kernel(
    static_shapes=True,
    config=helion.Config(
        block_sizes=[1, 128, 1, 128, 1, 128],
        pid_type="persistent_blocked",
    ),
)
def three_phase_helion(
    x: torch.Tensor,       # [1, N]
    weight: torch.Tensor,   # [N]
    eps: float,
) -> torch.Tensor:
    """
    3-phase kernel:
      Phase 1: RMSNorm        y = x * rsqrt(mean(x^2) + eps) * weight
      Phase 2: SiLU            z = y * sigmoid(y)
      Phase 3: Scale           output = z * 0.5
    """
    N = hl.specialize(x.size(1))
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    output = torch.empty_like(x)

    # Phase 1: RMSNorm
    for tile_m, tile_n in hl.tile([1, N]):
        x_tile = x[tile_m, tile_n].to(torch.float32)
        var = torch.mean(x_tile * x_tile, dim=-1)
        inv_rms = torch.rsqrt(var + eps)
        y[tile_m, tile_n] = (x_tile * inv_rms[:, None] * weight[tile_n].to(torch.float32)).to(x.dtype)

    hl.barrier()

    # Phase 2: SiLU
    for tile_m, tile_n in hl.tile([1, N]):
        y_tile = y[tile_m, tile_n].to(torch.float32)
        z[tile_m, tile_n] = (y_tile * torch.sigmoid(y_tile)).to(x.dtype)

    hl.barrier()

    # Phase 3: Scale
    for tile_m, tile_n in hl.tile([1, N]):
        z_tile = z[tile_m, tile_n].to(torch.float32)
        output[tile_m, tile_n] = (z_tile * 0.5).to(x.dtype)

    return output


# ==============================================================================
# Equivalent hand-written Triton kernel
# ==============================================================================

@triton.jit
def three_phase_triton(
    x_ptr, weight_ptr, y_ptr, z_ptr, output_ptr,
    x_grid_sem,
    eps,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)

    # Phase 1: RMSNorm
    num_tiles = tl.cdiv(N, BLOCK_N)
    # First pass: compute variance
    var_acc = tl.zeros([1], tl.float32)
    for i in range(0, N, BLOCK_N):
        offs = i + tl.arange(0, BLOCK_N)
        mask = offs < N
        val = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        var_acc += tl.sum(val * val, 0)
    inv_rms = libdevice.rsqrt(var_acc / N + eps)
    # Second pass: normalize
    for tile_idx in tl.range(pid, num_tiles, num_sms):
        offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        val = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = val * inv_rms * w
        tl.store(y_ptr + offs, result.to(tl.bfloat16), mask=mask)

    triton_helpers.x_grid_barrier(x_grid_sem)

    # Phase 2: SiLU
    for tile_idx in tl.range(pid, num_tiles, num_sms):
        offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        val = tl.load(y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = val * tl.sigmoid(val)
        tl.store(z_ptr + offs, result.to(tl.bfloat16), mask=mask)

    triton_helpers.x_grid_barrier(x_grid_sem)

    # Phase 3: Scale
    for tile_idx in tl.range(pid, num_tiles, num_sms):
        offs = tile_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = offs < N
        val = tl.load(z_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        result = val * 0.5
        tl.store(output_ptr + offs, result.to(tl.bfloat16), mask=mask)


from torch._inductor.runtime.triton_compat import libdevice


def run_triton(x, weight, eps):
    N = x.shape[1]
    device = x.device
    dtype = x.dtype
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    output = torch.empty_like(x)
    sem = torch.zeros(1, device=device, dtype=torch.uint32)
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count

    _default_launcher(
        three_phase_triton,
        (num_sms,),
        x, weight, y, z, output, sem,
        eps,
        N=N,
        BLOCK_N=128,
        num_warps=4,
        num_stages=1,
        launch_cooperative_grid=True,
    )
    return output


# ==============================================================================
# Reference implementation
# ==============================================================================

def reference(x, weight, eps):
    x_f = x.float()
    var = (x_f * x_f).mean(-1, keepdim=True)
    y = (x_f * torch.rsqrt(var + eps) * weight.float()).to(x.dtype)
    y_f = y.float()
    z = (y_f * torch.sigmoid(y_f)).to(x.dtype)
    z_f = z.float()
    return (z_f * 0.5).to(x.dtype)


# ==============================================================================
# Test: correctness
# ==============================================================================

def test_correctness():
    """Verify both kernels match the reference."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N = 256

    x = torch.randn(1, N, device=device, dtype=torch.bfloat16)
    weight = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1 + 1.0

    ref = reference(x, weight, 1e-6)
    helion_out = three_phase_helion(x.clone(), weight, 1e-6)
    triton_out = run_triton(x.clone(), weight, 1e-6)

    helion_err = (ref - helion_out).abs().max().item()
    triton_err = (ref - triton_out).abs().max().item()

    print(f"  Helion vs ref atol: {helion_err:.6f}")
    print(f"  Triton vs ref atol: {triton_err:.6f}")

    assert helion_err < 0.05, f"Helion output incorrect (atol={helion_err})"
    assert triton_err < 0.05, f"Triton output incorrect (atol={triton_err})"
    print("  PASS: Both kernels match reference")


# ==============================================================================
# Test: codegen structure
# ==============================================================================

def _compile_helion_kernel():
    """Call the Helion kernel to trigger compilation."""
    torch.manual_seed(42)
    device = torch.device("cuda")
    N = 256
    x = torch.randn(1, N, device=device, dtype=torch.bfloat16)
    weight = torch.randn(N, device=device, dtype=torch.bfloat16) * 0.1 + 1.0
    three_phase_helion(x, weight, 1e-6)


def get_helion_generated_code():
    """Return the generated Triton source code from the Helion kernel.

    Uses Helion's internal API to find the compiled code file.
    """
    import glob

    # Search for recently generated helion code files
    user = os.environ.get("USER", "*")
    base = f"/tmp/torchinductor_{user}"
    candidates = []
    for path in glob.glob(f"{base}/**/*.py", recursive=True):
        try:
            with open(path) as f:
                first_lines = f.read(500)
            if "_helion_three_phase" in first_lines:
                candidates.append((os.path.getmtime(path), path))
        except (OSError, UnicodeDecodeError):
            pass

    if not candidates:
        raise RuntimeError(
            "Could not find Helion generated code file. "
            "Make sure the kernel was compiled first."
        )

    # Pick the most recently modified file
    candidates.sort(reverse=True)
    best_path = candidates[0][1]
    print(f"  [Found Helion generated code: {best_path}]")
    with open(best_path) as f:
        return f.read()


def get_triton_source():
    """Return the hand-written Triton kernel source as a string."""
    # triton.jit wraps the function; raw_src is a list of lines
    try:
        raw = three_phase_triton.raw_src
        if isinstance(raw, list):
            return "".join(raw)
        return raw
    except AttributeError:
        pass
    # Fallback: read from this file between markers
    with open(__file__) as f:
        src = f.read()
    start = src.index("@triton.jit\ndef three_phase_triton(")
    end = src.index("\n\nfrom torch._inductor", start)
    return src[start:end]


def test_codegen_no_duplication():
    """
    Test that Helion's generated code does NOT duplicate phase logic across
    barrier segments.

    EXPECTED (ideal codegen):
      - 3 phases, each appearing ONCE
      - 2 barriers between them
      - Total kernel lines comparable to hand-written Triton (~50 lines)

    ACTUAL (current codegen):
      - 3 phases, each appearing 3 TIMES (once per barrier segment)
      - 2 barriers
      - Total kernel lines ~3x larger than necessary
    """
    print("\n--- Codegen Analysis ---")

    helion_code = get_helion_generated_code()
    triton_code = get_triton_source()

    # Count key metrics
    helion_lines = len(helion_code.strip().split("\n"))
    triton_lines = len(triton_code.strip().split("\n"))

    helion_barriers = helion_code.count("x_grid_barrier")
    triton_barriers = triton_code.count("x_grid_barrier")

    helion_virtual_pid_loops = len(re.findall(r"for virtual_pid", helion_code))
    helion_sigmoid_calls = len(re.findall(r"tl\.sigmoid", helion_code))
    helion_rsqrt_calls = len(re.findall(r"rsqrt", helion_code))
    helion_scale_05 = len(re.findall(r"0\.5", helion_code))

    triton_sigmoid_calls = len(re.findall(r"tl\.sigmoid", triton_code))
    triton_rsqrt_calls = len(re.findall(r"rsqrt", triton_code))

    print(f"  Hand-written Triton kernel:")
    print(f"    Lines:           {triton_lines}")
    print(f"    Barriers:        {triton_barriers}")
    print(f"    sigmoid calls:   {triton_sigmoid_calls}")
    print(f"    rsqrt calls:     {triton_rsqrt_calls}")

    print(f"  Helion-generated Triton kernel:")
    print(f"    Lines:           {helion_lines}")
    print(f"    Barriers:        {helion_barriers}")
    print(f"    virtual_pid loops: {helion_virtual_pid_loops}")
    print(f"    sigmoid calls:   {helion_sigmoid_calls}")
    print(f"    rsqrt calls:     {helion_rsqrt_calls}")
    print(f"    '0.5' scale:     {helion_scale_05}")

    line_ratio = helion_lines / triton_lines if triton_lines > 0 else float("inf")
    print(f"\n  Line ratio (Helion / Triton): {line_ratio:.1f}x")

    # --- Assertions ---

    # Both should have exactly 2 barriers (between 3 phases)
    assert helion_barriers == 2, (
        f"Expected 2 barriers in Helion code, got {helion_barriers}"
    )
    assert triton_barriers == 2, (
        f"Expected 2 barriers in Triton code, got {triton_barriers}"
    )

    # There should be exactly 3 virtual_pid loops (one per phase)
    assert helion_virtual_pid_loops == 3, (
        f"Expected 3 virtual_pid loops, got {helion_virtual_pid_loops}"
    )

    # KEY ASSERTION: Each phase's unique operation should appear exactly ONCE.
    # Phase 1 uses rsqrt (RMSNorm), Phase 2 uses sigmoid (SiLU), Phase 3 uses 0.5 (scale).
    # If code is duplicated, these will appear N times instead of 1.
    assert helion_sigmoid_calls == triton_sigmoid_calls, (
        f"sigmoid (Phase 2 SiLU) appears {helion_sigmoid_calls}x in Helion "
        f"but {triton_sigmoid_calls}x in Triton. "
        f"Each phase's code should appear exactly once, not be duplicated "
        f"across barrier segments."
    )

    assert helion_rsqrt_calls == triton_rsqrt_calls, (
        f"rsqrt (Phase 1 RMSNorm) appears {helion_rsqrt_calls}x in Helion "
        f"but {triton_rsqrt_calls}x in Triton. "
        f"Each phase's code should appear exactly once."
    )

    # The generated code should not be dramatically larger than hand-written
    assert line_ratio < 3.0, (
        f"Helion code is {line_ratio:.1f}x larger than hand-written Triton. "
        f"Expected < 3x. The full phase dispatch is being duplicated in every "
        f"barrier segment."
    )

    print("  PASS: No code duplication detected")


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    print("Test 1: Correctness")
    test_correctness()

    print("\nTest 2: Codegen structure (no duplication across barriers)")
    test_codegen_no_duplication()
