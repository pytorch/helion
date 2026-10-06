"""Runtime numerics for the tcgen05 fixed-cost trims (sm_100 GPU).

The register-direct output store must be bitwise identical to the
SMEM-staged TMA store of the same config (same conversion, same values), the
one-shot path reached from a swizzled config must match the swizzle-1 kernel,
and the folded L2-grouping decode must visit every tile exactly once.  Every
case launches many times: a TMEM leak, a barrier-count mismatch or a bad
alignment claim shows up as a hang, a fault or a wrong result within a few
launches per SM.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch

from test.test_cute_tcgen05_fixed_chain import _batched_gemm
from test.test_cute_tcgen05_fixed_chain import _bias_gemm
from test.test_cute_tcgen05_fixed_chain_gpu import _addmm_gemm
from test.test_cute_tcgen05_fixed_chain_gpu import _config
from test.test_cute_tcgen05_fixed_chain_gpu import _gemm
from test.test_cute_tcgen05_fixed_cost import FRESH_FACTORY_GEMMS
from test.test_cute_tcgen05_fixed_cost import TMA_STORE_ILLEGAL_OUTPUTS
from test.test_cute_tcgen05_fixed_cost import _gemm_empty_strided_column_major
from test.test_cute_tcgen05_fixed_cost import _gemm_into
from test.test_cute_tcgen05_fixed_cost import _output_view
from test.test_cute_tcgen05_fixed_cost import _row_bias_gemm

import helion
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = [
    skipUnlessBackends(["cute"]),
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
        reason="tcgen05 requires an sm_100 GPU",
    ),
]

LAUNCHES = 40


def _outputs(
    kernel_fn, args: tuple[torch.Tensor, ...], config: helion.Config
) -> tuple[list[torch.Tensor], str]:
    kernel = helion.kernel(kernel_fn, backend="cute", static_shapes=True)
    bound = kernel.bind(args)
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    outputs = [bound(*args) for _ in range(LAUNCHES)]
    torch.cuda.synchronize()
    return outputs, source


def _assert_direct_matches_tma(
    kernel_fn,
    args: tuple[torch.Tensor, ...],
    config: helion.Config,
    expected: torch.Tensor,
    **tol: float,
) -> None:
    normal, normal_source = _outputs(kernel_fn, args, config)
    direct, direct_source = _outputs(
        kernel_fn,
        args,
        helion.Config(**{**config.config, "tcgen05_c_store_mode": "direct"}),
    )
    assert "cute.copy(tcgen05_tma_store_atom" in normal_source
    assert "tcgen05_tma_store_atom" not in direct_source
    assert "cute.nvgpu.CopyR2GOp()" in direct_source
    # The 16-byte alignment proof held: vector stores, no per-element loop.
    assert "assumed_align=min(" in direct_source, direct_source
    assert "_edge_i" not in direct_source
    for out in normal:
        torch.testing.assert_close(out, expected.to(out.dtype), **tol)
    for out in direct:
        assert torch.equal(out, normal[0])


def test_direct_store_plain_fp16_one_tile_per_cta() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved"),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


def test_direct_store_plain_bf16_128_wide_tile() -> None:
    torch.manual_seed(0)
    a = torch.randn(1024, 512, device=DEVICE, dtype=torch.bfloat16)
    b = torch.randn(512, 1024, device=DEVICE, dtype=torch.bfloat16)
    # 8 x 8 = 64 tiles of 128x128: four 32-column subtiles per tile.
    _assert_direct_matches_tma(
        _addmm_gemm,
        (a, b),
        _config(block_sizes=[128, 128, 64], pid_type="persistent_interleaved"),
        a @ b,
        rtol=2e-2,
        atol=2e-2,
    )


def test_direct_store_persistent_tiles() -> None:
    torch.manual_seed(0)
    a = torch.randn(2048, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 2048, device=DEVICE, dtype=torch.float16)
    # 16 x 32 = 512 tiles > 148 SMs: the persistent while, several tiles per
    # CTA, each stored straight from registers.
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(pid_type="persistent_blocked", l2_groupings=[8]),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


def test_direct_store_batched_output() -> None:
    torch.manual_seed(0)
    a = torch.randn(8, 256, 512, device=DEVICE, dtype=torch.float16)
    b = torch.randn(8, 512, 256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _batched_gemm,
        (a, b),
        _config(block_sizes=[1, 128, 64, 64], loop_orders=[[0, 1, 2]]),
        torch.bmm(a, b),
        rtol=1e-2,
        atol=1e-2,
    )


def test_direct_store_64_row_tile() -> None:
    # bm=64 uses a different TMEM load atom (and so a different per-thread
    # chunk layout); the alignment claim must hold for it too.
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(block_sizes=[64, 16, 128], pid_type="persistent_interleaved"),
        a @ b,
        rtol=1e-2,
        atol=1e-2,
    )


def test_direct_store_fp8_narrow_tile() -> None:
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    b = (
        torch.randn(256, 256, device=DEVICE, dtype=torch.float32)
        .to(torch.float8_e4m3fn)
        .T.contiguous()
        .T
    )
    expected = a.float() @ b.float()
    _assert_direct_matches_tma(
        _gemm,
        (a, b),
        _config(
            block_sizes=[128, 16, 128],
            indexing=["pointer", "pointer", "pointer"],
            pid_type="persistent_interleaved",
        ),
        expected,
        rtol=5e-2,
        atol=5e-2,
    )


def test_direct_store_bias_row_pre_wait() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(512, 512, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _bias_gemm,
        (a, b, bias),
        _config(
            pid_type="persistent_interleaved",
            tcgen05_aux_load_placement="pre_acc_wait",
        ),
        (a @ b).float() + bias.float(),
        rtol=1e-2,
        atol=1e-2,
    )


def test_one_shot_path_from_a_swizzled_config_matches_swizzle_one() -> None:
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    plain, plain_source = _outputs(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=1),
    )
    swizzled, swizzled_source = _outputs(
        _gemm,
        (a, b),
        _config(pid_type="persistent_interleaved", tcgen05_l2_swizzle_size=4),
    )
    assert "while tcgen05_role_local" not in swizzled_source
    assert "swizzle_size" not in swizzled_source
    for out in plain:
        torch.testing.assert_close(out, (a @ b).to(out.dtype), rtol=1e-2, atol=1e-2)
    for out in swizzled:
        assert torch.equal(out, plain[0])


@pytest.mark.parametrize(
    ("m", "grouping", "folded"),
    [
        (2048, 32, "group_size_m = num_pid_m"),
        (2048, 2, "group_size_m = 2"),
        (768, 4, "group_size_m = min(num_pid_m - first_pid_m, 4)"),
    ],
    ids=["single_group", "full_groups", "partial_group"],
)
def test_folded_l2_grouping_visits_every_tile(
    m: int, grouping: int, folded: str
) -> None:
    torch.manual_seed(0)
    a = torch.randn(m, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 2048, device=DEVICE, dtype=torch.float16)
    grouped, source = _outputs(
        _gemm, (a, b), _config(pid_type="persistent_blocked", l2_groupings=[grouping])
    )
    assert folded in source
    ungrouped, _ = _outputs(
        _gemm, (a, b), _config(pid_type="persistent_blocked", l2_groupings=[1])
    )
    for out in ungrouped:
        torch.testing.assert_close(out, (a @ b).to(out.dtype), rtol=1e-2, atol=1e-2)
    # Same tiles, same per-tile math: the raster order cannot change a value.
    for out in grouped:
        assert torch.equal(out, ungrouped[0])


def test_direct_store_row_vector_bias_pre_wait() -> None:
    # The matmul s0 winner's shape: a trailing-axis bias row on a 128x16 tile
    # loaded ahead of the accumulator wait (the promoted FP32 row stage) and
    # stored straight from registers.
    torch.manual_seed(0)
    a = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 256, device=DEVICE, dtype=torch.float16)
    bias = torch.randn(256, device=DEVICE, dtype=torch.float16)
    _assert_direct_matches_tma(
        _row_bias_gemm,
        (a, b, bias),
        _config(
            block_sizes=[128, 16, 64],
            indexing=["tensor_descriptor"] * 4,
            pid_type="persistent_interleaved",
            tcgen05_aux_load_placement="pre_acc_wait",
        ),
        (a.float() @ b.float() + bias.float()).to(torch.float16),
        rtol=2e-2,
        atol=2e-2,
    )


# --------------------------------------------------------------------------
# N=8 tiles: six role warps, no CTA-wide barrier on the role-local chain
# --------------------------------------------------------------------------

N8_INLINE_CONFIG: dict[str, object] = {
    # The campaign's bmm s0 winner: 64x8 tiles, inline ``xyz`` path, SMEM-staged
    # TMA store.  Its N=8 tile used to launch 64-lane rows (twelve warps for six
    # roles); the spare warps raced the TMA warp through the pipeline-init
    # barrier ahead of warp 0's mbarrier init, a flaky illegal-instruction
    # fault on the second launch of a process.
    "block_sizes": [1, 64, 8, 64],
    "loop_orders": [[0, 1, 2]],
    "l2_groupings": [16],
    "indexing": ["pointer", "tensor_descriptor", "tensor_descriptor"],
    "pid_type": "xyz",
    "tcgen05_persistence_model": "non_persistent",
    "tcgen05_cluster_m": 1,
    "tcgen05_cluster_n": 1,
    "tcgen05_ab_stages": 4,
    "tcgen05_acc_stages": 2,
    "tcgen05_c_stages": 4,
    "tcgen05_tvm_ffi_launch": False,
}


def test_narrow_n8_inline_store_is_stable_across_streams_and_graph_replay() -> None:
    from helion.autotuner.benchmarking import _make_cudagraph_replay

    torch.manual_seed(0)
    a = torch.randn(4, 64, 128, device=DEVICE, dtype=torch.float16)
    b = torch.randn(4, 128, 128, device=DEVICE, dtype=torch.float16)
    config = helion.Config(**N8_INLINE_CONFIG)
    bound = helion.kernel(_batched_gemm, backend="cute", static_shapes=True).bind(
        (a, b)
    )
    bound.set_config(config)
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    source = bound.to_code(normalized)
    assert "block=(32, 6, 1)" in source
    assert "cute.copy(tcgen05_tma_store_atom" in source
    # The fault needed a launch after the first (the launcher's fast relaunch)
    # on another stream; alternate streams and allocate a fresh output each
    # time, as the autotuner's capture helper does.
    side = torch.cuda.Stream()
    outputs: list[torch.Tensor] = []
    for launch in range(LAUNCHES):
        stream = side if launch % 2 else torch.cuda.default_stream()
        with torch.cuda.stream(stream):
            outputs.append(bound(a, b))
    torch.cuda.synchronize()
    replay = _make_cudagraph_replay(lambda: bound(a, b))
    torch.cuda.synchronize()
    replays: list[torch.Tensor] = []
    for _ in range(5):
        replays.append(replay().clone())
        torch.cuda.synchronize()
    torch.testing.assert_close(outputs[0], torch.bmm(a, b), rtol=1e-2, atol=1e-2)
    for out in [*outputs[1:], *replays]:
        assert torch.equal(out, outputs[0])


def _launch_batched_gemm_in_subprocess(
    tmp_path: Path,
    shape: tuple[int, int, int, int],
    config: dict[str, object],
    *,
    timeout: float,
) -> None:
    """Compile, launch and check a bmm config in a child process under a timeout.

    A deadlocked kernel cannot be interrupted in-process (``synchronize``
    blocks); the child is killed by ``subprocess.run`` instead and the test
    fails with the timeout.  The probe is a file: Helion reads the kernel's
    source through ``inspect``.
    """
    probe = textwrap.dedent(
        f"""
        import json
        import torch
        import helion
        import helion.language as hl

        def bmm(a, b):
            batch, m, k = a.shape
            _, _, n = b.shape
            out = torch.empty((batch, m, n), dtype=a.dtype, device=a.device)
            for tile_b, tile_m, tile_n in hl.tile((batch, m, n)):
                acc = hl.zeros([tile_b, tile_m, tile_n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.baddbmm(
                        acc, a[tile_b, tile_m, tile_k], b[tile_b, tile_k, tile_n]
                    )
                out[tile_b, tile_m, tile_n] = acc.to(out.dtype)
            return out

        torch.manual_seed(0)
        batch, m, n, k = {shape!r}
        a = torch.randn(batch, m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(batch, k, n, device="cuda", dtype=torch.float16)
        bound = helion.kernel(bmm, backend="cute", static_shapes=True).bind((a, b))
        bound.set_config(helion.Config(**json.loads({json.dumps(config)!r})))
        first = bound(a, b)
        torch.cuda.synchronize()
        torch.testing.assert_close(first, torch.bmm(a, b), rtol=1e-2, atol=1e-2)
        for _ in range(8):
            assert torch.equal(bound(a, b), first)
        torch.cuda.synchronize()
        print("PROBE_OK", helion.__file__)
        """
    )
    probe_path = tmp_path / "n8_probe.py"
    probe_path.write_text(probe)
    # A script's own directory heads ``sys.path``, not the cwd: point the child
    # at the Helion this process imported (the checkout under test), not at an
    # installed one.
    helion_root = str(Path(helion.__file__).resolve().parents[1])
    env = {**os.environ, "HELION_BACKEND": "cute"}
    env["PYTHONPATH"] = os.pathsep.join(
        [helion_root, *filter(None, [env.get("PYTHONPATH")])]
    )
    try:
        result = subprocess.run(
            [sys.executable, str(probe_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired as timed_out:
        raise AssertionError(
            f"the launch did not complete within {timeout:.0f}s "
            f"(stdout={timed_out.stdout!r}, stderr={str(timed_out.stderr)[-2000:]!r})"
        ) from None
    assert result.returncode == 0 and "PROBE_OK" in result.stdout, result.stderr[-4000:]


@pytest.mark.parametrize("ab_stages", [1, 2], ids=["ab1", "ab2"])
def test_narrow_n8_one_shot_chain_completes_with_more_k_tiles_than_stages(
    tmp_path: Path, ab_stages: int
) -> None:
    # 4 x 1 x 16 = 64 tiles of 64x8 <= 148 SMs: the one-shot role-local chain
    # with 2 K tiles over a 1- or 2-stage ring.  A CTA-wide barrier between
    # the hoisted TMA-load role and the MMA role deadlocked the single-stage
    # ring (every ``tcgen05_ab_stages=1`` autotune sample timed out).
    _launch_batched_gemm_in_subprocess(
        tmp_path,
        (4, 64, 128, 128),
        {
            "block_sizes": [1, 64, 8, 64],
            "loop_orders": [[0, 1, 2]],
            "indexing": ["tensor_descriptor"] * 3,
            "pid_type": "persistent_interleaved",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": ab_stages,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 4,
            "tcgen05_tvm_ffi_launch": False,
        },
        timeout=300.0,
    )


def test_narrow_n8_one_shot_chain_completes_with_four_k_tiles_over_two_stages(
    tmp_path: Path,
) -> None:
    # 8 x 2 x 16 = 256 tiles would not be one-shot; 2 x 1 x 32 = 64 tiles of
    # 128x8 are, with 4 K tiles over a 2-stage ring.
    _launch_batched_gemm_in_subprocess(
        tmp_path,
        (2, 128, 256, 256),
        {
            "block_sizes": [1, 128, 8, 64],
            "loop_orders": [[0, 1, 2]],
            "indexing": ["tensor_descriptor"] * 3,
            "pid_type": "persistent_interleaved",
            "tcgen05_persistence_model": "static_persistent",
            "tcgen05_cluster_m": 1,
            "tcgen05_cluster_n": 1,
            "tcgen05_ab_stages": 2,
            "tcgen05_acc_stages": 2,
            "tcgen05_c_stages": 4,
            "tcgen05_tvm_ffi_launch": False,
        },
        timeout=300.0,
    )


_UNPROVEN_OUTPUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float8_e4m3fn,
)


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.uint8)


@pytest.mark.parametrize("dtype", _UNPROVEN_OUTPUT_DTYPES, ids=str)
@pytest.mark.parametrize("case", TMA_STORE_ILLEGAL_OUTPUTS)
def test_unproven_output_views_are_stored_exactly(
    case: str, dtype: torch.dtype
) -> None:
    # An output view that violates a TensorMap constraint (row stride that is
    # not a 16-byte multiple, N-major storage, 8-byte-aligned base) used to
    # go through the TMA store and came back mostly wrong.  It now takes the
    # SIMT store: bitwise equal to the TMA-stored result of the same kernel
    # on a contiguous output, and close to the reference.
    torch.manual_seed(0)
    in_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16
    a = torch.randn(512, 256, device=DEVICE, dtype=in_dtype)
    b = torch.randn(256, 512, device=DEVICE, dtype=in_dtype)
    config = _config(pid_type="persistent_interleaved")
    contiguous = _output_view("contiguous", dtype, device=DEVICE)
    _, tma_source = _outputs(_gemm_into, (a, b, contiguous), config)
    assert "cute.copy(tcgen05_tma_store_atom" in tma_source
    view = _output_view(case, dtype, device=DEVICE)
    outputs, source = _outputs(_gemm_into, (a, b, view), config)
    assert "tcgen05_tma_store_atom" not in source
    assert "cute.nvgpu.CopyR2GOp()" in source
    assert torch.equal(_bits(view), _bits(contiguous))
    expected = a.float() @ b.float()
    if dtype == torch.float8_e4m3fn:
        tolerance = {"rtol": 1e-1, "atol": 5e-1}
    else:
        tolerance = {"rtol": 2e-2, "atol": 2e-2}
    torch.testing.assert_close(view.float(), expected, **tolerance)


@pytest.mark.parametrize("factory", sorted(FRESH_FACTORY_GEMMS))
def test_tensor_method_factory_outputs_keep_the_tma_store(factory: str) -> None:
    # An output allocated through ``a.new_*`` or ``torch.empty_strided`` is
    # proven fresh like ``torch.empty``: the TMA store is kept and every
    # launch is bitwise the ``torch.empty`` kernel's result.
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    config = _config(pid_type="persistent_interleaved")
    reference, reference_source = _outputs(_gemm, (a, b), config)
    assert "cute.copy(tcgen05_tma_store_atom" in reference_source
    outputs, source = _outputs(FRESH_FACTORY_GEMMS[factory], (a, b), config)
    assert "cute.copy(tcgen05_tma_store_atom" in source
    assert "cute.nvgpu.CopyR2GOp()" not in source
    for out in outputs:
        assert torch.equal(out, reference[0])
    torch.testing.assert_close(
        outputs[0].float(), a.float() @ b.float(), rtol=2e-2, atol=2e-2
    )


def test_fresh_column_major_strided_output_is_stored_exactly() -> None:
    # Fresh but N-major: not what the staged row-major D tile addresses, so
    # the SIMT store writes it, bitwise equal to the TMA-stored result.
    torch.manual_seed(0)
    a = torch.randn(512, 256, device=DEVICE, dtype=torch.float16)
    b = torch.randn(256, 512, device=DEVICE, dtype=torch.float16)
    config = _config(pid_type="persistent_interleaved")
    reference, _ = _outputs(_gemm, (a, b), config)
    outputs, source = _outputs(_gemm_empty_strided_column_major, (a, b), config)
    assert "tcgen05_tma_store_atom" not in source
    assert "cute.nvgpu.CopyR2GOp()" in source
    for out in outputs:
        assert out.stride() == (1, 512)
        assert torch.equal(out.contiguous(), reference[0])
