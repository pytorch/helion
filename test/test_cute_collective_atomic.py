from __future__ import annotations

import ast

import pytest
import torch

from test.test_cute_collective_effects import _lower

import helion
from helion import exc
from helion._compiler.cute.collective_matmul import _region_write_roots
from helion._compiler.cute.state import cute_env_state
from helion._testing import import_example
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import mock_cuda_unavailable
from helion.autotuner.benchmarking import _make_cudagraph_replay
import helion.language as hl

matmul_split_k = import_example("matmul_split_k").matmul_split_k
segmented_reduction = import_example("segment_reduction").segmented_reduction_helion

CUDA_DEVICE = "cuda"


@pytest.mark.parametrize("scope", ["", ", scope='gpu'", ", scope='cta'"])
def test_unused_relaxed_atomic_add_has_known_output_effect(scope: str) -> None:
    body = ast.parse(
        "if active:\n"
        "    cute.arch.atomic_add("
        "(out.iterator + cute.crd2idx((m, n), out.layout)).llvm_ptr, "
        f"val=cutlass.Float32(value), sem='relaxed'{scope})\n"
    ).body
    original = ast.dump(ast.Module(body=body, type_ignores=[]))
    assert _region_write_roots(body, 0) == {"out"}
    assert ast.dump(ast.Module(body=body, type_ignores=[])) == original


@pytest.mark.parametrize(
    "source",
    [
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='acquire')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='release')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='acq_rel')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem=ordering)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed', scope=scope)",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed', other=True)",
        "cute.arch.atomic_add(ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add(ptr.llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + other.iterator).llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + unknown_index()).llvm_ptr, val=v, sem='relaxed')",
        "cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=unknown_value(), sem='relaxed')",
        "old = cute.arch.atomic_add((out.iterator + i).llvm_ptr, val=v, sem='relaxed')",
    ],
)
def test_atomic_effect_requires_exact_relaxed_unused_form(source: str) -> None:
    assert _region_write_roots(ast.parse(source).body, 0) is None


@pytest.mark.parametrize(
    "pointer",
    [
        "out.iterator if active else ptr",
        "cute.where(active, out.iterator, ptr)",
        "ptr + cutlass.Int32(0)",
        "out.iterator + ptr",
    ],
)
def test_atomic_base_cannot_hide_pointer_alias(pointer: str) -> None:
    source = (
        "ptr = limits.iterator\n"
        f"cute.arch.atomic_add(({pointer}).llvm_ptr, "
        "val=cutlass.Int32(1), sem='relaxed')"
    )
    assert _region_write_roots(ast.parse(source).body, 0) is None
    with pytest.raises(exc.BackendUnsupported, match="unclassified effects"):
        _lower(source)


@pytest.mark.parametrize(
    "offset", ["cutlass.Int32(index)", "cutlass.Int64(index) * 8 + 2"]
)
def test_atomic_base_accepts_proven_integer_offsets(offset: str) -> None:
    body = ast.parse(
        f"cute.arch.atomic_add((out.iterator + {offset}).llvm_ptr, "
        "val=cutlass.Float32(value), sem='relaxed')"
    ).body
    assert _region_write_roots(body, 0) == {"out"}


def test_collective_staging_preserves_disjoint_atomic_statement() -> None:
    source = _lower(
        "cute.arch.atomic_add((out.iterator + 0).llvm_ptr, "
        "val=cutlass.Float16(0), sem='relaxed')"
    )
    assert "cute.gemm(" in source
    assert source.count("cute.arch.atomic_add(") == 1


@pytest.mark.parametrize("root", ["a", "b", "limits"])
def test_collective_staging_rejects_atomic_aliases(root: str) -> None:
    with pytest.raises(exc.BackendUnsupported, match="may alias row-loop writes"):
        _lower(
            f"cute.arch.atomic_add(({root}.iterator + 0).llvm_ptr, "
            "val=cutlass.Float16(0), sem='relaxed')"
        )


def test_collective_control_proof_includes_relaxed_atomics() -> None:
    with pytest.raises(exc.BackendUnsupported, match="control.flow"):
        _lower(
            root_prefix="count = (flags.iterator + 0).load()",
            outer_header="if count > 0",
            root_suffix="cute.arch.atomic_add((flags.iterator + 0).llvm_ptr, "
            "val=cutlass.Int32(1), sem='relaxed')",
        )


def _config(compute: str) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 32],
        num_threads=[2, 64, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async",
        split_k=16,
    )


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_split_k_collective_keeps_atomic_epilogue(compute: str) -> None:
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    inputs = (
        torch.empty((64, 1024), dtype=torch.float16),
        torch.empty((1024, 128), dtype=torch.float16),
    )
    with mock_cuda_unavailable(), cpu_target():
        bound = kernel._bind_isolated(inputs)
        code = bound.to_code(_config(compute))
    assert "cute.gemm(" in code
    assert code.count("cute.arch.atomic_add(") == 1
    assert "cpasync.CopyG2SOp(" in code or "cp_async_shared_global(" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
@pytest.mark.parametrize("bias", [False, True])
def test_collective_atomic_tails_and_mutated_graphs(
    dtype: torch.dtype, compute: str, bias: bool
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(4317)
    # Odd K/N test scalar tails while padded row strides admit vector copies.
    x = (torch.randn((97, 536), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :530]
    y = (torch.randn((530, 80), device=CUDA_DEVICE, dtype=dtype) * 0.025)[:, :71]
    b = torch.randn((71,), device=CUDA_DEVICE, dtype=dtype) * 0.025
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    args = (x, y, lambda acc, tiles: acc + b[tiles[1]]) if bias else (x, y)
    bound = kernel._bind_isolated(args)
    config = _config(compute)
    code = bound.to_code(config)
    assert "cute.gemm(" in code
    assert code.count("cute.arch.atomic_add(") == 1
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*args)

    def expected() -> torch.Tensor:
        result = x.float() @ y.float()
        if bias:
            result += b.float()
        return result.to(dtype)

    torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(3):
        x.uniform_(-0.05, 0.05)
        y.uniform_(-0.05, 0.05)
        b.uniform_(-0.05, 0.05)
        torch.testing.assert_close(run(), expected(), atol=2e-3, rtol=1e-2)
        torch.testing.assert_close(replay(), expected(), atol=2e-3, rtol=1e-2)


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_counting_tiles(
    a: torch.Tensor, b: torch.Tensor, tiles: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.empty((m, n), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        hl.atomic_add(tiles, [0], 1)
        hl.atomic_add(rows, [tile_m], 1)
    return out


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _matmul_counting_into_its_output(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    n = b.size(1)
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for tile_m, tile_n in hl.tile([m, n]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(k):
            acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, tile_n])
        out[tile_m, tile_n] = acc
        hl.atomic_add(out, [0, 0], 1.0)
    return out


def _counting_config(compute: str) -> helion.Config:
    return helion.Config(
        block_sizes=[64, 64, 32],
        num_threads=[2, 64, 1],
        cute_collective_mma=True,
        cute_collective_compute=compute,
        cute_collective_copy="async_cached",
        cute_collective_stages=2,
    )


def _atomic_guards(code: str) -> dict[str, str]:
    """The guard of each atomic call, by the tensor it updates."""
    guards: dict[str, str] = {}
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.If):
            for call in ast.walk(node.body[0]):
                if isinstance(call, ast.Call) and ast.unparse(call.func) == (
                    "cute.arch.atomic_add"
                ):
                    tensor = ast.unparse(call.args[0]).split(".iterator")[0]
                    guards[tensor.strip("(")] = ast.unparse(node.test)
    return guards


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_tile_counter_beside_a_collective_matmul_runs_at_the_first_lane(
    compute: str,
) -> None:
    # The epilogue runs the rows of the output tile a thread holds in a lane
    # loop.  The collective matmul's placeholder leaves the body to its later
    # lowering, undistributed; the tile counter covers no tile axis, so it is
    # pinned to the first lane of that loop.  The row counter varies with the
    # loop and keeps every lane.
    inputs = (
        torch.empty((256, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
        torch.zeros((1,), dtype=torch.int32),
        torch.zeros((256,), dtype=torch.int32),
    )
    with mock_cuda_unavailable(), cpu_target():
        bound = _matmul_counting_tiles._bind_isolated(inputs)
        code = bound.to_code(_counting_config(compute))
    assert "cute.gemm(" in code
    guards = _atomic_guards(code)
    assert "and lane_0 == 0" in guards["tiles"], code
    assert "lane_0" not in guards["rows"], code


@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_tile_counter_into_the_collective_output_rejects_the_config(
    compute: str,
) -> None:
    # Issued at the first lane only, the counter would run before the other
    # lanes' stores of the output it adds into.
    inputs = (
        torch.empty((256, 512), dtype=torch.float16),
        torch.empty((512, 256), dtype=torch.float16),
    )
    with mock_cuda_unavailable(), cpu_target():
        bound = _matmul_counting_into_its_output._bind_isolated(inputs)
        with pytest.raises(
            exc.BackendUnsupported,
            match="lane-invariant atomic on out issued at the first lane_0 would "
            "run beside another access of its tensors",
        ):
            bound.to_code(_counting_config(compute))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("compute", ["warp", "tcgen05"])
def test_tile_counter_beside_a_collective_matmul_counts_each_tile_once(
    compute: str,
) -> None:
    if compute == "tcgen05" and torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires SM100-family")
    torch.manual_seed(5203)
    a = torch.randn((256, 512), device=CUDA_DEVICE, dtype=torch.float16)
    b = torch.randn((512, 256), device=CUDA_DEVICE, dtype=torch.float16)
    tiles = torch.zeros((1,), device=CUDA_DEVICE, dtype=torch.int32)
    rows = torch.zeros((256,), device=CUDA_DEVICE, dtype=torch.int32)
    args = (a, b, tiles, rows)
    bound = _matmul_counting_tiles._bind_isolated(args)
    config = _counting_config(compute)
    assert "cute.gemm(" in bound.to_code(config)
    bound.set_config(config)
    out = bound(*args)
    torch.testing.assert_close(out, a.float() @ b.float(), atol=1e-1, rtol=1e-2)
    # 64 x 64 tiles of a 256 x 256 output.
    assert tiles.item() == 16
    assert rows.tolist() == [4] * 256


def _vector_epilogue_config() -> helion.Config:
    """The 32x64 warp-MMA split-K config of the catalogue's s1 winner with the
    vector epilogue (one 128-thread packet loop over the output tile)."""
    return helion.Config(
        block_sizes=[32, 64, 64],
        num_threads=[2, 64, 1],
        cute_vector_widths=[1, 1, 1, 1],
        cute_collective_mma=True,
        cute_collective_compute="warp",
        cute_collective_copy="async",
        cute_collective_stages=4,
        cute_collective_epilogue="vector",
        split_k=16,
    )


def _red_helper(dtype: torch.dtype, promoted: bool) -> str:
    if promoted:
        return "_cute_red_add_f32_vec("
    return (
        "_cute_red_add_f16_vec("
        if dtype is torch.float16
        else "_cute_red_add_bf16_vec("
    )


@pytest.mark.parametrize("promote", [True, False])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_k_vector_epilogue_packs_the_atomic_add(
    promote: bool, bias: bool, dtype: torch.dtype
) -> None:
    """The vector collective epilogue rewrites a relaxed atomic-add terminal
    into one packed ``red`` per lane packet: ``red.global.add.v4.f32`` into
    the promoted fp32 output, ``red.global.add.noftz.f16x2`` / ``bf16x2``
    into a half output (the scalar half atomic is a CAS loop on sm_100).
    The epilogue's ``if outer_k.begin == 0`` bias branch runs per lane; the
    original lane loop stays as the fallback behind the stride guard."""
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=promote,
    )
    x = torch.empty((64, 1024), dtype=dtype)
    y = torch.empty((1024, 128), dtype=dtype)
    b = torch.empty((128,), dtype=dtype)
    args = (x, y, lambda acc, tiles: acc + b[tiles[1]]) if bias else (x, y)
    with mock_cuda_unavailable(), cpu_target():
        bound = kernel._bind_isolated(args)
        code = bound.to_code(_vector_epilogue_config())
    # Only float16 outputs are promoted; a bfloat16 output always keeps its dtype.
    promoted = promote and dtype is torch.float16
    assert cute_env_state(bound.env).half_atomic_output_promotions == (
        {"out": torch.float16} if promoted else {}
    )
    assert code.count(_red_helper(dtype, promoted)) == 1
    assert "cute.gemm(" in code
    # 32x64 elements over 128 threads: packets of 4 fp32 / 2 halves.
    assert (
        "cutlass.range(4, unroll=1)" if promoted else "cutlass.range(8, unroll=1)"
    ) in code
    # The guarded fallback keeps the scalar lane loop and its atomic.
    assert code.count("cute.arch.atomic_add(") == 1
    assert ("return out.to(torch.float16)" in code) == promoted
    # The bias epilogue runs under its partition-zero branch inside the
    # per-lane value loop of the packet.
    assert ("epilogue_value_lane" in code) and (
        code.count(".load()" if bias else "_cute_join_cast(") >= 1
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("promote", [True, False])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_split_k_vector_epilogue_atomics_match_the_reference(
    promote: bool, bias: bool, dtype: torch.dtype
) -> None:
    torch.manual_seed(2026)
    x = torch.randn((64, 2048), device=CUDA_DEVICE, dtype=dtype) * 0.05
    y = torch.randn((2048, 128), device=CUDA_DEVICE, dtype=dtype) * 0.05
    b = torch.randn((128,), device=CUDA_DEVICE, dtype=dtype) * 0.05
    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=promote,
    )
    args = (x, y, lambda acc, tiles: acc + b[tiles[1]]) if bias else (x, y)
    bound = kernel._bind_isolated(args)
    config = _vector_epilogue_config()
    promoted = promote and dtype is torch.float16
    code = bound.to_code(config)
    assert code.count(_red_helper(dtype, promoted)) == 1
    bound.set_config(config)

    def run() -> torch.Tensor:
        return bound(*args)

    def expected() -> torch.Tensor:
        result = x.float() @ y.float()
        if bias:
            result += b.float()
        return result.to(dtype)

    # The half output accumulates 16 partitions in its own dtype when the
    # promotion is off: allow the rounding of that many adds.
    tolerance = (
        {"atol": 2e-3, "rtol": 1e-2} if promoted else {"atol": 1e-2, "rtol": 2e-2}
    )
    torch.testing.assert_close(run(), expected(), **tolerance)
    replay = _make_cudagraph_replay(run)
    for _iteration in range(2):
        x.uniform_(-0.05, 0.05)
        y.uniform_(-0.05, 0.05)
        b.uniform_(-0.05, 0.05)
        torch.testing.assert_close(run(), expected(), **tolerance)
        torch.testing.assert_close(replay(), expected(), **tolerance)


@pytest.mark.parametrize("promote", [True, False])
def test_half_atomic_output_seeds_carry_the_vector_epilogue(promote: bool) -> None:
    """With the promotion off the output stays float16 and every warp seed
    with a scalar epilogue gets a vector twin (the scalar half atomic is a
    CAS loop); with it on the fp32 output keeps the ordinary seed set."""
    from helion.autotuner.config_generation import ConfigGeneration

    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=promote,
    )
    x = torch.empty((64, 1024), dtype=torch.float16)
    y = torch.empty((1024, 128), dtype=torch.float16)
    with mock_cuda_unavailable(), cpu_target():
        bound = kernel._bind_isolated((x, y))
        with bound.env:
            seeds = [
                config
                for _flat, config in ConfigGeneration(
                    bound.config_spec
                ).seed_flat_config_pairs()
            ]
    warp = [
        seed
        for seed in seeds
        if seed.get("cute_collective_mma")
        and seed.get("cute_collective_compute", "warp") == "warp"
    ]
    assert warp
    scalar = [
        s for s in warp if s.get("cute_collective_epilogue", "scalar") == "scalar"
    ]
    vector = [
        s
        for s in warp
        if s.get("cute_collective_epilogue") in ("vector", "vector_unrolled")
    ]
    if promote:
        assert len(vector) < len(scalar)
    else:
        assert len(vector) >= len(scalar)
        twins = {
            tuple(
                sorted(
                    (k, repr(v))
                    for k, v in s.config.items()
                    if k != "cute_collective_epilogue"
                )
            )
            for s in vector
            if s.get("cute_collective_epilogue") == "vector_unrolled"
        }
        for s in scalar:
            key = tuple(
                sorted(
                    (k, repr(v))
                    for k, v in s.config.items()
                    if k != "cute_collective_epilogue"
                )
            )
            assert key in twins


@pytest.mark.parametrize(
    "capability,dtype,packed",
    [
        ((8, 0), torch.float16, True),
        ((8, 0), torch.bfloat16, False),
        ((8, 6), torch.float32, False),
        ((9, 0), torch.bfloat16, True),
        ((9, 0), torch.float32, True),
        (None, torch.float16, False),
    ],
)
def test_packed_atomic_epilogue_respects_the_target_capability(
    capability: tuple[int, int] | None, dtype: torch.dtype, packed: bool
) -> None:
    """``red.global.add.noftz.f16x2`` needs sm_70, the ``bf16x2`` and
    ``v4.f32`` forms sm_90; an unknown or older target keeps the scalar
    atomic lane loop."""
    from helion._testing.cute_codegen import cpu_bind

    kernel = helion.kernel(
        matmul_split_k.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
        cute_promote_half_atomic_outputs=dtype is torch.float32,
    )
    x = torch.empty(
        (64, 1024), dtype=torch.float16 if dtype is torch.float32 else dtype
    )
    y = torch.empty((1024, 128), dtype=x.dtype)
    with mock_cuda_unavailable():
        bound = cpu_bind(kernel, (x, y), capability=capability)
        code = bound.to_code(_vector_epilogue_config())
    assert ("_cute_red_add_" in code) == packed
    assert "cute.arch.atomic_add(" in code


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_output_vector_lanes_match_the_reference(dtype: torch.dtype) -> None:
    """GPU: the generic V-loop flush of a half output (segmented reduction, one
    thread per 4-wide strip) packs the lanes into one red and matches an fp32
    index_add rounded once (at most 64 fp16 adds per node)."""
    from helion._testing import DEVICE

    torch.manual_seed(7)
    num_elements, num_features, num_nodes = 4096, 128, 64
    indices = torch.sort(
        torch.randint(0, num_nodes, (num_elements,), device=DEVICE)
    ).values
    values = (torch.randn(num_elements, num_features, device=DEVICE) * 0.05).to(dtype)
    kernel = helion.kernel(
        segmented_reduction.fn,
        backend="cute",
        static_shapes=True,
        autotune_effort="none",
    )
    bound = kernel._bind_isolated((indices, values, num_nodes))
    # 16-row strips, one thread per 4-wide group of a 128-wide feature tile.
    strip = {
        "block_sizes": [16, 128],
        "num_threads": [1, 32],
        "cute_vector_widths": [1, 4],
    }
    config = helion.Config(**{**bound.config_spec.default_config().config, **strip})
    code = bound.to_code(config)
    assert (
        "_cute_red_add_f16_vec("
        if dtype is torch.float16
        else "_cute_red_add_bf16_vec("
    ) in code
    bound.set_config(config)
    out = bound(indices, values, num_nodes)
    expected = torch.zeros(num_nodes, num_features, device=DEVICE).index_add_(
        0, indices, values.float()
    )
    torch.testing.assert_close(out.float(), expected, atol=2e-2, rtol=2e-2)
