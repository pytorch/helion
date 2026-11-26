"""The original row-carry schedule owns MMA only for an actual BF16 dot."""

from __future__ import annotations

import ast
from unittest.mock import patch

from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import (
    _helion_aot_qwen3_30b_a3b_moe_cuda_sm100 as aot,
)
from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import qwen3_30b_a3b_moe as model
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.autotune.fragment_knobs import ROW_CARRY
from helion._compiler.cute.autotune.fragment_knobs import _row_carry_roots
from helion._compiler.cute.autotune.fragment_knobs import fragment_knob_heuristic
from helion._compiler.cute.autotune.heuristics.cute_warp_row_partition import (
    _singleton_grid_schedule,
)
from helion._compiler.cute.autotune.search_surface import normalize_after_sequences
from helion._compiler.cute.backend import CuteBackend
from helion._compiler.cute.warp_row_dot_spec import row_dot_geometry
from helion._compiler.cute.warp_row_partition import _pure
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion.autotuner.config_generation import ConfigGeneration
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@pytest.mark.slow
def test_relaxed_carries_bind_only_complete_rows_and_k_tiles() -> None:
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = tuple(torch.empty(s, dtype=d) for s, d in aot._TENSOR_SIGNATURES)
    with cpu_render_target(fake_device_sm_count=148):
        op = helion.kernel(backend="cute", fast_math=True, static_shapes=False)(
            model.qwen3_30b_a3b_moe.fn
        )
        bound = cpu_bind(op, args + aot._STATIC_ARGS)
        spec = bound.config_spec
        assert len(spec.cute_fragment_row_carry_root_ids) == 2
        assert {extent for _, extent in spec.cute_fragment_row_carry_unmasked_rows} == {
            model.HIDDEN,
            model.INTERMEDIATE,
        }
        gen = ConfigGeneration(spec)
        candidates = [
            c for _, c in gen.seed_flat_config_pairs() if c.get("cute_warp_row_dot")
        ]
        assert candidates
        # Check actual normalized tiles of each offered carry; no synthetic
        # mask is supplied for a tail missing from the relaxed graph.
        for cfg in candidates:
            assert all(
                extent % spec.block_sizes.config_get(cfg.block_sizes, axis) == 0
                for axis, extent in spec.cute_fragment_row_carry_unmasked_rows
            )
        # A 512-wide tile is partial on the 768-row gate/up domain.
        row = next(
            axis
            for axis, extent in spec.cute_fragment_row_carry_unmasked_rows
            if extent == model.INTERMEDIATE
        )
        bad = dict(candidates[-1].config)
        bad["block_sizes"] = list(bad["block_sizes"])
        bad["block_sizes"][spec.block_sizes.block_id_to_index(row)] = 512
        # The 768-row site now takes its own original walk; the independent
        # complete down projection stays legal.
        normalize_after_sequences(spec, bad, fix_invalid=False)
        assert _pure(
            ast.parse("cute.math.div(x, y, approx=True, ftz=True)", mode="eval").body,
            set(),
        )
        assert not _pure(
            ast.parse(
                "cute.math.div(touch_memory(), y, approx=True, ftz=True)", mode="eval"
            ).body,
            set(),
        )
        assert not _pure(
            ast.parse(
                "cute.math.div(x, y, approx=True, ftz=touch_memory())", mode="eval"
            ).body,
            set(),
        )
        # This 512-row fallback requires >2 MiB of shared storage; the
        # original backend limit still rejects it. The smaller two-root
        # fixture below exercises the viable mixed emitted path.
        with pytest.raises(exc.InvalidConfig, match="shared bytes"):
            bound.to_code(helion.Config.from_dict(bad))


@pytest.mark.slow
def test_ordinary_warp_dot_seed_owns_only_the_plain_down_projection() -> None:
    # This real graph contains both the positive BF16 matrix*vector pair
    # and a two-result gate/up carry with a computed, rounded vector.
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = tuple(torch.empty(s, dtype=d) for s, d in aot._TENSOR_SIGNATURES)
    with cpu_render_target(fake_device_sm_count=148):
        op = helion.kernel(
            backend="cute",
            fast_math=True,
            static_shapes=model.qwen3_30b_a3b_moe.settings.static_shapes,
        )(model.qwen3_30b_a3b_moe.fn)
        bound = cpu_bind(op, args + aot._STATIC_ARGS)
        spec = bound.config_spec
        assert len(spec.cute_fragment_row_carry_root_ids) == 2
        (root,) = spec.cute_warp_row_dot_roots
        assert root in spec.cute_fragment_row_carry_root_ids
        ((row, k),) = spec.cute_warp_row_dot_roots[root]
        assert (k, model.INTERMEDIATE) in spec.cute_fragment_row_carry_contractions
        assert not spec.default_config().get("cute_warp_row_dot")
        gen = ConfigGeneration(spec)
        configs = [c.config for _, c in gen.seed_flat_config_pairs()]
        dots = [c for c in configs if c.get("cute_warp_row_dot")]
        assert {spec.block_sizes.config_get(c["block_sizes"], row) for c in dots} == {
            16,
            128,
        }
        assert all(row_dot_geometry(spec, c) for c in dots)
        combined, reordered = [
            c
            for c in dots
            if c.get("cute_warp_row_reductions")
            and not c.get("cute_fragment_compact_rows")
        ]
        assert combined.get("cute_min_blocks_per_mp", 0) == 0
        orders = [[1, 0], [1, 2, 0], [1, 2, 0], [1, 0]]
        assert reordered == combined | {
            "loop_orders": orders,
            "cute_min_blocks_per_mp": 1,
        }
        # An already-normalized root order must not duplicate a regular seed.
        assert (
            _singleton_grid_schedule(bound.env, helion.Config.from_dict(reordered))
            is None
        )
        assert spec.block_sizes.config_get(combined["block_sizes"], row) == 128
        # A single shared launch must retain the direct router's lane mapping
        # as well as the independent nonlinear projection's scalar row carry.
        (scalar_row,) = spec.cute_fragment_row_carry_row_axes - {row}
        assert spec.block_sizes.config_get(combined["block_sizes"], scalar_row) == 64
        (striped,) = [c for c in dots if c.get("cute_fragment_compact_rows")]
        assert spec.cute_fragment_compact_row_extents == {
            scalar_row: model.INTERMEDIATE
        }
        assert striped == reordered | {"cute_fragment_compact_rows": True}
        for cfg in (combined, reordered, striped):
            flat, normalized = gen.strict_config_pair(helion.Config.from_dict(cfg))
            assert gen.canonicalize_flat(flat)[1].config == normalized.config == cfg
        regular_code = bound.to_code(helion.Config.from_dict(combined))
        assert "_cute_mma_bf16(" in regular_code
        assert f"_BLOCK_SIZE_{scalar_row} = 64" in regular_code
        code = bound.to_code(helion.Config.from_dict(striped))
        assert "_cute_mma_bf16(" in code
        assert f"_BLOCK_SIZE_{scalar_row} = 48" in code
        assert f"_BLOCK_SIZE_{row} = 128" in code
        assert "total_pids = 280" in code
        too_small = striped | {"block_sizes": list(striped["block_sizes"])}
        too_small["block_sizes"][spec.block_sizes.block_id_to_index(scalar_row)] = 32
        with pytest.raises(exc.InvalidConfig, match="three-quarter"):
            spec.normalize(dict(too_small))
        assert any(
            c.get("cute_fragment_vector_io")
            and c.get("cute_fragment_row_carry")
            and not c.get("cute_warp_row_dot")
            for c in configs
        )

        candidate = dots[-1]
        assert spec.block_sizes.config_get(candidate["block_sizes"], k) == 256
        ids = list(spec.block_sizes.valid_block_ids())
        bad_configs = [
            candidate | {"cute_fragment_vector_io": False},
            candidate | {"cute_fragment_reduction": "serial"},
        ]
        for axis, tile in ((k, 128), (row, 8)):
            blocks = list(candidate["block_sizes"])
            blocks[ids.index(axis)] = tile
            bad_configs.append(candidate | {"block_sizes": blocks})
        for bad in bad_configs:
            assert not row_dot_geometry(spec, bad)
            with pytest.raises(exc.InvalidConfig):
                spec.normalize(dict(bad))
            spec.normalize(bad, _fix_invalid=True)
            assert not bad.get("cute_warp_row_dot")


@helion.kernel(
    backend="cute", static_shapes=True, autotune_effort="none", fast_math=True
)
def _mixed_tail_projections(
    x: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    out0: torch.Tensor,
    out1: torch.Tensor,
):
    k = x.size(0)
    tmp = torch.empty_like(x)
    bk = hl.register_block_size(k)
    for a in hl.tile(k):
        tmp[a] = x[a] * 2
    for row in hl.tile(out0.size(0)):
        acc = hl.zeros([row, bk], dtype=torch.float32)
        for col in hl.tile(k, block_size=bk):
            acc = acc + w0[row, col].float() * tmp[col].float()[None, :]
        out0[row] = torch.sum(acc, dim=-1)
    for row in hl.tile(out1.size(0)):
        acc = hl.zeros([row, bk], dtype=torch.float32)
        for col in hl.tile(k, block_size=bk):
            acc = acc + w1[row, col].float() * tmp[col].float()[None, :]
        out1[row] = torch.sum(acc, dim=-1)
    return out0, out1


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def _cast_after_broadcast(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, cast_product: hl.constexpr
):
    k = x.size(0)
    tmp = torch.empty_like(x)
    bk = hl.register_block_size(k)
    for a in hl.tile(k):
        tmp[a] = x[a] * 2
    for row in hl.tile(out.size(0)):
        acc = hl.zeros([row, bk], dtype=torch.float32)
        for col in hl.tile(k, block_size=bk):
            if cast_product:
                acc = acc + (w[row, col] * tmp[col][None, :]).float()
            else:
                acc = acc + w[row, col].float() * tmp[col][None, :].float()
        out[row] = torch.sum(acc, dim=-1)


@pytest.mark.parametrize(
    ("cast_product", "static_shapes"), [(False, True), (True, True), (True, False)]
)
def test_row_dot_schema_ignores_elided_cast(
    cast_product: bool, static_shapes: bool
) -> None:
    op = helion.kernel(
        _cast_after_broadcast.fn,
        backend="cute",
        static_shapes=static_shapes,
        fast_math=True,
    )
    w = torch.empty_strided((192, 256), (265, 1), dtype=torch.bfloat16)
    out = torch.empty(192)
    with cpu_render_target(fake_device_sm_count=148):
        fingerprints = []
        for dtype in (torch.bfloat16, torch.float32):
            x = torch.empty(256, dtype=dtype)
            spec = cpu_bind(op, (x, w, out, cast_product)).config_spec
            assert not spec.cute_warp_row_dot_roots
            if cast_product:
                assert bool(spec.cute_fragment_row_carry_root_ids) == (
                    static_shapes and dtype == torch.float32
                )
            fingerprints.append(spec.structural_fingerprint())
        assert fingerprints[0] == fingerprints[1]


def test_row_carry_facts_can_be_registered_after_binding() -> None:
    x = torch.empty(256, dtype=torch.bfloat16)
    w = torch.empty((192, 256), dtype=torch.bfloat16)
    args = (x, w, w.clone(), torch.empty(192), torch.empty(192))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_mixed_tail_projections, args)
        spec = bound.config_spec
        expected = spec.cute_fragment_row_carry_root_ids
        assert len(expected) == 2
        assert not CompileEnvironment.has_current()
        facts = fragment_knob_heuristic(ROW_CARRY).register_facts(
            bound.env, bound.host_function.device_ir
        )
    assert facts == {"input_tensor_metadata"}
    assert spec.cute_fragment_row_carry_root_ids == expected
    assert not CompileEnvironment.has_current()


def test_symbolic_row_schema_does_not_request_exact_input_metadata() -> None:
    op = helion.kernel(
        _mixed_tail_projections.fn, backend="cute", static_shapes=False, fast_math=True
    )
    with cpu_render_target(fake_device_sm_count=148):
        for k in (256, 512):
            x = torch.empty(k, dtype=torch.bfloat16)
            w = torch.empty(192, k, dtype=torch.bfloat16)
            bound = cpu_bind(op, (x, w, w.clone(), torch.empty(192), torch.empty(192)))
            with bound.env, bound.host_function:
                facts = _row_carry_roots(bound.env, bound.host_function.device_ir)
            assert bound.config_spec.cute_warp_row_dot_structure
            # Other heuristics can independently specialize this kernel.
            assert not facts.root_ids and not facts.specialized


def test_shared_row_search_schema_preserves_per_binding_admission() -> None:
    class ReachedSearch(Exception):
        pass

    # The pitch changes without changing either weight's K extent. Then
    # change only the row count to remove complete 3/4 stripes. A different
    # one-word input type still has the same scalar computation.
    k = 256
    cases = [
        (
            torch.empty(k, dtype=torch.bfloat16),
            *(
                torch.empty_strided((rows, k), (pitch, 1), dtype=dtype)
                for _ in range(2)
            ),
            torch.empty(rows),
            torch.empty(rows),
        )
        for rows, pitch, dtype in (
            (192, 264, torch.bfloat16),
            (192, 265, torch.bfloat16),
            (193, 264, torch.bfloat16),
            (192, 265, torch.float16),
        )
    ]
    common = helion.Config(
        block_sizes=[256, 64, 64, 64],
        num_threads=[1, 32, 1, 1],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_fragment_row_carry=True,
        cute_fragment_reduction="warp",
        cute_fragment_vector_io=True,
    )
    dot = helion.Config.from_dict(common.config | {"cute_warp_row_dot": True})
    stripe = helion.Config.from_dict(
        common.config | {"cute_fragment_compact_rows": True}
    )
    op = helion.kernel(backend="cute", static_shapes=True, fast_math=True)(
        _mixed_tail_projections.fn
    )
    with cpu_render_target(fake_device_sm_count=148):
        # Use the normal bindings and fingerprint comparison, and intercept
        # only accelerator identity and the actual benchmarking entry.
        with (
            patch(
                "helion.runtime.kernel._canonicalize_multi_shape_device",
                return_value=torch.device("cuda", 0),
            ),
            patch("helion.runtime.kernel._current_device_index", return_value=0),
            patch(
                "helion.runtime.kernel.target_device_capability", return_value=(10, 0)
            ),
            patch.object(CuteBackend, "autotune", side_effect=ReachedSearch) as search,
            pytest.raises(ReachedSearch),
        ):
            op.autotune_multi(cases)
        search.assert_called_once()
        _, workload = search.call_args.args
        aligned, padded, tail, half = [bound for bound, _ in workload.cases]
        for bound in (aligned, padded, tail, half):
            slots = {entry[0] for entry in bound.config_spec.structural_fingerprint()}
            assert {"cute_warp_row_dot", "cute_fragment_compact_rows"} <= slots
            assert "_cute_mma_bf16(" not in bound.to_code(common)
        assert "_cute_mma_bf16(" in aligned.to_code(dot)
        assert aligned.config_spec.normalized_config(stripe).get(
            "cute_fragment_compact_rows"
        )
        assert padded.config_spec.normalized_config(stripe).get(
            "cute_fragment_compact_rows"
        )
        for no_dot in (padded, half):
            with pytest.raises(exc.InvalidConfig, match="cute_warp_row_dot"):
                no_dot.to_code(dot)
        with pytest.raises(exc.InvalidConfig, match="cute_fragment_compact_rows"):
            tail.to_code(stripe)


def test_mixed_row_projections_keep_partial_fallback_and_one_complete_mma() -> None:
    k = 256
    args = (
        torch.empty(k, dtype=torch.bfloat16),
        torch.empty(37, k, dtype=torch.bfloat16),
        torch.empty(64, k, dtype=torch.bfloat16),
        torch.empty(37),
        torch.empty(64),
    )
    cfg = helion.Config(
        block_sizes=[256, 32, 32, 32],
        num_threads=[1, 32, 1, 1],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_fragment_row_carry=True,
        cute_fragment_reduction="warp",
        cute_fragment_vector_io=True,
        cute_warp_row_dot=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_mixed_tail_projections, args)
        source = bound.to_code(cfg)
        all_partial = cpu_bind(
            _mixed_tail_projections,
            (
                *args[:2],
                torch.empty(33, k, dtype=torch.bfloat16),
                args[3],
                torch.empty(33),
            ),
        )
        with pytest.raises(exc.InvalidConfig, match="cute_warp_row_dot"):
            all_partial.to_code(cfg)
    # The 37-row weight is scalar-loaded in the original K walk. Only the
    # independent 64-row projection tests aligned W1 and emits the two MMAs.
    assert "cutlass.BFloat16((w0.iterator" in source
    assert "w0.iterator.alignment" not in source
    assert "w1.iterator.alignment" in source
    assert source.count("_cute_mma_bf16(") == 2


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _masked_broadcast(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    n = x.size(0)
    tmp = torch.empty_like(x)
    bk = hl.register_block_size(n)
    for a in hl.tile(n):
        tmp[a] = x[a] * 2
    for r in hl.tile(out.size(0)):
        acc = hl.zeros([r, bk], dtype=torch.float32)
        for c in hl.tile(n, block_size=bk):
            acc = acc + (tmp[c].float() + 1)
        out[r] = torch.sum(acc, dim=-1)
    return out


def test_masked_rank_one_addend_retains_ordinary_broadcast() -> None:
    args = (torch.empty(256, dtype=torch.bfloat16), torch.empty(37))
    with cpu_render_target(
        fake_device_sm_count=148,
        reduction_smem_budget_bytes=232448,
        smem_budget_bytes=232448,
    ):
        bound = cpu_bind(_masked_broadcast, args)
        assert not bound.config_spec.cute_fragment_row_carry_root_ids
        assert "_launcher(" in bound.to_code(
            helion.Config(
                block_sizes=[32, 32, 8],
                pid_type="persistent_interleaved",
                cross_loop_pipeline="barrier",
            )
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _opposite_carry_axes(
    x: torch.Tensor,
    w0: torch.Tensor,
    w1: torch.Tensor,
    rows: torch.Tensor,
    blocks: torch.Tensor,
):
    k = x.size(0)
    tmp = torch.empty_like(x)
    bk = hl.register_block_size(k)
    for a in hl.tile(k):
        tmp[a] = x[a] * 2
    for r in hl.tile(rows.size(0)):
        acc = hl.zeros([r, bk], dtype=torch.float32)
        for c in hl.tile(k, block_size=bk):
            acc = acc + w0[r, c].float() * tmp[c].float()[None, :]
        rows[r] = torch.sum(acc, dim=-1)
    for r in hl.tile(w1.size(1)):
        acc = hl.zeros([bk, r], dtype=torch.float32)
        for c in hl.tile(k, block_size=bk):
            acc = acc + (w1[c, r].float() + 1)
        blocks[r.begin] = torch.sum(torch.sum(acc, dim=-1))
    return rows, blocks


def test_transposed_masked_carry_leaves_ordinary_sibling_seeds() -> None:
    k, rows = 256, 64
    args = (
        torch.empty(k, dtype=torch.bfloat16),
        torch.empty(rows, k, dtype=torch.bfloat16),
        torch.empty(k, rows, dtype=torch.bfloat16),
        torch.empty(rows),
        torch.empty(rows),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_opposite_carry_axes, args)
        spec = bound.config_spec
        assert len(spec.cute_fragment_row_carry_root_ids) == 1
        assert spec.cute_fragment_row_carry_row_axes.isdisjoint(
            k for k, _ in spec.cute_fragment_row_carry_contractions
        )
        assert any(
            cfg.get("cute_fragment_row_carry")
            for _, cfg in ConfigGeneration(spec).seed_flat_config_pairs()
        )


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _offset_row_carry(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, start: hl.constexpr
) -> torch.Tensor:
    tmp = torch.empty_like(x)
    bk = hl.register_block_size(512)
    for a in hl.tile(x.size(0)):
        tmp[a] = x[a] * 2
    for row in hl.tile(w.size(0)):
        acc = hl.zeros([row, bk], dtype=torch.float32)
        for col in hl.tile(start, start + 512, block_size=bk):
            acc = acc + w[row, col].float() * tmp[col].float()[None, :]
        out[row] = torch.sum(acc, dim=-1)
    return out


@pytest.mark.parametrize(("start", "packet"), [(8, True), (1, False)])
def test_row_carry_packets_follow_the_emitted_k_loop(start: int, packet: bool) -> None:
    # Both intervals cover two full tiles in the same padded BF16 rows.
    # Normalization permits only the interval with an aligned start.
    args = (
        torch.empty(520, dtype=torch.bfloat16),
        torch.empty((17, 520), dtype=torch.bfloat16),
        torch.empty(17),
        start,
    )
    config = helion.Config(
        block_sizes=[256, 32, 8],
        num_threads=[1, 32, 8],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_fragment_row_carry=True,
        cute_fragment_reduction="warp",
        cute_fragment_vector_io=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_offset_row_carry, args)
        if not packet:
            with pytest.raises(exc.InvalidConfig, match="16-byte-packetizable"):
                bound.to_code(config)
            config = helion.Config.from_dict(
                config.config | {"cute_fragment_vector_io": False}
            )
        source = bound.to_code(config)
    (loop,) = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.For)
        and ast.unparse(node.iter) == f"range({start}, {start + 512}, 256)"
    ]
    assert isinstance(loop.target, ast.Name)
    induction = loop.target.id
    sources = {
        ast.unparse(node.value.left)
        for node in ast.walk(loop)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id.startswith("fragment_carry_pointer")
        and isinstance(node.value, ast.BinOp)
        and any(
            isinstance(value, ast.Name) and value.id == induction
            for value in ast.walk(node.value.right)
        )
    }
    assert ("w.iterator" in sources) == packet
    if not packet:
        assert "fragment_carry_pointer" not in ast.unparse(loop)
