"""A shared physical CTA may give a pure implicit row reduction its own warps."""

from __future__ import annotations

import ast
import copy
from types import SimpleNamespace
from unittest.mock import MagicMock
from unittest.mock import patch

from pretuned_kernels.megakernels.gemma4_a4b_moe import (
    _helion_aot_gemma4_a4b_moe_cuda_sm100 as gemma_aot,
)
from pretuned_kernels.megakernels.gemma4_a4b_moe import gemma4_a4b_moe as gemma
from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import (
    _helion_aot_qwen3_30b_a3b_moe_cuda_sm100 as aot,
)
from pretuned_kernels.megakernels.qwen3_30b_a3b_moe import qwen3_30b_a3b_moe as original
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import helion
from helion import exc
from helion._compiler.cute import single_row_publication as publication
from helion._compiler.cute.autotune.config_keys import PHASE_PARTICIPATION_KEY
from helion._compiler.cute.autotune.config_keys import ROW_DOT_KEY
from helion._compiler.cute.autotune.config_keys import SPARE_ROW_PREFETCH_KEY
from helion._compiler.cute.autotune.config_keys import UNROLL_WARP_ROWS_KEY
from helion._compiler.cute.autotune.heuristics.cute_row_publication import (
    add_publication_seeds,
)
from helion._compiler.cute.computed_fragment import FragmentCompiler
from helion._compiler.cute.parallel_row_sum import RowSumAnchor
from helion._compiler.cute.parallel_row_sum import parallel_sum
from helion._compiler.cute.parallel_row_sum import validate_parallel_sum
from helion._compiler.cute.single_row_publication import validate_publication_launch
from helion._compiler.cute.spare_row_prefetch import logical_geometry
from helion._compiler.cute.warp_row_partition import KEY
from helion._compiler.cute.warp_row_partition import _row_waves
from helion._compiler.cute.warp_row_partition import _single_row_owner
from helion._compiler.cute.warp_row_partition import validate_row_launch
from helion._compiler.cute.warp_row_unroll import WarpRowUnroll
from helion._compiler.cute.warp_row_unroll import unroll_proved_sweeps
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import default_cute_mma_support
import helion.language as hl
from helion.language import _tracing_ops

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True)
def _partitioned(
    x: torch.Tensor, weights: torch.Tensor, peer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(weights.size(0), dtype=torch.float32, device=x.device)
    dst = torch.empty_like(peer)
    for row in hl.tile(weights.size(0)):
        norm = (x[:].float() * x[:].float()).mean()
        out[row] = (weights[row, :].float() * (x[:].float() / (norm + 1))).sum(-1)
    hl.barrier()
    for j in hl.tile(peer.numel()):
        dst[j] = peer[j] + 1
    return out, dst


@helion.kernel(backend="cute", static_shapes=True)
def _dropped_sibling_axis(
    x: torch.Tensor, matrix: torch.Tensor, peer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty(matrix.size(0), dtype=torch.float32, device=x.device)
    dst = torch.empty_like(peer)
    for row in hl.tile(matrix.size(0)):
        norm = (x[:].float() * x[:].float()).mean()
        out[row] = (matrix[row, :].float() * (x[:].float() / (norm + 1))).sum(-1)
    for j in hl.tile(peer.numel()):
        for _unused in hl.tile(256):
            dst[j] = peer[j] + 1
    return out, dst


def test_singleton_warp_row_refuses_dropped_nested_launch_axis() -> None:
    with _target():
        bound = cpu_bind(_dropped_sibling_axis, _args(16, 128))
        options = dict(bound.config_spec.default_config()) | {
            "block_sizes": [1, 4, 64],
            "num_threads": [1, 4, 64, 1],
            "reduction_loops": [64],
            "pid_type": "persistent_interleaved",
        }
        control = bound.to_code(helion.Config.from_dict(options))
        assert "block=(4, 1, 1)" in control
        with pytest.raises(exc.InvalidConfig, match="full physical warp"):
            bound.to_code(helion.Config.from_dict(options | {KEY: True}))


@helion.kernel(backend="cute", static_shapes=True)
def _with_second_effect(
    x: torch.Tensor, weights: torch.Tensor, scratch: torch.Tensor
) -> torch.Tensor:
    out = torch.empty(weights.size(0), device=x.device)
    for row in hl.tile(weights.size(0)):
        acc = (weights[row, :].float() * x[:].float()).sum(-1)
        out[row] = acc
        scratch[row] = acc + 2
    hl.barrier()
    for j in hl.tile(scratch.numel()):
        scratch[j] += 1
    return out


@helion.kernel(backend="cute", static_shapes=False)
def _symbolic_publication(x: torch.Tensor, peer: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for _row in hl.tile(1):
        out[0, :] = x[0, :] + x[0, :].sum()
    hl.barrier()
    for j in hl.tile(peer.numel()):
        peer[j] = peer[j] + 1
    return out


@helion.kernel(backend="cute", static_shapes=True)
def _inplace_publication(x: torch.Tensor, peer: torch.Tensor) -> None:
    for _row in hl.tile(1):
        x[0, :] = x[0, :] + x[0, :].sum()
    hl.barrier()
    for j in hl.tile(peer.numel()):
        peer[j] = peer[j] + 1


@helion.kernel(backend="cute", static_shapes=True)
def _publish_to_buffer(
    x: torch.Tensor, target: torch.Tensor, peer: torch.Tensor
) -> None:
    for _row in hl.tile(1):
        target[0, :] = x[0, :] + x[0, :].sum()
    hl.barrier()
    for j in hl.tile(peer.numel()):
        peer[j] = peer[j] + 1


@helion.kernel(backend="cute", static_shapes=True)
def _publish_math(
    x: torch.Tensor, dst: torch.Tensor, peer: torch.Tensor, kind: hl.constexpr
) -> None:
    for _row in hl.tile(1):
        v = x[0, :] + x[0, :].sum()
        if kind == 0:
            v = torch.exp(v)
        elif kind == 1:
            v = torch.erf(v)
        elif kind == 2:
            v = torch.floor(v)
        else:
            v = torch.sinh(v)
        dst[0, :] = v
    hl.barrier()
    for j in hl.tile(peer.numel()):
        peer[j] = peer[j] + 1


def test_publication_declines_a_known_unemittable_math_call() -> None:
    args = (torch.empty(1, 128), torch.empty(1, 128), torch.empty(32))
    with _target():
        for kind, fast_math in (
            (0, False),
            (0, True),
            (1, False),
            (2, False),
            (3, False),
        ):
            op = helion.kernel(
                _publish_math.fn,
                backend="cute",
                static_shapes=True,
                fast_math=fast_math,
            )
            bound = cpu_bind(op, (*args, kind))
            assert not bound.config_spec.cute_single_row_publication_roots
            assert publication.KEY not in {
                key for key, _, _ in bound.config_spec.flat_key_layout()
            }
        # Exclude impossible extents before performing any storage comparisons.
        narrow = (torch.empty(1, 16), torch.empty(1, 16), torch.empty(32))
        with patch.object(
            publication, "tensor_pairs_with_proven_disjoint_storage"
        ) as pairs:
            assert not cpu_bind(
                _publish_to_buffer, narrow
            ).config_spec.cute_single_row_publication_roots
            pairs.assert_not_called()
        # Removing the math call leaves a valid option on the same binding.
        assert cpu_bind(
            _publish_to_buffer, args
        ).config_spec.cute_single_row_publication_roots


def test_publication_seed_respects_its_shared_cta() -> None:

    with _target():
        for width in (96, 256):
            args = (torch.empty(1, width), torch.empty(1, width), torch.empty(256))
            bound = cpu_bind(_publish_to_buffer, args)
            assert [
                n
                for _, n in bound.config_spec.cute_single_row_publication_roots.values()
            ] == [width]
            carrier = helion.Config(
                cute_staged_indexed_rows=True,
                cross_loop_pipeline="barrier",
                block_sizes=[1, 128],
                num_threads=[1, 128, 1],
                reduction_loops=[16],
            )
            seeds = [carrier]
            original = seeds
            with bound.env:
                seeds = add_publication_seeds(
                    bound.env, bound.host_function.device_ir, original
                )
            assert original == [carrier]
            assert seeds[0] is carrier
            assert [
                (bool(c.get(publication.KEY)), bool(c.get("cute_parallel_row_sum")))
                for c in seeds[1:]
            ] == ([(True, False), (True, True)] if width == 256 else [])
            if width == 96:
                # The same heuristic must honor a narrower carrier instead
                # of assuming that all staged options use the 128-thread seed.
                smaller = helion.Config.from_dict(
                    carrier.config | {"block_sizes": [1, 32], "num_threads": [1, 32, 1]}
                )
                with bound.env:
                    choices = add_publication_seeds(
                        bound.env, bound.host_function.device_ir, [smaller]
                    )
                assert any(c.get(publication.KEY) for c in choices[1:])
                # Three 32-lane sweeps still work; only the 128-thread seed is impossible.
                config = dict(bound.config_spec.default_config()) | {
                    "num_threads": [1, 32, 1],
                    "reduction_loops": [16],
                    publication.KEY: True,
                }
                assert "publication_thread" in bound.to_code(
                    helion.Config.from_dict(config)
                )


def test_publication_discovery_handles_symbolic_and_inplace_rows() -> None:
    args = (torch.empty(1, 128), torch.empty(32))
    with _target():
        # Discovery runs outside the host-function context during bind.
        symbolic = cpu_bind(_symbolic_publication, args)
        assert not symbolic.config_spec.cute_single_row_publication_roots
        spec = cpu_bind(_inplace_publication, args).config_spec
        assert not spec.cute_single_row_publication_roots
        x, peer = args
        ordinary = None
        for target, ok in (
            (torch.empty(1, 128), True),
            (torch.empty(1, 256)[:, ::2], False),
            (x.view_as(x), False),
        ):
            bound = cpu_bind(_publish_to_buffer, (x, target, peer))
            assert bool(bound.config_spec.cute_single_row_publication_roots) is ok
            fingerprint = bound.config_spec.structural_fingerprint()
            if target.stride() == x.stride():
                if ordinary is None:
                    ordinary = fingerprint
                else:
                    assert fingerprint == ordinary
            assert publication.KEY in {row[0] for row in fingerprint}
            if ok:
                # Distributing stores leaves this narrower serial reduction
                # unchanged. Only distributing its additions needs a full warp.
                config = dict(bound.config_spec.default_config()) | {
                    "num_threads": [1, 32, 1],
                    "reduction_loops": [16],
                    publication.KEY: True,
                }
                assert "publication_thread" in bound.to_code(
                    helion.Config.from_dict(config)
                )
                # Both serial partitions prove the same complete set of K
                # values. The final single-warp walk no longer uses either.
                flat = config | {"cute_parallel_row_sum": True}

                protocols = []
                for chunk in (16, 32):
                    module = ast.parse(
                        bound.to_code(
                            helion.Config.from_dict(flat | {"reduction_loops": [chunk]})
                        )
                    )
                    root = next(
                        n
                        for n in ast.walk(module)
                        if isinstance(n, ast.For)
                        and any(
                            isinstance(v, ast.Assign)
                            and isinstance(v.targets[0], ast.Name)
                            and v.targets[0].id == "publication_partial"
                            for v in ast.walk(n)
                        )
                    )
                    assert "publication_lane =" not in ast.unparse(root)
                    protocols.append(ast.dump(root))
                assert protocols[0] == protocols[1]
        wide_indices = helion.kernel(
            _publish_to_buffer.fn,
            backend="cute",
            static_shapes=True,
            index_dtype=torch.int64,
        )
        assert not cpu_bind(
            wide_indices, (x, torch.empty_like(x), peer)
        ).config_spec.cute_single_row_publication_roots


def _args(n: int, k: int) -> tuple[torch.Tensor, ...]:
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        return (
            torch.empty(k, dtype=torch.bfloat16),
            torch.empty(n, k, dtype=torch.bfloat16),
            torch.empty(256),
        )


def _target():
    return cpu_render_target(
        fake_device_sm_count=148,
        mma_support=default_cute_mma_support(),
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    )


@helion.kernel(backend="cute", static_shapes=True)
def _dependent_rows(
    x: torch.Tensor, weights: torch.Tensor, peer: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    sums = torch.empty(weights.size(0), dtype=torch.float32, device=x.device)
    copied = torch.empty_like(peer)
    for row in hl.tile(weights.size(0)):
        sums[row] = (weights[row, :].float() * x[:].float()).sum(-1)
    for j in hl.tile(peer.numel()):
        copied[j] = peer[j] + sums[(j.index + 1) % weights.size(0)]
    return sums, copied


@helion.kernel(backend="cute", static_shapes=True)
def _later_reduced_rows(
    x: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    vector = torch.empty_like(x)
    sums = torch.empty(weights.size(0), dtype=torch.float32, device=x.device)
    for col in hl.tile(x.numel()):
        vector[col] = x[col] + 1
    hl.barrier()
    for row in hl.tile(weights.size(0)):
        sums[row] = (weights[row, :].float() * vector[:].float()).sum(-1)
    return vector, sums


def test_spare_warp_prefetch_field_requires_sm10x_and_the_first_root() -> None:
    with cpu_render_target(capability=(9, 0), fake_device_sm_count=120):
        older = cpu_bind(_dependent_rows, _args(16, 256), capability=(9, 0))
        assert older.config_spec.target_device_capability == (9, 0)
    with _target():
        later = cpu_bind(_later_reduced_rows, _args(16, 256)[:2])
        assert later.config_spec.target_device_capability == (10, 0)
    for bound in (older, later):
        fields = {key for key, *_ in bound.config_spec.structural_fingerprint()}
        assert bound.config_spec.cute_warp_row_reduction_roots
        assert KEY in fields and SPARE_ROW_PREFETCH_KEY not in fields


def test_spare_warp_prefetch_owns_only_unused_threads_and_complete_matrix_rows() -> (
    None
):
    with _target():
        x = torch.empty(256, dtype=torch.bfloat16)
        matrix = torch.empty(16, 256, dtype=torch.bfloat16)
        peer = torch.empty(256)
        bound = cpu_bind(_dependent_rows, (x, matrix, peer))
        options = {
            "block_sizes": [1, 64],
            "num_threads": [1, 64, 1],
            "reduction_loops": [64],
            "pid_type": "persistent_interleaved",
            "cross_loop_pipeline": "barrier",
            KEY: True,
        }
        assert logical_geometry(bound.config_spec, options)
        assert not logical_geometry(
            bound.config_spec, options | {PHASE_PARTICIPATION_KEY: True}
        )
        ordinary = bound.to_code(helion.Config.from_dict(options))
        cfg = options | {SPARE_ROW_PREFETCH_KEY: True}
        source = bound.to_code(helion.Config.from_dict(cfg))
        tree = ast.parse(source)
        guards = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.If)
            and "weights.iterator.alignment >= 16" in ast.unparse(n.test)
        ]
        (guard,) = guards
        assert "weights.layout.stride[1] == 1" in ast.unparse(guard.test)
        assert ">= 32" in ast.unparse(guard) and "< 36" in ast.unparse(guard)
        for scope in ast.walk(tree):
            body = getattr(scope, "body", None)
            if isinstance(body, list) and guard in body:
                index = body.index(guard)
                assert ast.unparse(body[index + 1].test) == "cute.arch.warp_idx() == 0"
                body.remove(guard)
                break
        assert ast.dump(tree) == ast.dump(ast.parse(ordinary))

        # Every physical thread already belongs to warp zero.
        for nt in ([1, 32, 1], [1, 1, 1]):
            with pytest.raises(exc.InvalidConfig):
                bound.to_code(helion.Config.from_dict(cfg | {"num_threads": nt}))
        transposed = torch.empty(256, 16, dtype=torch.bfloat16).T
        unsafe = cpu_bind(_dependent_rows, (x, transposed, peer))
        assert unsafe.config_spec.cute_spare_row_prefetch_root is None
        assert SPARE_ROW_PREFETCH_KEY in {
            f[0] for f in unsafe.config_spec.structural_fingerprint()
        }
        with pytest.raises(exc.InvalidConfig):
            unsafe.to_code(helion.Config.from_dict(cfg))
        weak = torch.empty(1 + 16 * 256, dtype=torch.bfloat16)[1:].view(16, 256)
        assert weak.data_ptr() % 16 == 2
        unaligned = cpu_bind(_dependent_rows, (x, weak, peer))
        assert (
            unaligned.config_spec.structural_fingerprint()
            == bound.config_spec.structural_fingerprint()
        )
        weak_source = unaligned.to_code(helion.Config.from_dict(cfg))
        assert "weights.iterator.alignment >= 16" in weak_source
        nested = cfg | {"block_sizes": [[1], [64]], "num_threads": [[1], [64], [1]]}
        assert bound.to_code(helion.Config.from_dict(nested)) == source


def test_scalar_warp_unroll_preserves_the_two_fp32_reduction_bodies() -> None:
    # Two distinct summers share the same K coordinates. The unchanged warp
    # result of the first feeds the second through a division.
    with _target():
        bound = cpu_bind(_partitioned, _args(16, 256))
        options = {
            "block_sizes": [1, 64],
            "num_threads": [1, 64, 1],
            "reduction_loops": [64],
            "pid_type": "persistent_interleaved",
            KEY: True,
        }
        ordinary = ast.parse(bound.to_code(helion.Config.from_dict(options)))
        full = helion.Config.from_dict(options | {UNROLL_WARP_ROWS_KEY: True})
        unrolled = ast.parse(bound.to_code(full))
        nested = dict(full) | {
            "block_sizes": [[1], [64]],
            "num_threads": [[1], [64], [1]],
        }
        assert ast.dump(
            ast.parse(bound.to_code(helion.Config.from_dict(nested)))
        ) == ast.dump(unrolled)

        def sweeps(tree: ast.AST) -> list[ast.For]:
            (owner,) = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "cute.arch.warp_idx() == 0"
            ]
            return [n for n in owner.body if isinstance(n, ast.For)]

        # The ordinary fused two-pass load cache must survive the static
        # rewrite, including its uses in the second reduction.
        assert "_fuse_cache_" in ast.unparse(ordinary)
        before, after = sweeps(ordinary), sweeps(unrolled)
        assert len(before) == len(after) == 2
        for old, new in zip(before, after, strict=True):
            assert ast.unparse(new.iter) == "cutlass.range_constexpr(4)"
            assign, inner = new.body
            assert isinstance(assign, ast.Assign) and isinstance(inner, ast.For)
            assert ast.unparse(assign.targets[0]) == ast.unparse(old.target)
            assert (
                ast.unparse(assign.value)
                == f"cutlass.Int32({ast.unparse(new.target)} * 64)"
            )
            assert ast.unparse(inner.iter) == "cutlass.range_constexpr(2)"
            assert isinstance(old.body[0], ast.For)
            # Undo only the iterator representation. This must recover all
            # addresses, masks, accumulation order, and the eventual store.
            new.body, new.iter, new.target = [inner], old.iter, old.target
            inner.iter = old.body[0].iter
        assert ast.dump(unrolled) == ast.dump(ordinary)

        large = cpu_bind(_partitioned, _args(16, 8192))
        with pytest.raises(exc.InvalidConfig, match="bounded complete scalar warp row"):
            large.to_code(full)
        with pytest.raises(exc.InvalidConfig, match="bounded complete scalar warp row"):
            bound.to_code(
                helion.Config.from_dict(full.config | {"block_sizes": [2, 64]})
            )


def test_unroll_refuses_an_ambiguous_or_rebound_loop() -> None:
    (outer,) = ast.parse("for k in range(0, 256, 64):\n for i in range(2): x += i").body
    assert isinstance(outer, ast.For)
    inner = outer.body[0]
    assert isinstance(inner, ast.For)
    fn = MagicMock()
    state = fn.cute_state.lanes
    state.pending_warp_row_unrolls = [WarpRowUnroll.record(outer, inner, 256, 64)]
    state.unrolled_warp_row = False
    unroll_proved_sweeps(fn, [outer, outer])
    assert not state.unrolled_warp_row
    outer.target = ast.Name(id="other", ctx=ast.Store())
    unroll_proved_sweeps(fn, [outer])
    assert not state.unrolled_warp_row
    outer.target = ast.Name(id="k", ctx=ast.Store())
    inner.iter = ast.parse("range(4)", mode="eval").body
    unroll_proved_sweeps(fn, [outer])
    assert not state.unrolled_warp_row
    assert not fn.new_var.called


def test_semantic_admission_rejects_effects_and_partial_warps() -> None:
    with _target():
        ok = cpu_bind(_partitioned, _args(16, 128)).config_spec
        assert next(iter(ok.cute_warp_row_reduction_roots.values()))[-2:] == (128, 2)
        assert not cpu_bind(
            _partitioned, _args(16, 129)
        ).config_spec.cute_warp_row_reduction_roots
        side = cpu_bind(_with_second_effect, _args(16, 128))
        assert not side.config_spec.cute_warp_row_reduction_roots
        with pytest.raises(exc.InvalidConfig, match="pure sum"):
            side.to_code(
                helion.Config.from_dict(
                    dict(side.config_spec.default_config()) | {KEY: True}
                )
            )


@pytest.mark.slow
def test_original_seed_partitions_only_the_row_reductions() -> None:
    # The frozen production program also contains two rolled expert projections
    # and the output merge. Register and render the real ordinary mixed seed.
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = (
            tuple(
                torch.empty(shape, dtype=dtype)
                for shape, dtype in aot._TENSOR_SIGNATURES
            )
            + aot._STATIC_ARGS
        )
    op = helion.kernel(
        backend="cute", static_shapes=original.qwen3_30b_a3b_moe.settings.static_shapes
    )(original.qwen3_30b_a3b_moe.fn)
    with _target():
        bound = cpu_bind(op, args)
        spec = bound.config_spec
        # Combined seeds also partition the router, but switch the separate
        # down projection to MMA. Exercise the original scalar companion here.
        (requested,) = [
            c
            for c in spec.compiler_seed_configs
            if c.get(KEY) and not c.get(ROW_DOT_KEY)
        ]
        generation = spec.create_config_generation()
        flat, cfg = generation.strict_config_pair(requested)
        assert any(pair[0] == flat for pair in generation.seed_flat_config_pairs())
        # The normalizer may not reserve these lanes for the other reductions.
        assert max(cfg.config["num_threads"]) == 256
        src = ast.parse(bound.to_code(cfg))
        calls = [n for n in ast.walk(src) if isinstance(n, ast.Call)]
        (launcher,) = [
            n
            for n in calls
            if isinstance(n.func, ast.Name) and n.func.id == "_launcher"
        ]
        assert ast.literal_eval(
            next(kw.value for kw in launcher.keywords if kw.arg == "block")
        ) == (256, 1, 1)
        reductions = [
            n for n in calls if ast.unparse(n.func) == "cute.arch.warp_reduction_sum"
        ]
        # Each reachable phase is now emitted once. The router has the two
        # proved sums (RMS and dot), while the other three phases have none.
        (facts,) = spec.cute_warp_row_reduction_roots.values()
        assert len(reductions) == facts[-1] == 2 and all(
            ast.literal_eval(n.keywords[0].value) == 32 for n in reductions
        )
        leaders = [
            n
            for n in ast.walk(src)
            if isinstance(n, ast.Compare)
            and ast.unparse(n) == "cute.arch.thread_idx()[0] % 32 == 0"
        ]
        assert len(leaders) == len(spec.cute_warp_row_reduction_roots) == 1
        # A local store is valid only if every output has a complete warp.
        row, _, _, _ = next(iter(spec.cute_warp_row_reduction_roots.values()))
        bs = list(cfg.block_sizes)
        bs[spec.block_sizes.valid_block_ids().index(row)] = 32
        with pytest.raises(exc.InvalidConfig, match="linear CTA"):
            bound.to_code(helion.Config.from_dict(cfg.config | {"block_sizes": bs}))


def test_row_sweeps_require_uniform_origins_and_exact_physical_warps() -> None:
    # Use the setup that DeviceGridState retained for the root. Altering a
    # single coordinate or leaving another live lane must refuse the rewrite.
    grid = MagicMock()
    grid.lane_loops = [("s", 2)]
    grid.lane_loop_block_ids = {"s": frozenset({1})}
    grid.strategy.grid_origin_var.return_value = "base"
    grid.outer_prefix = []
    grid.hoist_parent_statements = []
    grid.deferred_vector_ops = []
    setup = ast.parse(
        "i = base + cutlass.Int32(cute.arch.thread_idx()[0]) * 2 + cutlass.Int32(s)\n"
        "m = cutlass.Int32(cute.arch.thread_idx()[0]) < 4 and i < 127"
    ).body
    grid._live_lane_setup.return_value = ({"s"}, setup, set())

    def admitted():
        return _row_waves(grid, [], 1, 8, "i", "m")

    wave = admitted()
    assert wave is not None and wave.warps == 4
    grid.strategy.grid_origin_var.return_value = "different_base"
    assert admitted() is None
    grid.strategy.grid_origin_var.return_value = "base"
    grid.lane_loops = [("s", 2), ("other", 2)]
    grid._live_lane_setup.return_value = ({"s", "other"}, setup, set())
    assert admitted() is None
    grid.lane_loops = [("s", 2)]
    grid._live_lane_setup.return_value = ({"s"}, setup, set())
    grid.hoist_parent_statements = ast.parse("base = cute.arch.thread_idx()[1]").body
    assert admitted() is None

    fn = MagicMock()
    state = fn.cute_state.lanes
    state.warp_row_threads = 0
    state.has_singleton_warp_rows = False
    state.warp_row_wave_counts = {4}
    state.launch_block = (4, 1, 1)
    with pytest.raises(exc.InvalidConfig, match="exact proved"):
        validate_row_launch(fn, (4, 32, 1))
    state.launch_block = (4, 32, 1)
    validate_row_launch(fn, (4, 32, 1))
    with pytest.raises(exc.InvalidConfig, match="exact proved"):
        validate_row_launch(fn, (4, 31, 1))
    with pytest.raises(exc.InvalidConfig, match="exact proved"):
        validate_row_launch(fn, (4, 64, 1))
    state.warp_row_wave_counts.add(8)
    with pytest.raises(exc.InvalidConfig, match="exact proved"):
        validate_row_launch(fn, (4, 32, 1))
    state.warp_row_wave_counts = {4}
    state.warp_row_threads = 128
    with pytest.raises(exc.InvalidConfig, match="linear CTA"):
        validate_row_launch(fn, (4, 32, 1))


def test_scalar_warp_row_requires_a_unique_uniform_pid_and_closed_inputs() -> None:
    fn, grid = MagicMock(), MagicMock()
    fn.pid.shared_pid_var = "pid"
    fn.arguments = [MagicMock()]
    fn.arguments[0].name = "src"
    grid.outer_prefix = grid.lane_loops = []
    grid.deferred_vector_ops = False
    grid._live_lane_setup.return_value = (set(), [], set())
    grid.strategy.grid_origin_var.return_value = "origin"

    def admitted(origin: str, value: str) -> bool:
        grid.hoist_parent_statements = ast.parse(
            f"unit = 1\norigin = {origin}\nrow = origin"
        ).body
        body = ast.parse(f"value = {value}\nsrc.iterator.store(value)").body
        return _single_row_owner(fn, grid, body, 0, "row")

    assert admitted("pid // unit", "src.iterator.load()")
    assert not admitted("pid // 2", "src.iterator.load()")
    assert not admitted("pid % unit", "src.iterator.load()")
    assert not admitted("cute.arch.thread_idx()[1]", "src.iterator.load()")
    assert not admitted("pid", "state_from_sibling")
    assert not admitted("pid", "src[row]")
    state = fn.cute_state.lanes
    state.has_singleton_warp_rows = True
    state.warp_row_threads, state.warp_row_wave_counts = 0, set()
    for dims in ((4, 32, 1), (32, 1, 1)):
        state.launch_block = dims
        validate_row_launch(fn, dims)
    state.launch_block = (4, 31, 1)
    with pytest.raises(exc.InvalidConfig, match="full physical warp"):
        validate_row_launch(fn, state.launch_block)


@pytest.mark.slow
def test_original_staged_indexed_seed_changes_only_the_first_row_root() -> None:
    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = (
            tuple(torch.empty(h, dtype=dt) for h, dt in gemma_aot._TENSOR_SIGNATURES)
            + gemma_aot._STATIC_ARGS
        )
    cfg_source = helion.kernel(
        backend="cute",
        fast_math=True,
        static_shapes=gemma.gemma4_a4b_moe.settings.static_shapes,
    )(gemma.gemma4_a4b_moe.fn)
    with _target():
        bound = cpu_bind(cfg_source, args)
        generator = bound.config_spec.create_config_generation()
        seed = next(
            cfg
            for _, cfg in generator.seed_flat_config_pairs()
            if cfg.get("cute_staged_indexed_rows")
        )
        assert not seed.get(KEY)
        _, neighbor = generator.strict_config_pair(
            helion.Config.from_dict(seed.config | {KEY: True})
        )
        assert seed.config | {KEY: True} == neighbor.config
        sources = [ast.parse(bound.to_code(c)) for c in (seed, neighbor)]
    roots = []
    for source in sources:
        dev, host = [node for node in source.body if isinstance(node, ast.FunctionDef)]
        phases = [stmt for stmt in dev.body if isinstance(stmt, ast.For)]
        assert len(phases) == 8
        roots.append(ast.unparse(phases[0]))
        dev.body.remove(phases[0])
    # Includes all seven siblings, the grid barriers and the original host launch.
    assert ast.dump(sources[0]) == ast.dump(sources[1])
    assert roots[0].count("threads_in_group=1)") == 2
    assert roots[1].count("threads_in_group=32)") == 2
    assert roots[1].count("cute.arch.lane_idx() == 0") == 1
    assert "cute.arch.warp_idx() < 4" in roots[1]


@pytest.mark.slow
def test_single_row_publication_partitions_only_independent_output_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = (
            tuple(torch.empty(h, dtype=dt) for h, dt in gemma_aot._TENSOR_SIGNATURES)
            + gemma_aot._STATIC_ARGS
        )
    op = helion.kernel(
        backend="cute",
        fast_math=True,
        static_shapes=gemma.gemma4_a4b_moe.settings.static_shapes,
    )(gemma.gemma4_a4b_moe.fn)
    seen = []
    partition = publication.partition_row_publication

    def check_alias(cg, grid, body):
        fn = cg.device_function
        root = cg.current_root_graph_info
        if (
            fn.config.get(publication.KEY)
            and root.graph_id in bound.config_spec.cute_single_row_publication_roots
        ):
            # The same output can be disjoint at bind and alias in a new binding.
            # Neither ambiguous storage nor a broadcast output can change owners.
            for method in (
                "proven_disjoint_tensor_pairs",
                "proven_tensor_stride_values",
            ):
                before_proof = [ast.dump(n) for n in body]
                with patch.object(fn, method, return_value={}):
                    partition(cg, grid, body)
                assert before_proof == [ast.dump(n) for n in body]
                assert not fn.cute_state.lanes.single_row_publication_dims
            # A subscript on the destination (including an indirect load)
            # cannot be mistaken for its pointer's immutable stride metadata.
            changed = [*body[:-1], ast.parse(ast.unparse(body[-1])).body[0]]
            innermost = changed[-1].body[0]
            write = innermost.body[-1].value
            write.args[0] = ast.parse("moe_branch[0, 0]", mode="eval").body
            unchanged = ast.dump(changed[-1])
            partition(cg, grid, changed)
            assert ast.dump(changed[-1]) == unchanged
            assert not fn.cute_state.lanes.single_row_publication_dims
            # Neither update is an independent calculation for one column.
            # The first carries a serial recurrence, the second redirects stores.
            for mutate_index in (False, True):
                draft = [
                    *ast.parse("carried = 0").body,
                    *body[:-1],
                    ast.parse(ast.unparse(body[-1])).body[0],
                ]
                loop = draft[-1].body[0]
                name = loop.body[0].targets[0].id if mutate_index else "carried"
                loop.body.insert(1, ast.parse(f"{name} = {name} + 1").body[0])
                loop.body[-1].value.args[0] = ast.Name(id=name, ctx=ast.Load())
                before = [ast.dump(n) for n in draft]
                partition(cg, grid, draft)
                assert [ast.dump(n) for n in draft] == before
                assert not fn.cute_state.lanes.single_row_publication_dims
            for expr, admitted in (
                ("cute.arch.thread_idx()[0]", False),
                ("2816", True),
            ):
                preamble = ast.parse(f"static_extent_probe = {expr}").body[0]
                draft = [preamble, *body]
                partition(cg, grid, draft)
                assert (len(draft) == len(body) + 2) == admitted
                assert draft[0] is preamble
                if admitted:
                    assert fn.cute_state.lanes.single_row_publication_dims
                    fn.cute_state.lanes.single_row_publication_dims.clear()
            seen.append(root.graph_id)
        partition(cg, grid, body)

    with _target():
        bound = cpu_bind(op, args)
        gen = bound.config_spec.create_config_generation()
        configs = [cfg for _, cfg in gen.seed_flat_config_pairs()]
        candidate, parallel, independent = [
            cfg
            for cfg in configs
            if cfg.get(publication.KEY) and not cfg.get("cute_indexed_row_epilogue")
        ]
        assert candidate.get(KEY) and not independent.get(KEY)
        # A 16-wide serial chunk cannot provide a complete warp to the sibling.

        staged = next(
            cfg
            for cfg in configs
            if cfg.get("cute_staged_indexed_rows") and not cfg.get(publication.KEY)
        )
        narrow = helion.Config.from_dict(staged.config | {"reduction_loops": [16]})
        with bound.env:
            preferred = add_publication_seeds(
                bound.env, bound.host_function.device_ir, [narrow]
            )[1:]
        assert len(preferred) == 2 and all(not cfg.get(KEY) for cfg in preferred)
        assert preferred[1].config == preferred[0].config | {
            "cute_parallel_row_sum": True
        }
        assert "publication_partial" in bound.to_code(preferred[1])

        assert "publication_thread" in bound.to_code(independent)
        control = helion.Config.from_dict(
            {k: v for k, v in candidate.config.items() if k != publication.KEY}
        )
        monkeypatch.setattr(publication, "partition_row_publication", check_alias)
        before = ast.parse(bound.to_code(control))
        after = ast.parse(bound.to_code(candidate))
        assert parallel.config == candidate.config | {"cute_parallel_row_sum": True}
        assert gen.strict_config_pair(parallel)[1] == parallel
        monkeypatch.setattr(publication, "partition_row_publication", partition)
        parallel_tree = ast.parse(bound.to_code(parallel))
    assert len(seen) == 1
    a, b = [
        next(n for n in module.body if isinstance(n, ast.FunctionDef))
        for module in (before, after)
    ]
    aphase = [n for n in a.body if isinstance(n, ast.For)][-1]
    bphase = [n for n in b.body if isinstance(n, ast.For)][-1]
    ag, bg = aphase.body[-1], bphase.body[-1]
    assert isinstance(ag, ast.If) and isinstance(bg, ast.If)
    assert isinstance(ag.body[-1], ast.For) and isinstance(bg.body[-1], ast.For)
    old_loop, new_loop = ag.body[-1], bg.body[-1]
    assert isinstance(new_loop.target, ast.Name)
    assert isinstance(bg.body[-2], ast.Assign)
    thread = bg.body[-2].targets[0]
    assert isinstance(thread, ast.Name)
    assert ast.unparse(bg.body[-2].value) == (
        "cutlass.Int32(cute.arch.thread_idx()[0]) + cutlass.Int32(cute.arch.thread_idx()[1]) * 4"
    )
    assert ast.unparse(new_loop.iter) == "range(22)"
    old_inner = old_loop.body[0]
    assert isinstance(old_inner, ast.For)
    assert isinstance(old_inner.body[0], ast.Assign)
    assert isinstance(new_loop.body[0], ast.Assign)
    assert ast.dump(new_loop.body[0].targets[0]) == ast.dump(
        old_inner.body[0].targets[0]
    )
    assert (
        ast.unparse(new_loop.body[0].value)
        == f"{thread.id} + cutlass.Int32({new_loop.target.id}) * 128"
    )
    assert [ast.dump(v) for v in new_loop.body[1:]] == [
        ast.dump(v) for v in old_inner.body[1:]
    ]
    # No change to the statistic, its rounding, any peer root, or the host ABI.
    assert [ast.dump(n) for n in ag.body[:-1]] == [ast.dump(n) for n in bg.body[:-2]]
    bg.body[-2:] = [old_loop]
    assert ast.dump(before) == ast.dump(after)

    # The sum uses every column once and exactly one FP32 partial per warp.
    # Earlier roots and the host signature are byte-for-byte the same AST.
    child = next(n for n in parallel_tree.body if isinstance(n, ast.FunctionDef))
    root = [n for n in child.body if isinstance(n, ast.For)][-1]
    guard = root.body[-1]
    assert isinstance(guard, ast.If)
    loops = [s for s in guard.body if isinstance(s, ast.For)]
    assert len(loops) == 2
    assert [ast.unparse(q.iter) for q in loops] == ["range(22)", "range(22)"]
    assert ast.dump(loops[0].body[0]) == ast.dump(loops[1].body[0])

    def addends(node):
        return [
            ast.dump(sub.value.right)
            for sub in ast.walk(node)
            if isinstance(sub, ast.Assign)
            and isinstance(sub.targets[0], ast.Name)
            and isinstance(sub.value, ast.BinOp)
            and isinstance(sub.value.op, ast.Add)
            and ast.dump(sub.value.left)
            == ast.dump(ast.Name(id=sub.targets[0].id, ctx=ast.Load()))
        ]

    original = next(s for s in ag.body if isinstance(s, ast.For))
    assert addends(loops[0]) and addends(loops[0]) == addends(original)
    calls = [v for v in ast.walk(guard) if isinstance(v, ast.Call)]
    assert (
        len(
            [
                v
                for v in calls
                if ast.unparse(v.func) == "cute.arch.warp_reduction_sum"
                and ast.unparse(v.keywords[0].value) == "32"
            ]
        )
        == 2
    )
    assert (
        len([v for v in calls if ast.unparse(v.func) == "cute.arch.sync_threads"]) == 2
    )
    assert sum(ast.unparse(v.func).endswith(".store") for v in calls) == 1
    # Expand any private per-sweep cache back to its proven producer load,
    # then compare all value calculations and the destination. This catches a
    # dropped scale or cast without requiring the optimizer to use a cache.
    from_loads = {}
    cached = {}
    for item in loops[0].body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            dst = item.targets[0]
            if isinstance(dst, ast.Name):
                from_loads[dst.id] = item.value
            elif isinstance(dst, ast.Subscript) and isinstance(dst.value, ast.Name):
                assert isinstance(item.value, ast.Name)
                cached[dst.value.id, ast.dump(dst.slice)] = from_loads[item.value.id]

    class ExpandCache(ast.NodeTransformer):
        def visit_Subscript(self, node):
            if isinstance(node.value, ast.Name):
                value = cached.get((node.value.id, ast.dump(node.slice)))
                if value is not None:
                    return value
            return self.generic_visit(node)

    assert [
        ast.dump(ExpandCache().visit(copy.deepcopy(stmt))) for stmt in loops[1].body[1:]
    ] == [ast.dump(stmt) for stmt in new_loop.body[1:]]
    names = {name for name, _ in cached}
    child.body[:] = [
        stmt
        for stmt in child.body
        if not (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in names
            and isinstance(stmt.value, ast.Call)
            and ast.unparse(stmt.value.func) == "cute.make_rmem_tensor"
        )
    ]
    child.body[child.body.index(root)] = aphase
    assert ast.dump(parallel_tree) == ast.dump(before)


def test_single_row_launch_refuses_a_larger_actual_block() -> None:

    dims = (4, 32, 1)
    # The host launch can widen the x axis after strategy planning. Extra
    # lanes would overlap or run beyond the last column.
    lanes = SimpleNamespace(single_row_publication_dims={dims}, launch_block=(8, 32, 1))
    fn = SimpleNamespace(config={}, cute_state=SimpleNamespace(lanes=lanes))
    with pytest.raises(exc.InvalidConfig, match="proved full CTA"):
        validate_publication_launch(fn, dims)
    lanes.launch_block = dims
    validate_publication_launch(fn, dims)
    lanes.single_row_publication_dims.clear()
    with pytest.raises(
        exc.InvalidConfig,
        match="cute_single_row_publication needs a proved independent",
    ):
        validate_publication_launch(fn, dims)


@pytest.mark.parametrize(
    "control", ["cute.arch.thread_idx()[0]", "prior", "buf.iterator.load()"]
)
def test_parallel_row_sum_rejects_nonuniform_executing_lanes(control: str) -> None:

    def check(expr):
        (fn,) = ast.parse(
            "def kernel(buf, n):\n"
            "    prior = cute.arch.lane_idx()\n"
            "    for i in range(n):\n"
            f"        if {expr}:\n"
            "            tid = cute.arch.thread_idx()[0]\n"
            "            partial = cute.arch.warp_reduction_sum(value, threads_in_group=32)\n"
        ).body
        assert isinstance(fn, ast.FunctionDef)
        first = next(
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "partial"
        )
        coordinate = ast.parse("tid = cute.arch.thread_idx()[0]").body[0]
        validate_parallel_sum(
            fn,
            {
                "partial": RowSumAnchor(
                    "tid", ast.dump(coordinate), 0, (ast.dump(first),)
                )
            },
        )

    check("cute.arch.block_idx()[0] + i < n")
    with pytest.raises(exc.InvalidConfig, match="dispatch"):
        check(control)


def test_parallel_sum_preserves_uniform_pid_clipping_and_rejects_divergent_barriers() -> (
    None
):

    (scope,) = ast.parse(
        "def kernel(n):\n"
        "    end = cute.arch.block_idx()[0] + 1\n"
        "    if end > n:\n"
        "        end = n\n"
        "    for tile in range(end):\n"
        "        pass\n"
    ).body
    assert isinstance(scope, ast.FunctionDef)
    loop = scope.body[-1]
    assert isinstance(loop, ast.For)
    setup, row = ast.parse(
        "tid = cute.arch.thread_idx()[0]\nfor sweep in range(4):\n    k = tid + 128 * sweep"
    ).body
    assert isinstance(row, ast.For)
    state = SimpleNamespace(parallel_row_sum_anchors={})
    fn = MagicMock()
    fn.cute_state.lanes = state
    fn.new_var.side_effect = lambda name, dce: name
    terms = ast.parse("acc = acc + values[k]").body
    emitted = parallel_sum(fn, setup, row, "acc", "stat", terms, 128)
    assert emitted is not None
    sweep = emitted[1]
    assert isinstance(sweep, ast.For)
    assert not set(ast.walk(sweep.target)) & set(ast.walk(row.target))
    assert not set(ast.walk(sweep.iter)) & set(ast.walk(row.iter))
    loop.body = emitted
    validate_parallel_sum(scope, state.parallel_row_sum_anchors)
    loop.body = [*emitted[:2], ast.parse("tid = tid % 32").body[0], *emitted[2:]]
    with pytest.raises(exc.InvalidConfig, match="physical row sum coordinate"):
        validate_parallel_sum(scope, state.parallel_row_sum_anchors)
    loop.body = emitted
    sites = [
        i
        for i, stmt in enumerate(emitted)
        if any(
            isinstance(n, ast.Call)
            and ast.unparse(n.func)
            in ("cute.arch.warp_reduction_sum", "cute.arch.sync_threads")
            for n in ast.walk(stmt)
        )
    ]
    assert len(sites) == 4
    for i in sites:
        moved = ast.parse("if cute.arch.warp_idx() == 0:\n    pass").body[0]
        assert isinstance(moved, ast.If)
        moved.body = [emitted[i]]
        loop.body = [*emitted[:i], moved, *emitted[i + 1 :]]
        with pytest.raises(exc.InvalidConfig, match="protocol"):
            validate_parallel_sum(scope, state.parallel_row_sum_anchors)
    # Move the readiness barrier and the pick initializer together to
    # immediately before the shared-memory store they're meant to publish.
    sync = sites[1]
    assert isinstance(emitted[sync - 1], ast.If)
    loop.body = [
        *emitted[: sync - 1],
        *emitted[sync : sync + 2],
        emitted[sync - 1],
        *emitted[sync + 2 :],
    ]
    with pytest.raises(exc.InvalidConfig, match="protocol"):
        validate_parallel_sum(scope, state.parallel_row_sum_anchors)
    loop.body = emitted
    clip = scope.body[1]
    assert isinstance(clip, ast.If)
    clip.body = ast.parse("end = cute.arch.thread_idx()[0]").body
    with pytest.raises(exc.InvalidConfig, match="dispatch"):
        validate_parallel_sum(scope, state.parallel_row_sum_anchors)
    clip.body = ast.parse("end = n").body
    scope.body.insert(
        -1, ast.parse("unused = (end := cute.arch.thread_idx()[0])").body[0]
    )
    with pytest.raises(exc.InvalidConfig, match="dispatch"):
        validate_parallel_sum(scope, state.parallel_row_sum_anchors)


def test_carry_shape_only_edge_does_not_evaluate_its_unselected_value() -> None:
    graph = torch.fx.Graph()
    producer = graph.call_function(_tracing_ops._new_var, (graph.placeholder("x"),))
    size = graph.call_function(torch.ops.aten.sym_size.int, (producer, 0))
    size.meta["val"] = 37
    compiler = object.__new__(FragmentCompiler)
    compiler.pending_barrier = False
    assert compiler.node(size, {}) == 37
