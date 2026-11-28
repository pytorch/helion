"""Admission, grid ordering and regeneration of cooperative compact row tiles."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.compile_environment import tensor_input_sources
from helion._compiler.cute.autotune.heuristics.cute_warp_row_partition import (
    _row_dot_launch_variant,
)
from helion._compiler.cute.autotune.heuristics.cute_warp_row_partition import (
    _singleton_grid_schedule,
)
from helion._compiler.cute.register_tile_admission import RegisterTileUnsupported
from helion._compiler.cute.state import cute_env_state
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
import helion.language as hl

if TYPE_CHECKING:
    from helion._compiler.compile_environment import CompileEnvironment
    from helion._compiler.device_ir import DeviceIR

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("extent", [1, 3])
def test_singleton_schedule_preserves_both_axis_orders(extent: int) -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def grid_copy(x: torch.Tensor, out: torch.Tensor) -> None:
        for i, j, k, col in hl.tile(x.shape):
            out[i, j, k, col] = x[i, j, k, col]

    x = torch.empty(extent, 5, extent, 7)
    with cpu_render_target(fake_device_sm_count=148):
        b = cpu_bind(grid_copy, (x, torch.empty_like(x)))
        base = helion.Config(loop_orders=[[2, 3, 0, 1]])
        next_seed = _singleton_grid_schedule(b.env, base)
        if extent == 3:
            assert next_seed is None
            variant = _row_dot_launch_variant(b.env, base)
            assert variant is not None
            assert variant.config == base.config | {"cute_min_blocks_per_mp": 1}
            assert _row_dot_launch_variant(b.env, variant) is None
        else:
            assert next_seed is not None
            assert next_seed.get("loop_orders") == [[3, 1, 2, 0]]
            assert base.get("loop_orders") == [[2, 3, 0, 1]]


def test_singleton_schedule_leaves_nested_device_loop_alone() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def nested_copy(x: torch.Tensor, out: torch.Tensor) -> None:
        for row in hl.tile(x.size(0)):
            for j, k in hl.tile([x.size(1), x.size(2)]):
                out[row, j, k] = x[row, j, k]

    x = torch.empty(5, 1, 7)
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(nested_copy, (x, torch.empty_like(x)))
        spec = bound.config_spec
        (nested,) = spec.loop_orders
        assert set(nested.block_ids).isdisjoint(spec.grid_block_ids)
        assert _singleton_grid_schedule(bound.env, helion.Config()) is None


@helion.kernel(backend="cute", static_shapes=True, autotune_effort="none")
def _observable_row_width(
    x: torch.Tensor,
    w: torch.Tensor,
    out: torch.Tensor,
    extra: torch.Tensor,
    mode: hl.constexpr,
) -> torch.Tensor:
    rows, k = w.shape
    published = torch.empty_like(x)
    bk = hl.register_block_size(k)
    for first in hl.tile(k):
        published[first] = x[first] * 2
    for row in hl.tile(rows):
        acc = hl.zeros([row, bk], dtype=torch.float32)
        for col in hl.tile(k, block_size=bk):
            acc = acc + w[row, col].float() * published[col].float()[None, :]
        total = torch.sum(acc, dim=-1)
        out[row] = total
        if mode == "permuted_overwrite":
            out[row.index ^ 32] = total + 1
        elif mode == "multiply":
            extra[row] = total * row.block_size
        elif mode == "load_index":
            extra[row] = x[row.block_size].float()
        elif mode == "store_index":
            width = row.block_size
            hl.store(extra, [width], x[0].float())
        elif mode == "explicit_tile":
            hl.store(extra, [row], hl.load(w, [row, 0]).float())
        elif mode == "derived_size":
            extra[row] = total * (acc.size(0) + 1)
        elif mode == "branch_value":
            if row.begin == 0:
                chosen = x[0]
            else:
                chosen = x[1]
            extra[row] = total * chosen
        elif mode == "first_rows":
            if row.begin == 0:
                extra[row] = total
        else:
            if row.begin == 0:
                extra[0] = x[0]
    return out


@pytest.mark.parametrize(
    "mode",
    [
        # Direct scalar writes, tile observation, explicit addressing and
        # a crossing write are independent possible ownership failures.
        "scalar_publication",
        "permuted_overwrite",
        "first_rows",
        "multiply",
        "derived_size",
        "branch_value",
        "load_index",
        "store_index",
        "explicit_tile",
    ],
)
def test_compact_rows_require_unobservable_width(mode: str) -> None:
    x = torch.empty(256, dtype=torch.bfloat16)
    w = torch.empty(768, 256, dtype=torch.bfloat16)
    out = torch.empty(768)
    with cpu_render_target(fake_device_sm_count=148):
        spec = cpu_bind(
            _observable_row_width, (x, w, out, torch.empty_like(out), mode)
        ).config_spec
        assert bool(spec.cute_fragment_compact_row_extents) == (
            mode in {"explicit_tile", "scalar_publication"}
        )


def test_shifted_scalar_publication_is_not_a_canonical_first_tile() -> None:
    @helion.kernel(backend="cute", static_shapes=True)
    def shifted(
        x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, scalar: torch.Tensor
    ) -> None:
        n = x.numel()
        bk = hl.register_block_size(n)
        tmp = torch.empty_like(x)
        for a in hl.tile(n):
            tmp[a] = x[a] * 2
        for r in hl.tile(64, 832):
            acc = hl.zeros([r, bk], dtype=torch.float32)
            for c in hl.tile(n, block_size=bk):
                acc = acc + w[r, c].float() * tmp[c].float()[None, :]
            out[r] = torch.sum(acc, dim=-1)
            if r.begin == 0:
                scalar[0] = x[0].float()

    with cpu_render_target(fake_device_sm_count=148):
        args = (
            torch.empty(256, dtype=torch.bfloat16),
            torch.empty(832, 256, dtype=torch.bfloat16),
            torch.empty(832),
            torch.empty(1),
        )
        bound = cpu_bind(shifted, args)
        assert bound.config_spec.cute_fragment_row_carry_root_ids
        assert not bound.config_spec.cute_fragment_compact_row_extents


@pytest.mark.parametrize(
    ("threads", "pipeline"),
    [(None, "barrier"), ([0, 0, 0], None), ([1, 32, 1], None)],
)
def test_compact_rows_resolve_ordinary_defaults(
    threads: list[int] | None, pipeline: str | None
) -> None:
    x = torch.empty(512, dtype=torch.bfloat16)
    w = torch.empty(768, 512, dtype=torch.bfloat16)
    outputs = (torch.empty(768), torch.empty(768))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_observable_row_width, (x, w, *outputs, "scalar_publication"))
        (axis,) = bound.config_spec.cute_fragment_compact_row_extents
        cfg = helion.Config(
            block_sizes=[256, 32, 64],
            pid_type="persistent_interleaved",
            num_sm_multiplier=1,
            cute_fragment_row_carry=True,
            cute_fragment_reduction="warp",
            cute_fragment_compact_rows=True,
        )
        if threads is not None:
            cfg.config["num_threads"] = threads
        if pipeline is not None:
            cfg.config["cross_loop_pipeline"] = pipeline
        if threads is None or not any(threads):
            # AUTO ties the launch width to the physical row width here;
            # 48 threads cannot execute the cooperative warp reduction.
            with pytest.raises(exc.InvalidConfig, match="one explicit thread"):
                bound.config_spec.normalize(dict(cfg.config))
            bound.config_spec.normalize(cfg.config, _fix_invalid=True)
            assert not cfg.get("cute_fragment_compact_rows")
            assert f"_BLOCK_SIZE_{axis} = 64" in bound.to_code(cfg)
            return
        bound.config_spec.normalize(cfg.config)
        assert cfg.get("cute_fragment_compact_rows")
        assert cfg.get("cross_loop_pipeline") == "barrier"
        assert f"_BLOCK_SIZE_{axis} = 48" in bound.to_code(cfg)
        with (
            bound.env.without_dispatch_guards(
                symbolic_strides=tensor_input_sources({"out": outputs[0]}, [outputs[0]])
            ),
            pytest.raises(exc.InvalidConfig, match="disjoint stripe ownership"),
        ):
            bound.to_code(cfg)


def test_compact_rows_reapply_after_register_tile_retry() -> None:
    x = torch.empty(512, dtype=torch.bfloat16)
    w = torch.empty(768, 512, dtype=torch.bfloat16)
    outputs = (torch.empty(768), torch.empty(768))
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_observable_row_width, (x, w, *outputs, "scalar_publication"))
        backend = bound.env.backend
        validate = backend.validate_codegen_config
        (axis,) = bound.config_spec.cute_fragment_compact_row_extents
        attempts: list[bool] = []

        def reject_first_physical_pass(
            env: CompileEnvironment, ir: DeviceIR, cfg: helion.Config
        ) -> None:
            validate(env, ir, cfg)
            width = env.config_spec.block_sizes.config_get(cfg.block_sizes, axis)
            if width == 48:
                disabled = cute_env_state(env).register_tile_disabled
                attempts.append(disabled)
                if not disabled:
                    raise RegisterTileUnsupported("exercise the visible tile on retry")

        config = helion.Config(
            block_sizes=[256, 32, 64],
            num_threads=[1, 32, 1],
            cross_loop_pipeline="barrier",
            pid_type="persistent_interleaved",
            num_sm_multiplier=1,
            cute_fragment_row_carry=True,
            cute_fragment_reduction="warp",
            cute_fragment_compact_rows=True,
        )
        with patch.object(
            backend, "validate_codegen_config", side_effect=reject_first_physical_pass
        ):
            code = bound.to_code(config)
        assert attempts == [False, True]
        assert f"_BLOCK_SIZE_{axis} = 48" in code
