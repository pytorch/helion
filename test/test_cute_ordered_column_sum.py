"""Physical ownership of independent, contiguous BF16 columns in an ordered slot sum."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compiler.cute.autotune.config_keys import ORDERED_COLUMN_PAIRS_KEY
from helion._compiler.cute.ordered_column_sum import add_column_seed
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_phase_liveness import ordered_column_pool

pytestmark = pytest.mark.cpu_only


def _config() -> helion.Config:
    return helion.Config(
        block_sizes=[128, 256],
        num_threads=[128, 4, 1],
        reduction_loops=[None],
    )


def _assets(
    slots: int = 4, columns: int = 384
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(1, slots, columns, dtype=torch.bfloat16),
        torch.empty(128),
        torch.empty(1, columns, dtype=torch.bfloat16),
    )


def test_packed_columns_lower_a_strict_neighbor_with_a_masked_scalar_tail() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(ordered_column_pool, _assets())
        assert len(bound.config_spec.cute_ordered_column_sum_roots) == 1
        plain = _config()
        with bound.env:
            original, packed = add_column_seed(
                bound.env, bound.host_function.device_ir, [plain]
            )
        assert original is plain and packed.config == plain.config | {
            ORDERED_COLUMN_PAIRS_KEY: True
        }
        generation = bound.config_spec.create_config_generation()
        flat, checked = generation.strict_config_pair(packed)
        assert generation.canonicalize_flat(flat)[1] == checked
        before = bound.to_code(plain)
        assert "_cute_column_pack_bf16x2" not in before
        src = bound.to_code(checked)
        (branch,) = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.If)
            and "iterator.alignment >= 4" in ast.unparse(n.test)
        ]
        cond = ast.unparse(branch.test)
        assert "tile_offset_2 <= 128" in cond and "stride[1] % 2 == 0" in cond
        assert cond.count("iterator.alignment >= 4") == 2
        fast = ast.unparse(branch.body)
        slow = ast.unparse(branch.orelse)
        assert (
            "cutlass.range_constexpr(4)" in fast
            and "cute.arch.add_packed_f32x2" in fast
        )
        assert "column_sum_thread * 2" in fast and "_cute_column_pack_bf16x2" in fast
        assert "cute.arch.add_packed_f32x2" not in slow and "indices_2 < 384" in slow


def test_ordered_column_option_refuses_alias_strided_storage_and_too_small_a_tile() -> (
    None
):
    with cpu_render_target(fake_device_sm_count=148):
        x, peer, out = _assets()
        for args in [
            (x, peer, x[:, 1]),
            (x.transpose(1, 2).contiguous().transpose(1, 2), peer, out),
            (x, peer, torch.empty(1, 768, dtype=out.dtype)[:, ::2]),
        ]:
            rejected = cpu_bind(ordered_column_pool, args)
            assert not rejected.config_spec.cute_ordered_column_sum_roots
        bound = cpu_bind(ordered_column_pool, (x, peer, out))
        too_small = helion.Config.from_dict(
            _config().config | {"block_sizes": [128, 64]}
        )
        with bound.env:
            assert add_column_seed(
                bound.env, bound.host_function.device_ir, [too_small]
            ) == [too_small]
        too_wide = helion.Config.from_dict(
            _config().config | {"block_sizes": [128, 512]}
        )
        with bound.env:
            last = add_column_seed(
                bound.env, bound.host_function.device_ir, [_config(), too_wide]
            )[-1]
        assert last.config == _config().config | {ORDERED_COLUMN_PAIRS_KEY: True}
        with pytest.raises(helion.exc.InvalidConfig, match="physical column tile"):
            bound.to_code(
                helion.Config.from_dict(
                    too_small.config | {ORDERED_COLUMN_PAIRS_KEY: True}
                )
            )


def test_pre_blackwell_has_no_packed_fp32_column_surface() -> None:
    with cpu_render_target(capability=(9, 0), fake_device_sm_count=132):
        bound = cpu_bind(ordered_column_pool, _assets(), capability=(9, 0))
        assert not bound.config_spec.cute_ordered_column_sum_roots
        with pytest.raises(helion.exc.InvalidConfig, match="ordered short FP32"):
            bound.to_code(
                helion.Config.from_dict(
                    _config().config | {ORDERED_COLUMN_PAIRS_KEY: True}
                )
            )


def test_multi_shape_column_field_survives_ineligible_batch_and_long_sum() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        specs = []
        for batch, slots in ((1, 4), (2, 4), (1, 20)):
            x = torch.empty(batch, slots, 384, dtype=torch.bfloat16)
            out = torch.empty(batch, 384, dtype=x.dtype)
            bound = cpu_bind(ordered_column_pool, (x, torch.empty(128), out))
            spec = bound.config_spec
            assert spec.cute_column_sum_structure
            assert bool(spec.cute_ordered_column_sum_roots) == (
                batch == 1 and slots == 4
            )
            assert ORDERED_COLUMN_PAIRS_KEY in {
                key for key, _, _ in spec.flat_key_layout()
            }
            if not spec.cute_ordered_column_sum_roots:
                with pytest.raises(
                    helion.exc.InvalidConfig, match="ordered short FP32"
                ):
                    bound.to_code(
                        helion.Config.from_dict(
                            _config().config | {ORDERED_COLUMN_PAIRS_KEY: True}
                        )
                    )
            specs.append(spec)
        assert (
            specs[0].structural_fingerprint()
            == specs[1].structural_fingerprint()
            == specs[2].structural_fingerprint()
        )


def test_exact_column_tiles_accept_the_unmasked_scalar_lowering() -> None:
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(ordered_column_pool, _assets(columns=512))
        with bound.env:
            control, packed = add_column_seed(
                bound.env, bound.host_function.device_ir, [_config()]
            )
        assert "indices_2 < 512" not in bound.to_code(control)
        assert "cute.arch.add_packed_f32x2" in bound.to_code(packed)
