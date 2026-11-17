"""CPU source and search-surface coverage for computed half-precision dots."""

from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_half_mma import batched_dot
from helion._testing.cute_half_mma import cast_dot
from helion._testing.cute_half_mma import computed_dot

pytestmark = pytest.mark.cpu_only
KEY = "cute_fragment_half_mma"


def _args(dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty(2, 24, 19, dtype=dtype),
        torch.empty(2, 19, 11, dtype=dtype),
        torch.empty(2, 24, 11, dtype=torch.float32),
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_half_mma_is_searchable_and_predicates_partial_tiles(
    dtype: torch.dtype,
) -> None:
    with cpu_target():
        bound = cpu_bind(computed_dot, _args(dtype))
        spec = bound.config_spec
        assert spec.cute_fragment_half_mma_root_ids
        assert KEY in spec._flat_fields()
        assert KEY not in spec.default_config()
        assert any(
            group.key == KEY and any(w.value is True for w in group.witnesses)
            for group in spec.compiler_coverage_groups
        )
        config = dict(spec.default_config()) | {KEY: True}
        code = bound.to_code(helion.Config.from_dict(config))
        kind = "bf16" if dtype == torch.bfloat16 else "f16"
        assert f"_helion_fragment_mma.mma_{kind}(" in code
        assert code.count(f"_helion_fragment_mma.pack_{kind}x2(") == 6
        assert "for fragment_mma_step in cutlass.range_constexpr(2):" in code
        assert "< 24" in code and "< 19" in code and "< 11" in code
        assert "for fragment_k in" not in code
        simt = bound.to_code(spec.default_config())
        assert "for fragment_k in" in simt
        assert f"_helion_fragment_mma.mma_{kind}(" not in simt


@pytest.mark.parametrize("capability", [(7, 5), None])
def test_half_mma_requires_a_supported_target(
    capability: tuple[int, int] | None,
) -> None:
    with cpu_target():
        bound = cpu_bind(computed_dot, _args(torch.bfloat16), capability=capability)
        assert not bound.config_spec.cute_fragment_half_mma_root_ids
        assert KEY not in bound.config_spec._flat_fields()
        config = dict(bound.config_spec.default_config()) | {KEY: True}
        with pytest.raises(helion.exc.InvalidConfig, match="FP16/BF16"):
            bound.config_spec.normalize(config)


def test_half_mma_does_not_capture_fp32_dots() -> None:
    with cpu_target():
        bound = cpu_bind(computed_dot, _args(torch.float32))
        assert not bound.config_spec.cute_fragment_half_mma_root_ids
        assert KEY not in bound.config_spec._flat_fields()


def test_half_mma_predicates_each_half_of_the_last_k_word() -> None:
    with cpu_target():
        bound = cpu_bind(
            batched_dot,
            (
                torch.empty(2, 19, 19, dtype=torch.bfloat16),
                torch.empty(2, 19, 11, dtype=torch.bfloat16),
            ),
        )
        config = dict(bound.config_spec.default_config()) | {KEY: True}
        code = bound.to_code(helion.Config.from_dict(config))
        assert "_helion_fragment_mma.mma_bf16(" in code
        assert "+ 1 < 19" in code
        assert "fragment_mma_operand = cutlass.Float32(0)" in code


def test_partial_cast_recipes_read_shared_memory_inside_the_guard() -> None:
    with cpu_target():
        bound = cpu_bind(
            cast_dot,
            (
                torch.empty(2, 8, 8, dtype=torch.bfloat16),
                torch.empty(2, 8, 8, dtype=torch.float16),
            ),
        )
        config = dict(bound.config_spec.default_config()) | {KEY: True}
        code = bound.to_code(helion.Config.from_dict(config))
    loops = [
        node
        for node in ast.walk(ast.parse(code))
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id.startswith("fragment_mma_step")
    ]
    assert loops
    reads = 0

    def check(node: ast.AST, guarded: bool = False) -> None:
        nonlocal reads
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id.startswith("fragment_buffer")
        ):
            reads += 1
            assert guarded, ast.unparse(node)
        for child in ast.iter_child_nodes(node):
            check(child, guarded or isinstance(node, ast.If))

    for loop in loops:
        check(loop)
    assert reads


def test_tf32_cast_keeps_its_original_operand_lane_stride() -> None:
    with cpu_target():
        bound = cpu_bind(
            cast_dot,
            (
                torch.empty(2, 16, 12, dtype=torch.bfloat16),
                torch.empty(2, 12, 8, dtype=torch.float32),
            ),
        )
        config = dict(bound.config_spec.default_config()) | {
            "cute_fp32_dot": "tf32_mma"
        }
        code = bound.to_code(helion.Config.from_dict(config))
    assert "_helion_fragment_mma.mma_tf32(" in code
    # The cast reads the 16x12 BF16 A tile in place because its 24-byte rows
    # are conflict-free for TF32 lanes; it needs no corresponding FP32 copy.
    assert "allocate_tensor(cutlass.BFloat16, cute.make_layout((192,))" in code
    assert "allocate_tensor(cutlass.Float32, cute.make_layout((192,))" not in code
