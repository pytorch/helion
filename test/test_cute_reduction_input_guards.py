from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_cluster_online_pair import softmax_two_pass_kernel
from test.test_cute_persistent_reduction import _persistent_shifted_direct_update
from test.test_cute_persistent_reduction import _persistent_state_update

import helion
from helion._compiler.autotuner_heuristics.cute import (
    CutePersistentSubwarpRowsHeuristic,
)
from helion._compiler.cute.hoist_warp_reduce import _try_hoist_one_vloop

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def _cpu_only() -> Iterator[None]:
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU only")),
        patch(
            "helion._compiler.reduction_strategy._cute_shared_memory_budget_bytes",
            return_value=232448,
        ),
    ):
        yield


def _i32(value: int) -> int:
    return (int(value) + (1 << 31)) % (1 << 32) - (1 << 31)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_resolved_persistent_mask_preserves_safe_tail_vector_pointer(
    dtype: torch.dtype,
) -> None:
    arguments = (
        torch.empty(17, 128, dtype=dtype),
        torch.empty(256, dtype=dtype),
        torch.empty(17, dtype=dtype),
    )
    bound = _persistent_state_update._bind_isolated(arguments)
    assert bound.host_function is not None
    config = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, bound.host_function.device_ir
    )
    assert config is not None
    source = bound.to_code(config)
    loads = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.load"
        and "state.iterator" in ast.unparse(node.args[0])
    ]
    assert len(loads) == 1
    pointer = compile(ast.Expression(loads[0].args[0]), "<vector pointer>", "eval")
    assert "mask_0" in ast.unparse(loads[0].args[0])
    for row in range(32):
        for lane in range(16):
            offset = eval(
                pointer,
                {},
                {
                    "cutlass": SimpleNamespace(Int32=_i32),
                    "state": SimpleNamespace(
                        iterator=0, layout=SimpleNamespace(stride=(128, 1))
                    ),
                    "indices_0": row,
                    "mask_0": row < 17,
                    "reduction_lane_base_1": lane * 8,
                },
            )
            assert offset == (row * 128 + lane * 8 if row < 17 else 0)
            assert offset >= 0 and offset + 8 <= arguments[0].numel()
    assert "_cute_store_u16_vec(next_state.iterator" in source


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["unaligned_base", "partial_fragment"])
def test_resolved_persistent_masks_keep_unsafe_fragments_scalar(
    dtype: torch.dtype, kind: str
) -> None:
    if kind == "unaligned_base":
        storage = torch.empty(17 * 128 + 1, dtype=dtype)
        state = storage.as_strided((17, 128), (128, 1), 1)
        kernel = _persistent_state_update
        arguments = (state, torch.empty(256, dtype=dtype), torch.empty(17, dtype=dtype))
    else:
        kernel = _persistent_shifted_direct_update
        arguments = (torch.empty(17, 128, dtype=dtype),)
    bound = kernel._bind_isolated(arguments)
    assert bound.host_function is not None
    config = CutePersistentSubwarpRowsHeuristic.get_seed_config(
        bound.env, bound.host_function.device_ir
    )
    if kind == "partial_fragment":
        # This one-reduction negative deliberately does not match the
        # persistent recurrence seed family. Request the same vector geometry
        # directly, so the address proof must still reject the shifted lane.
        spec = bound.config_spec
        reduction = next(
            block.block_id for block in bound.env.block_sizes if block.reduction
        )
        row = spec.block_sizes.valid_block_ids()[0]
        config = helion.Config(
            block_sizes=[16],
            num_threads=[
                16 if block == reduction else 8 if block == row else 0
                for block in spec.num_threads.valid_block_ids()
            ],
            cute_vector_widths=[
                8 if block == reduction else 1
                for block in spec.cute_vector_widths.valid_block_ids()
            ],
            cute_lane_layouts=["blocked"] * len(spec.cute_lane_layouts),
            cute_reduction_reloads=[
                "register" if block == reduction else "auto"
                for block in spec.cute_reduction_reloads.valid_block_ids()
            ],
            num_warps=4,
            num_stages=1,
            pid_type="flat",
        )
    assert config is not None
    source = bound.to_code(config)
    assert "cute.arch.load(state.iterator" not in source
    assert "state.iterator" in source and ").load()" in source


def _loop(source: str) -> ast.For:
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.For)
    return node


@pytest.mark.parametrize(
    "dtype,name",
    [
        (torch.float16, "Float16"),
        (torch.bfloat16, "BFloat16"),
        (torch.float32, "Float32"),
    ],
)
def test_explicit_fp32_reduce_input_retains_prior_narrowing(
    dtype: torch.dtype, name: str
) -> None:
    source = f"""
for lane in cutlass.range_constexpr(4):
    value = cutlass.{name}(values[lane])
    reduced = cute.arch.warp_reduction_max(cutlass.Float32(value), threads_in_group=32)
    result = cute.math.max(result, reduced, propagate_nan=True)
"""
    original = _loop(source)
    unchanged = ast.dump(original, include_attributes=False)
    transformed = _try_hoist_one_vloop(original, 4, [0], set())
    assert transformed is not None
    assert ast.dump(original, include_attributes=False) == unchanged
    converted = ast.unparse(ast.Module(transformed, []))
    assert f"value = cutlass.{name}(values[lane])" in converted
    assert "cutlass.Float32(value)" in converted

    def run(body: list[ast.stmt]) -> tuple[bytes, int]:
        calls = []

        def reduction(value: float, **kwargs: object) -> float:
            calls.append(value)
            return value

        def maximum(a: float, b: float, **kwargs: object) -> float:
            return max(a, b)

        constructors = {
            "Float32": lambda value: torch.tensor(value, dtype=torch.float32).item(),
            name: lambda value: torch.tensor(value, dtype=dtype).item(),
            "range_constexpr": range,
        }
        namespace = {
            "values": [1.00048, 1.0005, -3.99, 0.0],
            "result": float("-inf"),
            "cutlass": SimpleNamespace(**constructors),
            "cute": SimpleNamespace(
                arch=SimpleNamespace(warp_reduction_max=reduction, fmax=maximum),
                math=SimpleNamespace(max=maximum),
            ),
        }
        exec(
            compile(ast.fix_missing_locations(ast.Module(body, [])), "<hoist>", "exec"),
            namespace,
        )
        result = torch.tensor(namespace["result"], dtype=torch.float32)
        return result.reshape(1).view(torch.uint8).numpy().tobytes(), len(calls)

    before, before_calls = run([original])
    after, after_calls = run(transformed)
    assert before == after
    assert before_calls == 4 and after_calls == 1


@pytest.mark.parametrize("through_alias", [False, True])
@pytest.mark.parametrize("flagged", [False, True])
def test_cast_does_not_hide_a_loop_carried_reduction_input(
    through_alias: bool, flagged: bool
) -> None:
    update = (
        "alias = carry\n    carry = alias + values[lane]"
        if through_alias
        else "carry = carry + values[lane]"
    )
    loop = _loop(
        f"""
for lane in cutlass.range_constexpr(4):
    {update}
    result = cute.arch.warp_reduction_sum(cutlass.Float32(carry), threads_in_group=32)
"""
    )
    before = ast.dump(loop, include_attributes=False)
    assert _try_hoist_one_vloop(loop, 4, [0], {"carry"} if flagged else set()) is None
    assert ast.dump(loop, include_attributes=False) == before


@pytest.mark.parametrize("reset", [False, True])
def test_conditional_reset_preserves_the_exact_recurrence_counterexample(
    reset: bool,
) -> None:
    # The first FP32-input normalization incorrectly treated the conditional
    # reset as definite. It moved the reset outside the loop and changed the
    # reset=True result from (carry, result)=(4, 10) to (10, 20).
    loop = _loop(
        """for lane in cutlass.range_constexpr(4):
    if reset:
        carry = cutlass.Float32(0)
    carry = carry + values[lane]
    reduced = cute.arch.warp_reduction_sum(cutlass.Float32(carry), threads_in_group=32)
    result = result + reduced
"""
    )
    unchanged = ast.dump(loop, include_attributes=False)
    transformed = _try_hoist_one_vloop(loop, 4, [0], set())
    assert transformed is None
    assert ast.dump(loop, include_attributes=False) == unchanged
    namespace = {
        "cutlass": SimpleNamespace(
            Float32=lambda value: torch.tensor(value, dtype=torch.float32).item(),
            range_constexpr=range,
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(warp_reduction_sum=lambda value, **kwargs: value)
        ),
        "reset": reset,
        "carry": 0.0,
        "values": [1.0, 2.0, 3.0, 4.0],
        "result": 0.0,
    }
    exec(
        compile(ast.fix_missing_locations(ast.Module([loop], [])), "<reset>", "exec"),
        namespace,
    )
    expected = (4.0, 10.0) if reset else (10.0, 20.0)
    assert (namespace["carry"], namespace["result"]) == expected


@pytest.mark.parametrize(
    "prefix",
    [
        "carry = cutlass.Float32(0)\n    carry = carry + values[lane]",
        "carry = values[lane]\n    if reset:\n        carry = cutlass.Float32(0)",
        "if reset:\n        carry = values[lane]\n    else:\n        carry = cutlass.Float32(0)",
        "carry = cutlass.Float32(0)\n    carry += values[lane]",
        "carry: float = values[lane]",
        "for inner in range(1):\n        carry = values[lane]",
    ],
    ids=[
        "unconditional-reset",
        "conditional-overwrite",
        "both-branches",
        "augmented-assignment",
        "annotated-assignment",
        "nested-loop-definition",
    ],
)
@pytest.mark.parametrize("through_alias", [False, True])
def test_only_definite_single_assignment_recipes_name_fp32_inputs(
    prefix: str, through_alias: bool
) -> None:
    alias = "alias = carry" if through_alias else ""
    value = "alias" if through_alias else "carry"
    loop = _loop(
        f"""for lane in cutlass.range_constexpr(4):
    {prefix}
    {alias}
    reduced = cute.arch.warp_reduction_sum(cutlass.Float32({value}), threads_in_group=32)
    result = result + reduced
"""
    )
    unchanged = ast.dump(loop, include_attributes=False)
    assert _try_hoist_one_vloop(loop, 4, [0], set()) is None
    assert ast.dump(loop, include_attributes=False) == unchanged


def test_fp32_input_recipe_keeps_definite_transitive_narrowing() -> None:
    loop = _loop(
        """for lane in cutlass.range_constexpr(4):
    sample = cutlass.Float16(values[lane])
    value = cutlass.Float32(sample)
    reduced = cute.arch.warp_reduction_max(cutlass.Float32(value), threads_in_group=32)
    result = cute.math.max(result, reduced)
"""
    )
    unchanged = ast.dump(loop, include_attributes=False)
    transformed = _try_hoist_one_vloop(loop, 4, [0], set())
    assert transformed is not None
    assert ast.dump(loop, include_attributes=False) == unchanged
    source = ast.unparse(ast.Module(transformed, []))
    assert "sample = cutlass.Float16(values[lane])" in source
    assert "value = cutlass.Float32(sample)" in source
    assert source.count("cute.arch.warp_reduction_max(") == 1


@pytest.mark.parametrize(
    "value",
    [
        "cutlass.Float16(value)",
        "cutlass.Float64(value)",
        "cutlass.Float32(value + 1)",
        "opaque(value)",
    ],
)
def test_unmodelled_reduction_input_expression_keeps_original_loop(value: str) -> None:
    loop = _loop(
        f"""
for lane in cutlass.range_constexpr(4):
    value = values[lane]
    result = cute.arch.warp_reduction_max({value}, threads_in_group=32)
"""
    )
    before = ast.dump(loop, include_attributes=False)
    assert _try_hoist_one_vloop(loop, 4, [0], set()) is None
    assert ast.dump(loop, include_attributes=False) == before


@pytest.mark.parametrize("static_shapes", [False, True])
@pytest.mark.parametrize(
    "dtype,columns,cluster", [(torch.bfloat16, 32768, 2), (torch.float16, 65536, 4)]
)
def test_explicit_fp32_cast_preserves_cluster_admission(
    dtype: torch.dtype, columns: int, cluster: int, static_shapes: bool
) -> None:
    kernel = helion.kernel(
        softmax_two_pass_kernel.fn,
        backend="cute",
        static_shapes=static_shapes,
        fast_math=False,
        autotune_effort="none",
    )
    arguments = (torch.empty(32, columns, dtype=dtype),)
    source = kernel._bind_isolated(arguments).to_code(
        helion.Config(
            block_sizes=[1, columns],
            num_threads=[0, 256],
            cute_vector_widths=[1, 8],
            cute_lane_layouts=["blocked", "strided"],
            cute_cluster_n=cluster,
        )
    )
    if static_shapes:
        assert source.count("_cute_grouped_reduce_cluster_online_pair(") == 1
        assert source.count("_cute_grouped_reduce_cluster(") == 0
        assert "_cute_grouped_reduce_block(" in source
        assert "_pair_exp_cache_0" in source and "_pair_rescale_0" in source
    else:
        # Dynamic column bounds already normalize this config to a regular
        # CTA. In addition to the whole-row max/sum, preserve the independent
        # first-logical-tile max: a leading all--inf tile makes the original
        # online recurrence NaN even if a later tile is finite. Each of the
        # three reductions must run once per runtime column tile, outside all
        # element/lane loops and divergent branches.
        assert "_cute_grouped_reduce_cluster" not in source
        tree = ast.parse(source)
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        reductions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) == "_cute_grouped_reduce_shared_two_stage"
        ]
        assert [ast.literal_eval(node.args[1]) for node in reductions] == [
            "max",
            "max",
            "sum",
        ]
        for reduction in reductions:
            loops = []
            ancestor = reduction
            while ancestor in parents:
                ancestor = parents[ancestor]
                assert not isinstance(ancestor, (ast.If, ast.IfExp, ast.While))
                if isinstance(ancestor, ast.For):
                    loops.append(ancestor)
            assert len(loops) == 1
            assert isinstance(loops[0].iter, ast.Call)
            assert ast.unparse(loops[0].iter.func) == "range"
            assert len(loops[0].iter.args) == 3
    assert "fastmath=True" not in source
