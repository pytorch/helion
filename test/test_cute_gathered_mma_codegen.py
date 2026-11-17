from __future__ import annotations

import ast

import pytest
import torch

import helion
from helion._compat import supports_torch_compile_template_lowering
from helion._compiler.ast_extension import unparse
from helion._compiler.cute.autotune.heuristics.cute import CuteCollectiveMatmulHeuristic
from helion._testing import import_example
from helion._testing import skipUnlessBackends
from helion._testing.cute_codegen import cpu_bind
from helion.runtime.cute.launch_descriptor import wrapper_plan_bakes_tensor_shapes
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

moe_matmul_ogs = import_example("moe_matmul_ogs").moe_matmul_ogs

pytestmark = [pytest.mark.cpu_only, skipUnlessBackends(["cute"])]


def _bound(
    shape: tuple[int, int, int, int],
    dtype: torch.dtype,
    static: bool,
    *,
    lhs_offset: int = 0,
):
    rows, k, n, groups = shape
    lhs_storage = torch.empty(rows * k + lhs_offset, dtype=dtype)
    inputs = (
        lhs_storage[lhs_offset:].view(rows, k),
        torch.empty((groups, k, n), dtype=dtype),
        torch.empty(groups, dtype=torch.int32),
        torch.empty(groups + 1, dtype=torch.int32),
        torch.empty(rows, dtype=torch.int32),
        rows // groups + 32,
    )
    kernel = helion.kernel(
        moe_matmul_ogs.fn,
        backend="cute",
        static_shapes=static,
        autotune_effort="none",
    )
    return cpu_bind(kernel, inputs)


@pytest.mark.parametrize(
    "shape", [(256, 128, 128, 4), (1024, 512, 512, 8), (4096, 1024, 1024, 16)]
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("static", [False, True])
def test_original_gathered_contraction_emits_pipeline(shape, dtype, static) -> None:
    bound = _bound(shape, dtype, static)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
            cute_collective_mma=True,
            cute_collective_compute="tma_gather",
            cute_gathered_mma_n=512,
            cute_gathered_mma_stages=2,
        )
    )
    assert "gathered_mma_tma" in source
    assert "gather_four_rows" in source
    assert "num_threads=288" in source
    assert "_helion_pending_collective_mma" not in source
    assert "torch.zeros(" in source
    # The final store retains its row predicate and complete/tail vector check.
    assert "gathered_logical_row" in source
    assert "% 8 == 0" in source
    ast.parse(source)


@pytest.mark.skipif(
    not supports_torch_compile_template_lowering(),
    reason="this PyTorch build lacks Helion's Inductor template lowering",
)
def test_dynamic_inductor_render_runs_the_warp_lowering() -> None:
    """The gathered pipeline takes its operands' exact sizes and strides,
    which Inductor's render of a dynamic-shape graph cannot prove: it runs at
    every size the graph accepts, without the binding's dispatch guards.  The
    render takes the warp lowering, as without the knob, rather than refusing
    a config Inductor's tuner may have chosen; a static graph's keeps the
    pipeline."""
    from helion._compiler._inductor.template_buffer import render_inductor_ast

    bound = _bound((256, 128, 128, 4), torch.bfloat16, False)

    def render(compute: str, *, symbolic: bool) -> str:
        config = helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
            cute_collective_mma=True,
            cute_collective_compute=compute,
            cute_gathered_mma_n=512,
            cute_gathered_mma_stages=2,
        )
        return unparse(
            render_inductor_ast(bound, config, symbolic_input_layouts=symbolic)
        )

    assert "gathered_mma_tma" in render("tma_gather", symbolic=False)
    dynamic = render("tma_gather", symbolic=True)
    assert "gathered_mma_tma" not in dynamic
    assert dynamic == render("warp", symbolic=True)


def test_an_outer_trace_binding_refuses_the_gathered_config() -> None:
    """Only Inductor's render of a dynamic graph skips the binding's dispatch
    guards.  A binding from an outer trace's FakeTensors keeps them, but its
    hints are that trace's and prove no exact size, so the pinned gathered
    config is refused, as before, not replaced by the warp lowering."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    rows, k, n, groups = 256, 128, 128, 4
    with FakeTensorMode():
        inputs = (
            torch.empty((rows, k), dtype=torch.bfloat16),
            torch.empty((groups, k, n), dtype=torch.bfloat16),
            torch.empty(groups, dtype=torch.int32),
            torch.empty(groups + 1, dtype=torch.int32),
            torch.empty(rows, dtype=torch.int32),
            rows // groups + 32,
        )
        kernel = helion.kernel(
            moe_matmul_ogs.fn,
            backend="cute",
            static_shapes=False,
            autotune_effort="none",
        )
        bound = cpu_bind(kernel, inputs)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="gathered MMA requires one isolated"
    ):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_vector_widths=[1, 1, 1, 1],
                cute_collective_mma=True,
                cute_collective_compute="tma_gather",
                cute_gathered_mma_n=512,
                cute_gathered_mma_stages=2,
            )
        )


def test_gathered_seeds_keep_wide_columns_and_pipeline_depth() -> None:
    bound = _bound((1024, 512, 512, 8), torch.float16, False)
    assert bound.host_function is not None
    with bound.env:
        seeds = CuteCollectiveMatmulHeuristic.get_seed_configs(
            bound.env, bound.host_function.device_ir
        )
    gathered = [
        seed
        for seed in seeds
        if seed.config.get("cute_collective_compute") == "tma_gather"
    ]
    assert {
        (seed["cute_gathered_mma_n"], seed["cute_gathered_mma_stages"])
        for seed in gathered
    } == {(128, 2), (128, 3), (128, 4), (256, 2), (256, 3), (256, 4), (512, 2)}
    for seed in gathered:
        normalized = helion.Config.from_dict(dict(seed.config))
        bound.config_spec.normalize(normalized)
        assert normalized["cute_collective_compute"] == "tma_gather"
        assert normalized["cute_gathered_mma_n"] == seed["cute_gathered_mma_n"]
        assert (
            normalized["cute_gathered_mma_stages"] == seed["cute_gathered_mma_stages"]
        )


def test_gathered_wrapper_uses_runtime_m_extent_and_exact_tensor_shapes() -> None:
    plan: dict[str, object] = {
        "kind": "gathered_mma_tma",
        "lhs_idx": 0,
        "rhs_idx": 1,
        "out_idx": 2,
        "m_extent_idx": 3,
        "groups": 4,
        "n_size": 128,
        "bn": 512,
        "stages": 2,
        "grid_cap": 65535,
        "kernel_args": ["aa", "ba", "bt", "mma", "al", "bl"],
    }
    statements = []
    arguments = []
    _append_cute_wrapper_plan(statements, arguments, plan)
    text = "\n".join(statements)
    assert "cutlass.Int32(arg3)" in text
    assert "_helion_make_gathered_tma(arg0, arg1, 512, 2)" in text
    assert arguments == plan["kernel_args"]
    assert wrapper_plan_bakes_tensor_shapes(plan)


def test_oversized_gathered_pipeline_is_rejected_by_wrapper() -> None:
    plan: dict[str, object] = {
        "kind": "gathered_mma_tma",
        "bn": 512,
        "stages": 3,
        "kernel_args": ["aa", "ba", "bt", "mma", "al", "bl"],
    }
    with pytest.raises(helion.exc.BackendUnsupported, match="wrapper geometry"):
        _append_cute_wrapper_plan([], [], plan)


def test_gathered_mode_requires_collective_admission() -> None:
    bound = _bound((256, 128, 128, 4), torch.float16, False)
    with pytest.raises(helion.exc.BackendUnsupported, match="admitted collective"):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_vector_widths=[1, 1, 1, 1],
                cute_collective_mma=False,
                cute_collective_compute="tma_gather",
            )
        )


def test_gathered_mode_rejects_unaligned_input_after_binding() -> None:
    bound = _bound((256, 128, 128, 4), torch.float16, False, lhs_offset=1)
    with pytest.raises(helion.exc.BackendUnsupported, match="unproved gathered"):
        bound.to_code(
            helion.Config(
                block_sizes=[128, 32, 64],
                num_threads=[4, 32, 1],
                cute_vector_widths=[1, 1, 1, 1],
                cute_collective_mma=True,
                cute_collective_compute="tma_gather",
            )
        )


def test_group_metadata_loads_are_bound_once_ahead_of_the_pipeline() -> None:
    # The row recipe reads the group's offset and the row predicate its
    # token count; both are coordinate-free, so the pipeline binds each once
    # (the producer's row chain used to load them twice per row, the
    # epilogue's scatter once per 16-byte packet between its global stores,
    # which the compiler cannot reorder: 1.1 us of a 3.4 us kernel on the
    # 256 x 128 x 128 x 4 shape), to the kernel's own bindings when it has
    # them. The gather of the row index itself keeps its load: it depends on
    # the row.
    bound = _bound((256, 128, 128, 4), torch.float16, False)
    source = bound.to_code(
        helion.Config(
            block_sizes=[128, 32, 64],
            num_threads=[4, 32, 1],
            cute_vector_widths=[1, 1, 1, 1],
            cute_collective_mma=True,
            cute_collective_compute="tma_gather",
            cute_gathered_mma_n=128,
            cute_gathered_mma_stages=2,
        )
    )
    kernel = source[
        source.index("@cute.kernel") : source.index("_helion_cute_wrapper_plans")
    ]
    lines = [line.strip() for line in kernel.splitlines()]
    bound_meta = [line for line in lines if line.startswith("gathered_group_meta")]
    # The kernel loaded the group's offset and count for its ``if count != 0``
    # guard (``start`` / ``num_tokens``): the region reuses those registers
    # instead of re-issuing the loads ahead of its CTA-wide sync.
    assert bound_meta == [
        "gathered_group_meta = start",
        "gathered_group_meta_1 = num_tokens",
    ]
    region = kernel.index("gathered_group_meta")
    for name in ("expert_token_offsets", "expert_token_counts"):
        assert kernel.count(f"({name}.iterator", region) == 0, name
    # the gathered row index is the region's only load: it depends on the row
    assert kernel.count(").load()", region) == 1
    predicate = next(
        line
        for line in lines
        if line.startswith("if (operator.lt(gathered_logical_row")
    )
    assert ".load()" not in predicate
    assert "gathered_group_meta_1" in predicate
    row = next(line for line in lines if line.startswith("gathered_row = "))
    assert "sorted_to_orig_token_idx.iterator" in row and ".load()" in row
    assert "gathered_group_meta" in row
    live = next(line for line in lines if line.startswith("gathered_live_rows = "))
    assert ".load()" not in live and "gathered_group_meta_1" in live
    # the bindings precede the pipeline's first barrier
    assert kernel.index("gathered_group_meta") < kernel.index("mbarrier_init")
    ast.parse(source)
