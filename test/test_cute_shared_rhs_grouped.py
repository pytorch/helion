from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable
from test.cute_population_contracts import _cpu_target
from test.test_cute_full_slice_matmul import _CpuRefMode
from test.test_cute_grouped_gemm_split_sizes import _selected_config

import helion
from helion._compiler.compile_environment import CompileEnvironment
from helion._compiler.cute.cute_mma import analyze_tcgen05_grouped_worklist
from helion._hardware import HardwareInfo
from helion._testing import patch_cute_mma_support
from helion._testing import skipUnlessBackends
from helion.exc import BackendUnsupported
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_dynamic_ab_tensormaps
from helion.runtime.ref_mode import RefMode
from helion.runtime.ref_mode import RefModeContext

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from contextlib import ExitStack
    from typing import Any

    from helion.runtime.kernel import BoundKernel


def _offset_mm(a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    groups = offsets.size(0) - 1
    for group in hl.grid(groups):
        start = offsets[group]
        end = offsets[group + 1]
        count = end - start
        if count != 0:
            for row, column in hl.tile([count, n]):
                acc = hl.zeros([row, column], dtype=torch.float32)
                for reduction in hl.tile(k):
                    x = a[start + row.index, reduction]
                    y = b[reduction, column]
                    acc = torch.addmm(acc, x, y)
                out[start + row.index, column] = acc.to(out.dtype)
    return out


def _explicit_offset_mm(
    a: torch.Tensor, b: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    m, k = a.shape
    k2, n = b.shape
    assert k == k2
    out = torch.zeros((m, n), dtype=a.dtype, device=a.device)
    groups = offsets.size(0) - 1
    for group_tile, row, column in hl.tile([groups, m, n], block_size=[1, None, None]):
        group = group_tile.index.sum()
        start = offsets[group]
        count = offsets[group + 1] - start
        row_index = start + row.index
        valid = row.index < count
        acc = hl.zeros([row, column], dtype=torch.float32)
        for reduction in hl.tile(k):
            x = hl.load(a, [row_index, reduction], extra_mask=valid[:, None])
            y = b[reduction, column]
            acc = torch.addmm(acc, x, y)
        hl.store(out, [row_index, column], acc.to(out.dtype), extra_mask=valid[:, None])
    return out


def _target() -> ExitStack:
    stack = _cpu_target()
    stack.enter_context(
        patch_cute_mma_support(
            SimpleNamespace(
                universal=True,
                warp_f16bf16=True,
                warpgroup_f16bf16=True,
                tcgen05_f16bf16=True,
            )
        )
    )
    for name, value in (
        (
            "helion._compiler.cute.tcgen05_config.CuteTcgen05Config.per_cta_smem_capacity_bytes",
            232448,
        ),
        (
            "helion._hardware.get_hardware_info",
            HardwareInfo(
                device_kind="cuda",
                hardware_name="NVIDIA B200",
                runtime_version="13.0",
                compute_capability="sm100",
            ),
        ),
    ):
        stack.enter_context(patch(name, return_value=value))
    return stack


@pytest.fixture(autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
    ):
        yield
    torch.set_num_threads(previous)


def _bind(
    function: Callable[..., object],
    args: tuple[torch.Tensor, ...],
    *,
    enabled: bool = True,
) -> BoundKernel[Any]:
    kernel = helion.kernel(
        function,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=enabled,
    )
    with _target():
        return kernel._bind_isolated(args)


def _inputs(dtype: torch.dtype = torch.bfloat16) -> tuple[torch.Tensor, ...]:
    return (
        torch.empty((640, 128), dtype=dtype),
        torch.empty((128, 352), dtype=dtype),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )


def _plans(source: str) -> list[dict[str, object]]:
    return next(
        ast.literal_eval(statement.value)
        for statement in ast.parse(source).body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Attribute)
        and statement.targets[0].attr == "_helion_cute_wrapper_plans"
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("column_major", [False, True])
@skipUnlessBackends(["cute"])
def test_original_example_emits_shared_rank2_worklist(
    dtype: torch.dtype, column_major: bool
) -> None:
    args = _inputs(dtype)
    if column_major:
        args = (args[0], torch.empty((352, 128), dtype=dtype).T, args[2])
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        assert bound.host_function is not None
        assert bound.config_spec.cute_tcgen05_search_enabled
        facts = bound.config_spec.matmul_facts
        assert len(facts) == 1
        analysis = analyze_tcgen05_grouped_worklist(
            bound.env, bound.host_function.device_ir, facts[0]
        )
        assert analysis is not None
        assert analysis.seed_facts.groups_hint == 4
        assert analysis.seed_facts.n_hint == 352
        assert analysis.seed_facts.b_major == ("k" if column_major else "n")
        assert any(
            seed.config.get("tcgen05_grouped_mode") == "worklist_nm"
            for seed in bound.config_spec.compiler_seed_configs
        )
        source = bound.to_code(_selected_config(64))
    plans = _plans(source)
    grouped = next(
        plan for plan in plans if plan["kind"] == "tcgen05_grouped_static_persistent"
    )
    ab = next(plan for plan in plans if plan["kind"] == "tcgen05_ab_tma")
    assert grouped["shared_rhs"] is True
    assert grouped["scheduler_mode"] == "device_group_search"
    assert grouped["device_layout_kind"] == "offsets"
    assert ab["shared_rhs"] is True
    assert "lhs_rank3_grouped_nt" not in ab
    assert "B.layout.stride[2]" not in source
    assert "init_tensormap_from_atom(tma_atom_a" not in source
    assert "init_tensormap_from_atom(tma_atom_b" in source
    assert "update_tensormap((tcgen05_grouped_tensormap_real_b,)" in source
    assert "tma_desc_ptr=tcgen05_grouped_tensormap_a_desc_ptr" not in source
    assert "cute.gemm(" in source
    body: list[str] = []
    _append_cute_wrapper_plan(body, [], ab)
    wrapper = "\n".join(body)
    lhs_idx = ab["lhs_idx"]
    assert f"arg{lhs_idx}_shape2" not in wrapper
    assert f"stride=(arg{lhs_idx}_stride1, arg{lhs_idx}_stride0)" in wrapper


def _run_reference(
    bound: BoundKernel[Any], args: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    host = bound.host_function
    assert host is not None
    with bound.env, host:
        definition = host.codegen_function_def(list(host.body))
    definition = ast.parse(ast.unparse(ast.fix_missing_locations(definition))).body[0]
    assert isinstance(definition, ast.FunctionDef)
    for node in ast.walk(definition):
        if isinstance(node, ast.For):
            assert isinstance(node.iter, ast.Call)
            node.iter.keywords = [
                keyword for keyword in node.iter.keywords if keyword.arg != "block_size"
            ]
            value = ast.parse(
                "[1, 8, 16]" if isinstance(node.target, ast.Tuple) else "8", mode="eval"
            ).body
            node.iter.keywords.append(ast.keyword(arg="block_size", value=value))
    namespace = dict(bound.kernel.fn.__globals__)
    namespace["_default_cute_launcher"] = None
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[definition], type_ignores=[])),
            "<segmented-reference>",
            "exec",
        ),
        namespace,
    )
    with _cpu_target():
        env = CompileEnvironment(
            torch.device("cpu"),
            dataclasses.replace(bound.settings, ref_mode=RefMode.EAGER),
        )
    context = RefModeContext(env, None)
    context.func_mode = _CpuRefMode()
    with context:
        result = namespace[definition.name](*args)
    assert isinstance(result, torch.Tensor)
    return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "offsets",
    [
        [0, 0, 1, 13, 37],
        [4, 17, 17, 7, 21],
        [9, 9, 9, 9, 9],
        [-10, 45, 60, 65, 100],
        [-200, -1, 20, 100, 105],
    ],
)
@skipUnlessBackends(["cute"])
def test_normalization_preserves_offsets_empty_groups_and_untouched_rows(
    dtype: torch.dtype, offsets: list[int]
) -> None:
    generator = torch.Generator().manual_seed(691)
    a = torch.randint(-7, 8, (37, 16), generator=generator).to(dtype) / 8
    b = torch.randint(-7, 8, (16, 24), generator=generator).to(dtype) / 8
    args = (a, b, torch.tensor(offsets, dtype=torch.int64))
    bound = _bind(_offset_mm, args)
    assert bound.host_function is not None
    assert len(bound.host_function.device_ir.grid_block_ids[0]) == 3
    expected = torch.zeros((37, 24), dtype=dtype)
    for start, end in itertools.pairwise(offsets):
        first, last = max(0, start), min(37, end)
        if last > first:
            expected[first:last] = (a[first:last].float() @ b.float()).to(dtype)
    torch.testing.assert_close(_run_reference(bound, args), expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "before,after",
    [
        ("offsets[group + 1]", "offsets[group + 2]"),
        ("y = b[reduction, column]", "y = b[reduction, column] * 2"),
        (
            "acc = hl.zeros([row, column], dtype=torch.float32)",
            "acc = hl.full([row, column], 1.0, dtype=torch.float32)",
        ),
        (
            "end = offsets[group + 1]",
            "a[group, 0] = 0\n        end = offsets[group + 1]",
        ),
    ],
)
@skipUnlessBackends(["cute"])
def test_normalization_rejects_unproven_programs(before: str, after: str) -> None:
    source = inspect.getsource(_offset_mm)
    assert source.count(before) == 1
    module = PyCodeCache.load(
        "import torch\nimport helion.language as hl\n" + source.replace(before, after)
    )
    args = _inputs()
    bound = _bind(module._offset_mm, args)
    unchanged = _bind(module._offset_mm, args, enabled=False)
    assert bound.host_function is not None
    assert unchanged.host_function is not None
    root = next(node for node in bound.host_function.body if isinstance(node, ast.For))
    original = next(
        node for node in unchanged.host_function.body if isinstance(node, ast.For)
    )
    assert ast.dump(root) == ast.dump(original)


@skipUnlessBackends(["cute"])
def test_normalization_is_opt_in() -> None:
    bound = _bind(_offset_mm, _inputs(), enabled=False)
    assert bound.host_function is not None
    root = next(node for node in bound.host_function.body if isinstance(node, ast.For))
    assert isinstance(root.iter, ast.Call)
    assert ast.unparse(root.iter.func) == "hl.grid"


@pytest.mark.parametrize(
    "effect", ["b[0, 0] = 0", "hl.atomic_add(b, [0, 0], 1.0)", "offsets[0] = 0"]
)
@skipUnlessBackends(["cute"])
def test_shared_rhs_worklist_rejects_other_device_writes(effect: str) -> None:
    source = inspect.getsource(_explicit_offset_mm)
    anchor = "        group = group_tile.index.sum()"
    module = PyCodeCache.load(
        "import torch\nimport helion.language as hl\n"
        + source.replace(anchor, "        " + effect + "\n" + anchor)
    )
    args = _inputs()
    bound = _bind(module._explicit_offset_mm, args)
    host = bound.host_function
    assert host is not None
    fact = bound.config_spec.matmul_facts[0]
    assert analyze_tcgen05_grouped_worklist(bound.env, host.device_ir, fact) is None
    # This mutation is independent of the offset/MMA dataflow. Prove that
    # operand/scaffold use counts alone would otherwise admit it.
    with patch(
        "helion._compiler.cute.cute_mma._shared_rhs_region_is_exclusive",
        return_value=True,
    ):
        assert (
            analyze_tcgen05_grouped_worklist(bound.env, host.device_ir, fact)
            is not None
        )


@skipUnlessBackends(["cute"])
def test_shared_rhs_worklist_rejects_input_alias_output() -> None:
    source = inspect.getsource(_explicit_offset_mm).replace(
        "out = torch.zeros((m, n), dtype=a.dtype, device=a.device)", "out = a"
    )
    module = PyCodeCache.load("import torch\nimport helion.language as hl\n" + source)
    args = (
        torch.empty((640, 128), dtype=torch.bfloat16),
        torch.empty((128, 128), dtype=torch.bfloat16),
        torch.tensor([0, 0, 1, 129, 640], dtype=torch.int32),
    )
    bound = _bind(module._explicit_offset_mm, args)
    assert bound.host_function is not None
    assert (
        analyze_tcgen05_grouped_worklist(
            bound.env, bound.host_function.device_ir, bound.config_spec.matmul_facts[0]
        )
        is None
    )


@skipUnlessBackends(["cute"])
def test_shared_rhs_worklist_rejects_unknown_device_call() -> None:
    args = _inputs()
    bound = _bind(_explicit_offset_mm, args)
    host = bound.host_function
    assert host is not None
    fact = bound.config_spec.matmul_facts[0]
    assert analyze_tcgen05_grouped_worklist(bound.env, host.device_ir, fact) is not None
    root = host.device_ir.graphs[host.device_ir.root_ids[0]].graph

    def opaque() -> None:
        raise AssertionError("analysis must not execute the opaque call")

    with root.inserting_before(next(iter(root.nodes))):
        root.call_function(opaque)
    assert analyze_tcgen05_grouped_worklist(bound.env, host.device_ir, fact) is None


@skipUnlessBackends(["cute"])
def test_shared_rhs_worklist_rejects_unaligned_input_pointer() -> None:
    a, b, offsets = _inputs()
    b = torch.empty(b.numel() + 1, dtype=b.dtype)[1:].view(b.shape)
    args = (a, b, offsets)
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        with pytest.raises(
            BackendUnsupported,
            match="aligned_shared_rhs_bases|RHS TMA pipeline was disabled",
        ):
            bound.to_code(_selected_config(64))


@dataclasses.dataclass
class _OffsetPointer:
    values: torch.Tensor
    offset: int = 0

    def __add__(self, offset: object) -> _OffsetPointer:
        return _OffsetPointer(self.values, self.offset + int(cast("Any", offset)))

    def load(self) -> torch.Tensor:
        return self.values[self.offset]


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@skipUnlessBackends(["cute"])
def test_emitted_device_metadata_preserves_clipped_signed_ranges(
    offset_dtype: torch.dtype,
) -> None:
    a, b, offsets = _inputs()
    args = (a, b, offsets.to(offset_dtype))
    with _target():
        bound = _bind(_offset_mm, args)
        source = bound.to_code(_selected_config(64))
    setup = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.If)
        and any(
            isinstance(statement, ast.Assign)
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "tcgen05_grouped_raw_start"
            for statement in node.body
        )
    )
    code = compile(
        ast.Module(body=[setup], type_ignores=[]), "<device-offsets>", "exec"
    )

    def convert(value: object, dtype: torch.dtype) -> torch.Tensor:
        return (
            value.to(dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=dtype)
        )

    limits = torch.iinfo(offset_dtype)
    for routing in (
        [-50, 700, 1000, 1000, 1001],
        [4, 131, 131, 7, 617],
        [limits.min, -1, 640, limits.max, 0],
        [31, limits.min + 8, limits.max, limits.max, 0],
    ):
        values = torch.tensor(routing, dtype=offset_dtype)
        starts = torch.zeros(4, dtype=torch.int32)
        sizes = torch.zeros((4, 4), dtype=torch.int32)
        namespace = {
            "cutlass": SimpleNamespace(
                Int32=lambda value: convert(value, torch.int32),
                Int64=lambda value: convert(value, torch.int64),
            ),
            "cute": SimpleNamespace(arch=SimpleNamespace(thread_idx=lambda: (0, 0, 0))),
            "offsets": SimpleNamespace(
                iterator=_OffsetPointer(values), layout=SimpleNamespace(stride=(1,))
            ),
            "tcgen05_grouped_starts": starts,
            "tcgen05_grouped_problem_sizes": sizes,
        }
        exec(code, namespace)
        # Subtraction has the input integer dtype; Python's wider sum below
        # enumerates the logical signed range without overflowing its endpoint.
        lengths = (values[1:] - values[:-1]).tolist()
        for group, (start, length) in enumerate(zip(routing, lengths, strict=False)):
            first = min(max(start, 0), 640)
            count = sum(start <= row < start + max(length, 0) for row in range(640))
            assert int(starts[group]) == first
            assert sizes[group].tolist() == [352, count, 128, 1]


@pytest.mark.parametrize("column_major", [False, True])
def test_shared_rhs_runtime_validation_checks_rank_stride_and_alignment(
    column_major: bool,
) -> None:
    a = torch.empty((640, 128), dtype=torch.bfloat16)
    b = (
        torch.empty((352, 128), dtype=torch.bfloat16).T
        if column_major
        else torch.empty((128, 352), dtype=torch.bfloat16)
    )
    plan: dict[str, object] = {
        "lhs_idx": 0,
        "rhs_idx": 1,
        "shared_rhs": True,
        "orientation": "nm",
        "dynamic_ab_tensormaps": True,
        "dynamic_ab_tensormap_rank": 2,
        "m_size": 640,
        "n_size": 352,
        "k_total_size": 128,
    }

    def tensor_arg(
        plan: dict[str, object], args: tuple[object, ...], key: str, operand: str
    ) -> torch.Tensor:
        return cast("torch.Tensor", args[cast("int", plan[key])])

    with patch(
        "helion.runtime.cute.launcher._tcgen05_grouped_dynamic_ab_tensor_arg",
        tensor_arg,
    ):
        _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, (a, b))
        bad = torch.as_strided(
            torch.empty(b.numel() + 1, dtype=b.dtype)[1:], b.shape, b.stride()
        )
        with pytest.raises(BackendUnsupported, match="16-byte-aligned"):
            _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, (a, bad))
        with pytest.raises(BackendUnsupported, match="matching rank-2"):
            _validate_tcgen05_grouped_dynamic_ab_tensormaps(plan, (a, b.unsqueeze(0)))
