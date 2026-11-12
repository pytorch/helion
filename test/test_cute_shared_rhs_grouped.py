from __future__ import annotations

import ast
from copy import deepcopy
import dataclasses
import inspect
import itertools
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import cast
from unittest.mock import patch

import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_grouped_gemm_split_sizes import _selected_config
from test.cute_population_contracts import _target

import helion
from helion._compiler.cute.cute_mma import analyze_tcgen05_grouped_worklist
from helion._testing import import_example
from helion._testing.cute_codegen import compile_reference
from helion._testing.cute_codegen import reference_host_function
from helion._testing.cute_codegen import run_cpu_reference
from helion._testing.cute_structural_kernels import _offset_mm
from helion.exc import BackendUnsupported
import helion.language as hl
from helion.runtime.cute.launcher import _append_cute_wrapper_plan
from helion.runtime.cute.launcher import _validate_tcgen05_grouped_dynamic_ab_tensormaps

_examples_grouped_gemm = import_example("grouped_gemm")
grouped_gemm_jagged = _examples_grouped_gemm.grouped_gemm_jagged
grouped_gemm_jagged_persistent = _examples_grouped_gemm.grouped_gemm_jagged_persistent

# The B200 model, CUDA hidden and native compilation forbidden.
pytestmark = [pytest.mark.cpu_only, pytest.mark.cpu_render()]
if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from typing import Any

    from helion.runtime.kernel import BoundKernel


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


# ``_target`` (the B200 model with tcgen05 MMA support) is the population
# contracts' target; the files importing it from here keep working.


@pytest.fixture(autouse=True)
def _single_thread() -> Generator[None, None, None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
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
def test_original_example_emits_shared_rank2_worklist(
    dtype: torch.dtype, column_major: bool
) -> None:
    args = _inputs(dtype)
    if column_major:
        args = (args[0], torch.empty((352, 128), dtype=dtype).T, args[2])
    with _target():
        bound = _bind(grouped_gemm_jagged.fn, args)
        assert bound.host_function is not None
        assert bound.config_spec.tcgen05.search_enabled
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
    definition = reference_host_function(bound)
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
    reference = compile_reference(bound, definition, "<segmented-reference>")
    result = run_cpu_reference(bound, reference, args)
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


def test_normalization_is_opt_in() -> None:
    bound = _bind(_offset_mm, _inputs(), enabled=False)
    assert bound.host_function is not None
    root = next(node for node in bound.host_function.body if isinstance(node, ast.For))
    assert isinstance(root.iter, ast.Call)
    assert ast.unparse(root.iter.func) == "hl.grid"


@pytest.mark.parametrize(
    "effect", ["b[0, 0] = 0", "hl.atomic_add(b, [0, 0], 1.0)", "offsets[0] = 0"]
)
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


_METADATA_THREAD_NAMES = {
    "tcgen05_grouped_metadata_thread_idx": "cute.arch.thread_idx()",
    "tcgen05_grouped_metadata_block_dim": "cute.arch.block_dim()",
}


def _offset_initializer_statements(tree: ast.Module) -> list[ast.stmt]:
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_")
    )
    setup = next(
        node
        for node in function.body
        if isinstance(node, (ast.If, ast.For))
        and any(
            isinstance(child, ast.Assign)
            and ast.unparse(child.targets[0]) == "tcgen05_grouped_raw_start"
            for child in node.body
        )
    )
    if isinstance(setup, ast.If):
        return [setup]
    assert isinstance(setup.target, ast.Name)
    assert setup.target.id == "tcgen05_grouped_metadata_group"
    assert isinstance(setup.iter, ast.Call)
    assert ast.unparse(setup.iter.func) == "cutlass.range"
    assert len(setup.iter.args) == 3
    assert ast.unparse(setup.iter.args[0]) == (
        "tcgen05_grouped_metadata_thread_idx[0] + "
        "tcgen05_grouped_metadata_block_dim[0] * "
        "(tcgen05_grouped_metadata_thread_idx[1] + "
        "tcgen05_grouped_metadata_block_dim[1] * "
        "tcgen05_grouped_metadata_thread_idx[2])"
    )
    assert ast.unparse(setup.iter.args[2]) == (
        "tcgen05_grouped_metadata_block_dim[0] * "
        "tcgen05_grouped_metadata_block_dim[1] * "
        "tcgen05_grouped_metadata_block_dim[2]"
    )
    assert [
        (keyword.arg, ast.literal_eval(keyword.value))
        for keyword in setup.iter.keywords
    ] == [("unroll", 1)]
    assert not any(
        isinstance(node, ast.Call)
        and ast.unparse(node.func) == "cute.arch.sync_threads"
        for statement in setup.body
        for node in ast.walk(statement)
    )
    assert (
        ast.unparse(function.body[function.body.index(setup) + 1])
        == "cute.arch.sync_threads()"
    )
    prologue = [
        node
        for node in function.body[: function.body.index(setup)]
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) in _METADATA_THREAD_NAMES
    ]
    assert len(prologue) == 2
    for node in prologue:
        assert (
            ast.unparse(node.value)
            == _METADATA_THREAD_NAMES[ast.unparse(node.targets[0])]
        )
    return [*prologue, setup]


def _run_offset_initializer(
    statements: list[ast.stmt], namespace: dict[str, Any]
) -> None:
    """Execute the real initializer across a physical 3D block and check owners."""
    observed: list[int] = []

    class Observe(ast.NodeTransformer):
        def visit_Assign(self, node: ast.Assign) -> ast.Assign | list[ast.stmt]:
            target = node.targets[0]
            if (
                isinstance(target, ast.Subscript)
                and ast.unparse(target.value) == "tcgen05_grouped_starts"
                and isinstance(node.value, ast.Call)
                and ast.unparse(node.value)
                == "cutlass.Int32(tcgen05_grouped_visible_start)"
            ):
                record = ast.Expr(
                    ast.Call(
                        ast.Name("_observe_group", ast.Load()),
                        [deepcopy(target.slice)],
                        [],
                    )
                )
                return [node, record]
            return node

    tree = Observe().visit(ast.Module(body=deepcopy(statements), type_ignores=[]))
    code = compile(ast.fix_missing_locations(tree), "<device-offsets>", "exec")
    namespace["_observe_group"] = lambda group: observed.append(int(group))
    namespace["cutlass"].range = lambda *args, **kwargs: range(
        *(int(arg) for arg in args)
    )
    shape = (2, 2, 2)
    namespace["cute"].arch.block_dim = lambda: shape
    for coordinate in itertools.product(*(range(size) for size in shape)):
        namespace["cute"].arch.thread_idx = lambda coordinate=coordinate: coordinate
        exec(code, namespace)
    assert sorted(observed) == list(range(namespace["tcgen05_grouped_starts"].numel()))


def _serial_offset_initializer(tree: ast.Module) -> ast.Module:
    """Expand only independent parallel metadata owners into the old leader form.

    This retains every typed load, signed clipping operation, and record store.
    All other generated code still participates in the exact AST comparison.
    """
    result = deepcopy(tree)
    statements = _offset_initializer_statements(result)
    setup = statements[-1]
    if not isinstance(setup, ast.For):
        return result
    assert isinstance(setup.iter, ast.Call) and isinstance(setup.target, ast.Name)
    groups = ast.literal_eval(setup.iter.args[1])
    assert type(groups) is int and groups > 0
    group_name = setup.target.id
    body: list[ast.stmt] = []
    for group in range(groups):

        class Group(ast.NodeTransformer):
            def visit_Name(self, node: ast.Name, *, group: int = group) -> ast.expr:
                return ast.Constant(group) if node.id == group_name else node

            def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
                updated = self.generic_visit(node)
                assert isinstance(updated, ast.BinOp)
                node = updated
                if isinstance(node.left, ast.Constant) and isinstance(
                    node.right, ast.Constant
                ):
                    if (
                        type(node.left.value) is int
                        and type(node.right.value) is int
                        and isinstance(node.op, ast.Add)
                    ):
                        return ast.Constant(node.left.value + node.right.value)
                return node

        body.extend(Group().visit(deepcopy(node)) for node in setup.body)
    leader = ast.parse(
        "if cute.arch.thread_idx()[0] == 0 and cute.arch.thread_idx()[1] == 0 and cute.arch.thread_idx()[2] == 0:\n    pass"
    ).body[0]
    assert isinstance(leader, ast.If)
    leader.body = body
    function = next(
        node
        for node in result.body
        if isinstance(node, ast.FunctionDef) and setup in node.body
    )
    remainder = [node for node in function.body if node not in statements]
    assert not any(
        isinstance(node, ast.Name) and node.id in _METADATA_THREAD_NAMES
        for statement in remainder
        for node in ast.walk(statement)
    )
    function.body = [
        leader if node is setup else node
        for node in function.body
        if node not in statements[:-1]
    ]
    return result


@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_emitted_device_metadata_preserves_clipped_signed_ranges(
    offset_dtype: torch.dtype,
) -> None:
    a, b, offsets = _inputs()
    args = (a, b, offsets.to(offset_dtype))
    with _target():
        bound = _bind(_offset_mm, args)
        source = bound.to_code(_selected_config(64))
    statements = _offset_initializer_statements(ast.parse(source))

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
        _run_offset_initializer(statements, namespace)
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


# GPU numerics of the shared-RHS grouped GEMM: device offsets and graph
# replay (sm_100 family).


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("column_major", [False, True])
def test_shared_rhs_device_offsets_and_graph_replay(
    dtype: torch.dtype, offset_dtype: torch.dtype, column_major: bool
) -> None:
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100-family TCGEN05 required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(731)
    a = torch.randn((313, 128), device=device, dtype=dtype, generator=generator)
    b = torch.randn(
        (352, 128) if column_major else (128, 352),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if column_major:
        b = b.T
    offsets = torch.tensor([0, 0, 1, 129, 313], device=device, dtype=offset_dtype)
    args = (a, b, offsets)
    kernel = helion.kernel(
        _offset_mm,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    config = _selected_config(64)
    config.config["tcgen05_ab_stages"] = 6
    source = bound.to_code(config)
    assert "'shared_rhs': True" in source
    assert "cute.gemm(" in source
    bound.set_config(config)

    def expected(routing: list[int]) -> torch.Tensor:
        result = torch.zeros((313, 352), device=device, dtype=dtype)
        product = (a.float() @ b.float()).to(dtype)
        for start, end in itertools.pairwise(routing):
            first, last = max(0, start), min(313, end)
            if last > first:
                result[first:last] = product[first:last]
        return result

    torch.testing.assert_close(
        bound(*args), expected([0, 0, 1, 129, 313]), rtol=3e-2, atol=3e-2
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = bound(*args)
    for routing in (
        [5, 171, 171, 9, 287],  # Empty, overlapping, and uncovered intervals.
        [-50, 363, 380, 380, 400],  # Negative prefix and an extent larger than M.
        [7, 7, 7, 7, 7],  # No work remains, including the final epilogue.
    ):
        a.normal_(generator=generator)
        b.normal_(generator=generator)
        offsets.copy_(offsets.new_tensor(routing))
        graph.replay()
        torch.testing.assert_close(captured, expected(routing), rtol=3e-2, atol=3e-2)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_persistent_example_takes_the_shared_rhs_worklist() -> None:
    # The hand-rolled worklist of the persistent example normalizes to the
    # same segment tile space as ``_offset_mm``; its grouped tcgen05 render
    # computes the product over the covered rows (an empty group included).
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("SM100-family TCGEN05 required")
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(733)
    a = torch.randn(
        (313, 128), device=device, dtype=torch.bfloat16, generator=generator
    )
    b = torch.randn(
        (128, 352), device=device, dtype=torch.bfloat16, generator=generator
    )
    offsets = torch.tensor([0, 0, 1, 129, 313], device=device, dtype=torch.int32)
    args = (a, b, offsets)
    kernel = helion.kernel(
        grouped_gemm_jagged_persistent.fn,
        backend="cute",
        static_shapes=False,
        autotune_effort="none",
        cute_segmented_matmul_tiling=True,
    )
    bound = kernel.bind(args)
    config = _selected_config(64)
    source = bound.to_code(config)
    assert "'shared_rhs': True" in source
    assert "cute.gemm(" in source
    bound.set_config(config)
    expected = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(bound(*args), expected, rtol=3e-2, atol=3e-2)
