from __future__ import annotations

import ast
from copy import deepcopy
import inspect
import itertools
import operator
import random
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

from examples.grouped_gemm import grouped_gemm_jagged
import pytest
import torch
from torch._inductor.codecache import PyCodeCache

from test._cute_binding import _mock_cuda_unavailable
from test.test_cute_grouped_coverage_search import _bound
from test.test_cute_grouped_coverage_search import _search
from test.test_cute_grouped_full_coverage import _config as _old_config
from test.test_cute_grouped_full_coverage import _metadata
from test.test_cute_grouped_full_coverage import _metadata_code
from test.test_cute_grouped_gemm_split_sizes import _device_offsets_kernel
from test.test_cute_shared_rhs_grouped import _bind
from test.test_cute_shared_rhs_grouped import _explicit_offset_mm
from test.test_cute_shared_rhs_grouped import _OffsetPointer
from test.test_cute_shared_rhs_grouped import _plans
from test.test_cute_shared_rhs_grouped import _target

from helion._compiler.cute.grouped_row_union import CONFIG_KEY as KEY
from helion._compiler.cute.grouped_row_union import TRANSPOSED
from helion._compiler.cute.grouped_row_union import GroupedRowUnionPlan
from helion._compiler.cute.grouped_row_union import index_domain
from helion._compiler.program_id import ProgramIDs
from helion._compiler.program_id import Tcgen05PersistentProgramIDs
from helion._testing import skipUnlessBackends
from helion.autotuner.metrics import AutotuneMetrics
from helion.exc import BackendUnsupported
from helion.exc import InvalidConfig
from helion.runtime.cute.launcher import _append_cute_wrapper_plan

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Sequence

    import helion
    from helion.autotuner.base_search import PopulationMember
    from helion.runtime.kernel import BoundKernel


@pytest.fixture(scope="module", autouse=True)
def _cpu_only() -> Generator[None, None, None]:
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with (
        _mock_cuda_unavailable(),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("GPU forbidden")),
        _target(),
    ):
        yield
    torch.set_num_threads(old_threads)


def _selected(bound: BoundKernel) -> helion.Config:
    group = next(g for g in bound.config_spec.compiler_coverage_groups if g.key == KEY)
    config = group.witnesses[0].carrier
    config.config[KEY] = True
    return config


@pytest.fixture(scope="module")
def case() -> tuple[BoundKernel, tuple[torch.Tensor, ...], str, str]:
    bound, args = _bound(0)
    old = bound.to_code(_old_config(None))
    new = bound.to_code(_selected(bound))
    return bound, args, old, new


@skipUnlessBackends(["cute"])
def test_dense_plan_uses_original_allocation_and_ordinary_wrapper(case: tuple) -> None:
    bound, _args, old, new = case
    assert "tcgen05_grouped_" not in new
    assert new.count("StaticPersistentTileScheduler.create(") == 3
    assert new.count("PersistentTileSchedulerParams((5, 2, 1)") == 3
    (plan,) = _plans(new)
    assert plan["kind"] == "tcgen05_ab_tma"
    assert (plan["bm"], plan["bn"], plan["bk"], plan["ab_stage_count"]) == (
        128,
        64,
        128,
        2,
    )
    assert "shared_rhs" not in plan and "dynamic_ab_tensormaps" not in plan
    wrapper: list[str] = []
    _append_cute_wrapper_plan(wrapper, [], plan)
    assert ", arg1, cute.slice_(tma_atom_a_smem_layout" in "\n".join(wrapper)
    assert "arg2_stride0" in "\n".join(wrapper)

    def host(source: str) -> ast.FunctionDef:
        return next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "grouped_gemm_jagged"
        )

    old_host, new_host = host(old), host(new)
    # Only launch geometry and physical block roles change in the public host.
    for node in (old_host, new_host):
        launch = next(
            child.value
            for child in node.body
            if isinstance(child, ast.Expr)
            and isinstance(child.value, ast.Call)
            and ast.unparse(child.value.func) == "_launcher"
        )
        launch.args[1] = ast.Constant(value=0)
        launch.keywords = []
        grid_constants = [
            child
            for child in node.body
            if isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and isinstance(child.targets[0], ast.Name)
            and child.targets[0].id.startswith("_BLOCK_SIZE_")
        ]
        for child in grid_constants:
            assert isinstance(child.value, ast.Constant)
            assert child.value.value in (256, 128)
            assert not any(
                isinstance(use, ast.Name)
                and isinstance(use.ctx, ast.Load)
                and use.id == child.targets[0].id
                for use in ast.walk(node)
            )
            node.body.remove(child)
    assert ast.dump(old_host) == ast.dump(new_host)
    for node in (old_host, new_host):
        assert (
            sum(
                isinstance(child, ast.Call) and ast.unparse(child.func) == "torch.empty"
                for child in ast.walk(node)
            )
            == 1
        )
    assert "stride=(128, 1)" in new
    assert "stride=(0, row_union_store_pred.layout.stride[1])" in new
    assert bound.config_spec._cute_tcgen05_config.grouped_row_union_supported


@pytest.mark.parametrize("mode", [None, "fixed_tma_dense", "fixed_tma_dense_local"])
@skipUnlessBackends(["cute"])
def test_false_retains_old_config_and_source(case: tuple, mode: str | None) -> None:
    bound = case[0]
    old = _old_config(mode)
    explicit = deepcopy(old)
    explicit.config[KEY] = False
    assert bound.to_code(old) == bound.to_code(explicit)
    old_normalized = bound.config_spec.normalized_config(old)
    explicit_normalized = bound.config_spec.normalized_config(explicit)
    assert old_normalized == explicit_normalized and KEY not in explicit_normalized


@pytest.mark.parametrize(
    "updates",
    [
        {KEY: 1},
        {"block_sizes": [128, 64, 64]},
        {"tcgen05_ab_stages": 3},
        {"tcgen05_grouped_mode": "worklist_nm"},
        {"tcgen05_grouped_full_coverage": "fixed_tma_dense_local"},
        {"tcgen05_cluster_n": 2},
        {"tcgen05_c_store_mode": "skip_epilogue_store"},
        {"l2_groupings": [2]},
    ],
)
@skipUnlessBackends(["cute"])
def test_incompatible_explicit_tuple_rejects(case: tuple, updates: dict) -> None:
    config = _selected(case[0])
    config.config.update(updates)
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        case[0].to_code(config)


@pytest.mark.parametrize(
    "kind", ["fp16", "offset64", "m_tail", "n_tail", "k_tail", "transpose"]
)
@skipUnlessBackends(["cute"])
def test_metadata_fallback_is_conservative(case: tuple, kind: str) -> None:
    _bound0, args, _old, _new = case
    a, b, offsets = args
    if kind == "fp16":
        a, b = a.half(), b.half()
    elif kind == "offset64":
        offsets = offsets.long()
    elif kind == "m_tail":
        a = a[:-1]
    elif kind == "n_tail":
        b = torch.empty((128, 127), dtype=b.dtype)
    elif kind == "k_tail":
        a, b = a[:, :-1].contiguous(), b[:-1].contiguous()
    else:
        b = b.T
    bound = _bind(grouped_gemm_jagged.fn, (a, b, offsets))
    assert not any(g.key == KEY for g in bound.config_spec.compiler_coverage_groups)
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        bound.to_code(_selected(case[0]))


@pytest.mark.parametrize("operand", [0, 1])
@skipUnlessBackends(["cute"])
def test_original_tma_base_alignment_gate_is_retained(
    case: tuple, operand: int
) -> None:
    args = list(case[1])
    tensor = args[operand]
    args[operand] = torch.empty((tensor.numel() + 1,), dtype=tensor.dtype)[1:].view(
        tensor.shape
    )
    assert args[operand].is_contiguous() and args[operand].storage_offset() == 1
    bound = _bind(grouped_gemm_jagged.fn, tuple(args))
    # Symbolic storage offsets are not pointer-alignment facts. The existing
    # cache-key-backed TMA alignment gate must reject this actual unaligned base.
    with pytest.raises(BackendUnsupported, match="dense row-union requires"):
        bound.to_code(_selected(case[0]))


@pytest.mark.parametrize(
    "change", ["alias", "group_epilogue", "extra_store", "group_rhs"]
)
@skipUnlessBackends(["cute"])
def test_semantic_proof_is_still_required(case: tuple, change: str) -> None:
    _bound0, args, _old, _new = case
    if change == "group_rhs":
        a, b, offsets = args
        fn = _device_offsets_kernel.fn
        args = (a, torch.empty((4, *b.shape), dtype=b.dtype), offsets)
    else:
        original = inspect.getsource(_explicit_offset_mm)
        if change == "alias":
            source = original.replace(
                "torch.zeros((m, n), dtype=a.dtype, device=a.device)", "a"
            )
        elif change == "group_epilogue":
            source = original.replace(
                "acc.to(out.dtype)", "(acc + group).to(out.dtype)"
            )
        else:
            source = original.replace(
                "    return out", "        a[group, group] = 0\n    return out"
            )
        assert source != original
        module = PyCodeCache.load(
            "import torch\nimport helion.language as hl\n" + source
        )
        fn = module._explicit_offset_mm
    bound = _bind(fn, args)
    with pytest.raises((InvalidConfig, BackendUnsupported)):
        bound.to_code(_selected(case[0]))


@pytest.mark.parametrize("stride", [1, 2])
@skipUnlessBackends(["cute"])
def test_generated_int32_intervals_match_original_initializer(
    case: tuple, stride: int
) -> None:
    old, new = case[2:]
    old_code = _metadata_code(old)
    tree = ast.parse(new)
    statements = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("_helion_"):
            statements = [
                child
                for child in node.body
                if (
                    isinstance(child, ast.Assign)
                    and ast.unparse(child.targets[0])
                    in {"row_union_starts", "row_union_ends"}
                )
                or (
                    isinstance(child, ast.If)
                    and any(
                        isinstance(inner, ast.For)
                        and ast.unparse(inner.target) == "row_union_group"
                        for inner in child.body
                    )
                )
            ]
    assert len(statements) == 3
    code = compile(ast.Module(body=statements, type_ignores=[]), "<row-union>", "exec")
    for offsets in (
        [0, 1, 193, 623, 640],
        [320, 640, 0, 320, 320],
        [-17, 193, 657, 657, 657],
        [320] * 5,
        [-(1 << 31), (1 << 31) - 1, 0, 640, 640],
        [(1 << 31) - 1, -(1 << 31), 0, 640, 640],
    ):
        data = torch.full((len(offsets) * stride,), 17, dtype=torch.int32)
        data[::stride] = torch.tensor(offsets, dtype=torch.int32)

        def convert(value: object, dtype: torch.dtype) -> torch.Tensor:
            return (
                value.to(dtype)
                if isinstance(value, torch.Tensor)
                else torch.tensor(value, dtype=dtype)
            )

        namespace = {
            "cutlass": SimpleNamespace(
                Int32=lambda value: convert(value, torch.int32),
                Int64=lambda value: convert(value, torch.int64),
                range=lambda value, **kw: range(value),
            ),
            "cute": SimpleNamespace(
                make_layout=lambda value: value,
                make_rmem_tensor=lambda shape, dtype: torch.empty(
                    shape, dtype=torch.int32
                ),
            ),
            "group_offsets": SimpleNamespace(
                iterator=_OffsetPointer(data), layout=SimpleNamespace(stride=(stride,))
            ),
            "tcgen05_epi_active": True,
        }
        exec(code, namespace)
        starts, sizes = _metadata(old_code, offsets, stride)
        assert torch.equal(namespace["row_union_starts"], starts)
        assert torch.equal(namespace["row_union_ends"], starts + sizes[:, 1])


@pytest.mark.parametrize("strategy", [None, "from_random"])
@skipUnlessBackends(["cute"])
def test_full_search_delivers_new_witness_without_changing_old_seeds(
    strategy: str | None,
) -> None:
    bound, args = _bound(0, autotune_initial_population_strategy=strategy)
    old_seeds = deepcopy(bound.config_spec.compiler_seed_configs)
    assert all(KEY not in seed for seed in old_seeds)
    search = _search(bound, args)
    search._autotune_metrics = AutotuneMetrics()
    delivered = []

    class HeldBenchmark(Exception):
        pass

    def hold(members: Sequence[PopulationMember], *, desc: str) -> None:
        assert desc == "Initial population"
        delivered.extend(member.config for member in members)
        raise HeldBenchmark

    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
        patch.object(search, "benchmark_population", side_effect=hold),
        pytest.raises(HeldBenchmark),
    ):
        random.seed(2026091919)
        search._autotune()
    accepted = [
        entry.effective
        for entry in search.compiler_coverage_outcomes
        if entry.requested.get(KEY) is True
        and entry.outcome in ("added", "already_present")
    ]
    assert len(accepted) == 2 and all(c in delivered for c in accepted), [
        entry
        for entry in search.compiler_coverage_outcomes
        if entry.requested.get(KEY) is True
    ]
    assert accepted[0] in search._pinned_finalist_configs
    assert bound.config_spec.compiler_seed_configs == old_seeds
    assert accepted[0] is not None
    assert "row_union_starts" in bound.to_code(accepted[0])


@skipUnlessBackends(["cute"])
def test_explicit_false_prevents_automatic_new_mode() -> None:
    bound, args = _bound(0, autotune_config_overrides={KEY: False})
    search = _search(bound, args)
    with (
        bound.env,
        patch.object(search, "_find_similar_cached_configs", return_value=[]),
    ):
        rows = search._generate_initial_population_flat()
    assert all(
        search.config_gen.unflatten(row).get(KEY, False) is False for row in rows
    )


def test_dense_index_envelope() -> None:
    assert index_domain(16, 10240, 1024, 1024)
    assert not index_domain(1 << 30, 640, 128, 128)
    assert not index_domain(4, 640, 128, 129)
    assert not index_domain(True, 640, 128, 128)


@skipUnlessBackends(["cute"])
def test_absent_grouped_flat_overrides_are_not_explicit_none(case: tuple) -> None:
    bound, args = case[:2]
    search = _search(bound, args)
    config = _selected(bound)
    with bound.env:
        _flat, effective = search.config_gen.strict_config_pair(config)
        for key in (
            "tcgen05_grouped_mode",
            "tcgen05_grouped_worklist_source_m_tile",
        ):
            assert key not in effective
            invalid = deepcopy(config)
            invalid.config[key] = None
            with pytest.raises(InvalidConfig):
                bound.config_spec.normalized_config(invalid)


@pytest.mark.parametrize("shape", [(4, 640, 128), (8, 2560, 512), (16, 10240, 1024)])
def test_pid_mapping_retains_mn_for_every_original_axis_order(shape: tuple) -> None:
    groups, m, n = shape
    extents = {17: groups, 23: m // 128, 31: n // 64}
    plan = SimpleNamespace(
        row_union=GroupedRowUnionPlan(groups, m, n, 128, 17, 23, 31, "o", "u")
    )
    # The metadata holds only the already-proved single-CTA geometry. Both
    # the new linearization and original inverse decomposition are real code.
    for order in itertools.permutations(extents):
        strategy = SimpleNamespace(
            pid_info=[
                SimpleNamespace(
                    block_id=block,
                    pid_var=f"pid_{block}",
                    num_pids_expr=lambda *, is_device, value=extents[block]: str(value),
                )
                for block in order
            ],
            _tcgen05_plan=lambda: plan,
            _tcgen05_logical_m_coord_expr=lambda value: value,
        )
        expression = (
            Tcgen05PersistentProgramIDs._tcgen05_linear_virtual_pid_from_coords_expr(
                strategy, ["row", "column", "0"]
            )
        )
        state = SimpleNamespace(
            device_function=SimpleNamespace(new_var=lambda name: name)
        )
        statements = ProgramIDs._decompose_pid_to_statements(strategy, "virtual", state)
        code = compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[*ast.parse("virtual = " + expression).body, *statements],
                    type_ignores=[],
                )
            ),
            "<dense-row-union-pid>",
            "exec",
        )
        for row, column in itertools.product(range(extents[23]), range(extents[31])):
            namespace = {
                "row": row,
                "column": column,
                "cutlass": SimpleNamespace(Int32=int),
            }
            exec(code, namespace)
            assert (namespace["pid_17"], namespace["pid_23"], namespace["pid_31"]) == (
                0,
                row,
                column,
            )


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("transposed", [False, True])
def test_packet_predicate_stores_only_selected_in_bounds_elements(
    width: int, transposed: bool
) -> None:
    """Execute the rendered predicate with explicit packet coordinates.

    This checks store/value/mask semantics and predicate broadcast shape; the
    existing CuTe partitioner, not this model, supplies hardware coordinates.
    """
    coords = (
        [(2, column) for column in range(width)]
        + [(4, column) for column in range(width)]
        + [(3 + (column >= max(1, width // 2)), column) for column in range(width)]
        + [(6, 63 + column) for column in range(width)]
        + [(-1, column) for column in range(width)]
        + [(128, column) for column in range(width)]
    )
    values = list(range(17, 17 + len(coords)))
    output = [-999] * len(coords)

    class Matrix:
        def __init__(self, data: list, rows: int, columns: int) -> None:
            self.data = data
            self.rows = rows
            self.shape = (rows, (columns,))
            self.layout = SimpleNamespace(stride=(1, (rows,)))
            self.iterator = self

        def __getitem__(self, index: tuple[int, int]) -> object:
            row, column = index
            return self.data[column * self.rows + row]

        def __setitem__(self, index: tuple[int, int], value: object) -> None:
            row, column = index
            self.data[column * self.rows + row] = value

    class Predicate:
        def __init__(self, base: Matrix, layout: SimpleNamespace) -> None:
            assert layout.stride == (0, base.layout.stride[1])
            self.base = base
            self.shape = layout.shape

        def __getitem__(self, index: tuple[int, int]) -> object:
            return self.base[0, index[1]]

    def make_layout(shape: object, stride: object = None) -> SimpleNamespace:
        return SimpleNamespace(shape=shape, stride=stride)

    def logical_divide(data: list, layout: SimpleNamespace) -> Matrix:
        assert layout.shape == width
        return Matrix(data, width, len(data) // width)

    def make_rmem(shape: tuple, dtype: object) -> Matrix:
        rows, (columns,) = shape
        return Matrix([False] * (rows * columns), rows, columns)

    def copy(
        atom: object, source: Matrix, destination: Matrix, *, pred: Predicate
    ) -> None:
        assert source.shape == destination.shape == pred.shape
        rows, (columns,) = source.shape
        for column in range(columns):
            for row in range(rows):
                if pred[row, column]:
                    destination[row, column] = source[row, column]

    plan = GroupedRowUnionPlan(
        2,
        128,
        64,
        128,
        0,
        1,
        2,
        "offsets",
        "u",
        schedule=TRANSPOSED if transposed else None,
    )
    namespace = {
        "cutlass": SimpleNamespace(
            BFloat16=SimpleNamespace(width=16),
            Boolean=bool,
            const_expr=lambda value: value,
            range=lambda value, **kw: range(value),
        ),
        "cute": SimpleNamespace(
            logical_divide=logical_divide,
            make_layout=make_layout,
            make_rmem_tensor=make_rmem,
            make_tensor=Predicate,
            size=operator.itemgetter(0),
            elem_less=lambda left, right: all(
                a < b for a, b in zip(left, right, strict=True)
            ),
            copy=copy,
        ),
        "u_starts": [2, 6],
        "u_ends": [4, 8],
        "source": values,
        "destination": output,
        "coordinates": [(n, m) for m, n in coords] if transposed else coords,
        "bits": width * 16,
        "atom": None,
    }
    exec(
        plan.masked_copy(
            source="source",
            destination="destination",
            coordinates="coordinates",
            bits="bits",
            atom="atom",
        ),
        namespace,
    )
    expected = [
        value if (2 <= row < 4 or 6 <= row < 8) and 0 <= column < 64 else -999
        for value, (row, column) in zip(values, coords, strict=True)
    ]
    assert output == expected
