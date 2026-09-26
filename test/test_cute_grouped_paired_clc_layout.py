from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from itertools import pairwise
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Any

import pytest
import torch

from helion._compiler.cute.grouped_row_union import PAIRED_CLC
from helion._compiler.cute.grouped_row_union import GroupedRowUnionPlan
from helion._testing import skipUnlessBackends

if TYPE_CHECKING:
    from collections.abc import Iterator


def _plan(groups: int = 16) -> GroupedRowUnionPlan:
    return GroupedRowUnionPlan(
        groups, 10240, 1024, 1024, 0, 1, 2, "offsets", "union", schedule=PAIRED_CLC
    )


@contextmanager
def _sdk() -> Iterator[SimpleNamespace]:
    import cutlass
    from cutlass._mlir import ir
    from cutlass._mlir import passmanager
    from cutlass._mlir.dialects import func
    import cutlass.cute as cute
    from cutlass.utils.gemm import sm100

    initialized = torch.cuda.is_initialized()
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            yield SimpleNamespace(
                cutlass=cutlass,
                cute=cute,
                ir=ir,
                func=func,
                passmanager=passmanager,
                epi=sm100,
                module=module,
            )
    assert torch.cuda.is_initialized() == initialized


def _geometry(sdk: SimpleNamespace) -> SimpleNamespace:
    c, cute = sdk.cutlass, sdk.cute
    profile = PAIRED_CLC
    mma = c.utils.blackwell_helpers.make_trivial_tiled_mma(
        c.BFloat16,
        c.BFloat16,
        cute.nvgpu.OperandMajorMode.MN,
        cute.nvgpu.OperandMajorMode.K,
        c.Float32,
        cute.nvgpu.tcgen05.CtaGroup.TWO,
        (profile.mma_m, profile.mma_n),
        cute.nvgpu.tcgen05.OperandSource.SMEM,
    )
    desc = SimpleNamespace(
        cta_tile_shape_mnk=(profile.mma_m, profile.mma_n, profile.block_k),
        c_layout=c.utils.layout.LayoutEnum.COL_MAJOR,
        c_dtype=c.BFloat16,
        acc_dtype=c.Float32,
        epilog_sync_bar_id=2,
        epilogue_warp_id=(0, 1, 2, 3),
        num_c_stage=2,
        use_2cta_instrs=True,
    )
    tile = (cute.make_layout(profile.epi_m), cute.make_layout(profile.epi_n))
    frag = mma.make_fragment_C(
        cute.append(mma.partition_shape_C((profile.mma_m, profile.mma_n)), 2)
    )
    tacc = cute.make_tensor(
        cute.make_ptr(c.Float32, 0, cute.AddressSpace.tmem, assumed_align=16),
        frag.layout,
    )
    tacc = sdk.epi.transform_partitioned_tensor_layout(tacc)
    rest = cute.make_layout(1, stride=0)
    global_c = cute.make_tensor(
        cute.make_ptr(c.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((1024, 10240, 1), stride=(1, 1024, 0)),
    )
    shared_layout = c.utils.blackwell_helpers.make_smem_layout_epi(
        c.BFloat16, desc.c_layout, tile, 2
    )
    shared = cute.make_tensor(
        cute.recast_ptr(
            cute.make_ptr(c.BFloat16, 0, cute.AddressSpace.smem, assumed_align=1024),
            shared_layout.inner,
            dtype=c.BFloat16,
        ),
        shared_layout.outer,
    )
    return SimpleNamespace(
        mma=mma,
        desc=desc,
        tile=tile,
        tacc=tacc,
        rest=rest,
        global_c=global_c,
        shared=shared,
        shared_layout=shared_layout,
    )


def _partition(
    sdk: SimpleNamespace, g: SimpleNamespace, tensor: Any, rank: int
) -> tuple[Any, Any]:
    cute = sdk.cute
    partition = sdk.epi.transform_partitioned_tensor_layout(
        g.mma.get_slice(rank).partition_C(tensor)
    )
    planned = cute.make_tensor(
        partition.iterator,
        cute.append(cute.append(cute.append(partition.layout, g.rest), g.rest), g.rest),
    )
    return partition, planned


@skipUnlessBackends(["cute"])
def test_paired_ab_stage_payload() -> None:
    with _sdk() as sdk:
        c, cute = sdk.cutlass, sdk.cute
        g = _geometry(sdk)
        layouts = [
            make(g.mma, g.desc.cta_tile_shape_mnk, c.BFloat16, PAIRED_CLC.ab_stages)
            for make in (
                c.utils.blackwell_helpers.make_smem_layout_a,
                c.utils.blackwell_helpers.make_smem_layout_b,
            )
        ]
        stages = [cute.slice_(layout, (None, None, None, 0)) for layout in layouts]
        sizes = [cute.cosize(stage.outer) * 2 for stage in stages]
        assert sizes == [16384, 12288]
        assert sum(sizes) == 28672
        assert sum(sizes) * PAIRED_CLC.cluster_m == 57344
        assert [cute.cosize(layout.outer) * 2 for layout in layouts] == [
            size * 7 for size in sizes
        ]
        selected = cute.select(layouts[1], mode=[0, 1, 2])
        assert str(selected) == str(stages[1])
        explicit_b = c.utils.blackwell_helpers.make_smem_layout_b(
            g.mma, g.desc.cta_tile_shape_mnk, c.BFloat16, 7, is_k_major=True
        )
        assert str(explicit_b) == str(layouts[1])


@skipUnlessBackends(["cute"])
def test_all_fallback_coordinates_and_membership_projection() -> None:
    emitted = ast.parse(
        _plan().masked_copy(
            source="source",
            destination="destination",
            coordinates="coordinate",
            bits="bits",
            atom="atom",
        )
    )
    row_loop = next(node for node in emitted.body if isinstance(node, ast.For))
    representative_code = compile(
        ast.Module(body=row_loop.body[:2], type_ignores=[]), "<row-projection>", "exec"
    )
    with _sdk() as sdk:
        cute = sdk.cute
        g = _geometry(sdk)
        cells: Counter[tuple[int, int]] = Counter()
        membership_layout = cute.make_layout((2, 2, 4, 2), stride=(1, 0, 2, 0))
        mapping = [int(membership_layout(i)) for i in range(32)]
        assert set(mapping) == set(range(8))
        for rank in range(2):
            identity, planned = _partition(
                sdk, g, cute.make_identity_tensor((256, 192)), rank
            )
            copy, _, _ = sdk.epi.epilogue_tmem_copy_and_partition(
                g.desc, 0, g.tacc, planned, g.tile, True
            )
            divided = cute.flat_divide(identity, g.tile)
            for tid in range(128):
                part = copy.get_slice(tid).partition_D(divided)
                for subtile in range(6):
                    fragment = part[None, None, None, 0, subtile]
                    coords = cute.logical_divide(fragment, cute.make_layout(1))
                    assert coords.shape == (1, (2, 2, 4, 2))
                    assert cute.size(coords.shape[1]) == 32
                    values = [tuple(coords[0, i]) for i in range(32)]
                    rows = [row for _, row in values]
                    representatives = []
                    for i in range(8):
                        scope = {
                            "union_row_i": i,
                            "union_store_coord": {
                                (0, j): v for j, v in enumerate(values)
                            },
                        }
                        exec(representative_code, scope)
                        assert scope["union_representative"] == i % 2 + 4 * (i // 2)
                        representatives.append(scope["union_representative_coord"])
                    assert len(set(representatives)) == 8
                    assert set(Counter(rows).values()) == {4}
                    assert rows == [representatives[mapping[i]] for i in range(32)]
                    assert all(32 * subtile <= row < 32 * (subtile + 1) for row in rows)
                    assert all(
                        rank * 128 <= col < (rank + 1) * 128 for col, _ in values
                    )
                    cells.update(values)
        assert len(cells) == 256 * 192 and set(cells.values()) == {1}


@skipUnlessBackends(["cute"])
def test_emitted_scalar_membership_view_roundtrip() -> None:
    with _sdk() as sdk:
        c, cute, ir = sdk.cutlass, sdk.cute, sdk.ir
        function = sdk.func.FuncOp(
            "membership", ([ir.IntegerType.get_signless(1)] * 8, [])
        )
        block = function.add_entry_block()
        with ir.InsertionPoint(block):
            g = _geometry(sdk)
            identity, planned = _partition(
                sdk, g, cute.make_identity_tensor((256, 192)), 0
            )
            copy, _, registers = sdk.epi.epilogue_tmem_copy_and_partition(
                g.desc, 0, g.tacc, planned, g.tile, True
            )
            coordinate = copy.get_slice(0).partition_D(
                cute.flat_divide(identity, g.tile)
            )[None, None, None, 0, 0]
            source = cute.make_rmem_tensor(registers.shape, c.BFloat16)
            destination = cute.make_tensor(
                cute.make_ptr(c.BFloat16, 0, cute.AddressSpace.gmem, assumed_align=16),
                source.layout,
            )
            body = ast.parse(
                _plan().masked_copy(
                    source="source",
                    destination="destination",
                    coordinates="coordinate",
                    bits="bits",
                    atom="atom",
                )
            ).body
            row_loop = next(
                i for i, node in enumerate(body) if isinstance(node, ast.For)
            )
            scope = {
                "cutlass": c,
                "cute": cute,
                "source": source,
                "destination": destination,
                "coordinate": coordinate,
                "bits": c.BFloat16.width,
            }
            exec(
                compile(
                    ast.Module(body=body[:row_loop], type_ignores=[]),
                    "<membership>",
                    "exec",
                ),
                scope,
            )
            assert scope["union_vector"] == 1
            storage, view = scope["union_membership"], scope["union_membership_copy"]
            assert storage.element_type == c.Boolean and view.element_type == c.Int8
            assert cute.cosize(storage.layout) == 8
            for i, value in enumerate(block.arguments):
                storage[i] = c.Boolean(value)
            copy_loop = body[row_loop + 1]
            assert isinstance(copy_loop, ast.For)
            reload = next(
                node
                for node in copy_loop.body
                if isinstance(node, ast.Assign)
                and ast.unparse(node.targets[0]) == "union_first_selected"
            )
            values = []
            for i in range(32):
                scope["union_copy_i"] = i
                exec(
                    compile(
                        ast.Module(body=[reload], type_ignores=[]),
                        "<membership-load>",
                        "exec",
                    ),
                    scope,
                )
                values.append(scope["union_first_selected"].ir_value())
            function.attributes["function_type"] = ir.TypeAttr.get(
                ir.FunctionType.get(
                    [v.type for v in block.arguments], [v.type for v in values]
                )
            )
            sdk.func.ReturnOp(values)
        assert sdk.module.operation.verify()
        sdk.passmanager.PassManager.parse(
            "builtin.module(canonicalize,cute-desugar,cute-expand-ops,cute-fold-static,"
            "canonicalize,convert-cute-to-core,canonicalize)"
        ).run(sdk.module.operation)
        assert sdk.module.operation.verify()
        returned = list(block.operations[-1].operands)
        assert returned == [
            block.arguments[i % 2 + 2 * ((i // 4) % 4)] for i in range(32)
        ]


@skipUnlessBackends(["cute"])
def test_two_tma_issuers_and_original_tail_packets() -> None:
    with _sdk() as sdk:
        cute = sdk.cute
        g = _geometry(sdk)
        atom, tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
            cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
            g.global_c,
            cute.slice_(g.shared_layout, (None, None, 0)),
            g.tile,
        )
        for rank in range(2):
            tile = cute.local_tile(tensor[None, None, 0], (256, 192), (0, 53))
            _, planned = _partition(sdk, g, tile, rank)
            shared, target = cute.nvgpu.cpasync.tma_partition(
                atom,
                0,
                cute.make_layout(1),
                cute.group_modes(g.shared, 0, 2),
                cute.group_modes(cute.flat_divide(planned, g.tile), 0, 2),
            )
            target = target[None, None, None, 0, 0, 0]
            target = cute.group_modes(target, 1, cute.rank(target))
            assert cute.size(target.shape[1]) == 6
            commits: list[list[tuple[int, int, bool]]] = [[], []]
            for subtile in range(6):
                packets = []
                for warp in (0, 2):
                    issuer = warp // 2
                    src = shared[(None, issuer), subtile % 2]
                    dst = target[(None, issuer), subtile]
                    assert cute.size(src.shape) == cute.size(dst.shape) == 64 * 32
                    coords = {tuple(dst[i])[:2] for i in range(cute.size(dst.shape))}
                    assert coords == {
                        (col, row)
                        for col in range(
                            rank * 128 + issuer * 64, rank * 128 + (issuer + 1) * 64
                        )
                        for row in range(
                            10176 + subtile * 32, 10176 + (subtile + 1) * 32
                        )
                    }
                    valid = [row < 10240 for _, row in coords]
                    assert all(valid) == any(valid)  # whole32-row packets only
                    commits[issuer].append((subtile, subtile % 2, any(valid)))
                    packets.append(coords)
                assert not packets[0] & packets[1]
            # Each issuer's six-packet history includes four empty commits.
            # The emitter's acquire/commit ordering is tested separately.
            for packets in commits:
                assert [valid for _, _, valid in packets] == [
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ]
                assert [slot for _, slot, _ in packets] == [0, 1, 0, 1, 0, 1]


class _OffsetPointer:
    def __init__(self, data: torch.Tensor, index: int = 0) -> None:
        self.data, self.index = data, index

    def __add__(self, offset: torch.Tensor) -> _OffsetPointer:
        return _OffsetPointer(self.data, self.index + int(offset))

    def load(self) -> torch.Tensor:
        return self.data[self.index]


def _cooperative(offsets: list[int], stride: int) -> tuple[bool, list[int], list[int]]:
    plan = _plan(len(offsets) - 1)
    tree = ast.parse(plan.cooperative_setup(warp="warp", lane="lane"))
    branch = tree.body[-1]
    assert isinstance(branch, ast.If)
    data = torch.full((len(offsets) * stride,), 123, dtype=torch.int32)
    data[::stride] = torch.tensor(offsets, dtype=torch.int32)

    def convert(value: Any, dtype: torch.dtype) -> torch.Tensor:
        return (
            value.to(dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(value, dtype=dtype)
        )

    scopes: list[dict[str, Any]] = []
    collective: dict[str, Any] = {}
    calls: Counter[str] = Counter()

    def shuffle(value: torch.Tensor, lane: torch.Tensor) -> torch.Tensor:
        calls["shuffle"] += 1
        return scopes[int(lane)][collective["source"]]

    def vote(value: torch.Tensor) -> bool:
        calls["vote"] += 1
        return all(bool(scope[collective["source"]]) for scope in scopes)

    namespace = {
        "cutlass": SimpleNamespace(
            Int32=lambda x: convert(x, torch.int32),
            Int64=lambda x: convert(x, torch.int64),
        ),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                alloc_smem=lambda dtype, count: torch.full(
                    (count,), -777, dtype=torch.int32
                ),
                shuffle_sync=shuffle,
                vote_all_sync=vote,
            ),
            make_layout=lambda shape: shape,
            make_tensor=lambda pointer, layout: pointer,
        ),
        "offsets": SimpleNamespace(
            iterator=_OffsetPointer(data), layout=SimpleNamespace(stride=(stride,))
        ),
    }
    exec(
        compile(
            ast.Module(body=tree.body[:-1], type_ignores=[]),
            "<coverage-storage>",
            "exec",
        ),
        namespace,
    )
    for lane in range(32):
        scopes.append(
            dict(
                namespace,
                lane=torch.tensor(lane, dtype=torch.int32),
                warp=torch.tensor(7, dtype=torch.int32),
            )
        )
    # Lockstep statement evaluation supplies the real full-warp shuffle/vote
    # semantics while all integer arithmetic executes on CPU Int32 tensors.
    for statement in branch.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            name = ast.unparse(statement.value.func)
            if name in ("cute.arch.shuffle_sync", "cute.arch.vote_all_sync"):
                collective["source"] = ast.unparse(statement.value.args[0])
        code = compile(
            ast.Module(body=[statement], type_ignores=[]),
            "<cooperative-coverage>",
            "exec",
        )
        for scope in scopes:
            exec(code, scope)
    assert calls == {"shuffle": 64, "vote": 32}
    flag = bool(namespace[plan.coverage_flag][0])
    return flag, namespace[plan.starts].tolist(), namespace[plan.ends].tolist()


def _normalized(offsets: list[int]) -> list[tuple[int, int]]:
    def int32(value: int) -> int:
        return (value + (1 << 31)) % (1 << 32) - (1 << 31)

    rows = []
    for start, end in pairwise(offsets):
        extent = max(int32(end - start), 0)
        low = min(max(start, 0), 10240)
        length = min(max(int32(extent + min(start, 0)), 0), 10240 - low)
        rows.append((low, low + length))
    return rows


@pytest.mark.parametrize("groups", [1, 31])
@pytest.mark.parametrize("stride", [1, 3])
def test_typed_interval_vote_and_invalid_lane_propagation(
    groups: int, stride: int
) -> None:
    full = [10240 * i // groups for i in range(groups + 1)]
    cases = [
        full,
        [-17, *full[1:]],
        [1, *full[1:]],
        [*full[:-1], 10239],
        [-(1 << 31), (1 << 31) - 1, *full[2:]],
        [(1 << 31) - 1, -(1 << 31), *full[2:]],
    ]
    if groups == 31:
        descending = full.copy()
        descending[15:17] = [6000, 10]
        cases.append(descending)
    for offsets in cases:
        expected = _normalized(offsets)
        covered = (
            expected[0][0] == 0
            and expected[-1][1] == 10240
            and all(a[1] == b[0] for a, b in pairwise(expected))
        )
        flag, starts, ends = _cooperative(offsets, stride)
        assert flag is covered
        if flag:
            assert starts == ends == [-777] * groups
        else:
            assert list(zip(starts, ends, strict=True)) == expected
