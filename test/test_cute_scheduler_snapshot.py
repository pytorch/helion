from __future__ import annotations

import ast
from itertools import starmap
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

from ._cute_aux import _config
from ._cute_aux import _cpu_codegen
from ._cute_aux import _rank_two_code
from ._cute_aux import _rank_two_inputs
from .test_cute_grouped_gemm_split_sizes import _device_split_sizes_kernel
from .test_cute_grouped_gemm_split_sizes import _selected_config
from .test_cute_scheduler_mailbox import _plain_matmul
from helion import exc
from helion._compiler import program_id
from helion._compiler.device_function import DeviceFunction
from helion._compiler.program_id import _build_sched_pipeline_consumer_wait_block
from helion._compiler.program_id import _literal_mailbox_access_fields
from helion._compiler.program_id import _sched_mailbox_load_expr
from helion._compiler.program_id import _SchedMailboxSnapshot
from helion._testing import DEVICE
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def test_normal_wait_keeps_exact_body_and_allocates_no_snapshot_names() -> None:
    function = Mock(spec=DeviceFunction)
    function.config = {"tcgen05_sched_consumer_wait_mode": "normal"}
    assert _SchedMailboxSnapshot.create(function, "role", (0, 1, 2)) is None
    function.new_var.assert_not_called()
    with patch.object(DeviceFunction, "current", return_value=function):
        body = _build_sched_pipeline_consumer_wait_block(
            sched_pipeline="pipe",
            sched_consumer_state="state",
            work_tile_smem="mailbox",
            valid_var="valid",
        )
    assert [ast.unparse(stmt) for stmt in body] == [
        "pipe.consumer_wait(state)",
        "cute.arch.fence_view_async_shared()",
        "cute.arch.sync_warp()",
        "valid = mailbox[cutlass.Int32(3)] != cutlass.Int32(0)",
    ]


def test_snapshot_names_use_device_function_allocator() -> None:
    function = Mock(spec=DeviceFunction)
    function.config = {"tcgen05_sched_consumer_wait_mode": "warp_leader"}
    function.new_var.side_effect = lambda name: f"{name}_unique"
    snapshot = _SchedMailboxSnapshot.create(function, "role", (0, 2))
    assert snapshot is not None
    assert snapshot.valid == "role_mailbox_valid_unique"
    assert snapshot.fields == {0: "role_mailbox_0_unique", 2: "role_mailbox_2_unique"}


@pytest.mark.parametrize(
    "valid_field,payload",
    [(3, (0, 1, 2)), (2, (0, 1, 3, 4, 5, 6, 7, 8)), (3, (2,)), (3, ())],
)
@pytest.mark.parametrize("valid", [False, True])
@pytest.mark.parametrize("stage", [None, "state.index"])
def test_snapshot_valid_first_and_empty_stream_payload_safety(
    valid_field: int, payload: tuple[int, ...], valid: bool, stage: str | None
) -> None:
    snapshot = _SchedMailboxSnapshot(
        "flag", {field: f"field_{field}" for field in payload}
    )
    body = snapshot.read_block("mailbox", "valid", valid_field, stage)
    base = 0 if stage is None else 16
    values = {base + valid_field: int(valid)}
    if valid:
        values.update({base + field: field + 100 for field in payload})
    reads: list[int] = []

    class Pointer:
        def __init__(self, offset: int) -> None:
            self.offset = offset

        def __add__(self, offset: int) -> Pointer:
            return Pointer(self.offset + offset)

        def toint(self) -> int:
            return self.offset

    class Mailbox:
        iterator = Pointer(0)

        def __getitem__(self, index: tuple[None, int]) -> SimpleNamespace:
            assert index == (None, 4)
            return SimpleNamespace(iterator=Pointer(16))

    def load(args: tuple[int, int], **kwargs: object) -> int:
        assert args[1] == 0
        assert kwargs["constraints"] == "=&r,r,r,~{memory}"
        assert kwargs["is_pure"] is False
        assert "@leader ld.volatile.shared.u32" in cast("str", kwargs["asm"])
        reads.append(args[0])
        # Missing payload on an invalid/sentinel record deliberately raises.
        return values[args[0]]

    namespace = {
        "mailbox": Mailbox(),
        "state": SimpleNamespace(index=4),
        "cutlass": SimpleNamespace(Int32=int),
        "cute": SimpleNamespace(
            arch=SimpleNamespace(
                lane_idx=lambda: 0, shuffle_sync=lambda value, source: value
            )
        ),
        "_cute_inline_asm_elementwise": load,
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])),
            "<snapshot>",
            "exec",
        ),
        namespace,
    )
    assert namespace["valid"] is valid
    assert reads == [
        base + valid_field,
        *([base + field for field in payload] if valid else []),
    ]
    for field in payload:
        assert namespace[f"field_{field}"] == (field + 100 if valid else 0)


def test_publication_scanner_accepts_exact_snapshot_and_literal_fields() -> None:
    snapshot = _SchedMailboxSnapshot(
        "flag", {field: f"field_{field}" for field in (0, 1, 3, 8)}
    )
    tree = ast.Module(
        body=snapshot.read_block("mailbox", "valid", 2, None), type_ignores=[]
    )
    assert _literal_mailbox_access_fields(tree, "mailbox") == ({0, 1, 2, 3, 8}, set())


@pytest.mark.parametrize(
    "change", ["assembly", "pure", "constraints", "overlap", "dynamic", "escape"]
)
def test_publication_scanner_rejects_unproven_snapshot_loads(change: str) -> None:
    source = ast.unparse(_sched_mailbox_load_expr("mailbox", 2))
    if change == "assembly":
        source = source.replace("ld.volatile.shared", "ld.shared")
    elif change == "pure":
        source = source.replace("is_pure=False", "is_pure=True")
    elif change == "constraints":
        source = source.replace(",~{memory}", "")
    elif change == "overlap":
        source = source.replace("=&r", "=r")
    elif change == "dynamic":
        source = source.replace("cutlass.Int32(2)", "cutlass.Int32(index)")
    else:
        source = "consume(mailbox.iterator)"
    with pytest.raises(AssertionError, match="direct literal field"):
        _literal_mailbox_access_fields(ast.parse(source), "mailbox")


def _assert_snapshots(code: str) -> None:
    assert "@leader ld.volatile.shared.u32" in code
    assert "constraints='=&r,r,r,~{memory}'" in code
    tree = ast.parse(code)
    # The only direct scalar subscripts left are producer stores. All
    # consumer reads are explicit predicated loads into register snapshots.
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "tcgen05_work_tile_smem"
            and isinstance(node.ctx, ast.Load)
        ):
            assert isinstance(node.slice, ast.Tuple)
            assert isinstance(node.slice.elts[0], ast.Constant)
            assert node.slice.elts[0].value is None  # stage view, not scalar load
    assert "cute.arch.shuffle_sync(" in code


@pytest.mark.parametrize("tile", [[64, 256, 64], [128, 128, 64]])
@pytest.mark.parametrize("order", [[0, 1], [1, 0]])
def test_aux_all_consumers_use_snapshots(tile: list[int], order: list[int]) -> None:
    _assert_snapshots(
        _rank_two_code(
            _rank_two_inputs(),
            _config(
                block_sizes=tile,
                loop_orders=[order],
                tcgen05_sched_consumer_wait_mode="warp_leader",
            ),
        )
    )


@pytest.mark.parametrize("stages", [1, 2])
def test_clc_staged_leader_snapshot_codegen(stages: int) -> None:
    args = (
        torch.empty((1024, 256), dtype=torch.bfloat16),
        torch.empty((256, 1024), dtype=torch.bfloat16),
    )
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
        tcgen05_persistence_model="clc_persistent",
        tcgen05_sched_stage_count=stages,
        tcgen05_sched_consumer_wait_mode="warp_leader",
    )
    with _cpu_codegen():
        code = _plain_matmul._bind_isolated(args).to_code(config)
    _assert_snapshots(code)
    if stages == 2:
        assert (
            "tcgen05_work_tile_smem[None, tcgen05_sched_pipeline_consumer_state.index].iterator"
            in code
        )


def test_selected_grouped_snapshots_preserve_publication_proof() -> None:
    args = (
        torch.empty((2048, 128), dtype=torch.bfloat16),
        torch.empty((8, 224, 128), dtype=torch.bfloat16),
        torch.tensor((0, 1, 127, 224, 256, 449, 0, 991), dtype=torch.int32),
    )
    config = _selected_config(128)
    with _cpu_codegen():
        bound = _device_split_sizes_kernel._bind_isolated(args)
        unsupported = _selected_config(128)
        unsupported.config["tcgen05_sched_consumer_wait_mode"] = "warp_leader"
        with pytest.raises(exc.InvalidConfig, match="only supported with"):
            bound.to_code(unsupported)

        # The selected scheduler is internally promoted; its public config
        # does not currently admit leader-wait mode. Exercise ONLY the generic
        # consumer builder's future-proof schema, after normal validation.
        # This is not a runnable/admitted selected-leader configuration.
        original_create = _SchedMailboxSnapshot.create
        original_wait = _build_sched_pipeline_consumer_wait_block

        def create_snapshot(
            function: DeviceFunction, prefix: str, fields: tuple[int, ...]
        ) -> _SchedMailboxSnapshot | None:
            with patch.dict(
                function.config.config,
                {"tcgen05_sched_consumer_wait_mode": "warp_leader"},
            ):
                return original_create(function, prefix, fields)

        def wait_snapshot(
            *,
            sched_pipeline: str,
            sched_consumer_state: str,
            work_tile_smem: str,
            valid_var: str,
            valid_slot_index: int = 3,
            work_tile_stage_index: str | None = None,
            snapshot: _SchedMailboxSnapshot | None = None,
        ) -> list[ast.stmt]:
            with patch.dict(
                DeviceFunction.current().config.config,
                {"tcgen05_sched_consumer_wait_mode": "warp_leader"},
            ):
                return original_wait(
                    sched_pipeline=sched_pipeline,
                    sched_consumer_state=sched_consumer_state,
                    work_tile_smem=work_tile_smem,
                    valid_var=valid_var,
                    valid_slot_index=valid_slot_index,
                    work_tile_stage_index=work_tile_stage_index,
                    snapshot=snapshot,
                )

        with (
            patch.object(_SchedMailboxSnapshot, "create", side_effect=create_snapshot),
            patch.object(
                program_id,
                "_build_sched_pipeline_consumer_wait_block",
                side_effect=wait_snapshot,
            ),
        ):
            code = bound.to_code(config)
    _assert_snapshots(code)
    assert "tcgen05_grouped_selected_sched" in code
    for field in range(9):
        assert f"tcgen05_work_tile_smem[cutlass.Int32({field})] =" in code


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires TCgen05 GPU")
@pytest.mark.parametrize("clc,stages", [(False, 1), (True, 1), (True, 2)])
def test_leader_snapshot_clustered_multiwave_runtime(clc: bool, stages: int) -> None:
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("requires a TCgen05-capable GPU")
    config = _config(
        block_sizes=[256, 256, 128],
        tcgen05_cluster_m=2,
        tcgen05_aux_load_mode="simt",
        tcgen05_warp_spec_c_input_warps=0,
        tcgen05_sched_stage_count=stages,
        tcgen05_sched_consumer_wait_mode="warp_leader",
    )
    if clc:
        config.config["tcgen05_persistence_model"] = "clc_persistent"
    for seed in range(5):
        torch.manual_seed(seed)
        args = (
            torch.randn((8192, 256), dtype=torch.bfloat16, device=DEVICE) * 0.1,
            torch.randn((256, 4096), dtype=torch.bfloat16, device=DEVICE) * 0.1,
        )
        before = tuple(value.clone() for value in args)
        expected = args[0].double() @ args[1].double()
        bound = _plain_matmul._bind_isolated(args)
        bound.set_config(config)
        first = bound(*args)
        assert torch.equal(bound(*args), first)
        torch.testing.assert_close(first.double(), expected, atol=0.002, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = bound(*args)
        graph.replay()
        assert torch.equal(captured, first)
        for _ in range(3):
            captured.fill_(float("nan"))
            graph.replay()
            assert torch.equal(captured, first)
        assert all(starmap(torch.equal, zip(args, before, strict=True)))
