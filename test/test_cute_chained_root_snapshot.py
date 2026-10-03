from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from . import test_cute_chained_snapshot_compile as native
from . import test_cute_chained_tmem_snapshot_mapping as mapping
from .test_cute_chained_root_pair import _source
from .test_cute_chained_tcgen05 import _tcgen_inputs
from helion import exc
from helion._compiler.cute import chained_matmul as chain
from helion._compiler.cute import chained_root_stage as roots
from helion._compiler.cute import chained_tcgen05 as root
from helion._compiler.cute import chained_tcgen_stage as stages
from helion._compiler.cute.chained_execution import ChainedExecution
from helion._compiler.cute.chained_tmem_snapshots import OrderedSnapshotAccess
from helion._compiler.cute.chained_tmem_snapshots import PackedSnapshotPanels
from helion._compiler.cute.chained_tmem_snapshots import emit_streamed_snapshot


def _stream(dtype=torch.bfloat16, mode="scan", columns=64, tile_columns=32):
    original = root.codegen_shared_root_sequence

    def selected(cg, plan):
        return original(cg, plan, snapshot_tile_columns=tile_columns)

    with patch.object(root, "codegen_shared_root_sequence", selected):
        return _source(dtype, mode, columns)


@pytest.mark.parametrize("width", [64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize("source", [0, 16, 32, 64])
def test_ordered_word_access_does_not_overwrite_future_reads(width, source):
    destination = source // 32 * 32
    proof = OrderedSnapshotAccess((128, width), source, destination, 512)
    panels = PackedSnapshotPanels((128, width), source, destination, 512, proof)
    for row in range(128):
        memory: dict[tuple[int, int], object] = {
            (row, source + j): (row, j) for j in range(width)
        }
        for panel in range(panels.count):
            loaded = [memory[row, source + panel * 32 + j] for j in range(32)]
            assert loaded == [(row, panel * 32 + j) for j in range(32)]
            for j in range(16):
                memory[row, destination + panel * 16 + j] = (
                    loaded[2 * j],
                    loaded[2 * j + 1],
                )


def test_overlap_requires_explicit_safe_order_and_matching_record():
    with pytest.raises(ValueError, match="disjoint"):
        PackedSnapshotPanels((128, 128), 0, 0, 256)
    with pytest.raises(ValueError, match="ordered access"):
        PackedSnapshotPanels(
            (128, 128),
            0,
            32,
            256,
            OrderedSnapshotAccess((128, 128), 0, 32, 256),
        )
    with pytest.raises(ValueError, match="ordered access"):
        PackedSnapshotPanels(
            (128, 128),
            0,
            0,
            256,
            OrderedSnapshotAccess((128, 64), 0, 0, 256),
        )


@pytest.mark.parametrize("dtype_name", ["BFloat16", "Float16"])
@pytest.mark.parametrize("width", [64, 128, 256])
def test_actual_ordered_dynamic_native_load_and_store(dtype_name, width):
    def emit(shape, dtype):
        panels = PackedSnapshotPanels(
            shape, 32, 32, 448, OrderedSnapshotAccess(shape, 32, 32, 448)
        )
        return emit_streamed_snapshot(
            panels,
            "snapshot",
            "original",
            dtype,
            ("row", "col"),
            [],
            f"{dtype}(original_values[snapshot_index])",
            execution=ChainedExecution(128),
        )

    with patch.object(mapping, "_emit", emit):
        mapping.test_actual_dynamic_emitter_load_pack_store_cpu(dtype_name, width)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("mode", ["plain", "scan"])
@pytest.mark.parametrize("columns", [32, 64, 128])
def test_default_exact_and_streamed_root_uses_both_shared_issues(dtype, mode, columns):
    before = torch.cuda.is_initialized()
    with patch.object(roots, "supports_root_pair", return_value=False):
        legacy = _source(dtype, mode, columns)
    assert _stream(dtype, mode, columns, 0) == legacy
    with (
        patch.object(
            root, "codegen_chained_tcgen05", side_effect=AssertionError("legacy")
        ),
        patch.object(stages, "emit_stage", wraps=stages.emit_stage) as issued,
    ):
        actual = _stream(dtype, mode, columns)
    assert [call.args[3] for call in issued.call_args_list] == [0, 1]
    assert "chain_0_copy =" not in actual
    assert "for chain_1_bridge_panel in cutlass.range(4, unroll=1)" in actual
    assert "chain_1_copy =" in actual
    assert actual.index("cp_async_wait_group(0)") < actual.index(
        "chain_1_bridge_segment0"
    )
    assert actual.index("fence_view_async_tmem_store()") < actual.index("chain_1_acc =")
    sequence = issued.call_args_list[0].kwargs["root_actions"].sequence
    assert sequence.next_stage == 2
    assert sequence.snapshot_ready and sequence.snapshot_consumed
    assert torch.cuda.is_initialized() == before


@pytest.mark.parametrize("value", [True, False, None, 1, 64, 32.0, "32"])
def test_strict_private_selection(value):
    with pytest.raises(exc.BackendUnsupported, match="tile columns"):
        _stream(tile_columns=value)


@pytest.mark.parametrize(
    "case",
    [
        "source",
        "stride",
        "scan",
        "boundary",
        "ready",
        "consumed",
        "record",
        "neighbor",
        "extra_user",
        "domain",
        "participants",
        "panels",
        "order",
    ],
)
def test_stale_or_wrong_expression_fails_before_kernel_install(case):
    original = stages.emit_stage
    checked = []

    def stage(*args, **kwargs):
        if args[3] != 1:
            return original(*args, **kwargs)
        cg, plan, boundaries = args[:3]
        action = kwargs["root_actions"]
        sequence = action.sequence
        snapshot = sequence.snapshot
        assert snapshot is not None
        before = list(cg.device_function.body)
        if case == "source":
            plan.dots[0].kwargs = {**plan.dots[0].kwargs, "unexpected": [1]}
        elif case == "stride":
            old = plan.dots[0].meta["val"]
            plan.dots[0].meta["val"] = old.T
        elif case == "scan":
            sequence.scans = (
                replace(sequence.scans[0], shared="unpublished"),
                *sequence.scans[1:],
            )
        elif case == "boundary":
            boundaries[plan.dots[1].args[0]] = "unpublished"
        elif case == "ready":
            sequence.snapshot_ready = False
        elif case == "consumed":
            sequence.snapshot_consumed = True
        elif case == "record":
            sequence.snapshot = replace(snapshot, boundaries=())
            sequence.snapshot = replace(sequence.snapshot, options=None)
        elif case == "extra_user":
            with plan.dots[0].graph.inserting_after(plan.dots[0]):
                plan.dots[0].graph.call_function(torch.neg, (plan.dots[0],))
        elif case == "domain":
            plan.dots[0].meta["val"] = plan.dots[0].meta["val"][:, :-1]
        elif case == "participants":
            kwargs["execution"] = ChainedExecution(256)
        elif case == "panels":
            object.__setattr__(snapshot.panels, "destination_offset", 32)
        elif case == "order":
            object.__setattr__(snapshot.panels.ordered, "destination_offset", 32)
        elif case == "neighbor":
            value = chain._Expression.value

            def shifted(expression, node, coordinates):
                if (
                    node is snapshot.bridge.source
                    and snapshot.bridge.source in expression.fragments
                ):
                    coordinates = (coordinates[0], coordinates[1] + " + 1")
                return value(expression, node, coordinates)

            with (
                patch.object(chain._Expression, "value", shifted),
                pytest.raises(chain._UnsupportedChain),
            ):
                original(*args, **kwargs)
            assert list(cg.device_function.body) == before
            checked.append(case)
            raise chain._UnsupportedChain("tested rejection")
        with pytest.raises(chain._UnsupportedChain):
            original(*args, **kwargs)
        assert list(cg.device_function.body) == before
        checked.append(case)
        raise chain._UnsupportedChain("tested rejection")

    with (
        patch.object(stages, "emit_stage", stage),
        pytest.raises(exc.BackendUnsupported, match="tested rejection"),
    ):
        _stream()
    assert checked == [case]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_streamed_root_original_host_native_compile(dtype, tmp_path, monkeypatch):
    before = torch.cuda.is_initialized()
    source = _stream(dtype)
    original_module = native._module

    def module(name, text, **bindings):
        result = original_module(name, text, **bindings)
        # The reusable compile harness calls this attribute; the original host
        # callable, its body, arguments and captured launcher stay untouched.
        result.__dict__["_sequence"] = result.__dict__["_tcgen_chain"]
        return result

    monkeypatch.chdir(tmp_path)
    with (
        patch.object(native, "_module", module),
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CUDA forbidden")),
    ):
        ptx, launch = native._compile_original_host(
            source, (*_tcgen_inputs("cpu", dtype, n=64), "scan"), tmp_path
        )
    assert "tcgen05.ld" in ptx and "tcgen05.st" in ptx
    assert launch["block"] == (128, 1, 1)
    assert torch.cuda.is_initialized() == before
