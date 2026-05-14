from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

import helion
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import RolledReductionInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._testing import DEVICE
from helion._testing import RefEagerTestBase
from helion._testing import TestCase
from helion._testing import code_and_output
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipIfTileIR
import helion.exc as exc
import helion.language as hl


@helion.kernel()
def barrier_dep_single(x: torch.Tensor) -> torch.Tensor:
    tmp = torch.empty_like(x)
    out = torch.empty_like(x)

    for t in hl.tile(x.size(0)):
        tmp[t] = x[t] * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = tmp[t] + 1

    return out


@helion.kernel()
def barrier_multiple(x: torch.Tensor) -> torch.Tensor:
    buf1 = torch.empty_like(x)
    buf2 = torch.empty_like(x)
    out = torch.empty_like(x)

    for t in hl.tile(x.size(0)):
        buf1[t] = x[t] + 3

    hl.barrier()

    for t in hl.tile(x.size(0)):
        buf2[t] = buf1[t] * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = buf2[t] - 5

    return out


@helion.kernel()
def barrier_groups(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    buf = torch.empty_like(x)
    buf2 = torch.empty_like(x)
    out = torch.empty_like(x)

    # group 1: independent loops
    for t in hl.tile(x.size(0)):
        buf[t] = x[t] + 1
    for t in hl.tile(x.size(0)):
        buf2[t] = y[t] + 5

    hl.barrier()

    # group 2: consumes both buffers
    for t in hl.tile(x.size(0)):
        out[t] = (buf[t] + buf2[t]) * 2

    hl.barrier()

    for t in hl.tile(x.size(0)):
        out[t] = out[t] + 7

    return out


@onlyBackends(["triton"])
class TestBarrier(RefEagerTestBase, TestCase):
    @skipIfTileIR("TileIR does not support barrier operations")
    def test_dep_across_barrier(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_dep_single,
            (x,),
            block_sizes=[8, 8],
            pid_type="persistent_blocked",
        )
        expected = x * 2 + 1
        torch.testing.assert_close(out, expected)

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_multiple_barriers(self) -> None:
        x = torch.arange(6, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_multiple,
            (x,),
            block_sizes=[8, 8, 8],
            pid_type="persistent_blocked",
        )
        expected = (x + 3) * 2 - 5
        torch.testing.assert_close(out, expected)

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_multiple_loops_between_barriers(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        y = torch.arange(8, device=DEVICE, dtype=torch.float32) * 3
        code, out = code_and_output(
            barrier_groups,
            (x, y),
            block_sizes=[8, 8, 8, 8],
            pid_type="persistent_blocked",
        )
        expected = ((x + 1) + (y + 5)) * 2 + 7
        torch.testing.assert_close(out, expected)

    @skipIfRefEager("pid_type validation is only enforced in compiled mode")
    def test_non_persistent_pid_type_errors(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(exc.BarrierRequiresPersistent, "requires pid_type"):
            code_and_output(
                barrier_dep_single,
                (x,),
                block_sizes=[4, 4],
                pid_type="flat",
            )

    @skipIfTileIR("TileIR does not support barrier operations")
    def test_default_config_is_persistent(self) -> None:
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_dep_single,
            (x,),
            block_sizes=[4, 4],
            pid_type="persistent_blocked",
        )
        expected = x * 2 + 1
        torch.testing.assert_close(out, expected)
        # Can't see pid_type in ref-mode code; rely on normalization to succeed.

    @skipIfRefEager(
        "DeviceIR mutation test does not execute a kernel in ref eager mode"
    )
    def test_apply_rolling_preserves_root_phase_metadata(self) -> None:
        class _FakeReductionLoopSpecs(list[object]):
            def config_get(
                self, values: list[int | None], block_id: int, default: object
            ) -> object:
                for idx, spec in enumerate(self):
                    spec_obj = cast("SimpleNamespace", spec)
                    if spec_obj.block_id == block_id:
                        return values[idx]
                return default

        class _FakeReductionRoller:
            def __init__(self, *args: object, **kwargs: object) -> None:
                pass

            def process(self, graph: torch.fx.Graph) -> torch.fx.Graph:
                return graph

        device_ir = DeviceIR()
        graph0 = torch.fx.Graph()
        graph0.output(())
        graph1 = torch.fx.Graph()
        graph1.output(())
        device_ir.add_root_graph(graph0)
        device_ir.add_root_graph(graph1)

        root0 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[0]])
        root1 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[1]])
        root0.phase_index = 0
        root1.phase_index = 1

        block_id = 7
        device_ir.rolled_reductions = [
            RolledReductionInfo(
                rolled_block_ids=[block_id],
                original_graph_id=1,
                used_rdim=True,
                can_be_rolled_by_caller=True,
            )
        ]

        reduction_specs = _FakeReductionLoopSpecs([SimpleNamespace(block_id=block_id)])
        fake_env = SimpleNamespace(
            block_sizes=[SimpleNamespace(block_id=block_id, reduction=True)],
            config_spec=SimpleNamespace(reduction_loops=reduction_specs),
        )

        with (
            patch(
                "helion._compiler.device_ir.CompileEnvironment.current",
                return_value=fake_env,
            ),
            patch("helion._compiler.device_ir.ReductionRoller", _FakeReductionRoller),
        ):
            device_ir._apply_rolling(helion.Config(reduction_loops=[16]))

        rolled_root1 = cast("RootGraphInfo", device_ir.graphs[device_ir.root_ids[1]])
        self.assertEqual(rolled_root1.phase_index, 1)

    @skipIfRefEager(
        "DeviceIR mutation test does not execute a kernel in ref eager mode"
    )
    def test_register_rollable_reductions_does_not_mutate_graphs(self) -> None:
        class _FakeRDim:
            block_id = 7
            reduction = True

            def size_hint(self) -> int:
                return 16

        class _FakeReductionRoller:
            def __init__(
                self,
                device_ir: DeviceIR,
                rdim: _FakeRDim,
                graph_id_to_info: dict[int, RolledReductionInfo],
            ) -> None:
                self.device_ir = device_ir
                self.rdim = rdim
                self.graphs_added: list[int] = []
                self.outer_count = 0

            def has_matmul_with_rdim(self, graph: torch.fx.Graph) -> bool:
                return False

            def has_stack_tensor_with_rdim(self, graph: torch.fx.Graph) -> bool:
                return False

            def process(self, graph: torch.fx.Graph) -> torch.fx.Graph:
                reduction_graph = torch.fx.Graph()
                reduction_graph.output(())
                graph_id = self.device_ir.add_reduction_loop_graph(
                    reduction_graph,
                    block_index=self.rdim.block_id,
                    node_args=[],
                )
                self.graphs_added.append(graph_id)
                return graph

        device_ir = DeviceIR()
        graph = torch.fx.Graph()
        graph.output(())
        device_ir.add_root_graph(graph)
        original_graph_count = len(device_ir.graphs)

        fake_env = SimpleNamespace(
            block_sizes=[_FakeRDim()],
            config_spec=SimpleNamespace(reduction_loops=[]),
            backend_name="triton",
        )

        with (
            patch(
                "helion._compiler.device_ir.CompileEnvironment.current",
                return_value=fake_env,
            ),
            patch("helion._compiler.device_ir.ReductionRoller", _FakeReductionRoller),
        ):
            device_ir.register_rollable_reductions()

        # Sub-graphs (ReductionLoopGraphInfo) are kept so that
        # _count_device_loads_and_stores can account for their loads/stores.
        self.assertEqual(len(device_ir.graphs), original_graph_count + 1)
        self.assertEqual(len(device_ir.rolled_reductions), 1)
        self.assertEqual(len(fake_env.config_spec.reduction_loops), 1)


@onlyBackends(["cute"])
class TestCuteBarrier(RefEagerTestBase, TestCase):
    def test_barrier_requires_real_backend_support(self) -> None:
        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        with self.assertRaisesRegex(exc.BackendUnsupported, "hl.barrier"):
            code_and_output(
                barrier_dep_single,
                (x,),
                block_sizes=[8, 8],
                pid_type="persistent_blocked",
            )
