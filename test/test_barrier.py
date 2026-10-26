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
from helion._testing import skipIfNotCUDA
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


@onlyBackends(["triton", "cute"])
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

    @onlyBackends(["triton"])
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
        code, out = code_and_output(barrier_dep_single, (x,))
        expected = x * 2 + 1
        torch.testing.assert_close(out, expected)
        # Can't see pid_type in ref-mode code; rely on normalization to succeed.

    @skipIfRefEager("reference configs are only used in compiled mode")
    @skipIfTileIR("TileIR does not support barrier operations")
    def test_autotune_reference_config_is_persistent(self) -> None:
        x = torch.arange(257, device=DEVICE, dtype=torch.float32)
        bound = barrier_dep_single.bind((x,))
        spec = bound.config_spec
        config = spec.autotune_reference_config()
        self.assertEqual(config.config["pid_type"], "persistent_blocked")
        generation = spec.create_config_generation()
        self.assertEqual(generation.unflatten(generation.flatten(config)), config)
        compiled = bound.compile_config(config)
        torch.testing.assert_close(compiled(x), x * 2 + 1)

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
            # A static reduction extent so register_rollable_reductions can register the
            # reduction_loops spec (reads rdim.size).
            size = 16

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

            def has_gather_along_rdim(self, graph: torch.fx.Graph) -> bool:
                return False

            def has_stack_tensor_with_rdim(self, graph: torch.fx.Graph) -> bool:
                return False

            def has_unrollable_reduction(self, graph: torch.fx.Graph) -> bool:
                return False

            def has_pinned_rdim(self) -> bool:
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

        fake_backend = SimpleNamespace(
            register_reduction_loop_config_slots=lambda *_: None
        )
        fake_env = SimpleNamespace(
            block_sizes=[_FakeRDim()],
            config_spec=SimpleNamespace(
                reduction_block_ids=set(),
                reduction_loops=[],
            ),
            backend_name="triton",
            backend=fake_backend,
        )

        # The fake roller adds an empty subgraph (only an output node), so the
        # FX walker in _reduction_fx_inter_loop_rw_names never dereferences the
        # HostFunction; we just need HostFunction.current() not to raise.
        fake_host = SimpleNamespace()
        with (
            patch(
                "helion._compiler.device_ir.CompileEnvironment.current",
                return_value=fake_env,
            ),
            patch("helion._compiler.device_ir.ReductionRoller", _FakeReductionRoller),
            patch(
                "helion._compiler.device_ir.HostFunction.current",
                return_value=fake_host,
            ),
        ):
            device_ir.register_rollable_reductions()

        # Sub-graphs (ReductionLoopGraphInfo) are kept so that
        # _collect_memory_op_facts can account for their loads/stores.
        self.assertEqual(len(device_ir.graphs), original_graph_count + 1)
        self.assertEqual(len(device_ir.rolled_reductions), 1)
        self.assertEqual(len(fake_env.config_spec.reduction_loops), 1)

    @skipIfNotCUDA()
    @skipIfRefEager("promoted-seed pid_type is only materialized in compiled mode")
    @skipIfTileIR("TileIR does not support barrier operations")
    def test_reduction_seed_default_config_is_persistent(self) -> None:
        """Regression: the promoted sm100 reduction seed hardcodes pid_type='flat',
        but hl.barrier() forces a persistent kernel. With autotuning off (no explicit
        config, so the promoted seed IS the default) the kernel must still compile with
        a persistent pid_type and match the reference -- before the fix this raised
        helion.exc.BarrierRequiresPersistent. No pid_type/config is passed, so this
        exercises config_spec.default_config() (the promoted seed)."""

        @helion.kernel(autotune_effort="none")
        def barrier_reduction(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                partial[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] * 2.0
            return out

        x = torch.randn([256, 8192], device=DEVICE, dtype=torch.float32)
        expected = (x.double().sum(-1) * 2.0).float()
        _code, out = code_and_output(barrier_reduction, (x,))
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-2)


@onlyBackends(["cute"])
class TestCuteBarrier(RefEagerTestBase, TestCase):
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

    @skipIfRefEager("config normalization is only enforced in compiled mode")
    def test_multi_phase_requires_one_cta_per_sm(self) -> None:
        """The grid barrier waits for every CTA, and the CuTe launcher has no
        cooperative launch to reject a grid the GPU cannot hold at once: an
        oversubscribed persistent grid would hang instead of failing."""

        @helion.kernel(autotune_effort="none")
        def implicit_dependency(x: torch.Tensor) -> torch.Tensor:
            tmp = torch.empty_like(x)
            out = torch.empty_like(x)
            for t in hl.tile(x.size(0)):
                tmp[t] = x[t] * 2
            for t in hl.tile(x.size(0)):
                out[t] = tmp[t] + 1
            return out

        x = torch.arange(8, device=DEVICE, dtype=torch.float32)
        config = helion.Config(
            block_sizes=[8, 8], pid_type="persistent_blocked", num_sm_multiplier=2
        )
        for kernel in (barrier_dep_single, implicit_dependency):
            with self.subTest(kernel=kernel.name):
                bound = kernel.bind((x,))
                with self.assertRaisesRegex(
                    exc.InvalidConfig, "multi-phase kernels .* num_sm_multiplier=1"
                ):
                    bound.to_code(config)
                fixed = helion.Config(**config.config)
                bound.config_spec.normalize(fixed, _fix_invalid=True)
                self.assertEqual(fixed.config.get("num_sm_multiplier", 1), 1)

    @skipIfRefEager("CuTe lane-loop lowering is unavailable in ref eager mode")
    def test_tile_reduction_preserves_lane_loop_metadata(self) -> None:
        @helion.kernel(backend="cute", autotune_effort="none")
        def barrier_tile_sum(x: torch.Tensor) -> torch.Tensor:
            m, n = x.shape
            tmp = torch.empty([m], dtype=x.dtype, device=x.device)
            out = torch.empty_like(tmp)
            for tile_m, tile_n in hl.tile([m, n]):
                tmp[tile_m] = x[tile_m, tile_n].sum(dim=1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = tmp[tile_m]
            return out

        x = torch.randn([8, 128], device=DEVICE, dtype=torch.float32)
        code, out = code_and_output(
            barrier_tile_sum,
            (x,),
            block_sizes=[1, 128, 1],
            num_threads=[1, 32, 1],
            pid_type="persistent_blocked",
        )

        torch.testing.assert_close(out, x.sum(dim=1))
        self.assertNotIn("_helion_lane_reduce", code)

    @skipIfRefEager("pid_type is only materialized in compiled mode")
    def test_default_config_is_persistent(self) -> None:
        """CuTe's flat config has no pid_type coordinate, so the default config
        used to fall back to ``flat`` and raise ``BarrierRequiresPersistent``."""
        x = torch.arange(4, device=DEVICE, dtype=torch.float32)
        spec = barrier_dep_single.bind((x,)).config_spec
        self.assertEqual(spec.allowed_pid_types, ("persistent_blocked",))
        self.assertEqual(spec.default_config().pid_type, "persistent_blocked")
        # A partial user config must keep the persistent choice through normalize
        # (a stripped pid_type would fall back to flat and fail codegen).
        code, out = code_and_output(barrier_dep_single, (x,), block_sizes=[4, 4])
        torch.testing.assert_close(out, x * 2 + 1)
        self.assertIn("_cute_grid_barrier", code)

    @skipIfRefEager("promoted-seed pid_type is only materialized in compiled mode")
    def test_reduction_seed_default_config_is_persistent(self) -> None:
        @helion.kernel(autotune_effort="none")
        def barrier_reduction(x: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                partial[tile_m] = x[tile_m, :].to(torch.float32).sum(-1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] * 2.0
            return out

        x = torch.randn([256, 8192], device=DEVICE, dtype=torch.float32)
        expected = (x.double().sum(-1) * 2.0).float()
        _code, out = code_and_output(barrier_reduction, (x,))
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-2)

    @skipIfRefEager("exercises compiled multi-phase thread layouts")
    def test_split_k_matmul_default_config(self) -> None:
        """The example's default config (no explicit config) must launch a
        persistent kernel whose two phases agree on the thread layout."""
        from examples.split_k_barrier import split_k_matmul

        torch.manual_seed(0)
        a = torch.randn(16, 4096, device=DEVICE)
        b = torch.randn(16, 4096, device=DEVICE).T
        bound = split_k_matmul.bind((a, b))
        config = bound.config_spec.default_config()
        self.assertEqual(config.pid_type, "persistent_blocked")
        out = bound.compile_config(config)(a, b)
        torch.testing.assert_close(out, a @ b, rtol=1e-4, atol=1e-2)

    @skipIfRefEager("exercises compiled multi-phase thread layouts")
    def test_split_k_matmul_phase_thread_layouts(self) -> None:
        """Both phases share one launch block.  Phase 2's split-K sum owns
        thread axis 0 in every phase, so phase 1's K contraction must group its
        lanes per axis-0 thread instead of shuffling consecutive lanes, and the
        launch must match the axes the body indexes (previously block=(16, 8, 1)
        was launched for a body indexing axes 1 and 2)."""
        from examples.split_k_barrier import split_k_matmul

        torch.manual_seed(0)
        a = torch.randn(16, 4096, device=DEVICE)
        b = torch.randn(16, 4096, device=DEVICE).T
        bound = split_k_matmul.bind((a, b))
        base = dict(bound.config_spec.default_config().config)
        for block_sizes in ([1, 1, 16, 1, 1], [16, 8, 16, 1, 1], [16, 8, 16, 16, 16]):
            config = helion.Config.from_dict(
                base | {"block_sizes": block_sizes, "pid_type": "persistent_blocked"}
            )
            out = bound.compile_config(config)(a, b)
            torch.testing.assert_close(
                out, a @ b, rtol=1e-4, atol=1e-2, msg=f"block_sizes={block_sizes}"
            )

    @skipIfRefEager("launch-layout checks only run in compiled mode")
    def test_rejects_reduction_narrower_than_phase_launch(self) -> None:
        """Phase 1's rolled 64-lane reduction widens the launch to 64 threads on
        axis 0; phase 2's persistent reduction only spans 16 of them, so the
        surplus lanes would be folded into (or race on) its result.  Without the
        launcher check this compiled and returned nondeterministic sums."""

        @helion.kernel(autotune_effort="none")
        def two_phase_row_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                partial[tile_m] = x[tile_m, :].sum(-1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] + y[tile_m, :].sum(-1)
            return out

        x = torch.randn([64, 1024], device=DEVICE)
        y = torch.randn([64, 512], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction block 3 spans 16 lanes on thread axis 0 but the "
            r"hl\.barrier\(\) launch \(64, 1, 1\) runs 64 there",
        ):
            code_and_output(
                two_phase_row_sums,
                (x, y),
                block_sizes=[1, 1],
                reduction_loops=[64, None],
                pid_type="persistent_blocked",
            )
        # Both reductions persistent: the phases agree on the launch.
        _code, out = code_and_output(
            two_phase_row_sums,
            (x, y),
            block_sizes=[1, 1],
            reduction_loops=[None, None],
            pid_type="persistent_blocked",
        )
        torch.testing.assert_close(out, x.sum(-1) + y.sum(-1), rtol=1e-4, atol=1e-2)

    @skipIfRefEager("launch-layout checks only run in compiled mode")
    def test_rejects_tile_reduction_narrower_than_phase_launch(self) -> None:
        """A reduction over a 128-thread tile axis in phase 1 shares thread axis
        0 with phase 2's 512-thread tile, so the launch runs more lanes on that
        axis than the reduction spans."""

        @helion.kernel(autotune_effort="none")
        def two_phase_tile_sums(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, n = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m, tile_n in hl.tile([m, n]):
                partial[tile_m] = x[tile_m, tile_n].sum(dim=1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] + y[tile_m, :].sum(-1)
            return out

        x = torch.randn([64, 1024], device=DEVICE)
        y = torch.randn([64, 1024], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"reduction over tile block 1 spans 128 lanes on thread axis 0 but "
            r"the hl\.barrier\(\) launch \(512, 1, 1\) runs 512 there",
        ):
            code_and_output(
                two_phase_tile_sums,
                (x, y),
                block_sizes=[1, 128, 512],
                reduction_loops=[128],
                pid_type="persistent_blocked",
            )

    @skipIfRefEager("launch-layout checks only run in compiled mode")
    def test_rejects_staged_matmul_sum_layout_mismatch(self) -> None:
        """Phase 1's scalar-fallback K contraction records the static launch
        plan (16, 32, 1); phase 2's free ``hl.arange`` lands on a synthetic
        thread axis the plan never saw, so the final launch (16, 32, 2) differs
        from the recorded layout.  The launcher rejects any such difference
        (conservatively: the contraction does not span axis 2)."""

        @helion.kernel(autotune_effort="none")
        def matmul_then_arange(
            a: torch.Tensor, b: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            m, k = a.size()
            _, n = b.size()
            prod = torch.empty([m, n], dtype=torch.float32, device=a.device)
            out = torch.empty([m, 12], dtype=torch.float32, device=a.device)
            for tile_m in hl.tile(m):
                acc = hl.zeros([tile_m, n], dtype=torch.float32)
                for tile_k in hl.tile(k):
                    acc = torch.addmm(acc, a[tile_m, tile_k], b[tile_k, :])
                prod[tile_m, :] = acc
            hl.barrier()
            for row in hl.grid(m):
                column = hl.arange(12)
                out[row, column] = y[row, column] * 2
            return out

        a = torch.randn([16, 256], device=DEVICE)
        b = torch.randn([256, 16], device=DEVICE)
        y = torch.randn([16, 12], device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported,
            r"staged matmul product sum spans 1 lanes on thread axis 2 but the "
            r"hl\.barrier\(\) launch \(16, 32, 2\) runs 2 there",
        ):
            code_and_output(
                matmul_then_arange,
                (a, b, y),
                block_sizes=[1, 32],
                pid_type="persistent_blocked",
            )

    @skipIfRefEager("launch-layout checks only run in compiled mode")
    def test_persistent_reduction_group_spans_the_barrier_launch(self) -> None:
        """Two reductions in one tile loop put 2 lanes on axis 0 and a 64-lane
        persistent reduction on axis 1, and phase 2's tile takes 8 threads on
        axis 2: the launch is (2, 64, 8).  Every strategy's axis counts toward
        the reduction's thread group, so its shared-memory reduce keys on the
        full linear thread id and gives each of phase 1's 8 replicas on axis 2
        its own group, interleaving the 2 lanes below it."""

        @helion.kernel(
            config=helion.Config(block_sizes=[1, 8], pid_type="persistent_blocked")
        )
        def two_reductions_then_wide(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            m, _ = x.size()
            partial = torch.empty([m], dtype=torch.float32, device=x.device)
            out = torch.empty([m], dtype=torch.float32, device=x.device)
            for tile_m in hl.tile(m):
                partial[tile_m] = x[tile_m, :].sum(-1) + y[tile_m, :].sum(-1)
            hl.barrier()
            for tile_m in hl.tile(m):
                out[tile_m] = partial[tile_m] * 2.0
            return out

        for seed in range(4):
            torch.manual_seed(seed)
            x = torch.randn([64, 2], device=DEVICE)
            y = torch.randn([64, 64], device=DEVICE)
            code, out = code_and_output(two_reductions_then_wide, (x, y))
            self.assertIn("block=(2, 64, 8)", code)
            self.assertIn("pre=2, group_span=128, group_count=8", code)
            torch.testing.assert_close(out, (x.sum(-1) + y.sum(-1)) * 2.0)


@onlyBackends(["cute"])
class TestCuteOrderedPhases(RefEagerTestBase, TestCase):
    @skipIfRefEager("checks separately launched grid phases")
    def test_explicit_barrier_launch_order(self) -> None:
        x = torch.randn(137, device=DEVICE)
        for kernel, expected, phases in (
            (barrier_dep_single, x * 2 + 1, 2),
            (barrier_multiple, (x + 3) * 2 - 5, 3),
        ):
            with self.subTest(phases=phases):
                code, output = code_and_output(
                    kernel, (x,), block_sizes=[32] * phases, pid_type="flat"
                )
                torch.testing.assert_close(output, expected)
                self.assertIn("_helion_cute_kernels", code)
                self.assertIn(f"_helion_cute_region_{phases - 1}_module", code)

    @skipIfRefEager("checks differently shaped phase grids")
    def test_explicit_barrier_grid_to_row_reduction(self) -> None:
        @helion.kernel(static_shapes=True)
        def transform_reduce(x: torch.Tensor) -> torch.Tensor:
            scratch = torch.empty_like(x)
            output = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            for row, col in hl.tile(x.shape):
                scratch[row, col] = x[row, col] * 2
            hl.barrier()
            for row in hl.tile(x.size(0)):
                output[row] = scratch[row, :].sum(-1)
            return output

        x = torch.randn(17, 65, device=DEVICE)
        _, output = code_and_output(
            transform_reduce, (x,), block_sizes=[4, 32, 4], pid_type="flat"
        )
        torch.testing.assert_close(output, (x * 2).sum(-1))

    @skipIfRefEager("checks a shared registered tile across different grid ranks")
    def test_explicit_barrier_shared_tile_loop_order(self) -> None:
        @helion.kernel(static_shapes=True)
        def transform_reduce(x: torch.Tensor) -> torch.Tensor:
            row_block = hl.register_block_size(x.size(0))
            scratch = torch.empty_like(x)
            output = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            for row, col in hl.tile(x.shape, block_size=[row_block, None]):
                scratch[row, col] = x[row, col] * 2
            hl.barrier()
            for row in hl.tile(x.size(0), block_size=row_block):
                output[row] = scratch[row, :].sum(-1)
            return output

        x = torch.randn(17, 65, device=DEVICE)
        for order in ([0, 1], [1, 0]):
            with self.subTest(order=order):
                _, output = code_and_output(
                    transform_reduce,
                    (x,),
                    block_sizes=[4, 32],
                    loop_orders=[order],
                    pid_type="flat",
                )
                torch.testing.assert_close(output, (x * 2).sum(-1))

    @skipIfRefEager("checks scan and custom-reduction combine graphs across phases")
    def test_explicit_barrier_scan_and_custom_reduction(self) -> None:
        @helion.kernel(static_shapes=True)
        def scan_reduce(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scratch = torch.empty_like(x)
            prefix = torch.empty_like(x)
            total = torch.empty((x.size(0),), dtype=x.dtype, device=x.device)
            for row, col in hl.tile(x.shape):
                scratch[row, col] = x[row, col] * 2
            hl.barrier()
            for row in hl.tile(x.size(0)):
                values = scratch[row, :]
                prefix[row, :] = hl.cumsum(values, dim=-1)
                total[row] = hl.reduce(torch.add, values, dim=-1)
            return prefix, total

        x = torch.randn(17, 65, device=DEVICE)
        _, (prefix, total) = code_and_output(
            scan_reduce, (x,), block_sizes=[4, 32, 4], pid_type="flat"
        )
        torch.testing.assert_close(prefix, (x * 2).cumsum(-1))
        torch.testing.assert_close(total, (x * 2).sum(-1))

    @skipIfRefEager("checks runtime input aliases across phases")
    def test_explicit_barrier_input_alias(self) -> None:
        @helion.kernel(static_shapes=True)
        def mutate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            for tile in hl.tile(x.numel()):
                out[tile] = x[tile] * 2
            hl.barrier()
            for tile in hl.tile(x.numel()):
                x[tile] = out[tile] + 1
            return x

        for alias in (False, True):
            with self.subTest(alias=alias):
                x = torch.randn(137, device=DEVICE)
                expected = x * 2 + 1
                out = x if alias else torch.empty_like(x)
                _, actual = code_and_output(
                    mutate, (x, out), block_sizes=[32, 32], pid_type="flat"
                )
                torch.testing.assert_close(actual, expected)

    @skipIfRefEager("checks ordered launch graph replay")
    def test_explicit_barrier_graph_replay_with_alias(self) -> None:
        @helion.kernel(
            static_shapes=True,
            autotune_effort="none",
            config=helion.Config(block_sizes=[32, 32], pid_type="flat"),
        )
        def mutate(x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            scratch = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                scratch[tile] = x[tile] * 2
            hl.barrier()
            for tile in hl.tile(x.numel()):
                out[tile] = scratch[tile] + 1
            return out

        x = torch.randn(137, device=DEVICE)
        mutate(x, x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = mutate(x, x)
        for _repeat in range(3):
            source = torch.randn_like(x)
            x.copy_(source)
            graph.replay()
            torch.testing.assert_close(output, source * 2 + 1)

    @skipIfRefEager("checks unsupported phase ownership")
    def test_explicit_barrier_grouped_roots_rejected(self) -> None:
        x = torch.randn(137, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "each phase must contain one top-level grid loop"
        ):
            code_and_output(
                barrier_groups, (x, x), block_sizes=[32] * 4, pid_type="flat"
            )

    @skipIfRefEager("checks host effects between grid phases")
    def test_explicit_barrier_host_effect_rejected(self) -> None:
        @helion.kernel(static_shapes=True)
        def allocate_between_phases(x: torch.Tensor) -> torch.Tensor:
            scratch = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                scratch[tile] = x[tile] * 2
            hl.barrier()
            output = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                output[tile] = scratch[tile] + 1
            return output

        x = torch.randn(137, device=DEVICE)
        with self.assertRaisesRegex(
            exc.TopLevelStatementBetweenLoops, "Statements cannot appear between"
        ):
            code_and_output(
                allocate_between_phases, (x,), block_sizes=[32, 32], pid_type="flat"
            )

    @skipIfRefEager("checks runtime conditional ownership")
    def test_explicit_barrier_runtime_conditional_rejected(self) -> None:
        @helion.kernel(static_shapes=True)
        def conditional(x: torch.Tensor) -> torch.Tensor:
            scratch = torch.empty_like(x)
            output = torch.empty_like(x)
            for tile in hl.tile(x.numel()):
                if tile.begin == 0:
                    scratch[tile] = x[tile] * 2
                else:
                    scratch[tile] = x[tile] * 3
            hl.barrier()
            for tile in hl.tile(x.numel()):
                output[tile] = scratch[tile] + 1
            return output

        x = torch.randn(137, device=DEVICE)
        with self.assertRaisesRegex(
            exc.BackendUnsupported, "runtime conditionals inside grid phases"
        ):
            code_and_output(conditional, (x,), block_sizes=[32, 32], pid_type="flat")
