"""The ``Backend`` capability hooks reproduce the literal backend-name
predicates they replaced, the loop-graph walker descends for-loops only, and
``BackendDispatchTable`` keeps the registries' lookup semantics."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pytest
import torch
from torch.fx import Graph

import helion
from helion import exc
from helion._compiler import backend_registry
from helion._compiler.aten_lowering import AtenLowering
from helion._compiler.backend import Backend
from helion._compiler.backend import _loop_contains_matmul
from helion._compiler.device_ir import ForLoopGraphInfo
from helion._compiler.device_ir import IfGraphInfo
from helion._compiler.device_ir import RootGraphInfo
from helion._compiler.device_ir import iter_loop_graph_nodes
from helion._compiler.per_thread_planning import _graph_used_block_ids
from helion._testing import skipIfRefEagerMark
from helion._testing.cute_codegen import cpu_bind
from helion.autotuner.config_spec import BlockSizeSpec
from helion.autotuner.config_spec import ConfigSpec
from helion.autotuner.config_spec import NumThreadsSpec
from helion.autotuner.search_surface import BackendSearchSurface
import helion.language as hl
from helion.language import _tracing_ops
from helion.language._decorators import BackendDispatchTable


def _backends() -> dict[str, Backend]:
    return {
        name: backend_registry.get_backend_class(name)()
        for name in backend_registry.list_backends()
    }


class TestCapabilityHooks(unittest.TestCase):
    """Each hook, over every registered backend, against the predicate it
    replaced (``== "cute"`` and friends are allowed here: this is the table)."""

    def test_unsupported_names_the_backend(self) -> None:
        for name, backend in _backends().items():
            error = backend.unsupported("a detail")
            self.assertIsInstance(error, exc.BackendUnsupported)
            self.assertEqual(str(error), f"Backend '{name}' does not support: a detail")

    def test_table(self) -> None:
        for registered, backend in _backends().items():
            codegen_name = backend.codegen_name
            # The gluon backend answers as Triton, whose lowering it compiles.
            name = "triton" if registered == "gluon" else registered
            with self.subTest(backend=registered):
                self.assertEqual(
                    backend.reserved_device_names(),
                    frozenset({"_"}) if name == "cute" else frozenset(),
                )
                self.assertEqual(backend.output_only_names(), [])
                self.assertEqual(
                    backend.lowers_integer_powers, codegen_name in ("triton", "cute")
                )
                self.assertEqual(backend.requires_proved_integer_powers, name == "cute")
                self.assertEqual(backend.tracks_statement_owners(), name == "cute")
                self.assertEqual(
                    backend.canonicalizes_reduction_loop_blocks(), name == "cute"
                )
                self.assertFalse(backend.prefers_wall_clock_bench())
                triton_barriers = codegen_name == "triton" and name != "tileir"
                self.assertEqual(backend.uses_triton_compiler, name == "triton")
                self.assertEqual(backend.supports_inter_loop_barriers, triton_barriers)
                self.assertEqual(
                    backend.supports_intra_loop_barriers(
                        in_divergent_control_flow=False
                    ),
                    name == "cute" or triton_barriers,
                )
                self.assertEqual(
                    backend.supports_intra_loop_barriers(
                        in_divergent_control_flow=True
                    ),
                    triton_barriers,
                )
                self.assertEqual(
                    backend.lowers_chunk_unbind, name in ("triton", "cute")
                )
                for pid_type in ("flat", "xyz", "persistent_interleaved"):
                    self.assertEqual(
                        backend.barrier_allows_pid_type(pid_type),
                        pid_type == "flat" and name == "cute",
                    )
                self.assertEqual(backend.supports_augmented_assign, name != "cute")
                self.assertEqual(backend.folds_static_program_counts, name == "cute")
                self.assertEqual(
                    backend.reduction_loops_allow_non_power_of_two, name == "cute"
                )
                self.assertEqual(
                    backend.reduction_loops_allow_full_extent, name == "cute"
                )
                self.assertEqual(
                    backend.merges_reduction_dims, codegen_name in ("triton", "cute")
                )
                self.assertEqual(backend.requires_merged_reduction_dims, name == "cute")
                self.assertEqual(
                    backend.warns_fast_math_unused, name in ("triton", "cute")
                )
                self.assertEqual(
                    backend.supports_autotune_multi,
                    name in ("triton", "tileir", "cute", "pallas"),
                )
                # Thread layout and lane loops of the tile strategies.
                self.assertEqual(
                    backend.loop_var_prefixes(),
                    ("tile_offset", "tile_step")
                    if name == "cute"
                    else ("offset", "step"),
                )
                self.assertEqual(backend.plans_launch_thread_axes(), name == "cute")
                self.assertEqual(
                    backend.single_thread_tile_claims_axis(), name != "cute"
                )
                self.assertEqual(
                    backend.flat_strided_tile_walks_elements(), name == "cute"
                )
                self.assertEqual(backend.tracks_flat_grid_origins(), name == "cute")
                self.assertEqual(backend.reflattens_pointwise_tiles(), name == "cute")
                self.assertEqual(backend.tunes_lane_vector_widths(), name == "cute")
                self.assertEqual(
                    backend.thread_axis_limit(), 3 if name == "cute" else None
                )
                self.assertEqual(
                    backend.thread_axis_extent_limits(),
                    (1024, 1024, 64) if name == "cute" else None,
                )
                self.assertEqual(
                    backend.rng_streams(),
                    ("auto", "word0", "philox4") if name == "cute" else ("word0",),
                )
                self.assertEqual(
                    backend.final_rebenchmark_top_k(), 32 if name == "cute" else None
                )
                self.assertEqual(backend.final_rebenchmark_isolated(), name == "cute")
                profile = backend.llm_prompt_profile()
                self.assertEqual(
                    profile.label,
                    {"cute": "CuTe", "triton": "Triton"}.get(name, name),
                )
                self.assertEqual(profile.large_gemm_guidance, name != "cute")

    def test_transform_launch_grid_default_is_identity(self) -> None:
        grid = object()
        for name, backend in _backends().items():
            if name != "cute":
                self.assertIs(
                    backend.transform_launch_grid(SimpleNamespace(), grid), grid
                )

    def test_codegen_hook_defaults_are_no_ops(self) -> None:
        """The hooks that replaced the CuTe branches of generate_ast,
        device_function, device_ir, program_id and tile_dispatch keep their
        documented no-op defaults on every other backend: ``None`` / ``False``
        answers, the input returned unchanged, and no mutation of the lists
        they are handed.  The defaults never look at their arguments."""
        arg = SimpleNamespace()
        ctx = SimpleNamespace(
            device_function=SimpleNamespace(triton_unbounded_uint64_axes=set())
        )
        for name, backend in _backends().items():
            if name == "cute":
                continue
            with self.subTest(backend=name):
                error = backend.unsupported("a root")
                body: list[object] = [arg]
                self.assertIsNone(backend.tile_begin_expr(arg, 0))
                self.assertFalse(backend.rewrite_lowered_root(arg, arg, body))
                self.assertEqual(body, [arg])
                self.assertIs(backend.explain_root_failure(arg, error), error)
                self.assertIsNone(backend.validate_codegen_config(arg, arg, arg))
                self.assertIsNone(backend.after_device_codegen(arg, arg, arg))
                self.assertIsNone(backend.prepare_device_loop(arg, arg, arg))
                outputs: list[object] = [arg]
                self.assertIsNone(
                    backend.finish_device_loop_body(arg, arg, outputs, {})
                )
                self.assertEqual(outputs, [arg])
                if name != "gluon":  # gluon finalizes its ring and pipelining
                    self.assertEqual(backend.finalize_kernel_def(arg, [], ctx), [])
                statements: list[object] = [arg]
                result = backend.finalize_host_statements(arg, arg, statements)
                self.assertIs(result[0], statements)
                self.assertEqual(result[1], [])
                self.assertEqual(statements, [arg])
                post = backend.split_persistent_post_loop(arg, body)
                self.assertIs(post[0], body)
                self.assertEqual(post[1], [])
                self.assertIsNone(backend.collective_sibling_blocks(arg, arg))
                self.assertFalse(backend.tensors_proven_disjoint(arg, arg, arg))
                self.assertIsNone(backend.packed_grid_block(arg))
                self.assertIsNone(backend.packed_grid_block_id(arg))
                self.assertIsNone(backend.record_launch_thread_extent(arg, "i", 0, 1))
                self.assertIsNone(backend.record_grid_thread_extent(arg, "i", 0, 1))
                self.assertIsNone(
                    backend.record_vloop_wrapper(
                        arg,
                        arg,
                        block_id=0,
                        lane_var="lane",
                        index_var="i",
                        mask_var=None,
                        begin=0,
                        extent=1,
                        step=None,
                    )
                )
                self.assertIsNone(backend.apply_lane_cluster_split(arg, arg))
                self.assertIsNone(backend.check_lane_loop_nest(arg))
                self.assertIsNone(backend.add_lane_nest_barriers(arg, body, [], {}))
                self.assertEqual(body, [arg])
                self.assertFalse(backend.demote_grid_tile_vec_stores(arg, body, None))
                self.assertEqual(body, [arg])
                self.assertIsNone(backend.demote_loop_tile_vec_stores(arg, [0]))
                self.assertEqual(backend.synthetic_thread_axis_sizes(arg), {})
                self.assertFalse(backend.consume_root_lane_loop_suppression(arg))
                self.assertIs(backend.reorder_post_loop_statements(arg, body), body)
                self.assertFalse(backend.roots_fissioned(arg))
                self.assertIsNone(backend.new_env_state())
                self.assertIsNone(backend.new_codegen_state())
                self.assertIsNone(backend.new_device_state(arg))
                surface = backend.config_spec_extension()
                self.assertIsNone(surface.explain_unavailable_key("block_sizes", arg))
                self.assertIsNone(
                    surface.eviction_policy_search_choices(arg, ("", "evict_last"))
                )
                self.assertIsNone(surface.new_spec_state(arg))
                self.assertEqual(surface.eviction_policies(), ("",))
                self.assertEqual(surface.reduction_loop_limits(arg, 0, 7, 9), (7, 9))
                self.assertEqual(surface.explicit_reduction_chunk(arg, arg, 5), 5)
                self.assertEqual(surface.family_fragments(arg, "fa4"), {})


class TestLoopGraphWalker(unittest.TestCase):
    """``iter_loop_graph_nodes`` yields the loop's graphs in pre-order, descends
    ``_for_loop`` / ``_for_loop_step`` and never an ``_if`` graph."""

    def _fixture(self) -> tuple[SimpleNamespace, Graph, Graph, Graph]:
        root = Graph()
        x = root.placeholder("x")
        root.call_function(_tracing_ops._for_loop_step, args=(1, [0], [8], [x]))
        root.call_function(_tracing_ops._if, args=(x, 2, -1, [x]))
        root.output(x)

        step = Graph()
        acc = step.placeholder("acc")
        step.call_function(torch.ops.aten.mm.default, args=(acc, acc))
        step.output(acc)

        branch = Graph()
        y = branch.placeholder("y")
        branch.call_function(torch.ops.aten.addmm.default, args=(y, y, y))
        branch.output(y)

        fn = SimpleNamespace(
            codegen=SimpleNamespace(
                codegen_graphs=[
                    RootGraphInfo(graph_id=0, graph=root, phase_index=0),
                    ForLoopGraphInfo(
                        graph_id=1, graph=step, node_args=[], block_ids=[3]
                    ),
                    IfGraphInfo(graph_id=2, graph=branch, node_args=[]),
                ]
            )
        )
        return fn, root, step, branch

    def test_pre_order_descends_for_loop_steps_only(self) -> None:
        fn, root, step, branch = self._fixture()
        host = SimpleNamespace(
            device_ir=SimpleNamespace(root_ids=[0], grid_block_ids=[[0, 1]])
        )
        with patch(
            "helion._compiler.host_function.HostFunction.current", return_value=host
        ):
            nodes = list(iter_loop_graph_nodes(fn, [0, 1]))
        root_nodes = list(root.nodes)
        step_nodes = list(step.nodes)
        # The root's nodes, with the step graph's nodes right after the node
        # that names it; the if-branch graph is never descended.
        self.assertEqual(
            nodes,
            [root_nodes[0], root_nodes[1], *step_nodes, root_nodes[2], root_nodes[3]],
        )
        self.assertFalse(set(branch.nodes) & set(nodes))
        # The for-loop graph itself, by its block ids.
        with patch(
            "helion._compiler.host_function.HostFunction.current", return_value=host
        ):
            self.assertEqual(list(iter_loop_graph_nodes(fn, [3])), step_nodes)
            self.assertEqual(list(iter_loop_graph_nodes(fn, [7])), [])
            self.assertTrue(_loop_contains_matmul(fn, [0, 1]))
            self.assertTrue(_loop_contains_matmul(fn, [3]))

    def test_roots_in_one_phase_only_match_their_own_grid(self) -> None:
        fn, _root, _step, _branch = self._fixture()
        other_root = Graph()
        x = other_root.placeholder("x")
        other_root.output(x)
        fn.codegen.codegen_graphs.append(
            RootGraphInfo(graph_id=3, graph=other_root, phase_index=0)
        )
        host = SimpleNamespace(
            device_ir=SimpleNamespace(root_ids=[0, 3], grid_block_ids=[[0, 1], [4]])
        )
        with patch(
            "helion._compiler.host_function.HostFunction.current", return_value=host
        ):
            self.assertTrue(_loop_contains_matmul(fn, [0, 1]))
            self.assertFalse(_loop_contains_matmul(fn, [4]))
            self.assertEqual(
                list(iter_loop_graph_nodes(fn, [4])), list(other_root.nodes)
            )

    def test_root_without_grid_matches_nothing(self) -> None:
        fn, _root, _step, _branch = self._fixture()
        host = SimpleNamespace(
            device_ir=SimpleNamespace(root_ids=[], grid_block_ids=[])
        )
        with patch(
            "helion._compiler.host_function.HostFunction.current", return_value=host
        ):
            self.assertEqual(list(iter_loop_graph_nodes(fn, [0, 1])), [])


@pytest.mark.cpu_only
@skipIfRefEagerMark("reads compiler state, which ref mode does not build")
def test_used_block_ids_follow_roots_not_barrier_phases() -> None:
    @helion.kernel(backend="triton", static_shapes=True)
    def kernel(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        first = torch.empty_like(x)
        second = torch.empty_like(x)
        third = torch.empty_like(x)
        for row, col in hl.tile(x.shape):
            first[row, col] = x[row, col] + 1
        for row, col in hl.tile(x.shape):
            second[row, col] = x[row, col] + 2
        hl.barrier()
        for row, col in hl.tile(x.shape):
            third[row, col] = x[row, col] + 3
        return first, second, third

    bound = cpu_bind(kernel, (torch.zeros(8, 16),))
    host = bound.host_function
    ir = host.device_ir
    fn = SimpleNamespace(codegen=SimpleNamespace(codegen_graphs=ir.graphs))
    assert len(ir.phases) == 2
    assert len(ir.root_ids) == 3
    with bound.env, host:
        for grid in ir.grid_block_ids:
            assert _graph_used_block_ids(fn, grid) == set(grid)


class TestBackendDispatchTable(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch("helion._compiler.backend_registry.repair_backend_codegen")
        self.repair = patcher.start()
        self.addCleanup(patcher.stop)

    def test_lookup_falls_back_and_names_the_missing_implementation(self) -> None:
        common = object()
        table: BackendDispatchTable[object] = BackendDispatchTable({"common": common})
        self.assertIs(
            table.lookup("pallas", backend_name="pallas", what="codegen for op"), common
        )
        with self.assertRaises(exc.BackendImplementationMissing) as caught:
            BackendDispatchTable().lookup(
                "triton", backend_name="tileir", what="codegen for API function f"
            )
        self.assertEqual(
            str(caught.exception),
            "Backend 'tileir' is missing required implementation: codegen for API "
            "function f",
        )
        with self.assertRaises(KeyError):
            BackendDispatchTable()["triton"]

    def test_aten_lowering_dispatches_through_the_table(self) -> None:
        def handler(ctx: object, node: object) -> str:
            return "common"

        self.assertIsInstance(AtenLowering().codegen_impls, BackendDispatchTable)
        lowering = AtenLowering(codegen_impls=BackendDispatchTable({"common": handler}))
        self.assertIs(lowering.codegen_impls["cute"], handler)


@pytest.mark.cpu_only
class TestSpecWithoutBackendState(unittest.TestCase):
    """A spec of a backend whose search surface stores no per-spec state
    (``new_spec_state`` returns ``None`` for every backend but CuTe) builds
    its default config: the per-block thread knobs ask
    ``BackendSearchSurface.pins_thread_knobs`` instead of reading the CuTe
    tcgen05 facade the slot holds for CuTe only."""

    @staticmethod
    def _spec(backend: Backend) -> ConfigSpec:
        spec = ConfigSpec(backend=backend, device=torch.device("cpu"))
        spec.block_sizes.append(BlockSizeSpec(block_id=0, size_hint=1024))
        if spec.supports_config_key("num_threads"):
            spec.num_threads.append(NumThreadsSpec(block_id=0, size_hint=1024))
        return spec

    def test_metal_spec_with_a_thread_count_builds_its_default_config(self) -> None:
        # Metal declares ``num_threads``, so every tiled loop registers a
        # ``NumThreadsSpec``: the fragment that dereferenced the CuTe state.
        spec = self._spec(backend_registry.get_backend_class("metal")())
        self.assertIsNone(spec.tcgen05)
        self.assertEqual(len(spec.num_threads), 1)
        # Metal adds its own launch keys (``pid_type`` since #4258); the
        # regression here was the thread fragment, so pin those two entries.
        config = dict(spec.default_config())
        self.assertEqual(config["block_sizes"], [32])
        self.assertEqual(config["num_warps"], 4)

    def test_every_non_cute_spec_admits_a_search_proposal_as_it_is(self) -> None:
        # A proposal that reaches a search from outside its flat fields (the
        # configs of an LLM's reply) is benchmarked as normalization left it
        # unless its backend keeps one of its options out of its searches,
        # and none of these does
        # (``BackendSearchSurface.repair_search_proposal``).
        default = BackendSearchSurface.repair_search_proposal
        for name, backend in _backends().items():
            if name == "cute":
                continue
            with self.subTest(backend=name):
                spec = self._spec(backend)
                surface = type(spec.search_surface)
                self.assertIs(surface.repair_search_proposal, default)
                config = spec.default_config()
                before = config.to_json()
                proposal, returned = spec.search_proposal(config)
                self.assertIs(proposal, config)
                self.assertEqual(returned, ())
                self.assertEqual(config.to_json(), before)

    def test_every_non_cute_spec_builds_its_default_config(self) -> None:
        for name, backend in _backends().items():
            if name == "cute":
                continue
            with self.subTest(backend=name):
                spec = self._spec(backend)
                self.assertIsNone(spec.tcgen05)
                self.assertFalse(spec.search_surface.pins_thread_knobs(spec, 0))
                self.assertEqual(spec.default_config().block_sizes, [32])

    def test_every_spec_compiles_an_unsearched_config_as_it_is(self) -> None:
        # ``autotune_config_overrides`` act on a search. A config the kernel
        # selects without one (a cache hit, the default of a kernel that
        # does not autotune) is compiled as it is unless its backend holds
        # such a config to some of them, and none does
        # (``BackendSearchSurface.overrides_without_search``).
        default = BackendSearchSurface.overrides_without_search
        held = BackendSearchSurface.overrides_held_without_search
        for name, backend in _backends().items():
            with self.subTest(backend=name):
                spec = self._spec(backend)
                surface = type(spec.search_surface)
                self.assertIs(surface.overrides_without_search, default)
                self.assertIs(surface.overrides_held_without_search, held)
                config = spec.default_config()
                before = config.to_json()
                for overrides in (
                    None,
                    {},
                    {"block_sizes": [1]},
                    {"num_warps": 1, "indexing": "block_ptr"},
                    {"cute_pdl": False},
                    {"cute_pdl_capture_overlap": False},
                ):
                    self.assertIs(spec.unsearched_config(config, overrides), config)
                    # So no override is part of what an AOT export is keyed by.
                    self.assertIsNone(spec.unsearched_overrides_identity(overrides))
                    # A custom autotuner may hand back a plain dict of
                    # options, which ``set_config`` reads as well.
                    options = dict(config.config)
                    self.assertIs(spec.unsearched_config(options, overrides), options)
                self.assertEqual(config.to_json(), before)


if __name__ == "__main__":
    unittest.main()
