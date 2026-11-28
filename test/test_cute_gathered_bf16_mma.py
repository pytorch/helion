"""The BF16 reduction uses its actual page table and complete K fold."""

from __future__ import annotations

import ast
from contextlib import ExitStack
import copy
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler._inductor.template_buffer import inductor_call_facts
from helion._compiler.cute.analysis.gathered_bf16_mma import gathered_phase_region
from helion._compiler.cute.analysis.gathered_bf16_mma import prove_gathered_bf16_mma
from helion._compiler.cute.analysis.gathered_softmax import prove_gathered_softmax
from helion._compiler.cute.analysis.split_weighted_merge import (
    prove_split_weighted_merge,
)
from helion._compiler.cute.autotune.heuristics.cute_gathered_bf16_mma import (
    CuteGatheredBf16MmaSeed,
)
from helion._compiler.cute.autotune.heuristics.cute_gathered_bf16_mma import (
    gathered_phase_carrier,
)
from helion._compiler.cute.gathered_softmax import _seed_base_is_aligned
from helion._compiler.cute.gathered_softmax import can_pack_bf16_k_pairs
from helion._compiler.cute.gathered_softmax import can_pipeline_page
from helion._compiler.cute.memory_ops import (
    _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY,
)
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import cpu_target
from helion._testing.cute_codegen import default_cute_mma_support
from helion._testing.cute_gathered_bf16 import masked_page_stream
from helion._testing.cute_gathered_bf16 import routed_transposed_product
from helion._testing.cute_gathered_bf16 import split_paged_attention
from helion.language import _tracing_ops
from helion.language import memory_ops

pytestmark = pytest.mark.cpu_only


def _bind(
    seed: float = 0,
    dtype: torch.dtype = torch.bfloat16,
    page_shift: int = 0,
    columns: int = 32,
    depth: int = 80,
):
    return cpu_bind(
        routed_transposed_product,
        (
            torch.empty((32, depth), dtype=dtype),
            torch.empty((5, columns, depth), dtype=dtype),
            torch.empty((7,), dtype=torch.int64),
            seed,
            page_shift,
            False,
        ),
    )


def test_native_gathered_reduction() -> None:
    with cpu_target():
        bound = _bind()
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            proofs = [
                result
                for info in ir.graphs
                for node in info.graph.nodes
                if (result := prove_gathered_bf16_mma(node, ir.graphs, bound.env))
            ]
            assert len(proofs) == 1
            assert (proofs[0].rows, proofs[0].columns, proofs[0].depth) == (32, 32, 80)
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": True,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        source = bound.to_code(helion.Config.from_dict(config))
        assert "gathered_mma_page" in source
        assert "_helion_fragment_mma.mma_bf16(" in source


@pytest.mark.parametrize("enabled", [True, False])
def test_checked_gather_keeps_unmasked_page_bounds(enabled: bool) -> None:
    with cpu_target():
        bound = cpu_bind(
            routed_transposed_product,
            (
                torch.empty((32, 80), dtype=torch.bfloat16),
                torch.empty((5, 32, 80), dtype=torch.bfloat16),
                torch.full((7,), 5, dtype=torch.int64),
                0,
                0,
                True,
            ),
        )
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": enabled,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        with pytest.raises(helion.exc.BackendUnsupported, match="not proved in bounds"):
            bound.to_code(helion.Config.from_dict(config))


def test_wide_page_reduction_rolls_k_instead_of_unrolling_the_full_axis() -> None:
    with cpu_target():
        bound = _bind(depth=65536)
        config = helion.Config(
            block_sizes=bound.config_spec.default_config().block_sizes,
            cute_fragment_half_mma=True,
            cute_mma_dag="off",
        )
        source = bound.to_code(config)
        assert "gathered_mma_page" in source
        assert "for fragment_mma_step in range(4096)" in source


@pytest.mark.parametrize(
    "seed,dtype,page_shift,columns",
    [
        (1.0, torch.bfloat16, 0, 32),
        (0.0, torch.float16, 0, 32),
        (0.0, torch.bfloat16, 1, 32),
        (0.0, torch.bfloat16, 0, 24),
    ],
)
def test_unproved_reduction_falls_back(
    seed: float, dtype: torch.dtype, page_shift: int, columns: int
) -> None:
    with cpu_target():
        bound = _bind(seed, dtype, page_shift, columns)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            assert not any(
                prove_gathered_bf16_mma(node, graphs, bound.env)
                for graph in graphs
                for node in graph.graph.nodes
            )
        config = {
            "block_sizes": bound.config_spec.default_config().block_sizes,
            "cute_fragment_half_mma": True,
        }
        if "cute_mma_dag" in bound.config_spec._flat_fields():
            config["cute_mma_dag"] = "off"
        assert "gathered_mma_page" not in bound.to_code(helion.Config.from_dict(config))


def test_ordered_producer_keeps_the_combines_launch_and_barrier() -> None:
    # Keep this real CPU pointer outside the traced host's fake-tensor context.
    misaligned = torch.empty(33, dtype=torch.bfloat16)[1:]
    assert misaligned.data_ptr() % 16 != 0
    with torch.device("meta"):
        inputs = (
            torch.empty((1, 1, 16, 576), dtype=torch.bfloat16),
            torch.empty((1024, 1, 64, 576), dtype=torch.bfloat16),
            torch.empty((1, 1024), dtype=torch.int32),
            torch.empty((1,), dtype=torch.int32),
            576**-0.5,
        )
    with cpu_render_target(
        fake_device_sm_count=148,
        mma_support=default_cute_mma_support(),
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    ):
        kernel = helion.kernel(backend="cute", static_shapes=False)(
            split_paged_attention
        )
        gate = "helion._compiler.cute.cooperative_grid_init.grid_toolkit_installed"
        rejected = "helion._compiler.cute.cooperative_grid_init.grid_sync_rejected"
        with patch(gate, return_value=True), patch(rejected, return_value=False):
            bound = cpu_bind(kernel, inputs)
            cache_key = bound.config_spec.cache_fingerprint_hash()
        # A compiler can reject the cooperative instruction on first use.
        # Keep the seed roster and its disk key; viability declines that
        # specific config while the cached software-counter winner survives.
        with patch(gate, return_value=True), patch(rejected, return_value=True):
            after_rejection = cpu_bind(
                helion.kernel(backend="cute", static_shapes=False)(
                    split_paged_attention
                ),
                inputs,
            )
            assert after_rejection.config_spec.cache_fingerprint_hash() == cache_key
            carriers = [
                c
                for c in after_rejection.config_spec.compiler_seed_configs
                if c.get("cute_in_kernel_grid_init")
                and c.get("cute_gathered_page_pipeline")
            ]
            assert len(carriers) == 1
            spec_after = after_rejection.config_spec
            assert not spec_after.backend.can_reuse_autotune_result(
                spec_after, carriers[0]
            )
            software = next(
                c
                for c in spec_after.compiler_seed_configs
                if not c.get("cute_in_kernel_grid_init")
                and c.get("cute_gathered_page_stages") == 8
            )
            assert spec_after.backend.can_reuse_autotune_result(spec_after, software)
        # Discovery snapshots its candidates at bind time. A separate kernel
        # ensures the toolkit-absent binding cannot reuse the first one.
        with patch(gate, return_value=False):
            unsupported = cpu_bind(
                helion.kernel(backend="cute", static_shapes=False)(
                    split_paged_attention
                ),
                inputs,
            )
        assert not any(
            c.get("cute_gathered_page_stages") and c.get("cute_in_kernel_grid_init")
            for c in unsupported.config_spec.compiler_seed_configs
        )
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            ir = bound.host_function.device_ir
            merge_graph = ir.graphs[ir.root_ids[-1]].graph

            def merge_proof():
                return prove_split_weighted_merge(
                    merge_graph, ir.graphs, bound.env, *ir.grid_block_ids[-1]
                )

            assert merge_proof() is not None
            # Corrupt the update selected by the first loop's conditional.
            # This must decline even if the score and value shapes still match.
            max_loop = next(
                iter(
                    merge_graph.find_nodes(
                        op="call_function", target=_tracing_ops._for_loop
                    )
                )
            )
            cond_loop = ir.graphs[max_loop.args[0]].graph
            if_node = next(
                iter(cond_loop.find_nodes(op="call_function", target=_tracing_ops._if))
            )
            first_branch = ir.graphs[if_node.args[1]]
            first_max = next(
                iter(
                    first_branch.graph.find_nodes(
                        op="call_function", target=torch.ops.aten.maximum.default
                    )
                )
            )
            with patch.object(first_max, "target", torch.ops.aten.minimum.default):
                assert merge_proof() is None
            with patch.object(
                first_branch, "branches_outputs", [(0, first_branch.if_arg_names[0])]
            ):
                assert merge_proof() is None
            merge_div = next(
                iter(
                    merge_graph.find_nodes(
                        op="call_function", target=torch.ops.aten.div.Tensor
                    )
                )
            )
            with patch.object(merge_div, "target", torch.ops.aten.mul.Tensor):
                assert merge_proof() is None
            cond = next(
                iter(
                    merge_graph.find_nodes(
                        op="call_function", target=torch.ops.aten.eq.Scalar
                    )
                )
            )
            original_args = cond.args
            try:
                cond.args = (cond.args[0], 0)
                assert merge_proof() is None
            finally:
                cond.args = original_args
            producer = gathered_phase_region(ir, bound.env, ir.graphs)
            assert producer is not None
            pages = [
                (g, p)
                for g in ir.graphs
                if (p := prove_gathered_softmax(g.graph, ir.graphs, bound.env))
                is not None
            ]
            assert len(pages) == 1
            page, proof = pages[0]
            query, bank = (n.meta["val"] for n in (proof.qk.lhs, proof.qk.rhs))
            assert can_pipeline_page(bound.env, proof, 128, 128)
            # Seed admission sees the current arguments before the binding
            # snapshots its pointer facts. Do not advertise an unaligned seed.
            with patch.object(bound.env, "live_example_input", return_value=misaligned):
                assert not can_pipeline_page(
                    bound.env, proof, 128, 128, require_bound_pointer=False
                )
            assert not can_pipeline_page(bound.env, proof, 64, 128)
            assert not can_pipeline_page(bound.env, proof, 128, 64)
            # A two-element BF16 offset still permits existing 32-bit pairs,
            # but not a 16-byte async copy. Check every operand independently.
            for operand in (proof.qk.lhs, proof.qk.rhs, proof.value):
                saved = operand.meta["val"]
                with patch.dict(operand.meta, {"val": saved[..., 2:]}):
                    assert can_pack_bf16_k_pairs(bound.env, operand.meta["val"])
                    assert not can_pipeline_page(bound.env, proof, 128, 128)
            assert can_pack_bf16_k_pairs(bound.env, query)
            assert can_pack_bf16_k_pairs(bound.env, bank)
            alternatives = (query[:, :-1], query[:, 1:])
            joined = torch.empty_like(alternatives[0])
            bound.env.register_tensor_storage_join(joined, alternatives)
            assert not can_pack_bf16_k_pairs(bound.env, joined)
            assert not can_pack_bf16_k_pairs(bound.env, query[:, 1:])
            assert not can_pack_bf16_k_pairs(bound.env, query[:, ::2])
            assert not can_pack_bf16_k_pairs(
                bound.env, query.as_strided((16, 16), (33, 1))
            )
            with inductor_call_facts(
                bound,
                symbolic_input_layouts=True,
                symbolic_strides=tuple(bound.env.input_sources.values()),
            ):
                assert not can_pipeline_page(bound.env, proof, 128, 128)
                assert not can_pack_bf16_k_pairs(bound.env, query)
                assert not can_pack_bf16_k_pairs(bound.env, bank)
            nodes = list(page.graph.nodes)
            with ExitStack() as omitted:
                for n in nodes:
                    if n.target in (
                        torch.ops.aten.amax.default,
                        torch.ops.aten.sum.dim_IntList,
                    ):
                        omitted.enter_context(patch.dict(n.meta, lowering=None))
                assert gathered_phase_carrier(bound.env, ir, producer) is None

            # Each mutation breaks a different semantic edge. Restoring it
            # lets all six adversarial cases share one traced/compiled kernel.
            converted = next(
                n
                for n in nodes
                if n.target is torch.ops.prims.convert_element_type.default
            )
            safe = next(n for n in nodes if n.target is torch.ops.aten.eq.Scalar)
            choice = next(n for n in nodes if n.target is torch.ops.aten.lt.Tensor)
            loaded = next(n for n in nodes if n.args and n.args[0] is proof.value)
            cast_input, _ = converted.args
            near = next(
                n
                for n in nodes
                if n.target is torch.ops.aten.exp2.default and n is not cast_input
            )
            ssum = next(n for n in nodes if n.target is torch.ops.aten.sum.dim_IntList)
            for node, replacement in (
                (converted, (cast_input, torch.float16)),
                (safe, (safe.args[0], 0)),
                (choice, tuple(reversed(choice.args))),
                (
                    loaded,
                    (proof.value, [proof.logical, *loaded.args[1][1:]], None, None),
                ),
                (converted, (near, torch.bfloat16)),
                (ssum, (ssum.args[0], [0])),
            ):
                original = node.args
                try:
                    node.args = replacement
                    assert (
                        prove_gathered_softmax(page.graph, ir.graphs, bound.env) is None
                    )
                finally:
                    node.args = original
            assert bound.config_spec.cute_fragment_half_mma_root_ids == {ir.root_ids[0]}
            assert ir.root_ids[1] not in {g.graph_id for g in producer}
            for attribute, value in (
                ("implicit_dependency_starts", frozenset()),
                ("root_ids", list(reversed(ir.root_ids))),
                ("tile_dependency_graph", None),
            ):
                changed = copy.copy(ir)
                setattr(changed, attribute, value)
                assert gathered_phase_region(changed, bound.env, ir.graphs) is None
        group = next(
            g
            for g in bound.config_spec.compiler_coverage_groups
            if g.key == "cute_fragment_half_mma"
        )
        witness = next(w for w in group.witnesses if w.value is True)
        config = witness.carrier.config | {group.key: witness.value}
        # The search repairs incompatible variants without compiling them;
        # the positive witness survives normalization unchanged.
        assert (
            tuple(config["num_threads"])
            in bound.config_spec.cute_gathered_phase_search_lanes
        )
        with bound.env, bound.host_function:
            seeds = CuteGatheredBf16MmaSeed.get_seed_configs(bound.env, ir)
        # The ordinary seven remain available without a cooperative toolkit.
        # Exercise joint seed discovery without probing NVRTC on a CPU job.
        original = [s for s in seeds if not s.get("cute_in_kernel_grid_init")]
        extra = [s for s in seeds if s.get("cute_in_kernel_grid_init")]
        assert len(extra) == 1
        assert extra[0].config == original[-1].config | {
            "cute_in_kernel_grid_init": True
        }
        seeds = original
        assert len(seeds) == 7
        registered = [
            s
            for s in bound.config_spec.compiler_seed_configs
            if s.config.get("cute_fragment_half_mma")
            and not s.get("cute_in_kernel_grid_init")
        ]
        assert [s.config for s in registered] == [s.config for s in seeds]
        assert (
            seeds[0].config
            == CuteGatheredBf16MmaSeed.get_seed_config(bound.env, ir).config
        )
        assert [s.block_sizes[2] for s in seeds[1:]] == [128, 64, 256, 128, 128, 128]
        assert seeds[2].loop_orders[0] == [1, 0]
        assert not seeds[1].config.get("loop_orders")
        assert all(
            tuple(s.num_threads) in bound.config_spec.cute_gathered_phase_search_lanes
            for s in seeds
        )
        pipeline = seeds[-1]
        assert pipeline.config.get("cute_gathered_page_pipeline") is True
        assert pipeline.config.get("cute_gathered_page_stages") == 8
        pipelined = bound.to_code(pipeline)
        fast = cpu_bind(
            helion.kernel(backend="cute", static_shapes=False, fast_math=True)(
                split_paged_attention
            ),
            inputs,
        )
        fast_source = fast.to_code(pipeline)
        math_calls = {
            node.func.attr
            for node in ast.walk(ast.parse(fast_source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and any(
                kw.arg == "fast_math"
                and isinstance(kw.value, ast.Constant)
                and kw.value.value is True
                for kw in node.keywords
            )
        }
        assert {"gathered_softmax_page", "split_weighted_merge"} <= math_calls
        assert fast.host_function is not None
        with fast.env, fast.host_function:
            fast_ir = fast.host_function.device_ir
            root = fast_ir.graphs[fast_ir.root_ids[-1]].graph
            axes = fast_ir.grid_block_ids[-1]
            positive = prove_split_weighted_merge(root, fast_ir.graphs, fast.env, *axes)
            assert positive is not None
            # Losing the implicit zero mask is safe only for the matched
            # weight/product: the partial must still have the same live mask.
            load = next(
                n
                for g in fast_ir.graphs
                for n in g.graph.nodes
                if n.target is memory_ops.load and n.args[0] is positive.values
            )
            old_load_args = load.args
            try:
                load.args = (old_load_args[0], old_load_args[1], None, old_load_args[3])
                assert (
                    prove_split_weighted_merge(root, fast_ir.graphs, fast.env, *axes)
                    is None
                )
            finally:
                load.args = old_load_args

        def page_call(code: str) -> ast.Call:
            return next(
                n
                for n in ast.walk(ast.parse(code))
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "gathered_softmax_page"
            )

        alternate = bound.to_code(seeds[1])
        assert len(page_call(pipelined).args) == len(page_call(alternate).args) + 2
        assert isinstance(page_call(pipelined).args[-2], ast.Name)
        assert ast.literal_eval(page_call(pipelined).args[-1]) == 8
        # The producer owns the 128-wide V panel; it does not own the merge
        # tile width. The same eligible producer must survive unrelated tuning.
        different_merge = copy.deepcopy(pipeline.config)
        different_merge["block_sizes"][-1] = 256
        bound.config_spec.normalize(different_merge)
        other_call = page_call(bound.to_code(helion.Config.from_dict(different_merge)))
        assert ast.literal_eval(other_call.args[-1]) == 8
        with (
            patch(
                "helion._compiler.cute.computed_fragment.gathered_softmax_page",
                return_value=None,
            ),
            pytest.raises(
                helion.exc.InvalidConfig,
                match="did not emit the configured asynchronous",
            ),
        ):
            bound.to_code(pipeline)
        # A packed producer does not implicitly require the async slab: at a
        # tight budget its ordinary fused schedule still fits, but the
        # explicitly selected 8-stage ring alone already exceeds this limit.
        with patch(
            "helion._compiler.cute.target_facts.per_cta_smem_capacity_bytes",
            return_value=64 * 1024,
        ):
            with pytest.raises(helion.exc.InvalidConfig, match="shared bytes"):
                bound.to_code(pipeline)
            assert ".gathered_softmax_page(" in bound.to_code(seeds[1])
        invalid_pipeline = copy.deepcopy(
            seeds[2].config | {"cute_gathered_page_pipeline": True}
        )
        bound.config_spec.normalize(invalid_pipeline, _fix_invalid=True)
        assert not invalid_pipeline.get("cute_gathered_page_pipeline")
        assert "block=(128, 1, 1)" in alternate
        assert ".gathered_softmax_page(" in alternate
        assert ".split_weighted_merge(" in alternate
        page_update = next(
            n
            for n in ast.walk(ast.parse(alternate))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "gathered_softmax_page"
        )
        assert [ast.unparse(a) for a in page_update.args[4:7]] == [
            ast.unparse(a) for a in page_update.args[7:10]
        ]
        # The fully synchronized update is the end of the enclosing page
        # loop: no snapshot/copy-back or extra CTA barrier is necessary.
        assert any(
            isinstance(loop, ast.For)
            and isinstance(loop.body[-1], ast.Expr)
            and isinstance(loop.body[-1].value, ast.Call)
            and isinstance(loop.body[-1].value.func, ast.Attribute)
            and loop.body[-1].value.func.attr == "gathered_softmax_page"
            for loop in ast.walk(ast.parse(alternate))
        )
        assert seeds[1].pid_type == "persistent_interleaved"
        compatible = copy.deepcopy(config)
        bound.config_spec.normalize(compatible, _fix_invalid=True)
        assert compatible.get("cute_fragment_half_mma")
        for changes in (
            {"num_threads": [1, 1, 4, 1, 4, 1, 64]},
            {"cross_loop_pipeline": "static"},
        ):
            incompatible = copy.deepcopy(config) | changes
            bound.config_spec.normalize(incompatible, _fix_invalid=True)
            assert not incompatible.get("cute_fragment_half_mma")
        source = bound.to_code(helion.Config.from_dict(config))
        assert source.count("_launcher(") == 1
        assert source.count("_cute_grid_barrier(x_grid_sem.iterator)") == 1
        assert "block=(64, 2, 1)" in source
        assert ".split_weighted_merge(" not in source
        assert "q2d_stride_1" in source and "kv3_stride_2" in source
        # Every warp leaving a producer iteration (even on a false predicate)
        # rendezvous before the persistent CTA reuses its local storage.
        assert any(
            isinstance(node, ast.If)
            and node.body
            and ast.unparse(node.body[-1]) == "cute.arch.sync_threads()"
            and any(
                isinstance(n, ast.Attribute) and n.attr == "gathered_softmax_page"
                for n in ast.walk(node)
            )
            for node in ast.walk(ast.parse(source))
        )
        assert ".gathered_softmax_page(" in source
        assert "gathered_mma_page" not in source
        conflict = config | {"num_threads": [1, 1, 4, 1, 4, 1, 64]}
        with pytest.raises(
            helion.exc.InvalidConfig, match="needs 128 physical threads"
        ):
            bound.to_code(helion.Config.from_dict(conflict))


def test_fused_gathered_page_applies_the_limits_slab_offset() -> None:
    args = (
        torch.empty((64, 64), dtype=torch.bfloat16),
        torch.empty((8, 64, 64), dtype=torch.bfloat16),
        torch.empty((1, 8), dtype=torch.int32),
        torch.empty((2, 64), dtype=torch.int32),
        0.35,
    )
    with cpu_target():
        bound = cpu_bind(masked_page_stream, args)
        sizes = list(bound.config_spec.default_config().block_sizes)
        assert bound.host_function is not None
        with bound.env, bound.host_function:
            graphs = bound.host_function.device_ir.graphs
            page = next(
                proof
                for g in graphs
                if (proof := prove_gathered_softmax(g.graph, graphs, bound.env))
                is not None
            )
            for i, item in enumerate(bound.config_spec.block_sizes):
                if item.block_id == page.value_bid:
                    sizes[i] = 64
        cfg = helion.Config(
            block_sizes=sizes,
            cute_fragment_slab_loads=True,
            cute_fragment_half_mma=True,
        )
        source = bound.to_code(cfg)
        tree = ast.parse(source)
        call = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "gathered_softmax_page"
        )
        limit = ast.unparse(call.args[3])
        # The helper's limit argument must be the tile filled from the
        # second row of the slab, not the slab beginning at state[0, :].
        assert any(
            isinstance(store, ast.Assign)
            and isinstance(store.targets[0], ast.Subscript)
            and ast.unparse(store.targets[0].value) == limit
            and any(
                isinstance(read, ast.Subscript)
                and ast.unparse(read.value) != limit
                and ast.unparse(read.slice).startswith("1 * 64 +")
                for read in ast.walk(store.value)
            )
            for store in ast.walk(tree)
        )


def test_seed_pointer_offset_uses_each_aliased_dtype_in_bytes() -> None:
    # The FP32 owner begins 16 bytes into storage. The BF16 view of that
    # address has offset 8, while its owner has offset 4.
    owner = torch.empty(68, dtype=torch.float32)[4:]
    assert owner.data_ptr() % 16 == 0
    alias = owner.view(torch.bfloat16)
    env = Mock()
    env.input_sources = (owner,)
    registered = Mock()
    registered.sources = (env.tensor_input_source.return_value,)
    env.runtime_input_specializations = {
        _PERSISTENT_VEC_ALIGNMENT_SPECIALIZATION_KEY: registered
    }
    env.input_residues_hold_modulo.return_value = True
    env.live_example_input.return_value = owner
    assert _seed_base_is_aligned(env, alias, 16)
    assert not _seed_base_is_aligned(env, alias[4:], 16)

    # An unaligned input and an offset can cancel for this one example.
    # The binding only proves individually aligned input and view offsets.
    misaligned = torch.empty(32, dtype=torch.bfloat16)[1:]
    env.input_sources = (misaligned,)
    env.live_example_input.return_value = misaligned
    assert misaligned[7:].data_ptr() % 16 == 0
    assert not _seed_base_is_aligned(env, misaligned[7:], 16)

    # An int64 input may be viewed as BF16, but the binding does not
    # register an int64 input's pointer residue in its cache key.
    integer_owner = torch.empty(32, dtype=torch.int64)
    env.input_sources = (integer_owner,)
    env.live_example_input.return_value = integer_owner
    registered.sources = ()
    assert not _seed_base_is_aligned(env, integer_owner.view(torch.bfloat16), 16)
