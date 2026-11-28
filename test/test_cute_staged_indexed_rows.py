from __future__ import annotations

import ast
from copy import deepcopy
import inspect
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.analysis.indexed_row_epilogue import _pure
from helion._compiler.cute.analysis.indexed_row_epilogue import indexed_row_epilogue
from helion._compiler.cute.analysis.indexed_row_phases import indexed_row_phases
from helion._compiler.cute.analysis.match_common import host_tensor_name
from helion._compiler.cute.autotune.heuristics.cute_staged_indexed_rows import (
    add_indexed_row_seed,
)
from helion._compiler.cute.indexed_row_epilogue import _arithmetic
from helion._compiler.cute.passes import launch_attributes
from helion._compiler.cute.thread_block_projection import update_launch_block
from helion._compiler.device_ir import codegen_node_order
from helion._testing import DEVICE
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
from helion._testing.cute_codegen import load_generated_module
from helion._testing.cute_staged_indexed_rows import selected_projection_pair
from helion._testing.cute_staged_indexed_rows import selected_projection_with_epilogue
import helion.language as hl
from helion.language import _tracing_ops
from helion.language import memory_ops

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def _select_rows(
    x: torch.Tensor,
    bank: torch.Tensor,
    route: torch.Tensor,
    score: torch.Tensor,
    row_tile: hl.constexpr = None,
) -> torch.Tensor:
    batch, slots, kx = x.shape
    groups, extent, k = bank.shape
    assert k == kx
    flat = bank.view(groups * extent, k)
    out = torch.empty((batch, slots, extent), device=x.device, dtype=x.dtype)
    # A real preceding root forces the unpaired projection to share a CTA.
    source = torch.empty_like(x)
    for b, s, col in hl.tile([batch, slots, k], block_size=[1, 1, 4]):
        source[b.begin, s.begin, col] = x[b.begin, s.begin, col] + 1
    for b, s, row in hl.tile([batch, slots, extent], block_size=[1, 1, row_tile]):
        r = route[b.begin, s.begin] * extent + row.index
        acc = hl.zeros([row], dtype=torch.float32)
        for tile in hl.tile(k):
            xv = source[b.begin, s.begin, tile].to(torch.float32)
            w = flat[r, tile].to(torch.float32)
            acc = acc + torch.sum(xv * w, dim=-1)
        out[b.begin, s.begin, row] = (acc * score[b.begin, s.begin]).to(out.dtype)
    return out


def test_indexed_projection_requires_matching_outer_and_inner_coordinates() -> None:
    args = (
        torch.empty((2, 3, 96), dtype=torch.bfloat16),
        torch.empty((7, 64, 96), dtype=torch.bfloat16),
        torch.empty((2, 3), dtype=torch.int32),
        torch.empty((2, 3)),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_select_rows, args)
        ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            (site,) = indexed_row_phases(bound.env, ir)
            assert (
                not site.invariant_rows
                and not bound.config_spec.cute_reusable_row_inputs
            )
            assert (site.k, site.rows, site.groups, site.paired) == (96, 64, 7, False)
            root = ir.graphs[site.root].graph
            (loop,) = root.find_nodes(op="call_function", target=_tracing_ops._for_loop)
            inner = ir.graphs[loop.args[0]].graph
            (matrix,) = [
                n
                for n in inner.nodes
                if n.target is memory_ops.load and n.meta["val"].ndim == 2
            ]
            before = matrix.args
            wrong_k = next(
                n for n in root.nodes if n.target is _tracing_ops._get_symnode
            )
            matrix.args = (before[0], [before[1][0], wrong_k], None, None)
            try:
                assert not indexed_row_phases(bound.env, ir)
            finally:
                matrix.args = before
            (st,) = root.find_nodes(op="call_function", target=memory_ops.store)
            before = st.args
            st.args = (
                before[0],
                [before[1][1], before[1][0], before[1][2]],
                before[2],
                None,
            )
            try:
                assert not indexed_row_phases(bound.env, ir)
            finally:
                st.args = before


def _fixed_row_config() -> helion.Config:
    return helion.Config(
        block_sizes=[64],
        num_threads=[2],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_staged_indexed_rows=True,
        cute_defer_tile_reduce=True,
    )


@pytest.mark.parametrize("row_tile", [8, 64])
def test_fixed_projection_row_tile_uses_its_declared_geometry(row_tile: int) -> None:
    args = (
        torch.empty((2, 3, 256), dtype=torch.bfloat16),
        torch.empty((7, 64, 256), dtype=torch.bfloat16),
        torch.empty((2, 3), dtype=torch.int32),
        torch.empty((2, 3)),
        row_tile,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_select_rows, args)
        candidate = _fixed_row_config()
        with bound.env:
            _, admitted = (
                bound.config_spec.create_config_generation().strict_config_pair(
                    candidate
                )
            )
        source = bound.to_code(admitted)
        assert ("_cute_staged_indexed_row_dot(" in source) == (row_tile == 64)
        # The same signed row must guard the matrix read and feed the wrapped address.
        (guard,) = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.IfExp)
            and isinstance(node.body, ast.Call)
            and isinstance(node.body.func, ast.Attribute)
            and node.body.func.attr == "load"
            and "flat.iterator" in ast.unparse(node.body)
        ]
        (wrapped,) = [n for n in ast.walk(guard.body) if isinstance(n, ast.IfExp)]
        raw = ast.unparse(wrapped.orelse)
        assert ast.unparse(wrapped) == f"{raw} + 448 if {raw} < 0 else {raw}"
        assert ast.unparse(guard.test) == f"{raw} >= -448 and {raw} < 448"
        assert ast.unparse(guard.orelse) == "cutlass.BFloat16(0)"
        if row_tile == 64:

            def replace_block(fn):
                assert (
                    update_launch_block(fn.codegen.host_statements, fn.name, (64, 1, 1))
                    == 1
                )

            with (
                patch.object(launch_attributes, "project_launch_block", replace_block),
                pytest.raises(exc.InvalidConfig, match="exact final launch block"),
            ):
                bound.to_code(admitted)


@pytest.mark.parametrize("host_collision", [True, False])
def test_staged_call_resolves_to_its_import_with_user_names(
    tmp_path: Path, host_collision: bool
) -> None:
    base = "_cute_staged_indexed_row_dot"
    host, argument = (base, base + "_") if host_collision else (base + "_", base)
    source = inspect.getsource(selected_projection_pair.fn).split("\n", 1)[1]
    source = source.replace("selected_projection_pair", host).replace(
        "scores", argument
    )
    path = tmp_path / "authored_row.py"
    path.write_text(source)
    namespace = selected_projection_pair.fn.__globals__.copy()
    exec(compile(source, str(path), "exec"), namespace)
    op = helion.kernel(
        namespace[host], backend="cute", static_shapes=True, fast_math=True
    )
    args = (
        torch.empty(2, 256, dtype=torch.bfloat16),
        torch.empty(5, 64, 256, dtype=torch.bfloat16),
        torch.empty(5, 64, 64, dtype=torch.bfloat16),
        torch.empty(2, 3, dtype=torch.int32),
        torch.empty(2, 3),
        torch.empty(256, dtype=torch.bfloat16),
        3.0,
    )
    config = helion.Config(
        block_sizes=[16, 256, 64, 64],
        num_threads=[4, 32, 4, 2, 1],
        reduction_loops=[64],
        pid_type="persistent_interleaved",
        cross_loop_pipeline="barrier",
        cute_defer_tile_reduce=True,
        cute_staged_indexed_rows=True,
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(op, args)
        rendered = bound.to_code(config)
        path = tmp_path / "staged_collision_render.py"
        path.write_text(rendered)
        module = load_generated_module(path)
    tree = ast.parse(rendered)
    (alias,) = [
        a.asname
        for stmt in tree.body
        if isinstance(stmt, ast.ImportFrom)
        and stmt.module == "helion._compiler.cute.device.staged_bf16_rows"
        for a in stmt.names
        if a.name == "staged_row_dot"
    ]
    assert alias is not None
    # The import survives the later host definition, and a device argument
    # must not shadow it when CuTe traces the actual helper call.
    assert getattr(module, alias) is not getattr(module, host)
    assert alias != argument
    assert (
        sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == alias
            for node in ast.walk(tree)
        )
        == 2
    )


@pytest.mark.parametrize("shared_phase", [False, True])
def test_staging_cannot_expose_an_unguarded_bank_on_fallback(
    tmp_path: Path, shared_phase: bool
) -> None:
    source = inspect.getsource(_select_rows.fn).split("\n", 1)[1]
    if shared_phase:
        # A second independent output belongs to the same dependency phase.
        source = source.replace(
            "    source = torch.empty_like(x)",
            "    source = torch.empty_like(x)\n    other = torch.empty_like(x)",
        )
        source = source.replace(
            "    return out",
            """    for i in hl.tile(batch, block_size=1):
        other[i.begin, 0, 0] = x[i.begin, 0, 0]
    return out, other""",
        )
    else:
        source = source.replace("score", "_stage_indexed_score")
    path = tmp_path / "renamed_projection.py"
    path.write_text(source)
    namespace = _select_rows.fn.__globals__.copy()
    exec(compile(source, str(path), "exec"), namespace)
    op = helion.kernel(
        namespace["_select_rows"], backend="cute", static_shapes=True, fast_math=True
    )
    args = (
        torch.empty(2, 3, 256, dtype=torch.bfloat16),
        torch.empty(7, 64, 256, dtype=torch.bfloat16),
        torch.empty(2, 3, dtype=torch.int32),
        torch.empty(2, 3),
        64,  # Ordinary operand names stage this exact tile.
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(op, args)
        if shared_phase:
            assert any(len(p.roots) > 1 for p in bound.host_function.device_ir.phases)
            with pytest.raises(exc.InvalidConfig, match="scalar phase"):
                bound.to_code(_fixed_row_config())
        else:
            rendered = bound.to_code(_fixed_row_config())
            assert "_cute_staged_indexed_row_dot(" not in rendered
            (read,) = [
                ast.unparse(n)
                for n in ast.walk(ast.parse(rendered))
                if isinstance(n, ast.IfExp)
                and isinstance(n.body, ast.Call)
                and "flat.iterator" in ast.unparse(n.body)
            ]
            assert " + 448 if " in read and ">= -448 and " in read


@pytest.mark.cuda
@pytest.mark.parametrize("row_tile", [8, 64])
def test_fixed_rows_replay_signed_inputs_and_negative_bank_indices(
    row_tile: int,
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("BF16 warp MMA requires SM80+")
    rng = torch.Generator(device=DEVICE).manual_seed(612)
    x = torch.randint(-4, 5, (2, 3, 256), generator=rng, device=DEVICE).bfloat16()
    bank = torch.randint(-5, 6, (7, 64, 256), generator=rng, device=DEVICE).bfloat16()
    route = torch.tensor([[-1, 0, 1], [-7, 6, 4]], device=DEVICE, dtype=torch.int32)
    score = torch.ones_like(route, dtype=torch.float32)
    args = (x, bank, route, score, row_tile)
    call = _select_rows.bind(args).compile_config(_fixed_row_config())
    call(*args)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call(*args)
    for _ in range(2):
        graph.replay()
        expected = ((x.float() + 1).unsqueeze(-2) * bank[route.long()].float()).sum(-1)
        torch.testing.assert_close(out, expected.bfloat16(), rtol=0, atol=0)
        x.neg_()
        route.copy_(route.flip(1))


def test_unpaired_projection_seed_supplies_its_own_four_warps() -> None:
    args = (
        torch.empty((2, 3, 256), dtype=torch.bfloat16),
        torch.empty((7, 64, 256), dtype=torch.bfloat16),
        torch.empty((2, 3), dtype=torch.int32),
        torch.empty((2, 3)),
    )
    with cpu_render_target(fake_device_sm_count=148):
        bound = cpu_bind(_select_rows, args)
        with bound.env, bound.host_function:
            carrier = helion.Config(
                block_sizes=[64, 64],
                num_threads=[4, 4],
                pid_type="persistent_interleaved",
                cross_loop_pipeline="barrier",
            )
            original, candidate = add_indexed_row_seed(bound.env, [carrier])
            assert original is carrier
            gen = bound.config_spec.create_config_generation()
            flat, checked = gen.strict_config_pair(candidate)
            assert gen.canonicalize_flat(flat)[1].config == checked.config
        source = bound.to_code(checked)
    (launch,) = [
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_launcher"
    ]
    block = next(kw.value for kw in launch.keywords if kw.arg == "block")
    assert ast.literal_eval(block) == (4, 32, 1)
    assert source.count("= _cute_staged_indexed_row_dot(") == 1


def _gemma_kernel_args():
    from pretuned_kernels.megakernels.gemma4_a4b_moe import (
        _helion_aot_gemma4_a4b_moe_cuda_sm100 as aot,
    )
    from pretuned_kernels.megakernels.gemma4_a4b_moe import gemma4_a4b_moe as model
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode(allow_non_fake_inputs=True), torch.device("cpu"):
        args = tuple(
            torch.empty(shape, dtype=dtype) for shape, dtype in aot._TENSOR_SIGNATURES
        )
    return model.gemma4_a4b_moe, args + aot._STATIC_ARGS


@pytest.mark.slow
def test_shared_forward_keeps_a_scalar_seed_and_stages_both_complete_projections() -> (
    None
):
    # This model supplies two different owned outputs in one cooperative grid:
    # the pair publishes two metadata arrays, the other consumes both.
    model, args = _gemma_kernel_args()
    with cpu_render_target(
        fake_device_sm_count=148,
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    ):
        op = helion.kernel(
            backend="cute", fast_math=True, static_shapes=model.settings.static_shapes
        )(model.fn)
        bound = cpu_bind(op, args)
        with bound.env:
            generation = bound.config_spec.create_config_generation()
            seeds = [seed for _, seed in generation.seed_flat_config_pairs()]
            assert any(not seed.get("cute_staged_indexed_rows") for seed in seeds)
            (registered,) = [
                c
                for c in seeds
                if c.get("cute_staged_indexed_rows")
                and not c.get("cute_single_row_publication")
            ]
            flat, strict = generation.strict_config_pair(registered)
            assert generation.canonicalize_flat(flat)[1].config == strict.config
        source = bound.to_code(strict)
        assert source.count("= _cute_staged_indexed_row_dot(") == 2
        assert "_stage_indexed_inv" in source
        assert source.count("_stage_indexed_scratch = cute.arch.alloc_smem(") == 1

        fused, spread, reused, network = [
            c for c in seeds if c.get("cute_indexed_row_epilogue")
        ]
        # The independent row root may instead give one complete scalar row
        # to each persistent task. Its ordinary block/lane sizes are the
        # only differences; the epilogue and all other choices stay fixed.
        (row_info,) = bound.config_spec.cute_warp_row_reduction_roots.values()
        row = row_info[0]
        bs = list(fused.block_sizes)
        ts = list(fused.config["num_threads"])
        bs[bound.config_spec.block_sizes.block_id_to_index(row)] = 1
        ts[bound.config_spec.num_threads.block_id_to_index(row)] = 1
        assert spread.config == fused.config | {"block_sizes": bs, "num_threads": ts}
        _, strict_spread = generation.strict_config_pair(spread)
        assert strict_spread.config == spread.config
        spread_body = ast.parse(bound.to_code(strict_spread))
        owners = [
            n
            for n in ast.walk(spread_body)
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "cute.arch.warp_idx() == 0"
            and "warp_reduction_sum(" in ast.unparse(n)
        ]
        assert len(owners) == 1
        route = ast.unparse(owners[0])
        assert route.count("threads_in_group=32") == row_info[-1]
        assert route.count("cute.arch.lane_idx() == 0") == route.count(".store(") == 1
        assert reused.config == spread.config | {"cute_reuse_indexed_row_input": True}
        flat, checked = generation.strict_config_pair(reused)
        assert generation.canonicalize_flat(flat)[1].config == checked.config
        assert network.config == reused.config | {"cute_fragment_topk_network": True}
        network_flat, strict_network = generation.strict_config_pair(network)
        assert (
            generation.canonicalize_flat(network_flat)[1].config
            == strict_network.config
        )
        parsed = ast.parse(bound.to_code(checked))
        (device,) = [
            f
            for f in parsed.body
            if isinstance(f, ast.FunctionDef)
            and any(
                isinstance(n, ast.Name) and n.id == "_stage_indexed_inv"
                for n in ast.walk(f)
            )
        ]
        (i,) = [
            i
            for i, n in enumerate(device.body)
            if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id == "_stage_indexed_inv"
        ]
        assert isinstance(device.body[i + 1], ast.For)  # publish the normalized input
        assert ast.unparse(device.body[i + 2]) == "cute.arch.sync_threads()"
        phase = device.body[i + 3]
        assert isinstance(
            phase, ast.For
        ) and "_cute_staged_indexed_row_dot(" in ast.unparse(phase)
        assert "_stage_indexed_inv" not in ast.unparse(phase)
        (row_axis,) = bound.config_spec.cute_reusable_row_inputs
        # When every block owns at most one tile, there is no reuse.
        assert not any(c.get("cute_reuse_indexed_row_input") for c in seeds[:-2])
        for facts in ({row_axis: 1}, {}):
            with (
                patch.dict(
                    bound.config_spec.cute_reusable_row_inputs, facts, clear=True
                ),
                pytest.raises(exc.InvalidConfig, match="multiple row tiles"),
            ):
                bound.config_spec.normalize(dict(checked.config))
        # A wider consumer requires a different physical program. Refuse it
        # during config normalization, not as an ineffective search candidate.
        (axes,) = bound.config_spec.cute_indexed_row_epilogue_axes
        for axis, width in ((axes[0], 4), (axes[2], 2)):
            widened = dict(fused.config)
            blocks = list(fused.block_sizes)
            blocks[bound.config_spec.block_sizes.block_id_to_index(axis)] = width
            widened["block_sizes"] = blocks
            with pytest.raises(exc.InvalidConfig, match="independent pointwise"):
                bound.config_spec.normalize(widened.copy())
            bound.config_spec.normalize(widened, _fix_invalid=True)
            assert not widened.get("cute_indexed_row_epilogue")

        (parent,) = [
            c
            for c in seeds
            if c.get("cute_parallel_row_sum") and not c.get("cute_indexed_row_epilogue")
        ]
        assert fused.config == parent.config | {"cute_indexed_row_epilogue": True}
        flat, candidate = generation.strict_config_pair(fused)
        assert generation.canonicalize_flat(flat)[1].config == candidate.config
        emitted = bound.to_code(candidate)
        assert emitted.count("_cute_grid_barrier(x_grid_sem.iterator)") == 6
        assert emitted.count("(gate_up.iterator") == 4  # producer stores survive
        assert emitted.count("(activation_flat.iterator") == 2  # one per address path
        assert "if cute.arch.thread_idx()[1] == 0:" in emitted

        _assert_fused_recipe(bound.to_code(parent), emitted)

        # Failure to prove either physical ownership path must reject the
        # request even when the typed source matched it.
        with (
            patch(
                "helion._compiler.cute.indexed_row_epilogue.fuse_row_epilogue",
                return_value=False,
            ),
            pytest.raises(exc.InvalidConfig, match="cannot fuse row epilogue"),
        ):
            bound.to_code(candidate)
        # Each mutation invalidates a different part of the FX certificate.
        ir = bound.host_function.device_ir
        with bound.env, bound.host_function:
            paired, down = indexed_row_phases(bound.env, ir)
            proof = indexed_row_epilogue(bound.env, ir, paired)
            assert proof is not None and not indexed_row_epilogue(bound.env, ir, down)
            g = ir.graphs[proof.consumer].graph
            (store,) = g.find_nodes(op="call_function", target=memory_ops.store)
            lo, hi = g.find_nodes(op="call_function", target=memory_ops.load)
            index = hi.args[1][1]
            old = index.args
            try:
                index.args = (old[0], proof.columns - 1)
                assert not indexed_row_epilogue(bound.env, ir, paired)
            finally:
                index.args = old
            # Keep the exact fresh contiguous destination: changing its
            # storage identity in just one nested producer operand must reject.
            nested = next(
                n
                for gid, n in codegen_node_order(
                    ir.graphs, [paired.root], unreached=False
                )
                if gid != paired.root
                and host_tensor_name(n) == "expert_gate_up_flattened_weight"
            )
            with patch.dict(nested.meta, {"val": store.args[0].meta["val"]}):
                assert not indexed_row_epilogue(bound.env, ir, paired)
            with patch.object(
                ir,
                "phases",
                ir.phases[: ir.phase_for_root(ir.root_ids.index(paired.root)) + 2],
            ):
                assert not indexed_row_epilogue(bound.env, ir, paired)
            (op,) = g.find_nodes(op="call_function", target=torch.ops.aten.tanh.default)
            try:
                op.target = torch.ops.aten.cumsum.default
                assert not indexed_row_epilogue(bound.env, ir, paired)
            finally:
                op.target = torch.ops.aten.tanh.default


def _assert_fused_recipe(control: str, fused: str) -> None:
    """Check both original producer stores feed the unmodified consumer recipe."""

    def assignments(tree: ast.AST) -> dict[str, ast.expr]:
        return {
            n.targets[0].id: n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        }

    source = ast.parse(control)
    defs = assignments(source)
    tree = ast.parse(fused)
    # Each physical address path writes the two rounded per-row results and
    # produces the corresponding activation. Compare the statements in program
    # order, independently on both paths, including casts and fma.
    events = sorted(
        (n for n in ast.walk(tree) if isinstance(n, (ast.Expr, ast.Assign))),
        key=lambda n: n.lineno,
    )
    gates = []
    addresses = []
    values = []
    prefix = "_stage_epilogue_"
    branch_defs: set[str] = set()
    branches = 0
    for n in events:
        expr = ast.unparse(n.value)
        if (
            isinstance(n, ast.Expr)
            and expr.startswith("(gate_up.iterator")
            and ".store(" in expr
        ):
            assert isinstance(n.value, ast.Call)
            gates.append(ast.dump(n.value.args[0]))
            addresses.append(ast.unparse(n.value.func))
        elif (
            isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id.startswith(prefix)
        ):
            name = n.targets[0].id.removeprefix(prefix)
            assert name in defs and name not in branch_defs
            branch_defs.add(name)
            if len(gates) > len(values):
                # The original address determines which half feeds this input.
                original_load = ast.unparse(defs[name])
                assert (
                    "gate_up.iterator" in original_load and ".load()" in original_load
                )
                assert (" if " in original_load) == bool(len(values) % 2)
                assert ast.dump(n.value) == gates[len(values)]
                values.append(name)
            else:
                assert expr.replace(prefix, "") == ast.unparse(defs[name])
        elif (
            isinstance(n, ast.Expr)
            and expr.startswith("(activation_flat.iterator")
            and ".store(" in expr
        ):
            assert isinstance(n.value, ast.Call)
            assert (
                ast.unparse(n.value.func).replace("activation_flat", "gate_up")
                == addresses[-2]
            )
            (original,) = (
                a
                for a in ast.walk(source)
                if isinstance(a, ast.Call)
                and ast.unparse(a.func).startswith("(activation_flat.iterator")
                and ast.unparse(a.func).endswith(".store")
            )
            assert ast.unparse(n.value.args[0]).replace(prefix, "") == ast.unparse(
                original.args[0]
            )
            # The branch must define every original intermediate, even when
            # a missing assignment leaves the same final variable name.
            leaves = set(values[-2:])
            dependencies = set()
            pending = [original.args[0]]
            while pending:
                for dep in ast.walk(pending.pop()):
                    if (
                        isinstance(dep, ast.Name)
                        and dep.id in defs
                        and dep.id not in dependencies
                    ):
                        dependencies.add(dep.id)
                        if dep.id not in leaves:
                            pending.append(defs[dep.id])
            assert dependencies == branch_defs
            branch_defs.clear()
            branches += 1
    assert branches == 2 and not branch_defs
    assert len(gates) == len(values) == 4
    assert values[:2] == values[2:]


def test_host_selected_projection_input_keeps_the_original_grid_join(
    tmp_path: Path,
) -> None:
    # A host conditional has its own fake storage. One of its possible
    # inputs is exactly the written activation: fusing the epilogue would
    # race another CTA still reading that input.
    original = selected_projection_with_epilogue
    source = inspect.getsource(original.fn).split("\n", 1)[1]
    source = source.replace("    eps: float,", "    eps: float,\n    choose: int,")
    source = source.replace("activation = torch.empty(", "activation = torch.zeros(")
    source = source.replace(
        "    mid3 = activation.view",
        "    x = activation[:batches, :] if choose > 0 else x\n    mid3 = activation.view",
    )
    path = tmp_path / "conditional_projection.py"
    path.write_text(source)
    ns = original.fn.__globals__.copy()
    exec(compile(source, str(path), "exec"), ns)
    conditional = helion.kernel(
        ns[original.fn.__name__], backend="cute", static_shapes=True, fast_math=True
    )
    args = (
        torch.empty(1, 128, dtype=torch.bfloat16),
        torch.empty(5, 256, 128, dtype=torch.bfloat16),
        torch.empty(5, 128, 128, dtype=torch.bfloat16),
        torch.empty(1, 3, dtype=torch.int32),
        torch.empty(1, 3),
        torch.empty(128, dtype=torch.bfloat16),
        0.25,
    )
    with cpu_render_target(fake_device_sm_count=148):
        safe = cpu_bind(original, args).config_spec
        joined = cpu_bind(conditional, (*args, 1)).config_spec
        assert safe.cute_indexed_row_epilogue_axes
        assert not joined.cute_indexed_row_epilogue_axes
        assert joined.cute_staged_indexed_row_axes == safe.cute_staged_indexed_row_axes
        for spec in (safe, joined):
            assert "cute_indexed_row_epilogue" in {
                f[0] for f in spec.structural_fingerprint()
            }


def test_signed_numeric_epilogue_constants() -> None:
    for literal in ("-0.7978845608028654", "+3"):
        assert _arithmetic(ast.parse(literal, mode="eval").body, set())
    assert not _arithmetic(ast.parse("-x", mode="eval").body, {"x"})


def test_deep_pointwise_fx_keeps_binding_iterative() -> None:
    graph = torch.fx.Graph()
    low = graph.placeholder("lo")
    high = graph.placeholder("hi")
    expr = graph.call_function(torch.ops.aten.mul.Tensor, (low, high))
    for _ in range(1100):
        expr = graph.call_function(torch.ops.aten.tanh.default, (expr,))
    assert _pure(expr, (low, high))
    expr.target = torch.ops.aten.sinh.default
    assert not _pure(expr, (low, high))


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["end_after_down", "preceding_pair"])
def test_normal_seed_fuses_the_right_pair_without_an_unrelated_root(
    tmp_path: Path, variant: str
) -> None:
    model, args = _gemma_kernel_args()
    tree = ast.parse(inspect.getsource(model.fn))
    (body,) = tree.body
    assert isinstance(body, ast.FunctionDef)
    body.decorator_list.clear()
    target = (
        "expert_reduce_tile_m"
        if variant == "end_after_down"
        else "expert_gate_up_tile_"
    )
    pos = next(
        i
        for i, stmt in enumerate(body.body)
        if isinstance(stmt, ast.For) and target in ast.unparse(stmt.target)
    )
    if variant == "end_after_down":
        # Keep the consumer and its following dependency but omit the terminal RMS.
        body.body[pos:] = [ast.Return(ast.Name("expert_outputs", ast.Load()))]
    else:
        # The first projection must not claim the only shared scratch ahead of
        # the producer whose two rounded stores feed a proved consumer.
        body.body.insert(pos, deepcopy(body.body[pos]))
    path = tmp_path / "variant.py"
    path.write_text(ast.unparse(tree))
    namespace = model.fn.__globals__.copy()
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    with cpu_render_target(
        fake_device_sm_count=148,
        smem_budget_bytes=232448,
        reduction_smem_budget_bytes=232448,
    ):
        op = helion.kernel(
            backend="cute", fast_math=True, static_shapes=model.settings.static_shapes
        )(namespace[body.name])
        bound = cpu_bind(op, args)
        space = bound.config_spec.create_config_generation()
        fused = [
            c
            for _, c in space.seed_flat_config_pairs()
            if c.get("cute_indexed_row_epilogue")
        ]
        assert fused
        seed = fused[0]
        flat, candidate = space.strict_config_pair(seed)
        assert space.canonicalize_flat(flat)[1] == candidate
        assert "_stage_epilogue_" in bound.to_code(candidate)
        if variant == "preceding_pair":
            # An earlier candidate competes for the paired scratch allocation.
            # A prelude is legal only on the producer where reuse was installed.
            reused = next(c for c in fused if c.get("cute_reuse_indexed_row_input"))
            code = bound.to_code(reused)
            device = next(
                n
                for n in ast.parse(code).body
                if isinstance(n, ast.FunctionDef)
                and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
            )
            roots = [
                i
                for i, stmt in enumerate(device.body)
                if isinstance(stmt, ast.For)
                and "_stage_indexed_" not in ast.unparse(stmt.target)
            ]
            prep = [
                i
                for i, stmt in enumerate(device.body)
                if isinstance(stmt, ast.Assign)
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "_stage_indexed_inv"
            ]
            assert len(prep) == 1
            assert roots[3] < prep[0] < roots[4]
