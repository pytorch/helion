from __future__ import annotations

import ast
from unittest.mock import Mock

import pytest
import torch

import helion
from helion import exc
from helion._compiler.cute.active_phase_joins import participants
from helion._compiler.cute.autotune.config_keys import EARLY_INDEXED_INPUT_KEY as KEY
from helion._compiler.cute.early_indexed_input import _row_phase_participants
from helion._compiler.cute.early_indexed_input import add_seed
from helion._compiler.cute.early_indexed_input import has_known_effects
from helion._compiler.cute.early_indexed_input import launch_allocation
from helion._compiler.cute.early_indexed_input import reserve
from helion._compiler.device_function import TensorArg
from helion._testing.cute_codegen import cpu_bind
from helion._testing.cute_codegen import cpu_render_target
import helion.language as hl

pytestmark = pytest.mark.cpu_only


@helion.kernel(backend="cute", static_shapes=True, fast_math=True)
def _routed_rows(
    x,
    kernel_bank,
    ids,
    scores,
    scale,
    writable,
    eps: float,
    tail,
    limit,
    dynamic: hl.constexpr,
):
    b, width = x.shape
    groups, twice, _ = kernel_bank.shape
    slots = ids.size(1)
    matrix = kernel_bank.view(groups * twice, width)
    rows = twice // 2
    chosen = torch.empty_like(ids)
    vals = torch.empty_like(scores)
    adjusted = torch.empty_like(scores)
    result = torch.empty_like(tail)
    out = torch.empty((b * slots, twice), dtype=x.dtype, device=x.device)
    for s in hl.tile(slots, block_size=1):
        adjusted[0, s.begin] = scores[0, s.begin] + 1
        writable[0, s.begin] = scores[0, s.begin].to(x.dtype)
    for ib, slot, r in hl.tile([b, slots, rows], block_size=[1, 1, None]):
        bi = ib.begin
        si = slot.begin
        selected = ids[bi, si]
        lo = selected * twice + r.index
        hi = lo + rows
        full = x[bi, :].float()
        inv = torch.rsqrt(torch.mean(full * full, dim=-1) + eps)
        a = hl.zeros([r], dtype=torch.float32)
        h = hl.zeros([r], dtype=torch.float32)
        for k in hl.tile(width):
            v = (x[bi, k].float() * inv).to(x.dtype)
            normal = (v * scale[k]).to(x.dtype).float()
            a = a + torch.sum(matrix[lo, k].float() * normal, dim=-1)
            h = h + torch.sum(matrix[hi, k].float() * normal, dim=-1)
        out[bi * slots + si, r] = a.to(x.dtype)
        out[bi * slots + si, r.index + rows] = h.to(x.dtype)
        if r.begin == 0:
            chosen[bi, si] = selected
            vals[bi, si] = adjusted[bi, si]
    # Reading chosen keeps this final phase separate; only its trip count changes.
    for t in hl.tile(limit if dynamic else 37, block_size=1):
        result[t.begin] = tail[t.begin] + chosen[0, 0]
    return out, chosen, vals, result


def test_bound_alias_refuses_to_read_the_row_before_a_write() -> None:
    x = torch.empty((1, 256), dtype=torch.bfloat16)
    args = (
        x,
        torch.empty((5, 256, 256), dtype=x.dtype),
        torch.empty((1, 23), dtype=torch.int32),
        torch.empty(1, 23),
        torch.empty(256, dtype=x.dtype),
    )
    with cpu_render_target(fake_device_sm_count=148):
        tail = torch.empty(37)
        limit = torch.tensor(31, dtype=torch.int32)
        safe = cpu_bind(
            _routed_rows, (*args, torch.empty_like(x), 0.25, tail, limit, False)
        )
        alias = cpu_bind(_routed_rows, (*args, x[:, :23], 0.25, tail, limit, False))
        with safe.env:
            assert safe.config_spec.cute_early_indexed_axis is not None
            gen = safe.config_spec.create_config_generation()
            # The held-out kernel has no specialized seed producers;
            # apply the same ordinary refinement to its legal resident carrier.
            carrier = helion.Config(
                block_sizes=[16, 256],
                num_threads=[4, 32, 1],
                reduction_loops=[64],
                pid_type="persistent_interleaved",
                cross_loop_pipeline="barrier",
                cute_defer_tile_reduce=True,
                cute_staged_indexed_rows=True,
                cute_reuse_indexed_row_input=True,
                cute_staged_projection_grid=2,
            )
            prior, offered = add_seed(safe.config_spec, [carrier])
            assert prior is carrier
            assert offered.config == carrier.config | {KEY: True}
            flat, early = gen.strict_config_pair(offered)
            assert gen.canonicalize_flat(flat)[1] == early
            repaired = helion.Config(**(offered.config | {"block_sizes": [8, 256]}))
            safe.config_spec.normalize(repaired, _fix_invalid=True)
            assert not repaired.get("cute_reuse_indexed_row_input")
            assert not repaired.get(KEY)
            safe.config_spec.normalize(repaired)
        source = safe.to_code(early)
        assert "_NUM_SM >= 25" in source and "early_input_scratch" in source
        assert "torch.zeros((192 if _NUM_SM >= 25 else 1,)" in source
        # A device-dependent third phase widens both earlier static phases.
        # All CTAs keep the original count read; only two spare CTAs normalize.
        wide = cpu_bind(
            _routed_rows,
            (
                *args,
                torch.empty_like(x),
                0.25,
                tail,
                limit,
                True,
            ),
        )
        with wide.env:
            _, wide_early = (
                wide.config_spec.create_config_generation().strict_config_pair(
                    helion.Config.from_dict(carrier.config | {KEY: True})
                )
            )
        backend = wide.to_code(wide_early)
        (definition,) = [
            n
            for n in ast.parse(backend).body
            if isinstance(n, ast.FunctionDef)
            and any(ast.unparse(d) == "cute.kernel" for d in n.decorator_list)
        ]
        first = next(i for i, n in enumerate(definition.body) if isinstance(n, ast.For))
        loop = definition.body[first]
        assert isinstance(loop, ast.For)
        assert ast.unparse(loop.iter).startswith("range(cutlass.Int64(")
        assert participants([loop]) == 23
        assert "cutlass.Int64(limit[0])" in ast.unparse(definition.body[first - 1])
        assert participants(definition.body[: first + 1]) is None
        # These checks still run on every CTA; only the first row loop's
        # participant count may discount a recognized, unchanged scheduler trap.
        preheader = definition.body[:first]
        traps = [s for s in preheader if isinstance(s, ast.If)]
        assert len(traps) == 2 and all("trap;" in ast.unparse(s) for s in traps)
        model = Mock(arguments=[TensorArg("limit", limit, "limit")], preamble=traps)
        with wide.env:
            assert _row_phase_participants(model, [*preheader, loop]) == 23
            model.preamble = []
            assert _row_phase_participants(model, [*preheader, loop]) is None
            # Even a compiler preamble cannot hide other possible writes.
            changed = ast.If(
                test=traps[0].test,
                body=ast.parse("out[0] = 1").body,
                orelse=[],
            )
            model.preamble = [changed, *traps[1:]]
            assert (
                _row_phase_participants(
                    model, [changed if s is traps[0] else s for s in [*preheader, loop]]
                )
                is None
            )
        assert "block_idx()[0]) >= 23" in backend and "block_idx()[0]) < 25" in backend
        assert "early_input_scratch" in backend and "_NUM_SM >= 25" in backend
        with alias.env:
            generation = alias.config_spec.create_config_generation()
            assert not any(c.get(KEY) for _, c in generation.seed_flat_config_pairs())
            with pytest.raises(exc.InvalidConfig, match="immutable"):
                generation.strict_config_pair(early)


def test_only_a_marked_launch_can_supply_the_private_allocation() -> None:
    definition = ast.parse("def compute(x, sem): pass").body[0]
    assert isinstance(definition, ast.FunctionDef)
    sem = TensorArg(
        "sem",
        torch.empty(32, device=torch.device("meta"), dtype=torch.uint32),
        "torch.zeros((32,), device=x.device, dtype=torch.uint32)",
    )
    invocation = (
        f"_launcher(compute, (_NUM_SM,), data, {sem.host_str()}, block=(4,32,1))"
    )

    def bind(source: str, *, mark: bool = True):
        body = ast.parse(source).body
        body[0]._is_kernel_call = mark
        return launch_allocation(list(body), definition, sem, 128)

    assert ast.unparse(bind(invocation)) == sem.host_str()
    for changed in (
        invocation.replace(sem.host_str(), "caller_buffer"),
        invocation.replace("data, " + sem.host_str(), sem.host_str() + ", data"),
        invocation.replace("(_NUM_SM,)", "(2,)"),
        invocation.replace("4,32,1", "2,32,1"),
    ):
        with pytest.raises(exc.InvalidConfig, match="private allocation"):
            bind(changed)
    with pytest.raises(exc.InvalidConfig, match="marked"):
        bind(invocation, mark=False)
    assert reserve(96, 2816) == (96, 1536)
    assert reserve(1, 256) == (32, 192)


def test_foreign_and_mutating_calls_decline_even_when_not_on_the_load_path() -> None:
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    assert has_known_effects(graph.call_function(torch.ops.aten.tanh.default, (x,)))
    assert not has_known_effects(
        graph.call_function(torch.ops.aten.add_.Tensor, (x, x))
    )
    assert not has_known_effects(graph.call_function(torch.linalg.cholesky, (x,)))
