"""Which load or store each per-memory-op config slot names.

Code generation numbers the ``indexing`` and ``load_eviction_policies`` slots
in the order it lowers the loads and stores: a root's loads ahead of a nested
loop take their slots before the loop's, a config that rolls a reduction
lowers the loads its loop does not need ahead of the loop, and a load of a
``tile.index`` value or of a stack tensor takes no ``indexing`` slot.  The
memory-op facts the seeds and the descriptor guards read must name the same
loads, or a seed's policy lands on another tensor and a guard on another
input.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler.autotuner_heuristics.triton import TritonReductionHeuristic
from helion._compiler.device_ir import MEMORY_OP_FACT_META
from helion._compiler.device_ir import DeviceIR
from helion._compiler.device_ir import GraphInfo
from helion._testing import DEVICE
from helion._testing import cpu_render_target
from helion._testing import skipIfNotCUDA
from helion._testing import skipUnlessBackends
from helion._testing import skipUnlessTensorDescriptor
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

if TYPE_CHECKING:
    from helion.autotuner.config_spec import MemoryOpFact
    from helion.runtime.kernel import BoundKernel

pytestmark = pytest.mark.cpu_only


def _scale_then_loop(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        scale = w[tile_m]
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] * scale[:, None]
    return out


def _relu_and_spatial_mean(
    x: torch.Tensor, shift: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, rows, channels = x.shape
    act = torch.empty_like(x)
    pooled = torch.empty([batch, channels], dtype=x.dtype, device=x.device)
    for tile_n, tile_c in hl.tile([batch, channels]):
        y = torch.relu(
            x[tile_n, :, tile_c].to(torch.float32) - shift[tile_c][None, None, :]
        )
        act[tile_n, :, tile_c] = y.to(act.dtype)
        pooled[tile_n, tile_c] = torch.mean(y, dim=1).to(pooled.dtype)
    return act, pooled


def _scaled_softmax(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        row = x[tile_m, :].to(torch.float32) * scale[tile_m][:, None]
        top = torch.amax(row, -1, keepdim=True)
        e = torch.exp(row - top)
        out[tile_m, :] = (e / e.sum(-1, keepdim=True)).to(x.dtype)
    return out


def _double_until(x: torch.Tensor, limits: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        acc = x[row]
        iteration = hl.full([], 0, dtype=torch.int32)
        while iteration < limits[row]:
            acc = acc * 2
            iteration = iteration + 1
        out[row] = acc
    return out


def _slots(facts: list[MemoryOpFact]) -> list[str]:
    return [f"{fact.kind} {fact.tensor_name}" for fact in facts]


def _evict_first_tensors(code: str, names: tuple[str, ...]) -> set[str]:
    """The tensors of ``names`` read by the generated loads that carry the
    ``first`` eviction policy (Triton's ``evict_first``, CuTe's policy
    load)."""
    found: set[str] = set()
    for line in code.splitlines():
        if "evict_first" in line or "'first'" in line:
            found.update(name for name in names if re.search(rf"\b{name}\b", line))
    return found


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_load_ahead_of_a_nested_loop_takes_the_first_slot() -> None:
    # ``w`` is loaded in the root before the inner loop's graph, which the
    # graph list holds first: its slots are the first ones.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_scale_then_loop, static_shapes=True),
            (torch.empty(64, 128), torch.empty(64)),
        )
        spec = bound.config_spec
        assert _slots(spec.memory_op_facts) == ["load w", "load x", "store out"]
        assert [fact.eviction_index for fact in spec.memory_op_facts] == [0, 1, None]
        assert spec.store_indices == [2]
        code = bound.to_code(
            helion.Config(block_sizes=[16, 32], load_eviction_policies=["first", ""])
        )
    assert _evict_first_tensors(code, ("w", "x")) == {"w"}


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_while_condition_lowered_again_takes_slots_by_backend() -> None:
    # Codegen lowers the condition ahead of the loop and again at the end of
    # its body.  On Triton the second lowering takes the next slots, ahead of
    # the store's; with one slot for both, Triton could not render the
    # default config (the store asked for an ``indexing`` slot past the
    # list).  On CuTe a load lowered again keeps its eviction slot.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_double_until, static_shapes=True),
            (torch.empty(8), torch.empty(8, dtype=torch.int32)),
        )
        spec = bound.config_spec
        triton = bound.env.backend_name != "cute"
        slots = ["load x", "load limits", "store out"]
        if triton:
            slots.insert(2, "load limits")
        assert _slots(spec.memory_op_facts) == slots
        evictions = [0, 1, 2, None] if triton else [0, 1, None]
        assert [fact.eviction_index for fact in spec.memory_op_facts] == evictions
        bound.to_code(spec.default_config())
        policies = ["", "", "first"] if triton else ["", "first"]
        code = bound.to_code(helion.Config(load_eviction_policies=policies))
    limits_loads = [
        line
        for line in code.splitlines()
        if re.search(r"\blimits\b", line) and "load" in line and "src[" not in line
    ]
    assert [bool(_evict_first_tensors(line, ("limits",))) for line in limits_loads] == [
        not triton,
        True,
    ], limits_loads


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_rolled_reduction_numbers_its_slots_in_loop_order() -> None:
    # The persistent reduction loads ``x`` before ``shift``; rolling the
    # reduction lowers ``shift``, which its loop does not need, ahead of the
    # loop.  The facts follow the persistent order, with the loop graph the
    # roller registers after them; ``config_memory_op_facts`` follows each
    # config, and a policy placed through it on the hoisted load reaches it.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_relu_and_spatial_mean, static_shapes=True),
            (torch.empty(4, 96, 64, dtype=torch.bfloat16), torch.empty(64)),
        )
        spec = bound.config_spec
        device_ir = bound.host_function.device_ir
        assert _slots(spec.memory_op_facts) == [
            "load x",
            "load shift",
            "store act",
            "store pooled",
            "load x",
            "store act",
        ]
        configs = {
            rolled: spec.normalized_config(
                helion.Config(block_sizes=[1, 4], reduction_loops=[rolled])
            )
            for rolled in (None, 32)
        }
        with bound.env:
            # Rolling nothing, the code keeps the facts' slots: no codegen
            # graphs are built to read them.
            with patch.object(
                type(device_ir), "build_codegen_graphs", side_effect=AssertionError
            ):
                persistent = device_ir.config_memory_op_facts(configs[None])
            rolled = device_ir.config_memory_op_facts(configs[32])
        assert _slots(persistent) == [
            "load x",
            "load shift",
            "store act",
            "store pooled",
        ]
        assert _slots(rolled) == ["load shift", "load x", "store act", "store pooled"]
        assert [fact.eviction_index for fact in rolled] == [0, 1, None, None]
        assert [fact.indexing_index for fact in rolled] == [0, 1, 2, 3]
        policies = [""] * spec.load_eviction_policies.length
        policies[rolled[0].eviction_index] = "first"
        code = bound.to_code(
            helion.Config(**{**configs[32].config, "load_eviction_policies": policies})
        )
    assert _evict_first_tensors(code, ("x", "shift")) == {"shift"}


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_the_reread_seed_keeps_the_first_read_of_its_input() -> None:
    # ``x`` feeds both reductions: the reduction seeds keep its lines in L2
    # for the next read.  Rolled, the row is read in three loops, after the
    # per-row ``scale`` the roller lowers ahead of them, so the input's first
    # read is the second load slot, not its fact's first.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_scaled_softmax, static_shapes=True),
            (torch.empty(64, 1024, dtype=torch.bfloat16), torch.empty(64)),
        )
        spec = bound.config_spec
        x_fact = next(fact for fact in spec.memory_op_facts if fact.tensor_name == "x")
        assert x_fact.eviction_index == 0 and x_fact.reductions_fed
        seed = {"block_sizes": [1], "reduction_loops": [64]}
        with bound.env:
            policies = TritonReductionHeuristic._reread_eviction_policies(
                bound.env, bound.host_function.device_ir, seed, x_fact.eviction_index
            )
        assert policies is not None
        assert policies[:2] == ["first", "last"] and policies.count("last") == 1
        code = bound.to_code(
            spec.normalized_config(
                helion.Config(**seed, load_eviction_policies=policies)
            )
        )
    loads = [line for line in code.splitlines() if "tl.load(" in line]
    scale_loads = [line for line in loads if "tl.load(scale " in line]
    x_loads = [line for line in loads if "tl.load(x " in line]
    assert loads[0] == scale_loads[0] and len(x_loads) >= 2, loads
    assert ["evict_last" in line for line in x_loads] == [True] + [False] * (
        len(x_loads) - 1
    ), loads
    assert not any("evict_last" in line for line in scale_loads), loads


def _bias_then_matmul(
    x: torch.Tensor, y: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    _, n = y.shape
    out = torch.empty([m, n], dtype=x.dtype, device=x.device)
    for tile_m in hl.tile(m):
        b = bias[tile_m]
        for tile_n in hl.tile(n):
            acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
            for tile_k in hl.tile(k):
                acc = torch.addmm(acc, x[tile_m, tile_k], y[tile_k, tile_n])
            out[tile_m, tile_n] = (acc + b[:, None]).to(out.dtype)
    return out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_descriptor_guards_take_the_slots_codegen_numbers() -> None:
    # ``bias`` is loaded in the root ahead of the nested loops: the store
    # takes the fourth slot, and so does its descriptor guard.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        bound = cpu_bind(
            helion.kernel(_bias_then_matmul, static_shapes=False),
            (torch.empty(64, 32), torch.empty(32, 48), torch.empty(64)),
        )
    assert _slots(bound.config_spec.memory_op_facts) == [
        "load bias",
        "load x",
        "load y",
        "store out",
    ]
    assert _guarded_slots(bound) == {"x": [1], "y": [2]}


def _guarded_slots(bound: BoundKernel) -> dict[str, list[int]]:
    """The slots of each tensor's descriptor layout guard."""
    return {
        source.local_name: sorted(guard.memory_op_indices)
        for source, guard in bound.env.tensor_descriptor_layout_guards.items()
    }


def _rows_then_loop(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(0)):
        rows = tile_m.index[:, None]
        for tile_n in hl.tile(x.size(1)):
            out[tile_m, tile_n] = x[tile_m, tile_n] + rows
    return out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_tile_index_load_takes_no_indexing_slot() -> None:
    # The root's load of ``tile_m.index`` is lowered first, with an eviction
    # slot and no indexing slot: ``x``'s load takes indexing slot 0 and the
    # store slot 1, as a descriptor on slot 1 shows.  ``x``'s guard keys on
    # slot 0, so a column-major ``x`` binds apart from a row-major one; with
    # an indexing slot for the ``tile.index`` load, the guard sat on the
    # store's slot and both shared one kernel.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        kernel = helion.kernel(
            _rows_then_loop,
            static_shapes=False,
            config=helion.Config(
                block_sizes=[16, 32],
                indexing=["tensor_descriptor", "pointer", "pointer"],
            ),
        )
        row_major = kernel.bind((torch.empty(64, 128),))
        column_major = kernel.bind((torch.empty(128, 64).t(),))
        code = cpu_bind(
            helion.kernel(_rows_then_loop, static_shapes=True),
            (torch.empty(64, 128),),
        ).to_code(
            helion.Config(
                block_sizes=[16, 32],
                indexing=["pointer", "tensor_descriptor", "pointer"],
            )
        )
    facts = row_major.config_spec.memory_op_facts
    assert _slots(facts) == ["load None", "load x", "store out"]
    assert [fact.indexing_index for fact in facts] == [2, 0, 1]
    assert [fact.eviction_index for fact in facts] == [0, 1, None]
    assert _guarded_slots(row_major) == {"x": [0]}
    assert row_major is not column_major
    assert "out_desc.store(" in code


def _stacked_sum_then_add(
    dev_ptrs: torch.Tensor, example: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m, tile_n in hl.tile(x.size()):
        stacked = hl.stacktensor_like(example, dev_ptrs[:])
        total = stacked[tile_m].sum(0)
        out[tile_m, tile_n] = x[tile_m, tile_n] + total[:, None]
    return out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_stack_tensor_load_takes_no_indexing_slot() -> None:
    # As a ``tile.index`` load: the stack load lowered between the pointer
    # table's and ``x``'s takes no indexing slot, so ``x``'s is slot 1 and
    # the store's slot 2.
    args = (torch.zeros(4, dtype=torch.uint64), torch.empty(64), torch.empty(64, 32))
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        bound = cpu_bind(
            helion.kernel(_stacked_sum_then_add, static_shapes=False), args
        )
        code = cpu_bind(
            helion.kernel(_stacked_sum_then_add, static_shapes=True), args
        ).to_code(
            helion.Config(
                block_sizes=[16, 32],
                indexing=["pointer", "pointer", "tensor_descriptor", "pointer"],
            )
        )
    facts = bound.config_spec.memory_op_facts
    assert _slots(facts) == ["load dev_ptrs", "load None", "load x", "store out"]
    assert [fact.indexing_index for fact in facts] == [0, 3, 1, 2]
    assert _guarded_slots(bound) == {"x": [1]}
    assert "out_desc.store(" in code


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_config_whose_code_cannot_be_built_still_binds() -> None:
    # Guard discovery reads each candidate's descriptor slots.  A config
    # that rolls no reduction keeps the facts' slots, without building its
    # codegen graphs: subtiling ``pooled``'s store breaks this kernel's graph
    # order (bind failed with "used before it has been defined").  One that
    # rolls fails to build them, as its compile will, which autotuning
    # skips: bind still succeeds, and ``x``'s guard keys the first config.
    # The configs list their slots: a scalar "tensor_descriptor" makes every
    # load and store a descriptor's without building any graphs.
    descriptors = ["tensor_descriptor"] * 6
    configs = [
        helion.Config(block_sizes=[1, 4], indexing=descriptors),
        helion.Config(block_sizes=[1, 4], epilogue_subtile=2, indexing=descriptors),
        helion.Config(
            block_sizes=[1, 4],
            reduction_loops=[32],
            epilogue_subtile=2,
            indexing=descriptors,
        ),
    ]
    kernel = helion.kernel(_relu_and_spatial_mean, static_shapes=False, configs=configs)
    shift = torch.empty(64)
    failed_builds: list[helion.Config] = []
    original = DeviceIR.build_codegen_graphs

    def recording(
        self: DeviceIR, config: helion.Config, **kwargs: bool
    ) -> list[GraphInfo]:
        try:
            return original(self, config, **kwargs)
        except RuntimeError:
            failed_builds.append(config)
            raise

    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch.object(DeviceIR, "build_codegen_graphs", recording),
    ):
        contiguous = kernel.bind((torch.empty(4, 96, 64, dtype=torch.bfloat16), shift))
        transposed = kernel.bind(
            (torch.empty(4, 64, 96, dtype=torch.bfloat16).transpose(1, 2), shift)
        )
    assert contiguous is not transposed
    assert failed_builds and all(
        config.reduction_loops == [32] for config in failed_builds
    ), failed_builds


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_memory_op_facts_error_is_not_taken_for_a_failed_build() -> None:
    # Only building a candidate's codegen graphs may fail quietly: a pass that
    # drops a load's facts is a compiler bug, and binding reports it rather
    # than counting every load and store as a descriptor's.
    original = DeviceIR.build_codegen_graphs

    def dropping_facts(
        self: DeviceIR, config: helion.Config, **kwargs: bool
    ) -> list[GraphInfo]:
        graphs = original(self, config, **kwargs)
        for info in graphs:
            for node in info.graph.nodes:
                node.meta = {
                    key: value
                    for key, value in node.meta.items()
                    if key != MEMORY_OP_FACT_META
                }
        return graphs

    # A list: under the one-string form every load and store takes a
    # descriptor, and guard discovery reads no slots.
    config = helion.Config(
        block_sizes=[1, 4], reduction_loops=[32], indexing=["tensor_descriptor"] * 6
    )
    kernel = helion.kernel(
        _relu_and_spatial_mean, static_shapes=False, configs=[config]
    )
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
        patch.object(DeviceIR, "build_codegen_graphs", dropping_facts),
        pytest.raises(AssertionError, match="has no memory-op facts"),
    ):
        kernel.bind((torch.empty(4, 96, 64, dtype=torch.bfloat16), torch.empty(64)))


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_one_descriptor_string_builds_no_codegen_graphs() -> None:
    # Under ``indexing="tensor_descriptor"`` every load and store takes a
    # descriptor, in any order: guard discovery builds no codegen graphs of a
    # config that rolls a reduction to read its slots.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        bound = cpu_bind(
            helion.kernel(_relu_and_spatial_mean, static_shapes=False),
            (torch.empty(4, 96, 64, dtype=torch.bfloat16), torch.empty(64)),
        )
        config = bound.config_spec.normalized_config(
            helion.Config(
                block_sizes=[1, 4], reduction_loops=[32], indexing="tensor_descriptor"
            )
        )
        with patch.object(
            DeviceIR, "config_codegen_graphs", side_effect=AssertionError
        ) as built:
            descriptors = bound._descriptor_memory_ops(config)
    assert not built.called
    assert descriptors == frozenset(range(6))


def _bias_then_batches(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile_m in hl.tile(x.size(1)):
        b = bias[tile_m]
        for tile_b in hl.tile(x.size(0)):
            out[tile_b, tile_m, :] = x[tile_b, tile_m, :] * 2 + b[None, :, None]
    return out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_subtiled_store_reads_its_own_descriptor_slot() -> None:
    # ``bias`` is loaded ahead of the nested loop, so ``out``'s store takes
    # slot 2.  A descriptor there has the subtiling split the store along
    # its full slice, to keep the descriptor's block shape.
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        code = cpu_bind(
            helion.kernel(_bias_then_batches, static_shapes=True),
            (torch.empty(8, 64, 32), torch.empty(64)),
        ).to_code(
            helion.Config(
                block_sizes=[16, 4],
                epilogue_subtile=2,
                indexing=["pointer", "pointer", "tensor_descriptor"],
            )
        )
    columns = re.findall(r"out_desc\.store\(\[offset_1, offset_0, (\d+)\]", code)
    assert columns == ["0", "16"], code


def _softmax_rows_then_double(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    a = torch.empty_like(x)
    out = torch.empty_like(x)
    for tile_m in hl.tile(m):
        row = x[tile_m, :].to(torch.float32)
        top = torch.amax(row, -1, keepdim=True)
        total = torch.exp(row - top).sum(-1, keepdim=True)
        for tile_n in hl.tile(n):
            a[tile_m, tile_n] = (torch.exp(x[tile_m, tile_n] - top) / total).to(a.dtype)
    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = a[tile_m, tile_n] * 2
    return out


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
@helion._testing.skipIfTileIR(
    "TileIR has no barrier between loops: the second loop's read of the "
    "first one's output is a LoopDependencyError there"
)
def test_cross_root_scheduling_reads_a_rolled_configs_descriptor_slots() -> None:
    # Rolled, the row is read in two loops, so the store of ``a`` in the
    # nested loop, which the second root reads, moves from slot 2 to slot 3.
    # A descriptor there keeps that loop from publishing its tiles to the
    # second root: the first root's task publishes them once done.  Read at
    # slot 2, ``x``'s in this code, the store was a pointer store, and the
    # loop published after each descriptor store.
    with (
        cpu_render_target(fake_device_sm_count=148),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        code = cpu_bind(
            helion.kernel(_softmax_rows_then_double, static_shapes=True),
            (torch.empty(64, 256),),
        ).to_code(
            helion.Config(
                block_sizes=[16, 32, 16, 32],
                reduction_loops=[64],
                pid_type="persistent_blocked",
                cross_loop_pipeline="dynamic",
                indexing=["pointer"] * 3 + ["tensor_descriptor"] + ["pointer"] * 3,
            )
        )
    nested = code.split("def tile_dependency_root_0(")[1].split("\ndef ")[0]
    assert "a_desc.store(" in nested, code
    assert "sem='release'" not in nested, code


def _count_until(counter: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for row in hl.grid(x.size(0)):
        while hl.atomic_add(counter, [row], 1) < 4:
            x[row] = x[row] * 2
        out[row] = x[row]
    return out


@skipUnlessBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_an_atomic_in_a_while_condition_takes_slots_by_backend() -> None:
    # As a load there: on Triton the condition's second lowering takes the
    # next ``atomic_indexing`` slot.  With one slot the default config failed
    # ("Atomic operation 1 exceeds atomic_indexing config length 1").
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_count_until, static_shapes=True),
            (torch.zeros(8, dtype=torch.int32), torch.empty(8)),
        )
        spec = bound.config_spec
        assert spec.atomic_indexing.length == (
            2 if bound.env.backend_name != "cute" else 1
        )
        bound.to_code(spec.default_config())


# Rolled (block_sizes=[16], reduction_loops=[64]), the code loads ``scale``
# first and ``x`` second: the descriptor on slot 1 is ``x``'s.
_ROLLED_X_DESCRIPTOR = helion.Config(
    block_sizes=[16],
    reduction_loops=[64],
    indexing=["pointer", "tensor_descriptor"] + ["pointer"] * 5,
)


@skipUnlessBackends(["triton"])
@helion._testing.skipIfRefEager("inspects config slots and code; ref mode has neither")
def test_a_rolled_config_keys_on_its_descriptor_layouts() -> None:
    # ``x``'s descriptor guard holds its slots in the facts' order, where it
    # is not slot 1.  Read in the config's order, the guard is active, so a
    # column-major ``x`` binds apart from a row-major one; it shared the
    # row-major kernel, whose descriptor Triton refused for that layout.
    kernel = helion.kernel(
        _scaled_softmax, static_shapes=False, config=_ROLLED_X_DESCRIPTOR
    )
    scale = torch.empty(64)
    with (
        cpu_render_target(),
        patch("helion._compat._supports_tensor_descriptor", return_value=True),
    ):
        row_major = kernel.bind((torch.empty(64, 256), scale))
        column_major = kernel.bind((torch.empty(256, 64).t(), scale))
    assert row_major is not column_major


@pytest.mark.cuda
@skipIfNotCUDA()
@skipUnlessBackends(["triton"])
@skipUnlessTensorDescriptor("Tensor descriptor support is required")
def test_a_rolled_config_keys_on_its_descriptor_alignment() -> None:
    # A base 4 bytes off 16 shared the aligned input's kernel, whose
    # descriptor then failed with a misaligned address.
    kernel = helion.kernel(
        _scaled_softmax, static_shapes=True, config=_ROLLED_X_DESCRIPTOR
    )
    m, n = 64, 256
    scale = torch.rand(m, device=DEVICE) + 0.5
    x = torch.randn(m, n, device=DEVICE)
    misaligned = torch.randn(m * n + 4, device=DEVICE)[1 : 1 + m * n].view(m, n)
    for value in (x, misaligned):
        torch.testing.assert_close(
            kernel(value, scale), torch.softmax(value * scale[:, None], -1)
        )
