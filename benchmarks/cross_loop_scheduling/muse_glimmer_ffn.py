"""Muse-Glimmer FFN probe for global list scheduling.

The five mathematical stages and all intermediate dtypes match the existing
Muse-Glimmer split-K Helion baseline.  ``muse_glimmer_ffn_keyed`` changes only
the logical coordinates of gate/up work: tasks belonging to one activation
slice are adjacent and therefore form an explicit readiness key for the
reduction, activation, and down-projection stages.

The benchmark's primary comparison is the best independently configured
standalone Helion pipeline versus the persistent list-scheduled kernel.  A
forced-local persistent build is retained only to attribute any improvement to
global list placement.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import inspect
import json
import linecache
from pathlib import Path
import sys
import textwrap
from typing import TYPE_CHECKING
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pretuned_kernels._bench import bench_pre_captured_cudagraphs
from pretuned_kernels._bench import capture_cuda_graph
from pretuned_kernels._bench import thermal_warmup
import torch

import helion
from helion._compiler import cross_loop_scheduler
import helion.language as hl

if TYPE_CHECKING:
    from collections.abc import Callable

    from helion.runtime.kernel import CompiledConfig
    from helion.runtime.kernel import Kernel


HIDDEN = 6656
INTERMEDIATE = 19968
GATE_SPLITS = 16
ACTIVATION_SPLITS = 16


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def splitk_mm_flat(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    splits: int,
) -> torch.Tensor:
    """Production-order split-K BF16 GEMV used by the standalone baseline."""
    m, k = x.size()
    weight_k, n = weight_t.size()
    assert m == 1
    assert k == weight_k
    assert k % splits == 0
    splits = hl.specialize(splits)
    split_k = k // splits
    hl.specialize(split_k)
    split_x = x.view(splits, split_k)
    partial = torch.empty((splits, n), dtype=torch.float32, device=x.device)
    for tile_s, tile_n in hl.tile([splits, n], block_size=[1, None]):
        accumulator = hl.zeros([tile_s, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(split_k):
            lhs = split_x[tile_s, tile_k].view(tile_s, 1, tile_k)
            global_k = (
                tile_s.index[:, None, None] * split_k + tile_k.index[None, :, None]
            )
            rhs = weight_t[global_k, tile_n.index[None, None, :]]
            accumulator = torch.baddbmm(accumulator, lhs, rhs)
        partial[tile_s, tile_n] = accumulator.squeeze(1)
    return partial


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gate_up_splitk_keyed(
    x: torch.Tensor,
    weight_t: torch.Tensor,
    gate_splits: int,
    activation_splits: int,
) -> torch.Tensor:
    """Split-K gate/up with tasks ordered by their consumer readiness key."""
    m, hidden = x.size()
    weight_hidden, twice_intermediate = weight_t.size()
    intermediate = twice_intermediate // 2
    assert m == 1
    assert hidden == weight_hidden
    assert hidden % gate_splits == 0
    assert intermediate % activation_splits == 0
    gate_splits = hl.specialize(gate_splits)
    activation_splits = hl.specialize(activation_splits)
    split_hidden = hidden // gate_splits
    split_intermediate = intermediate // activation_splits
    hl.specialize(split_hidden)
    hl.specialize(split_intermediate)
    partial = torch.empty(
        (gate_splits, 2, activation_splits, split_intermediate),
        dtype=torch.float32,
        device=x.device,
    )
    for tile_a, tile_half, tile_n, tile_s in hl.tile(
        [activation_splits, 2, split_intermediate, gate_splits],
        block_size=[1, 1, None, 1],
    ):
        split_id = tile_s.begin
        output_n = (
            tile_half.begin * intermediate
            + tile_a.begin * split_intermediate
            + tile_n.index
        )
        accumulator = hl.zeros([1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(split_hidden):
            global_k = split_id * split_hidden + tile_k.index
            lhs = x[:, global_k]
            rhs = weight_t[global_k[:, None], output_n[None, :]]
            accumulator = torch.addmm(accumulator, lhs, rhs)
        partial[split_id, tile_half.begin, tile_a.begin, tile_n] = accumulator.squeeze(
            0
        )
    return partial


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def gate_up_reduce_keyed(
    partial: torch.Tensor,
    activation_splits: int,
) -> torch.Tensor:
    """Reduce gate/up partials in the same activation-slice coordinates."""
    gate_splits, halves, stored_activation_splits, split_intermediate = partial.size()
    assert halves == 2
    assert stored_activation_splits == activation_splits
    activation_splits = hl.specialize(activation_splits)
    hl.specialize(gate_splits)
    hl.specialize(split_intermediate)
    output = torch.empty(
        (2, activation_splits, split_intermediate),
        dtype=torch.bfloat16,
        device=partial.device,
    )
    for tile_a, tile_half, tile_n in hl.tile(
        [activation_splits, 2, split_intermediate], block_size=[1, 1, None]
    ):
        values = partial[:, tile_half, tile_a, tile_n]
        output[tile_half, tile_a, tile_n] = torch.sum(values, dim=0).to(output.dtype)
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def silu_and_mul_keyed(
    gate_up: torch.Tensor,
    activation_splits: int,
) -> torch.Tensor:
    """Produce exactly one readiness key for every down-projection K split."""
    halves, stored_activation_splits, split_intermediate = gate_up.size()
    assert halves == 2
    assert stored_activation_splits == activation_splits
    activation_splits = hl.specialize(activation_splits)
    hl.specialize(split_intermediate)
    output = torch.empty(
        (activation_splits, split_intermediate),
        device=gate_up.device,
        dtype=gate_up.dtype,
    )
    for tile_a in hl.tile(activation_splits, block_size=1):
        for tile_i in hl.tile(split_intermediate):
            gate_values = gate_up[0, tile_a, tile_i]
            up_values = gate_up[1, tile_a, tile_i]
            silu = gate_values.to(torch.float32) * torch.sigmoid(
                gate_values.to(torch.float32)
            )
            output[tile_a, tile_i] = silu.to(up_values.dtype) * up_values
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def silu_and_mul_flat(gate_up: torch.Tensor) -> torch.Tensor:
    """Production-order pointwise activation used by the standalone baseline."""
    m, twice_intermediate = gate_up.size()
    intermediate = twice_intermediate // 2
    output = torch.empty((m, intermediate), device=gate_up.device, dtype=gate_up.dtype)
    for tile_m, tile_i in hl.tile([m, intermediate], block_size=[1, None]):
        gate = gate_up[tile_m, tile_i]
        up = gate_up[tile_m, tile_i + intermediate]
        silu = gate.to(torch.float32) * torch.sigmoid(gate.to(torch.float32))
        output[tile_m, tile_i] = silu.to(up.dtype) * up
    return output


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def down_splitk_keyed(
    activation: torch.Tensor,
    weight_t: torch.Tensor,
    activation_splits: int,
) -> torch.Tensor:
    """Down-projection partials keyed by the matching activation slice."""
    stored_activation_splits, split_intermediate = activation.size()
    weight_intermediate, hidden = weight_t.size()
    assert stored_activation_splits == activation_splits
    assert activation_splits * split_intermediate == weight_intermediate
    activation_splits = hl.specialize(activation_splits)
    hl.specialize(split_intermediate)
    partial = torch.empty(
        (activation_splits, hidden), dtype=torch.float32, device=activation.device
    )
    for tile_a, tile_n in hl.tile([activation_splits, hidden], block_size=[1, None]):
        accumulator = hl.zeros([tile_a, 1, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(split_intermediate):
            lhs = activation[tile_a, tile_k].view(tile_a, 1, tile_k)
            global_k = (
                tile_a.index[:, None, None] * split_intermediate
                + tile_k.index[None, :, None]
            )
            rhs = weight_t[global_k, tile_n.index[None, None, :]]
            accumulator = torch.baddbmm(accumulator, lhs, rhs)
        partial[tile_a, tile_n] = accumulator.squeeze(1)
    return partial


@helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")
def splitk_reduce(partial: torch.Tensor) -> torch.Tensor:
    """Reduce FP32 split-K partials to the production BF16 output."""
    _, n = partial.size()
    output = torch.empty((1, n), dtype=torch.bfloat16, device=partial.device)
    for tile_n in hl.tile(n):
        values = partial[:, tile_n]
        output[:, tile_n] = torch.sum(values, dim=0, keepdim=True).to(output.dtype)
    return output


class _KernelWithFunction(Protocol):
    fn: object


@dataclasses.dataclass(frozen=True)
class _Invocation:
    prefix: str
    kernel: _KernelWithFunction
    arguments: dict[str, str]
    outputs: dict[str, str]


class _AssignedNames(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)


class _RenameNames(ast.NodeTransformer):
    def __init__(self, names: dict[str, str]) -> None:
        self.names = names

    def visit_Name(self, node: ast.Name) -> ast.Name:
        renamed = self.names.get(node.id)
        if renamed is None:
            return node
        return ast.copy_location(ast.Name(id=renamed, ctx=node.ctx), node)


def _inline_invocation(invocation: _Invocation) -> tuple[list[ast.stmt], list[ast.For]]:
    """Alpha-rename a component and retain each scheduled outer loop verbatim."""
    source = textwrap.dedent(inspect.getsource(invocation.kernel.fn))
    module = ast.parse(source)
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    function = functions[0]
    parameters = [argument.arg for argument in function.args.args]
    assert set(parameters) == set(invocation.arguments)

    assigned = _AssignedNames()
    for statement in function.body:
        assigned.visit(statement)
    rename = {
        name: invocation.outputs.get(name, f"__muse_{invocation.prefix}_{name}")
        for name in set(parameters) | assigned.names
    }
    transformer = _RenameNames(rename)
    preamble: list[ast.stmt] = [
        ast.Assign(
            targets=[ast.Name(id=rename[parameter], ctx=ast.Store())],
            value=ast.parse(invocation.arguments[parameter], mode="eval").body,
        )
        for parameter in parameters
    ]
    loops: list[ast.For] = []
    for statement in function.body:
        if isinstance(statement, ast.Return):
            continue
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        transformed = transformer.visit(ast.fix_missing_locations(statement))
        assert isinstance(transformed, ast.stmt)
        if isinstance(transformed, ast.For):
            loops.append(transformed)
        else:
            preamble.append(transformed)
    assert loops
    return preamble, loops


def _build_persistent_kernel() -> Kernel:
    events = (
        _Invocation(
            "gate_main",
            gate_up_splitk_keyed,
            {
                "x": "ff_input",
                "weight_t": "gate_up_weight_t",
                "gate_splits": "gate_splits",
                "activation_splits": "activation_splits",
            },
            {"partial": "gate_partial"},
        ),
        _Invocation(
            "gate_reduce",
            gate_up_reduce_keyed,
            {
                "partial": "gate_partial",
                "activation_splits": "activation_splits",
            },
            {"output": "gate_up"},
        ),
        _Invocation(
            "activation",
            silu_and_mul_keyed,
            {"gate_up": "gate_up", "activation_splits": "activation_splits"},
            {"output": "activation"},
        ),
        _Invocation(
            "down_main",
            down_splitk_keyed,
            {
                "activation": "activation",
                "weight_t": "down_weight_t",
                "activation_splits": "activation_splits",
            },
            {"partial": "down_partial"},
        ),
        _Invocation(
            "down_reduce",
            splitk_reduce,
            {"partial": "down_partial"},
            {"output": "down"},
        ),
    )
    preamble: list[ast.stmt] = []
    loops: list[ast.For] = []
    for event in events:
        event_preamble, event_loops = _inline_invocation(event)
        preamble.extend(event_preamble)
        loops.extend(event_loops)
    arguments = (
        "ff_input",
        "gate_up_weight_t",
        "down_weight_t",
        "gate_splits",
        "activation_splits",
    )
    function = ast.FunctionDef(
        name="muse_glimmer_ffn_keyed_source",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in arguments],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *preamble,
            *loops,
            ast.Return(
                value=ast.Tuple(
                    elts=[
                        ast.Name(id=name, ctx=ast.Load())
                        for name in (
                            "down",
                            "gate_partial",
                            "gate_up",
                            "activation",
                            "down_partial",
                        )
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    source = ast.unparse(module) + "\n"
    filename = "<muse_glimmer_ffn_keyed_source>"
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    namespace = {"torch": torch, "hl": hl, "__name__": "_muse_ffn_probe"}
    exec(compile(source, filename, "exec"), namespace)
    return helion.kernel(static_shapes=True, autotune_effort="none", backend="triton")(
        namespace["muse_glimmer_ffn_keyed_source"]
    )


MUSE_GLIMMER_FFN_KEYED = _build_persistent_kernel()


_STANDALONE_OVERRIDES: dict[str, dict[str, object]] = {
    "gate_main": {
        "block_sizes": [32, 256],
        "num_warps": 1,
        "num_stages": 2,
        "range_num_stages": [0, 1],
        "range_flattens": [None, True],
        "range_unroll_factors": [0, 0],
    },
    "gate_reduce": {
        "block_sizes": [64],
        "num_warps": 2,
        "num_stages": 3,
    },
    "activation": {
        "block_sizes": [256],
        "num_warps": 1,
        "num_stages": 1,
    },
    "down_main": {
        "block_sizes": [32, 256],
        "num_warps": 1,
        "num_stages": 4,
        "range_num_stages": [0, 2],
        "range_flattens": [None, False],
        "range_unroll_factors": [0, 1],
    },
    "down_reduce": {
        "block_sizes": [64],
        "num_warps": 16,
        "num_stages": 1,
    },
}

_MATCHED_STANDALONE_OVERRIDES: dict[str, dict[str, object]] = {
    "gate_main": {
        "block_sizes": [32, 128],
        "num_warps": 1,
        "num_stages": 2,
    },
    "gate_reduce": {
        "block_sizes": [64],
        "num_warps": 1,
        "num_stages": 2,
    },
    "activation": {
        "block_sizes": [256],
        "num_warps": 1,
        "num_stages": 2,
    },
    "down_main": {
        "block_sizes": [32, 128],
        "num_warps": 1,
        "num_stages": 2,
    },
    "down_reduce": {
        "block_sizes": [64],
        "num_warps": 1,
        "num_stages": 2,
    },
}


def _compile(
    kernel: Kernel,
    args: tuple[object, ...],
    overrides: dict[str, object],
) -> tuple[CompiledConfig, dict[str, object]]:
    bound = kernel.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(overrides)
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    return bound.compile_config(config), dict(config)


def _compile_persistent(
    args: tuple[object, ...],
    *,
    multiplier: int,
    global_list: bool,
    gate_block_n: int,
    gate_block_k: int,
    down_block_k: int,
    maxnreg: int | None,
) -> tuple[CompiledConfig, tuple[dict[str, object], ...], dict[str, object]]:
    bound = MUSE_GLIMMER_FFN_KEYED.bind(args)
    values = dict(bound.config_spec.default_config())
    values.update(
        {
            "block_sizes": [
                gate_block_n,
                gate_block_k,
                64,
                256,
                32,
                down_block_k,
                64,
            ],
            "num_warps": 1,
            "num_stages": 2,
            "pid_type": "persistent_blocked",
            "cross_loop_schedule": "static_pipeline",
            "num_sm_multiplier": multiplier,
            "maxnreg": maxnreg,
            # Complete one readiness key at a time.  The first source axis is
            # the fastest-varying PID coordinate, so putting the activation
            # slice last makes it the outer grouping key for the two upstream
            # roots and the down-projection root.
            "loop_orders": [[2, 3, 1, 0], [2, 1, 0], [0, 1]],
        }
    )
    config = helion.Config.from_dict(values)
    bound.config_spec.normalize(config.config)
    original = cross_loop_scheduler._global_unit_list_schedule
    records: list[dict[str, object]] = []

    def traced(*call_args: object, **call_kwargs: object):  # noqa: ANN202
        result = original(*call_args, **call_kwargs)
        readiness_graph = call_args[0]
        source_schedule = call_args[1]
        records.append(
            {
                "returned_schedule": result is not None,
                "changed": result is not None and result != source_schedule,
                "root_sizes": [domain.size for domain in readiness_graph.root_domains],
                "source_stage_root": (
                    None
                    if (
                        source_segment
                        := cross_loop_scheduler._source_ticket_schedule_segment(
                            source_schedule
                        )
                    )
                    is None
                    else source_segment.root
                ),
                "readiness_counters": [repr(plan) for plan in call_args[2]],
                "root_barrier_edges": sorted(call_args[3]),
                "input_segments": len(source_schedule.segments),
                "output_segments": None if result is None else len(result.segments),
                "segments": None
                if result is None
                else [
                    {
                        "root": segment.root,
                        "tasks": segment.task_count,
                        "worker_begin": segment.worker_begin,
                        "worker_count": segment.worker_count,
                    }
                    for segment in result.segments
                ],
            }
        )
        return result

    try:
        cross_loop_scheduler._global_unit_list_schedule = (
            traced if global_list else lambda *unused_args, **unused_kwargs: None
        )
        bound._compile_cache.clear()
        compiled = bound.compile_config(config)
    finally:
        cross_loop_scheduler._global_unit_list_schedule = original
    return compiled, tuple(records), dict(config)


def _make_inputs(seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "ff_input": torch.randn((1, HIDDEN), device="cuda", dtype=torch.bfloat16),
        "gate_up_weight_t": (
            torch.randn((2 * INTERMEDIATE, HIDDEN), device="cuda", dtype=torch.bfloat16)
            * (0.5 / HIDDEN**0.5)
        ).T,
        "down_weight_t": (
            torch.randn((HIDDEN, INTERMEDIATE), device="cuda", dtype=torch.bfloat16)
            * (0.5 / INTERMEDIATE**0.5)
        ).T,
    }


def _canonical_outputs(
    outputs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    down, gate_partial, gate_up, activation, down_partial = outputs
    return (
        down.view(1, HIDDEN),
        gate_partial.view(GATE_SPLITS, 2 * INTERMEDIATE),
        gate_up.view(1, 2 * INTERMEDIATE),
        activation.view(1, INTERMEDIATE),
        down_partial.view(ACTIVATION_SPLITS, HIDDEN),
    )


def _standalone_pipeline(
    tensors: dict[str, torch.Tensor],
    *,
    keyed: bool,
    matched: bool,
) -> tuple[Callable[[], tuple[torch.Tensor, ...]], dict[str, object]]:
    configs: dict[str, object] = {}
    overrides = _MATCHED_STANDALONE_OVERRIDES if matched else _STANDALONE_OVERRIDES

    def build(name: str, kernel: Kernel, args: tuple[object, ...]) -> CompiledConfig:
        compiled, config = _compile(kernel, args, overrides[name])
        configs[name] = config
        return compiled

    gate_kernel = gate_up_splitk_keyed if keyed else splitk_mm_flat
    gate_args = (
        (
            tensors["ff_input"],
            tensors["gate_up_weight_t"],
            GATE_SPLITS,
            ACTIVATION_SPLITS,
        )
        if keyed
        else (tensors["ff_input"], tensors["gate_up_weight_t"], GATE_SPLITS)
    )
    gate_main = build("gate_main", gate_kernel, gate_args)
    gate_partial = gate_main(*gate_args)
    gate_reduce_kernel = gate_up_reduce_keyed if keyed else splitk_reduce
    gate_reduce_args = (gate_partial, ACTIVATION_SPLITS) if keyed else (gate_partial,)
    gate_reduce = build("gate_reduce", gate_reduce_kernel, gate_reduce_args)
    gate_up = gate_reduce(*gate_reduce_args)
    activation_kernel_source = silu_and_mul_keyed if keyed else silu_and_mul_flat
    activation_args = (gate_up, ACTIVATION_SPLITS) if keyed else (gate_up,)
    activation_kernel = build("activation", activation_kernel_source, activation_args)
    activation = activation_kernel(*activation_args)
    down_args = (activation, tensors["down_weight_t"], ACTIVATION_SPLITS)
    down_kernel = down_splitk_keyed if keyed else splitk_mm_flat
    down_main = build("down_main", down_kernel, down_args)
    down_partial = down_main(*down_args)
    down_reduce_args = (down_partial,)
    down_reduce = build("down_reduce", splitk_reduce, down_reduce_args)
    down_reduce(*down_reduce_args)
    torch.cuda.synchronize()

    def launch() -> tuple[torch.Tensor, ...]:
        current_gate_partial = gate_main(*gate_args)
        current_gate_up = gate_reduce(
            *(
                (current_gate_partial, ACTIVATION_SPLITS)
                if keyed
                else (current_gate_partial,)
            )
        )
        current_activation = activation_kernel(
            *((current_gate_up, ACTIVATION_SPLITS) if keyed else (current_gate_up,))
        )
        current_down_partial = down_main(
            current_activation, tensors["down_weight_t"], ACTIVATION_SPLITS
        )
        current_down = down_reduce(current_down_partial)
        return _canonical_outputs(
            (
                current_down,
                current_gate_partial,
                current_gate_up,
                current_activation,
                current_down_partial,
            )
        )

    return launch, configs


@torch.inference_mode()
def benchmark(
    *,
    multipliers: tuple[int, ...],
    gate_block_ns: tuple[int, ...],
    gate_block_ks: tuple[int, ...],
    down_block_ks: tuple[int, ...],
    maxnregs: tuple[int, ...],
    repetitions: int,
    warmup_ms: int,
    seed: int,
) -> dict[str, object]:
    if torch.cuda.get_device_capability()[0] < 10:
        raise RuntimeError("Muse-Glimmer probe requires an SM100 GPU")
    tensors = _make_inputs(seed)
    args = (
        tensors["ff_input"],
        tensors["gate_up_weight_t"],
        tensors["down_weight_t"],
        GATE_SPLITS,
        ACTIVATION_SPLITS,
    )
    standalone_production, standalone_production_configs = _standalone_pipeline(
        tensors, keyed=False, matched=False
    )
    standalone_keyed_tuned, standalone_keyed_tuned_configs = _standalone_pipeline(
        tensors, keyed=True, matched=False
    )
    standalone_matched, standalone_matched_configs = _standalone_pipeline(
        tensors, keyed=True, matched=True
    )
    production_outputs = standalone_production()
    tuned_outputs = standalone_keyed_tuned()
    matched_outputs = standalone_matched()
    for outputs in (tuned_outputs, matched_outputs):
        for value, expected in zip(outputs, production_outputs, strict=True):
            torch.testing.assert_close(value, expected, atol=5e-4, rtol=0.1)

    variants: list[tuple[str, CompiledConfig]] = []
    plans: dict[str, object] = {}
    configs: dict[str, object] = {}
    persistent_outputs: dict[str, tuple[torch.Tensor, ...]] = {}
    for multiplier in multipliers:
        for gate_block_n in gate_block_ns:
            for gate_block_k in gate_block_ks:
                for down_block_k in down_block_ks:
                    for maxnreg_value in maxnregs:
                        maxnreg = None if maxnreg_value == 0 else maxnreg_value
                        suffix = (
                            f"m{multiplier}_gn{gate_block_n}_gk{gate_block_k}_"
                            f"dk{down_block_k}_r{maxnreg_value or 'none'}"
                        )
                        for label, global_list in (
                            ("list", True),
                            ("local", False),
                        ):
                            name = f"persistent_{label}_{suffix}"
                            compiled, records, config = _compile_persistent(
                                args,
                                multiplier=multiplier,
                                global_list=global_list,
                                gate_block_n=gate_block_n,
                                gate_block_k=gate_block_k,
                                down_block_k=down_block_k,
                                maxnreg=maxnreg,
                            )
                            actual = _canonical_outputs(compiled(*args))
                            torch.cuda.synchronize()
                            for value, expected in zip(
                                actual, matched_outputs, strict=True
                            ):
                                torch.testing.assert_close(
                                    value, expected, atol=5e-4, rtol=0.1
                                )
                            persistent_outputs[name] = tuple(
                                value.clone() for value in actual
                            )
                            variants.append((name, compiled))
                            plans[name] = records
                            configs[name] = config

    for name, outputs in persistent_outputs.items():
        paired_name = name.replace("_list_", "_local_")
        if "_list_" not in name or paired_name not in persistent_outputs:
            continue
        for value, expected in zip(
            outputs, persistent_outputs[paired_name], strict=True
        ):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)

    standalone_production_graph, _ = capture_cuda_graph(standalone_production)
    standalone_keyed_tuned_graph, _ = capture_cuda_graph(standalone_keyed_tuned)
    standalone_matched_graph, _ = capture_cuda_graph(standalone_matched)
    persistent_graphs = [
        capture_cuda_graph(lambda compiled=compiled: compiled(*args))[0]
        for _name, compiled in variants
    ]
    thermal_warmup(warmup_ms)
    timings = bench_pre_captured_cudagraphs(
        (
            standalone_production_graph.replay,
            standalone_keyed_tuned_graph.replay,
            standalone_matched_graph.replay,
            *(graph.replay for graph in persistent_graphs),
        ),
        rep=repetitions,
    )
    return {
        "workload": "Muse-Glimmer-30B BF16 split-K FFN, batch 1",
        "device": torch.cuda.get_device_name(),
        "shape": {
            "hidden": HIDDEN,
            "intermediate": INTERMEDIATE,
            "gate_splits": GATE_SPLITS,
            "activation_splits": ACTIVATION_SPLITS,
        },
        "timings_us": {
            "standalone_production": timings[0] * 1000,
            "standalone_keyed_tuned": timings[1] * 1000,
            "standalone_matched": timings[2] * 1000,
            **{
                name: timing * 1000
                for (name, _compiled), timing in zip(variants, timings[3:], strict=True)
            },
        },
        "plans": plans,
        "persistent_configs": configs,
        "standalone_production_configs": standalone_production_configs,
        "standalone_keyed_tuned_configs": standalone_keyed_tuned_configs,
        "standalone_matched_configs": standalone_matched_configs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multipliers", type=int, nargs="+", default=(12,))
    parser.add_argument("--gate-block-n", type=int, nargs="+", default=(32,))
    parser.add_argument("--gate-block-k", type=int, nargs="+", default=(128,))
    parser.add_argument("--down-block-k", type=int, nargs="+", default=(128,))
    parser.add_argument("--maxnreg", type=int, nargs="+", default=(0,))
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--warmup-ms", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = benchmark(
        multipliers=tuple(args.multipliers),
        gate_block_ns=tuple(args.gate_block_n),
        gate_block_ks=tuple(args.gate_block_k),
        down_block_ks=tuple(args.down_block_k),
        maxnregs=tuple(args.maxnreg),
        repetitions=args.repetitions,
        warmup_ms=args.warmup_ms,
        seed=args.seed,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
