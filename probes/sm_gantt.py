# ruff: noqa: ANN001, ANN201, ANN202
"""Reusable per-SM tracing and stacked Gantt rendering for Helion probes."""

from __future__ import annotations

import ast
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import re
import statistics
from typing import Iterator

import torch
import triton

from helion._compiler.ast_extension import statement_from_string
from helion._compiler.program_id import ForEachProgramID
from helion._compiler.program_id import PersistentProgramIDs
from helion._compiler.program_id import ProgramIDs
from helion._compiler.program_id import _clone_ast_value
from helion._compiler.program_id import typed_program_id


TRACE_FIELDS = 3
TRACE_SEGMENTS = 8
TRACE_START = 0
TRACE_END = 1
TRACE_SM = 2
SINGLE_ROOT_TRACE_CAPACITY = 65_536
ROOT_HELPER = re.compile(r"tile_dependency_root_(\d+)(?:_scheduled_task)?$")
_TRACE_PADDING = itertools.count(1)

PALETTE = (
    "#2563EB",
    "#F97316",
    "#10B981",
    "#A855F7",
    "#E11D48",
    "#0891B2",
    "#CA8A04",
    "#4F46E5",
    "#DB2777",
    "#059669",
    "#EA580C",
    "#7C3AED",
    "#0F766E",
    "#B91C1C",
    "#65A30D",
    "#C026D3",
    "#0284C7",
    "#D97706",
    "#475569",
    "#84CC16",
)


@dataclass(frozen=True)
class TileInterval:
    root: int
    stage: str
    task: int
    segment: int
    sm: int
    start_ns: int
    end_ns: int


@dataclass
class TraceInstrumentation:
    task_counts: list[int] | None = None
    trace_numel: int | None = None
    instrumented_roots: set[int] | None = None

    def __post_init__(self) -> None:
        self.instrumented_roots = set()


@dataclass
class SingleRootInstrumentation:
    task_count: int | None = None
    task_capacity: int | None = None
    trace_numel: int | None = None


def _globaltimer_assignment(name: str) -> ast.stmt:
    return statement_from_string(
        f"{name} = tl.inline_asm_elementwise("
        "asm='mov.u64 $0, %globaltimer;', constraints='=l', args=[], "
        "dtype=tl.int64, is_pure=False, pack=1)"
    )


def _smid_assignment(name: str) -> ast.stmt:
    return statement_from_string(
        f"{name} = tl.inline_asm_elementwise("
        "asm='mov.u32 $0, %smid;', constraints='=r', args=[], "
        "dtype=tl.int32, is_pure=False, pack=1)"
    )


@contextmanager
def trace_cross_loop_roots() -> Iterator[TraceInstrumentation]:
    """Instrument every CLC root body without changing its task stream."""
    original_emit = ForEachProgramID._emit_cross_loop_schedule
    original_outline = ForEachProgramID._outline_cross_loop_region
    trace = TraceInstrumentation()
    task_counts_by_function: dict[int, list[int]] = {}
    shared_pid_by_function: dict[int, str] = {}
    trace_arg_by_function: dict[int, str] = {}
    segment_count_by_root: dict[tuple[int, int], int] = defaultdict(int)

    def traced_emit(self, strategy, device_function, *args, **kwargs):
        shared_pid = self.shared_pid_var
        if not isinstance(shared_pid, str):
            raise RuntimeError("cross-loop tracing requires a shared logical PID")
        shared_pid_by_function[id(device_function)] = shared_pid
        task_counts = self._static_case_task_counts(device_function)
        if task_counts is not None:
            task_counts = list(task_counts)
            task_counts_by_function[id(device_function)] = task_counts
            trace.task_counts = task_counts
            trace.trace_numel = (
                sum(task_counts) * TRACE_SEGMENTS * TRACE_FIELDS
                + next(_TRACE_PADDING)
            )
        return original_emit(self, strategy, device_function, *args, **kwargs)

    def traced_outline(
        device_function,
        *,
        name_hint,
        body,
        extra_argument_names=(),
        noinline=False,
    ):
        match = ROOT_HELPER.fullmatch(name_hint)
        if match is None:
            return original_outline(
                device_function,
                name_hint=name_hint,
                body=body,
                extra_argument_names=extra_argument_names,
                noinline=noinline,
            )

        root = int(match.group(1))
        task_counts = task_counts_by_function.get(id(device_function))
        if task_counts is None:
            raise RuntimeError("root tracing ran before task geometry was known")
        logical_pid = shared_pid_by_function.get(id(device_function))
        if logical_pid is None:
            raise RuntimeError("root tracing has no shared logical PID")
        traced_body = _clone_ast_value(body)
        if not isinstance(traced_body, list):
            raise RuntimeError("root body cloning did not produce a statement list")

        def find_pid_assignment(nodes: list[ast.stmt]):
            for index, statement in enumerate(nodes):
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == logical_pid
                    for target in statement.targets
                ):
                    return nodes, index
            for statement in nodes:
                for field in ("body", "orelse"):
                    child = getattr(statement, field, None)
                    if isinstance(child, list):
                        found = find_pid_assignment(child)
                        if found is not None:
                            return found
            return None

        pid_location = find_pid_assignment(traced_body)
        if pid_location is None:
            if trace.instrumented_roots is not None and root in trace.instrumented_roots:
                return original_outline(
                    device_function,
                    name_hint=name_hint,
                    body=body,
                    extra_argument_names=extra_argument_names,
                    noinline=noinline,
                )
            raise RuntimeError(f"root {root} has no logical-PID assignment")
        pid_body, pid_assignment = pid_location

        key = id(device_function)
        segment_key = (key, root)
        segment = segment_count_by_root[segment_key]
        segment_count_by_root[segment_key] += 1
        if segment >= TRACE_SEGMENTS:
            raise RuntimeError(
                f"root {root} has more than {TRACE_SEGMENTS} outlined segments"
            )
        trace_arg = trace_arg_by_function.get(key)
        if trace_arg is None:
            assert trace.trace_numel is not None
            trace_arg = ForEachProgramID._register_cross_loop_state(
                device_function,
                name_hint="tile_dependency_root_trace",
                numel=str(trace.trace_numel),
                dtype=torch.int64,
            )
            trace_arg_by_function[key] = trace_arg

        begin = device_function.new_var("tile_dependency_root_begin", dce=False)
        end = device_function.new_var("tile_dependency_root_end", dce=False)
        sm = device_function.new_var("tile_dependency_root_sm", dce=False)
        trace_pid = device_function.new_var("tile_dependency_root_pid", dce=False)
        slot = f"(({trace_pid}) * {TRACE_SEGMENTS} + {segment}) * {TRACE_FIELDS}"
        begin_statements = [
            statement_from_string(f"{trace_pid} = {logical_pid}"),
            _globaltimer_assignment(begin),
            _smid_assignment(sm),
            statement_from_string(
                f"tl.store({trace_arg} + ({slot}) + {TRACE_START}, {begin})"
            ),
            statement_from_string(
                f"tl.store({trace_arg} + ({slot}) + {TRACE_SM}, {sm})"
            ),
        ]
        end_statements = [
            _globaltimer_assignment(end),
            statement_from_string(
                f"tl.store({trace_arg} + ({slot}) + {TRACE_END}, {end})"
            ),
        ]
        pid_body[pid_assignment + 1 : pid_assignment + 1] = begin_statements
        pid_body.extend(end_statements)
        assert trace.instrumented_roots is not None
        trace.instrumented_roots.add(root)
        return original_outline(
            device_function,
            name_hint=name_hint,
            body=traced_body,
            extra_argument_names=extra_argument_names,
            noinline=noinline,
        )

    ForEachProgramID._emit_cross_loop_schedule = traced_emit
    ForEachProgramID._outline_cross_loop_region = staticmethod(traced_outline)
    try:
        yield trace
    finally:
        ForEachProgramID._emit_cross_loop_schedule = original_emit
        ForEachProgramID._outline_cross_loop_region = staticmethod(original_outline)


@contextmanager
def trace_single_root_program() -> Iterator[SingleRootInstrumentation]:
    """Trace each logical task in an ordinary one-root Helion kernel."""
    original_base_setup = ProgramIDs.setup_persistent_kernel
    original_setup = ForEachProgramID.setup_persistent_kernel
    original_persistent_setup = PersistentProgramIDs.setup_persistent_kernel
    trace = SingleRootInstrumentation()

    def register(device_function, strategy: ProgramIDs) -> str:
        if trace.trace_numel is not None:
            raise RuntimeError("single-root tracing registered more than once")
        axis_counts = [
            ForEachProgramID._static_block_axis_geometry(info.block_id, device_function)
            for info in strategy.pid_info
        ]
        static = all(geometry is not None for geometry in axis_counts)
        task_count = (
            math.prod(geometry[0] for geometry in axis_counts if geometry is not None)
            if static
            else None
        )
        task_capacity = task_count or SINGLE_ROOT_TRACE_CAPACITY
        trace_numel = task_capacity * TRACE_FIELDS + next(_TRACE_PADDING)
        trace_arg = ForEachProgramID._register_cross_loop_state(
            device_function,
            name_hint="standalone_root_trace",
            numel=str(trace_numel),
            dtype=torch.int64,
        )
        trace.task_count = task_count
        trace.task_capacity = task_capacity
        trace.trace_numel = trace_numel
        return trace_arg

    def statements(
        device_function, trace_arg: str, pid: str, *, begin: bool
    ) -> list[ast.stmt]:
        timestamp = device_function.new_var(
            "standalone_root_begin" if begin else "standalone_root_end",
            dce=False,
        )
        result = [_globaltimer_assignment(timestamp)]
        if begin:
            sm = device_function.new_var("standalone_root_sm", dce=False)
            result.extend(
                (
                    _smid_assignment(sm),
                    statement_from_string(
                        f"tl.store({trace_arg} + ({pid}) * {TRACE_FIELDS} + "
                        f"{TRACE_START}, {timestamp})"
                    ),
                    statement_from_string(
                        f"tl.store({trace_arg} + ({pid}) * {TRACE_FIELDS} + "
                        f"{TRACE_SM}, {sm})"
                    ),
                )
            )
        else:
            result.append(
                statement_from_string(
                    f"tl.store({trace_arg} + ({pid}) * {TRACE_FIELDS} + "
                    f"{TRACE_END}, {timestamp})"
                )
            )
        return result

    def traced_base_setup(self, device_function, total_pids_expr=None):
        emitted = original_base_setup(self, device_function, total_pids_expr)
        if trace.trace_numel is not None:
            return emitted
        if emitted is not None:
            raise RuntimeError("non-persistent ProgramID unexpectedly returned a body")
        trace_arg = register(device_function, self)
        pid = self.shared_pid_var or typed_program_id(0)
        device_function.body = [
            *statements(device_function, trace_arg, pid, begin=True),
            *device_function.body,
            *statements(device_function, trace_arg, pid, begin=False),
        ]
        return emitted

    def traced_persistent_setup(self, device_function, total_pids_expr=None):
        emitted = original_persistent_setup(self, device_function, total_pids_expr)
        if trace.trace_numel is not None:
            return emitted
        if emitted is None:
            raise RuntimeError("persistent ProgramID did not return its virtual loop")
        trace_arg = register(device_function, self)
        virtual_pid = self.virtual_pid_var
        matches = 0

        def rewrite(nodes: list[ast.stmt]) -> None:
            nonlocal matches
            for node in nodes:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == virtual_pid
                ):
                    node.body = [
                        *statements(device_function, trace_arg, virtual_pid, begin=True),
                        *node.body,
                        *statements(device_function, trace_arg, virtual_pid, begin=False),
                    ]
                    matches += 1
                    continue
                for field in ("body", "orelse"):
                    child = getattr(node, field, None)
                    if isinstance(child, list):
                        rewrite(child)

        rewrite(emitted)
        if matches != 1:
            raise RuntimeError(
                f"expected one persistent standalone task loop, found {matches}"
            )
        return emitted

    def traced_setup(self, device_function, total_pids_expr=None):
        task_counts = self._static_case_task_counts(device_function)
        if task_counts is None or len(task_counts) != 1:
            return original_setup(self, device_function, total_pids_expr)
        if trace.trace_numel is not None:
            raise RuntimeError("single-root tracing registered more than once")
        task_count = task_counts[0]
        trace_numel = task_count * TRACE_FIELDS + next(_TRACE_PADDING)
        trace_arg = ForEachProgramID._register_cross_loop_state(
            device_function,
            name_hint="standalone_root_trace",
            numel=str(trace_numel),
            dtype=torch.int64,
        )
        trace.task_count = task_count
        trace.task_capacity = task_count
        trace.trace_numel = trace_numel
        emitted = original_setup(self, device_function, total_pids_expr)
        if emitted is None:
            pid = typed_program_id(0)
            device_function.body = [
                *statements(device_function, trace_arg, pid, begin=True),
                *device_function.body,
                *statements(device_function, trace_arg, pid, begin=False),
            ]
            return emitted

        case = self.cases[0]
        parent = getattr(case, "parent_strategy", None)
        strategy = parent if parent is not None else case
        virtual_pid = getattr(strategy, "virtual_pid_var", None)
        if not isinstance(virtual_pid, str):
            raise RuntimeError("persistent standalone kernel has no virtual PID")
        matches = 0

        def rewrite(nodes: list[ast.stmt]) -> None:
            nonlocal matches
            for node in nodes:
                if (
                    isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name)
                    and node.target.id == virtual_pid
                ):
                    node.body = [
                        *statements(device_function, trace_arg, virtual_pid, begin=True),
                        *node.body,
                        *statements(device_function, trace_arg, virtual_pid, begin=False),
                    ]
                    matches += 1
                    continue
                for field in ("body", "orelse"):
                    child = getattr(node, field, None)
                    if isinstance(child, list):
                        rewrite(child)

        rewrite(emitted)
        if matches != 1:
            raise RuntimeError(
                f"expected one persistent standalone task loop, found {matches}"
            )
        return emitted

    ProgramIDs.setup_persistent_kernel = traced_base_setup
    PersistentProgramIDs.setup_persistent_kernel = traced_persistent_setup
    ForEachProgramID.setup_persistent_kernel = traced_setup
    try:
        yield trace
    finally:
        ProgramIDs.setup_persistent_kernel = original_base_setup
        PersistentProgramIDs.setup_persistent_kernel = original_persistent_setup
        ForEachProgramID.setup_persistent_kernel = original_setup


def persistent_traces(compiled, expected_numel: int) -> list[torch.Tensor]:
    matches: dict[int, torch.Tensor] = {}

    def inspect(value) -> None:
        namespace = getattr(value, "__dict__", {})
        for state in namespace.get("_helion_persistent_state_cache", {}).values():
            if state.dtype == torch.int64 and state.numel() == expected_numel:
                matches[state.data_ptr()] = state
        device_caches = getattr(value, "device_caches", None)
        if not device_caches or torch.cuda.current_device() not in device_caches:
            return
        cache = device_caches[torch.cuda.current_device()][0]
        for kernel in cache.values():
            for state in vars(kernel).get(
                "_helion_persistent_state_cache", {}
            ).values():
                if state.dtype == torch.int64 and state.numel() == expected_numel:
                    matches[state.data_ptr()] = state

    inspect(compiled)
    for value in list(getattr(compiled, "__globals__", {}).values()):
        inspect(value)
    if not matches:
        raise RuntimeError(
            f"expected at least one {expected_numel}-element trace, found none"
        )
    return list(matches.values())


class TracedCompiled:
    def __init__(
        self,
        compiled,
        *,
        stage_by_root: dict[int, str],
        task_counts: list[int],
        trace_numel: int,
        single_root: bool,
        root_by_stage: dict[str, int],
        infer_task_count: bool = False,
    ) -> None:
        self.compiled = compiled
        self.stage_by_root = stage_by_root
        self.task_counts = task_counts
        self.trace_numel = trace_numel
        self.single_root = single_root
        self.root_by_stage = root_by_stage
        self.infer_task_count = infer_task_count
        self.trace: torch.Tensor | None = None
        self.trace_candidates: list[torch.Tensor] = []

    def __call__(self, *args):
        return self.compiled(*args)

    def prepare_trace_replay(self) -> None:
        self.trace = None
        self.trace_candidates = persistent_traces(self.compiled, self.trace_numel)
        for candidate in self.trace_candidates:
            candidate.zero_()

    def select_populated_trace(self) -> None:
        populated = [
            candidate
            for candidate in self.trace_candidates
            if bool(torch.count_nonzero(candidate))
        ]
        if len(populated) != 1:
            raise RuntimeError(
                f"expected one populated trace for {self.stage_by_root}, found "
                f"{len(populated)} of {len(self.trace_candidates)}"
            )
        self.trace = populated[0]

    def intervals(self) -> list[TileInterval]:
        if self.trace is None:
            raise RuntimeError("traced standalone kernel has not run")
        if not self.single_root:
            local = collect_intervals(
                self.trace,
                self.task_counts,
                self.stage_by_root,
            )
            return [
                TileInterval(
                    root=self.root_by_stage[item.stage],
                    stage=item.stage,
                    task=item.task,
                    segment=item.segment,
                    sm=item.sm,
                    start_ns=item.start_ns,
                    end_ns=item.end_ns,
                )
                for item in local
            ]

        stage = self.stage_by_root[0]
        rows = self.trace[: self.task_counts[0] * TRACE_FIELDS].view(-1, 3)
        result: list[TileInterval] = []
        for task, row in enumerate(rows.cpu().tolist()):
            start_ns, end_ns, sm = map(int, row)
            if start_ns == end_ns == sm == 0 and self.infer_task_count:
                continue
            if not (end_ns > start_ns > 0):
                raise RuntimeError(
                    f"invalid standalone trace for {stage} task {task}: {row}"
                )
            result.append(
                TileInterval(
                    root=self.root_by_stage[stage],
                    stage=stage,
                    task=task,
                    segment=0,
                    sm=sm,
                    start_ns=start_ns,
                    end_ns=end_ns,
                )
            )
        return result


def compile_traced(
    bound,
    config,
    stage_by_root: dict[int, str],
    *,
    root_by_stage: dict[str, int],
) -> TracedCompiled:
    with trace_single_root_program() as single, trace_cross_loop_roots() as cross:
        compiled = bound.compile_config(config)
    if cross.instrumented_roots:
        if cross.task_counts is None or cross.trace_numel is None:
            raise RuntimeError("cross-loop trace metadata is incomplete")
        if set(stage_by_root) != set(range(len(cross.task_counts))):
            raise RuntimeError(
                f"stage map {sorted(stage_by_root)} does not match roots "
                f"{list(range(len(cross.task_counts)))}"
            )
        return TracedCompiled(
            compiled,
            stage_by_root=stage_by_root,
            task_counts=cross.task_counts,
            trace_numel=cross.trace_numel,
            single_root=False,
            root_by_stage=root_by_stage,
        )
    if single.task_capacity is None or single.trace_numel is None:
        raise RuntimeError("kernel did not use a traceable ProgramID schedule")
    if set(stage_by_root) != {0}:
        raise RuntimeError("single-root kernel requires one stage")
    return TracedCompiled(
        compiled,
        stage_by_root=stage_by_root,
        task_counts=[single.task_capacity],
        trace_numel=single.trace_numel,
        single_root=True,
        infer_task_count=single.task_count is None,
        root_by_stage=root_by_stage,
    )


def compile_traced_megakernel(bound, config, root_stages: dict[int, str]):
    untraced_lowered = bound.to_triton_code(config, output_origin_lines=True)
    with trace_cross_loop_roots() as instrumentation:
        compiled = bound.compile_config(config)
    if instrumentation.task_counts is None or instrumentation.trace_numel is None:
        raise RuntimeError("megakernel did not lower through the CLC scheduler")
    expected_roots = set(root_stages)
    if instrumentation.instrumented_roots != expected_roots:
        raise RuntimeError(
            f"instrumented roots {sorted(instrumentation.instrumented_roots or ())} "
            f"do not match expected roots {sorted(expected_roots)}"
        )
    if set(range(len(instrumentation.task_counts))) != expected_roots:
        raise RuntimeError(
            f"task counts have {len(instrumentation.task_counts)} roots, expected "
            f"{len(expected_roots)}"
        )
    with trace_cross_loop_roots():
        traced_lowered = bound.to_triton_code(config, output_origin_lines=True)
    return {
        "compiled": compiled,
        "task_counts": instrumentation.task_counts,
        "trace_numel": instrumentation.trace_numel,
        "untraced_lowered": untraced_lowered,
        "traced_lowered": traced_lowered,
        "config": dict(config),
    }


def clear_l2() -> int:
    driver = triton.runtime.driver.active
    buffer = driver.get_empty_cache_for_benchmark()
    size = buffer.numel() * buffer.element_size()
    expected = 256 * 1024 * 1024
    if size != expected:
        raise RuntimeError(f"expected a 256 MiB L2 flush buffer, got {size} bytes")
    driver.clear_cache(buffer)
    driver.get_device_interface().synchronize()
    return size


def capture_with_reset(fn, reset):
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(3):
            reset()
            output = fn()
        reset()
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        output = fn()
    torch.cuda.synchronize()
    return graph, output


def collect_intervals(
    raw_trace: torch.Tensor,
    task_counts: list[int],
    root_stages: dict[int, str],
) -> list[TileInterval]:
    expected_roots = set(range(len(task_counts)))
    if set(root_stages) != expected_roots:
        raise RuntimeError(
            f"stage-map roots {sorted(root_stages)} do not match {sorted(expected_roots)}"
        )
    record_count = sum(task_counts) * TRACE_SEGMENTS
    rows = raw_trace[: record_count * TRACE_FIELDS].view(
        -1, TRACE_SEGMENTS, TRACE_FIELDS
    )
    rows = rows.cpu().tolist()
    intervals: list[TileInterval] = []
    offset = 0
    for root, task_count in enumerate(task_counts):
        for task, segments in enumerate(rows[offset : offset + task_count]):
            valid_segments = 0
            for segment, row in enumerate(segments):
                start_ns, end_ns, sm = map(int, row)
                if start_ns == end_ns == sm == 0:
                    continue
                if not (end_ns > start_ns > 0):
                    raise RuntimeError(
                        f"invalid trace for root {root}, task {task}, "
                        f"segment {segment}: {row}"
                    )
                valid_segments += 1
                intervals.append(
                    TileInterval(
                        root=root,
                        stage=root_stages[root],
                        task=task,
                        segment=segment,
                        sm=sm,
                        start_ns=start_ns,
                        end_ns=end_ns,
                    )
                )
            if valid_segments == 0:
                raise RuntimeError(f"missing trace for root {root}, task {task}")
        offset += task_count
    return intervals


def trace_separate(graph, traced: list[TracedCompiled], reset):
    for item in traced:
        item.prepare_trace_replay()
    reset()
    l2_bytes = clear_l2()
    graph.replay()
    torch.cuda.synchronize()
    for item in traced:
        item.select_populated_trace()
    return [interval for item in traced for interval in item.intervals()], l2_bytes


def trace_megakernel(graph, compiled, trace_numel, task_counts, root_stages, reset):
    candidates = persistent_traces(compiled, trace_numel)
    for candidate in candidates:
        candidate.zero_()
    reset()
    l2_bytes = clear_l2()
    graph.replay()
    torch.cuda.synchronize()
    populated = [
        candidate for candidate in candidates if bool(torch.count_nonzero(candidate))
    ]
    if len(populated) != 1:
        raise RuntimeError(
            f"expected one populated megakernel trace, found "
            f"{len(populated)} of {len(candidates)}"
        )
    return collect_intervals(populated[0], task_counts, root_stages), l2_bytes


def merged_ranges(values: list[TileInterval]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for begin, end in sorted((item.start_ns, item.end_ns) for item in values):
        if not result or begin > result[-1][1]:
            result.append((begin, end))
        else:
            result[-1] = (result[-1][0], max(result[-1][1], end))
    return result


def summarize(intervals: list[TileInterval], stage_order: tuple[str, ...]):
    origin = min(item.start_ns for item in intervals)
    end = max(item.end_ns for item in intervals)
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    active_by_sm: dict[int, list[TileInterval]] = defaultdict(list)
    for item in intervals:
        active_by_sm[item.sm].append(item)
    active_ns = sum(
        finish - begin
        for values in active_by_sm.values()
        for begin, finish in merged_ranges(values)
    )
    stages = {}
    for stage in stage_order:
        values = [item for item in intervals if item.stage == stage]
        if not values:
            continue
        stages[stage] = {
            "logical_tasks": len({item.task for item in values}),
            "traced_segments": len(values),
            "first_start_us": (min(item.start_ns for item in values) - origin)
            / 1000.0,
            "last_end_us": (max(item.end_ns for item in values) - origin) / 1000.0,
            "active_union_us": sum(
                finish - begin for begin, finish in merged_ranges(values)
            )
            / 1000.0,
            "sm_time_us": sum(item.end_ns - item.start_ns for item in values)
            / 1000.0,
            "median_tile_us": statistics.median(
                (item.end_ns - item.start_ns) / 1000.0 for item in values
            ),
        }
    return {
        "span_us": (end - origin) / 1000.0,
        "sm_active_fraction": active_ns / ((end - origin) * sm_count),
        "tile_intervals": len(intervals),
        "stages": stages,
    }


def render_stacked_gantt(
    separate: list[TileInterval],
    megakernel: list[TileInterval],
    output: Path,
    *,
    title_text: str,
    stage_order: tuple[str, ...],
    separate_label: str,
    megakernel_label: str,
) -> None:
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFont

    if len(stage_order) > len(PALETTE):
        raise RuntimeError("not enough colors for stage legend")
    colors = dict(zip(stage_order, PALETTE, strict=False))
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    for values in (separate, megakernel):
        invalid = sorted({item.sm for item in values if not 0 <= item.sm < sm_count})
        if invalid:
            raise RuntimeError(f"trace contains invalid SM IDs: {invalid}")

    width = 2400
    left, right = 190, 55
    columns = 4
    legend_rows = math.ceil(len(stage_order) / columns)
    legend_top = 77
    legend_row_height = 39
    panel_top = legend_top + legend_rows * legend_row_height + 75
    panel_height = 720
    panel_gap = 285
    height = panel_top + 2 * panel_height + panel_gap + 90
    plot_width = width - left - right
    row_height = panel_height / sm_count
    max_us = max(
        (
            max(item.end_ns for item in values)
            - min(item.start_ns for item in values)
        )
        / 1000.0
        for values in (separate, megakernel)
    ) * 1.025

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    regular_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    regular = ImageFont.truetype(regular_path, 16)
    small = ImageFont.truetype(regular_path, 14)
    bold = ImageFont.truetype(bold_path, 20)
    title = ImageFont.truetype(bold_path, 30)
    box = draw.textbbox((0, 0), title_text, font=title)
    draw.text(((width - box[2]) / 2, 18), title_text, fill="#111827", font=title)

    cell_width = (width - 100) / columns
    for index, stage in enumerate(stage_order):
        row, column = divmod(index, columns)
        x = 52 + column * cell_width
        y = legend_top + row * legend_row_height
        draw.rectangle((x, y + 2, x + 20, y + 18), fill=colors[stage])
        draw.text((x + 29, y - 2), stage, fill="#111827", font=regular)

    panels = (
        (separate_label, separate),
        (megakernel_label, megakernel),
    )
    for panel_index, (label, values) in enumerate(panels):
        top = panel_top + panel_index * (panel_height + panel_gap)
        origin = min(item.start_ns for item in values)
        summary = summarize(values, stage_order)
        draw.text((left, top - 60), label, fill="#111827", font=bold)
        draw.text(
            (left, top - 31),
            f"instrumented tile span {summary['span_us']:.2f} us "
            f"(not CUDA-event latency); "
            f"{100 * float(summary['sm_active_fraction']):.1f}% SM-time occupied",
            fill="#4B5563",
            font=small,
        )
        for tick in range(11):
            tick_us = max_us * tick / 10
            x = left + plot_width * tick / 10
            draw.line((x, top, x, top + panel_height), fill="#E5E7EB")
            draw.text(
                (x - 16, top + panel_height + 8),
                f"{tick_us:.0f}",
                fill="#4B5563",
                font=regular,
            )
        for sm in range(0, sm_count, 16):
            y = top + sm * row_height
            draw.line((left, y, width - right, y), fill="#F3F4F6")
            draw.text((left - 49, y - 8), str(sm), fill="#6B7280", font=small)
        last_y = top + (sm_count - 1) * row_height
        draw.text(
            (left - 58, last_y - 8),
            str(sm_count - 1),
            fill="#6B7280",
            font=small,
        )

        by_sm: dict[int, list[TileInterval]] = defaultdict(list)
        for item in values:
            by_sm[item.sm].append(item)
        for sm, sm_values in by_sm.items():
            lane_ends: list[int] = []
            placements: list[tuple[TileInterval, int]] = []
            for item in sorted(
                sm_values, key=lambda value: (value.start_ns, value.end_ns)
            ):
                lane = next(
                    (
                        index
                        for index, lane_end in enumerate(lane_ends)
                        if lane_end <= item.start_ns
                    ),
                    len(lane_ends),
                )
                if lane == len(lane_ends):
                    lane_ends.append(item.end_ns)
                else:
                    lane_ends[lane] = item.end_ns
                placements.append((item, lane))
            lane_height = row_height / max(1, len(lane_ends))
            for item, lane in placements:
                x0 = left + (item.start_ns - origin) / 1000.0 / max_us * plot_width
                x1 = left + (item.end_ns - origin) / 1000.0 / max_us * plot_width
                y0 = top + sm * row_height + lane * lane_height
                y1 = y0 + max(0.7, lane_height * 0.88)
                draw.rectangle(
                    (x0, y0, max(x0 + 1.0, x1), y1),
                    fill=colors[item.stage],
                )
        draw.rectangle(
            (left, top, width - right, top + panel_height),
            outline="#6B7280",
            width=1,
        )
        draw.text((68, top + panel_height / 2), "SM ID", fill="#111827", font=bold)

    draw.text(
        (width / 2 - 125, height - 38),
        "time from first tile (us)",
        fill="#111827",
        font=bold,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def serialize_intervals(intervals: list[TileInterval]):
    return [
        {
            "root": item.root,
            "stage": item.stage,
            "task": item.task,
            "segment": item.segment,
            "sm": item.sm,
            "start_ns": item.start_ns,
            "end_ns": item.end_ns,
        }
        for item in intervals
    ]


def safe_name(value: str) -> str:
    return "".join(
        character.lower() if character.isalnum() else "_" for character in value
    ).strip("_")
