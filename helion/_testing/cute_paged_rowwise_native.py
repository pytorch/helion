"""Bounded native source comparisons for the explicit rowwise recurrence."""

from __future__ import annotations

import ast
import copy
import functools
from itertools import starmap
from typing import TYPE_CHECKING
from typing import cast

import torch

import helion
from helion._testing.cuda_graph import cuda_graph_for_replay
from helion._testing.cute_folded_attention import folded_attention_args
from helion._testing.cute_folded_attention import folded_attention_reference
from helion._testing.cute_optional_lse import initialize_paged_attention_inputs
from helion._testing.cute_optional_lse import optional_lse_config
from helion._testing.cute_paged_rowwise_lazy import paged_query_first_direct_attention
from helion._testing.cute_source import transform_kernel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from helion.runtime.kernel import Kernel


ROW_CASES = (
    "original",
    "changed_q",
    "zero_maximum",
    "negative_maximum",
    "mixed_rows",
    "nonfinite",
    "zero_v_nan_q",
    "restored",
)


def rowwise_config(query_len: int, *, scoped: bool = False) -> helion.Config:
    config = optional_lse_config()
    config.config.update(
        block_sizes=[helion.next_power_of_2(query_len * 8)],
        cute_mma_dag_pass_warps="attention_scoped" if scoped else "attention_paired",
        cute_min_blocks_per_mp=2 if scoped else 1,
    )
    if not scoped:
        config.config.update(
            cute_mma_dag_probability_schedule="early",
            cute_mma_dag_query_tma=True,
            cute_mma_dag_value_metadata_skip=True,
            cute_mma_dag_v_lag=3,
        )
    return config


def configure_rowwise_case(args: tuple[object, ...], case: str) -> None:
    initialize_paged_attention_inputs(args)
    q, k, v, ks, vs, pages, lengths = (cast("torch.Tensor", x) for x in args[:7])
    if case in ("original", "restored"):
        return
    if case == "changed_q":
        q.neg_()
        pages.add_(2).remainder_(k.size(0))
        lengths.copy_(lengths.new_tensor([129, 0]))
    elif case == "zero_maximum":
        k.zero_()
        lengths.copy_(lengths.new_tensor([129, 257]))
    elif case == "negative_maximum":
        q.fill_(0.5)
        k.fill_(-1)
        lengths.copy_(lengths.new_tensor([129, 257]))
    elif case == "mixed_rows":
        q.zero_()
        q[..., 0] = torch.tensor([1.0, 16.0] * 8, device=q.device)
        k.zero_()
        for page in range(k.size(0)):
            k[page, :, :, 0] = 3 - page
        ks.fill_(127)
        pages.copy_(torch.arange(pages.size(1), device=pages.device).expand_as(pages))
        lengths.fill_(257)
    elif case == "nonfinite":
        k.fill_(1)
        q[0, 0, 0, 0] = float("nan")
        q[1, -1, 1, 0] = float("inf")
        lengths.fill_(257)
    elif case == "zero_v_nan_q":
        v.fill_(-128)
        vs.zero_()
        q[0, 0, 0, 0] = float("nan")
        lengths.fill_(257)
    else:
        raise AssertionError(case)


def rowwise_reference(
    args: tuple[object, ...], *, threshold: float = 2.0, headwide: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    return folded_attention_reference(
        args,
        reference_scores=True,
        reverse_pages=True,
        base2_scores=True,
        output_reciprocal=True,
        key_post_scale_log2=-6,
        value_post_scale_log2=-22,
        output_scale_log2=22,
        value_positive_zero=True,
        rowwise_rescale_threshold=threshold,
        headwide_rescale=headwide,
    )


def _backings(args: tuple[object, ...]) -> list[torch.Tensor]:
    result = []
    seen = set()
    for value in args[:7]:
        tensor = cast("torch.Tensor", value)
        storage = tensor.untyped_storage()
        identity = (storage.data_ptr(), storage.nbytes())
        if identity in seen:
            continue
        seen.add(identity)
        result.append(
            torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(
                storage, 0, (storage.nbytes(),), (1,)
            )
        )
    return result


def check_rowwise_native(
    query_len: int,
    device: torch.device,
    *,
    threshold: int | None = None,
    scoped: bool = False,
) -> None:
    args_list = list(folded_attention_args(query_len, device=device))
    args_list[7] = 2047
    args = tuple(args_list)
    configure_rowwise_case(args, "original")
    config = rowwise_config(query_len, scoped=scoped)
    kernel = paged_query_first_direct_attention
    if threshold is not None:
        kernel = configurable_rowwise_kernel()
        config.config["rescale_threshold"] = threshold
    selected_threshold = 2.0 if threshold is None else float(threshold)
    helper = "paged_attention_partial_body" if scoped else "paged_attention_paired_body"
    fragments = (f"rowwise_rescale_threshold={selected_threshold!r}", helper + "(")
    if scoped:
        fragments += ("scoped_registers=True",)
    check_paged_recurrence_native(
        args,
        kernel,
        config,
        functools.partial(rowwise_reference, threshold=selected_threshold),
        None,
        fragments,
    )


def check_paged_recurrence_native(
    args: tuple[object, ...],
    kernel: Kernel,
    config: helion.Config,
    reference: Callable[[tuple[object, ...]], tuple[torch.Tensor, torch.Tensor]],
    directory: Path | None,
    source_fragments: tuple[str, ...],
    *,
    configure: Callable[[tuple[object, ...], str], None] = configure_rowwise_case,
) -> None:
    """Compare eager/captured outputs; save optional headwide diagnostic records."""
    query_len = cast("torch.Tensor", args[0]).shape[1]
    bound = kernel.bind(args)
    source = bound.to_code(config)
    assert all(fragment in source for fragment in source_fragments)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "generated.py").write_text(source)
    compiled = bound.compile_config(config)
    backings = _backings(args)
    supplied = cast("torch.Tensor", args[10])
    output_storage = supplied.untyped_storage()
    output_backing = torch.empty(0, device=supplied.device, dtype=torch.uint8).set_(
        output_storage, 0, (output_storage.nbytes(),), (1,)
    )
    written = torch.zeros(
        output_storage.nbytes() // supplied.element_size(),
        dtype=torch.bool,
        device=supplied.device,
    )
    written.as_strided(
        supplied.shape, supplied.stride(), supplied.storage_offset()
    ).fill_(True)
    written_bytes = written.repeat_interleave(supplied.element_size()).cpu()
    record_index = 0

    def verify(
        name: str,
        call: Callable[[tuple[torch.Tensor, torch.Tensor]], tuple[torch.Tensor, ...]],
    ) -> None:
        nonlocal record_index
        expected = reference(args)
        supplied.fill_(float("nan"))
        supplied.masked_fill_(torch.isnan(expected[0]), 0.0)
        before = [
            x.cpu().clone() if directory is not None else x.clone() for x in backings
        ]
        output_before = output_backing.cpu().clone()
        actual = call(expected)
        torch.cuda.synchronize()
        snapshots = (
            [x.detach().cpu().clone() for x in actual]
            if directory is not None
            else actual
        )
        expected_snapshots = (
            [x.detach().cpu().clone() for x in expected]
            if directory is not None
            else expected
        )
        after = [x.cpu().clone() if directory is not None else x for x in backings]
        output_after = output_backing.cpu().clone()
        unchanged = list(starmap(torch.equal, zip(before, after, strict=True)))
        record = {
            "name": name,
            "query_len": query_len,
            "actual": snapshots,
            "expected": expected_snapshots,
            "inputs_unchanged": unchanged,
            "input_before": before,
            "input_after": after,
            "output_backing_before": output_before,
            "output_backing_after": output_after,
            "output_written_bytes": written_bytes,
        }
        if directory is not None:
            with (directory / f"{record_index:02d}-{name}.pt").open("xb") as stream:
                torch.save(record, stream)
        record_index += 1
        assert all(unchanged)
        assert torch.equal(output_before[~written_bytes], output_after[~written_bytes])
        for got, want in zip(snapshots, expected_snapshots, strict=True):
            torch.testing.assert_close(
                got, want, rtol=0.002, atol=0.002, equal_nan=True
            )

    verify("primer", lambda expected: compiled(*args))
    with cuda_graph_for_replay() as graph:
        before = [
            x.cpu().clone() if directory is not None else x.clone() for x in backings
        ]
        with torch.cuda.graph(graph):
            captured = compiled(*args)
        torch.cuda.synchronize()
        after = [x.cpu().clone() if directory is not None else x for x in backings]
        if directory is not None:
            with (directory / "capture-inputs.pt").open("xb") as stream:
                torch.save({"before": before, "after": after}, stream)
        assert all(starmap(torch.equal, zip(before, after, strict=True)))
        for case in ROW_CASES:
            configure(args, case)
            verify(case + "-eager", lambda expected: compiled(*args))

            def replay(
                expected: tuple[torch.Tensor, torch.Tensor],
            ) -> tuple[torch.Tensor, ...]:
                for output, reference in zip(captured, expected, strict=True):
                    output.fill_(float("nan"))
                    output.masked_fill_(torch.isnan(reference), 0.0)
                graph.replay()
                return captured

            verify(case + "-graph", replay)
    assert record_index == 17
    if directory is not None:
        (directory / "completed.txt").write_text(
            "17 full O/LSE source comparisons passed\n"
        )


class _ConstexprThreshold(ast.NodeTransformer):
    def __init__(self) -> None:
        self.replaced = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.args.args.append(
            ast.arg(
                arg="threshold",
                annotation=ast.Attribute(
                    ast.Name("hl", ast.Load()), "constexpr", ast.Load()
                ),
            )
        )
        node.args.defaults.append(ast.Constant(2.0))
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if (
            ast.unparse(node.left) == "tile_maximum"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Gt)
        ):
            assert (
                len(node.comparators) == 1
                and ast.literal_eval(node.comparators[0]) == 2.0
            )
            node.comparators = [ast.Name("threshold", ast.Load())]
            self.replaced += 1
        return node


@functools.cache
def constexpr_rowwise_kernel() -> Kernel:
    return _threshold_kernel(_ConstexprThreshold())


class _ConfigThreshold(_ConstexprThreshold):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.body.insert(
            0,
            ast.parse(
                'threshold = hl.register_tunable("rescale_threshold", '
                "helion.autotuner.EnumFragment((2, 16)))"
            ).body[0],
        )
        return node


@functools.cache
def configurable_rowwise_kernel() -> Kernel:
    return _threshold_kernel(_ConfigThreshold())


def _threshold_kernel(transform: _ConstexprThreshold) -> Kernel:
    result = transform_kernel(
        paged_query_first_direct_attention,
        transform,
        name=paged_query_first_direct_attention.fn.__name__,
        filename=f"<rowwise_{type(transform).__name__}>",
    )
    assert transform.replaced == 1
    return helion.kernel(
        result.fn, settings=copy.deepcopy(paged_query_first_direct_attention.settings)
    )


def rowwise_input_bytes(query_len: int) -> tuple[int, ...]:
    return ((2 * (query_len * 4608 + 256) + 8) * 2, 7 * 33792, 7 * 33792, 24, 8)


def validate_rowwise_record(record: dict, query_len: int) -> int:
    """Recheck complete saved source comparisons without a CUDA context."""
    assert record["query_len"] == query_len
    actual, expected = record["actual"], record["expected"]
    assert len(actual) == len(expected) == 2
    shapes = ((2, query_len, 16, 128), (2, query_len, 16))
    for got, want, shape, dtype in zip(
        actual, expected, shapes, (torch.bfloat16, torch.float32), strict=True
    ):
        assert got.device.type == want.device.type == "cpu"
        assert tuple(got.shape) == tuple(want.shape) == shape
        assert got.dtype == want.dtype == dtype
        torch.testing.assert_close(got, want, rtol=0.002, atol=0.002, equal_nan=True)
    before, after = record["input_before"], record["input_after"]
    assert len(before) == len(after) == len(record["inputs_unchanged"]) == 5
    assert record["inputs_unchanged"] == [True] * 5
    for a, b, count in zip(before, after, rowwise_input_bytes(query_len), strict=True):
        assert a.device.type == b.device.type == "cpu"
        assert a.dtype == b.dtype == torch.uint8
        assert tuple(a.shape) == tuple(b.shape) == (count,)
        assert torch.equal(a, b)
    if "output_written_bytes" in record:
        written = record["output_written_bytes"]
        before = record["output_backing_before"]
        after = record["output_backing_after"]
        assert (
            written.dtype == torch.bool and before.dtype == after.dtype == torch.uint8
        )
        assert written.shape == before.shape == after.shape
        assert torch.equal(before[~written], after[~written])
    return sum(x.numel() for x in actual)
