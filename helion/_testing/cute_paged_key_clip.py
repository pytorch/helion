"""Host mirror and replay checks for the paired loader's boundary key clips."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import cast

import torch

from .cuda_graph import cuda_graph_for_replay
from .cute_capture_replay import capture_launches
from .cute_capture_replay import fill_nan
from .cute_optional_lse import initialize_paged_attention_inputs
from .cute_optional_lse import optional_lse_reference

if TYPE_CHECKING:
    from collections.abc import Sequence

    import helion
    from helion.runtime.kernel import BoundKernel


def expected_key_shift(
    key_base: int,
    length: int,
    *,
    query_len: int,
    group: int,
    query_columns: int,
    window: int | None,
    key_clip: str,
    key_count: int = 128,
) -> int:
    """logical_attention_key_shift for coordinates that do not wrap."""
    if key_clip == "off":
        return 0
    query_begin = length - query_len
    query_end = query_begin + (query_columns - 1) // group
    key_end = key_base + key_count - 1
    leading = trailing = 0
    if key_clip != "leading":
        trailing = min(max(key_end - query_end, 0), key_count)
    if key_clip != "trailing":
        assert window is not None
        leading = min(max(query_begin - window - key_base, 0), key_count)
    leading = min(leading & -4, key_count - 4)
    trailing = min(trailing & -4, key_count - 4)
    return -trailing if trailing > leading else leading


def unloaded_keys(shift: int, key_count: int = 128) -> range:
    """Page keys a tile with this shift leaves to the hardware's zero fill."""
    return range(shift) if shift >= 0 else range(key_count + shift, key_count)


def poison_unloaded_keys(
    args: tuple[object, ...],
    form: str,
    *,
    key_clip: str,
    page_ranges: Sequence[tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Write nonfinite K and V to every key the clip leaves out.

    The tiles are those each request's current length selects, or its
    [first, last) pages in ``page_ranges``. The page table must give each
    logical page its own physical page, so no visible key is written.
    Returns the leading and trailing key counts.
    """
    q, k, v, ks, vs, pages, length = (cast("torch.Tensor", arg) for arg in args[:7])
    if form == "mapped":
        query_len, group = q.size(1), q.size(2) // k.size(2)
        window = cast("int | None", args[7])
    else:
        query_len, group = cast("int", args[7]), cast("int", args[8])
        window = cast("int", args[9]) if cast("int", args[9]) >= 0 else None
    rows = query_len * group
    query_columns = 8 if rows <= 8 else 32 if rows <= 32 else 40
    token_axis = 0 if form == "mapped" else 1
    table = pages.tolist()
    counts = [0, 0]
    for request, request_length in enumerate(length.tolist()):
        begin = (
            max(request_length - query_len - window, 0) // 128
            if window is not None
            else 0
        )
        end = (request_length + 127) // 128
        if page_ranges is not None:
            begin, end = page_ranges[request]
        for tile in range(begin, end):
            shift = expected_key_shift(
                tile * 128,
                request_length,
                query_len=query_len,
                group=group,
                query_columns=query_columns,
                window=window,
                key_clip=key_clip,
            )
            keys = unloaded_keys(shift)
            if not keys:
                continue
            counts[shift < 0] += len(keys)
            page = table[request][tile]
            for raw, scale in ((k, ks), (v, vs)):
                # 1 * 2**(255 - 127) is infinite in FP32 and BF16.
                raw[page].narrow(token_axis, keys.start, len(keys)).fill_(1)
                scale[page].narrow(token_axis, keys.start, len(keys)).fill_(255)
    return counts[0], counts[1]


def _assert_outputs(
    actual: tuple[torch.Tensor, torch.Tensor | None],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    for value, reference in zip(actual, expected, strict=True):
        assert value is not None
        torch.testing.assert_close(
            value, reference, rtol=0.002, atol=0.002, equal_nan=True
        )


def check_key_clip_replay(
    bound: BoundKernel,
    config: helion.Config,
    args: tuple[object, ...],
    form: str,
    lengths: tuple[tuple[int, int], ...],
    *,
    key_clip: str,
    reference_scores: bool = False,
    reverse_pages: bool = False,
    base2_scores: bool = False,
    page_ranges: tuple[tuple[tuple[int, int], ...], ...] | None = None,
) -> tuple[int, int]:
    """Replay lengths with nonfinite K and V at every key the clip leaves out.

    Those keys are visible to no row, so a consistent loader and mask never
    read them and the clean inputs' reference is the result. A whole tile
    would instead multiply the infinite V by its zero probability (NaN), and
    a loader and mask that disagree by one key change finite outputs.
    ``page_ranges`` gives each replay's [first, last) pages per request for a
    kernel that reads them from its ``bounds`` argument; pages outside the
    range the lengths imply hold no visible key and leave the reference alone.
    Returns the leading and trailing key counts that were left out.
    """
    k, v, ks, vs, pages, length = (cast("torch.Tensor", arg) for arg in args[1:7])
    # One physical page per logical page: poison never reaches a visible key.
    assert k.size(0) >= pages.numel()
    initialize_paged_attention_inputs(args)
    assert pages.unique().numel() == pages.numel()
    clean = [tensor.clone() for tensor in (k, v, ks, vs)]
    bounds = None
    if page_ranges is not None:
        assert form != "mapped" and len(page_ranges) == len(lengths)
        bounds = cast("torch.Tensor", args[10])
    compiled = bound.compile_config(config)
    # The capture guard tracks only launches of a warmed kernel.
    compiled(*args)
    # A config that names no dependent-launch option takes the paired default.
    with bound.env:
        normalized = bound.config_spec.normalized_config(config)
    capture_overlap = normalized.config.get("cute_pdl_capture_overlap") is True
    counts = [0, 0]
    with cuda_graph_for_replay() as graph:
        # A second launch reads its inputs before the first finishes.
        captured = capture_launches(
            graph, compiled, args, 2 if capture_overlap else 1, overlap=capture_overlap
        )
        for replay, request_lengths in enumerate(lengths):
            for tensor, original in zip((k, v, ks, vs), clean, strict=True):
                tensor.copy_(original)
            length.copy_(length.new_tensor(request_lengths))
            if bounds is not None:
                assert page_ranges is not None
                bounds.copy_(bounds.new_tensor(page_ranges[replay]))
            expected = optional_lse_reference(
                args,
                form,
                reference_scores=reference_scores,
                reverse_pages=reverse_pages,
                base2_scores=base2_scores,
            )
            leading, trailing = poison_unloaded_keys(
                args,
                form,
                key_clip=key_clip,
                page_ranges=None if page_ranges is None else page_ranges[replay],
            )
            counts[0] += leading
            counts[1] += trailing
            fill_nan(captured)
            graph.replay()
            for output in captured:
                _assert_outputs(output, expected)
            _assert_outputs(compiled(*args), expected)
    return counts[0], counts[1]
