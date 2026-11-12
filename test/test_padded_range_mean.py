from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

import helion
from helion._compiler import indexing_strategy
from helion._testing import cpu_render_target
from helion._testing.cute_codegen import cpu_bind
import helion.language as hl

pytestmark = pytest.mark.cpu_only


def _reconverging_chain_mean(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(n)])
        for _ in hl.static_range(16):
            row = row + row * 0.5
        out[i] = row.mean(-1)
    return out


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "counts the compiler's walks; ref mode does not compile"
)
def test_a_mean_walks_each_value_of_a_reconverging_chain_once() -> None:
    # The mean's divisor walks back from its input to the padded ranges it
    # reads (``_padded_iota_dims``).  Each round of ``row + row * 0.5`` reads
    # the previous value twice, so a walk that visits a value once per path
    # reached the load 2**16 times, and a deeper chain never finished binding.
    walks = 0
    walk = indexing_strategy._padded_iota_dims

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal walks
        walks += 1
        assert walks <= 1000, "the padded-range walk revisits values"
        return walk(*args, **kwargs)

    with (
        patch.object(indexing_strategy, "_padded_iota_dims", counted),
        cpu_render_target(),
    ):
        bound = cpu_bind(
            helion.kernel(_reconverging_chain_mean, static_shapes=True),
            (torch.empty(4, 200),),
        )
        bound.to_code(bound.config_spec.default_config())
    assert 0 < walks <= 1000


def _reconverging_loads_mean(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        cols = hl.load(a, [hl.arange(n)])
        for _ in hl.static_range(12):
            cols = (hl.load(a, [cols]) + hl.load(b, [cols])) % n
        out[i] = hl.load(x, [i, cols]).mean(-1)
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager(
    "counts the compiler's walks; ref mode does not compile"
)
def test_a_walk_through_loads_meets_each_value_once() -> None:
    # The walks also follow a load back to the padded ranges its index reads
    # (``_padded_load_dims``), the mean's divisor's among them.  Each round
    # loads ``a`` and ``b`` at the previous ``cols``, so a walk that restarted
    # at each load met the first ``cols`` on 2**12 paths.
    walks = 0
    walk = indexing_strategy._padded_iota_dims

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal walks
        walks += 1
        assert walks <= 2000, "a padded-range walk revisits values"
        return walk(*args, **kwargs)

    index = torch.zeros(200, dtype=torch.int32)
    with (
        patch.object(indexing_strategy, "_padded_iota_dims", counted),
        cpu_render_target(),
    ):
        bound = cpu_bind(
            helion.kernel(_reconverging_loads_mean, static_shapes=True),
            (torch.empty(4, 200), index, index),
        )
        bound.to_code(bound.config_spec.default_config())
    assert 0 < walks <= 2000


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager(
    "counts the compiler's walks; ref mode does not compile"
)
def test_a_kernel_without_a_padded_range_does_not_walk_through_its_loads() -> None:
    # A range of 256 lanes pads nothing, so the walk back through the loads
    # (``_padded_load_dims``) has nothing to find and is skipped: through 160
    # rounds of ``cols = (a[cols] + b[cols]) % n`` it recursed past Python's
    # limit (RecursionError), where the kernel compiled before the walk
    # followed loads.
    followed = 0
    follow = indexing_strategy._padded_load_dims

    def counted(*args: object, **kwargs: object) -> object:
        nonlocal followed
        followed += 1
        return follow(*args, **kwargs)

    index = torch.zeros(256, dtype=torch.int32)
    with (
        patch.object(indexing_strategy, "_padded_load_dims", counted),
        cpu_render_target(),
    ):
        bound = cpu_bind(
            helion.kernel(_reconverging_loads_mean, static_shapes=True),
            (torch.empty(4, 256), index, index),
        )
        bound.to_code(bound.config_spec.default_config())
    assert followed == 0


def _padded_range_row_stats(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = x.shape
    mean = torch.empty([m], dtype=x.dtype, device=x.device)
    var = torch.empty([m, 1], dtype=x.dtype, device=x.device)
    top = torch.empty([m], dtype=x.dtype, device=x.device)
    total = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(n)])
        mean[i] = (row + 1.0).mean(-1)
        var[i, :], _ = torch.var_mean(row, dim=-1, keepdim=True, correction=0)
        top[i] = row.amax(-1)
        total[i] = (row * 2.0 + 1.0).sum(-1)
    return mean, var, top, total


def _padded_range_var_and_std(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    var = torch.empty([m], dtype=x.dtype, device=x.device)
    std = torch.empty([m], dtype=x.dtype, device=x.device)
    for i in hl.grid(m):
        row = hl.load(x, [i, hl.arange(n)])
        var[i] = torch.var(row, dim=-1)
        std[i] = torch.std(row, dim=-1)
    return var, std


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_triton_reductions_take_their_identity_past_a_padded_range() -> None:
    # The load over hl.arange(200) has 256 lanes, and the ops between it and
    # each reduction make the 56 past the range nonzero: ``row + 1.0``,
    # var_mean's squared deviations, ``row * 2.0 + 1.0``, and ``row`` itself
    # for an amax of a negative row.  Each reduction's input takes the
    # reduction's identity there.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_padded_range_row_stats, static_shapes=True),
            (torch.empty(4, 200),),
        )
        code = bound.to_code(bound.config_spec.default_config())
    masks = [
        line.split(", ")[-1]
        for line in code.splitlines()
        if " = tl.where(tl.arange(0, 256) < 200, " in line
        and line.strip().startswith("padded_range_masked_")
    ]
    assert masks == ["0)", "0)", "0)", "float('-inf'))", "0)"], code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton"])
def test_padded_range_reductions_count_only_the_range() -> None:
    # Rows of zeros gave a mean of 1.28 (256 / 200) for ``(row + 1.0)``, and
    # negative rows an amax of 0, from the lanes past the range.
    kernel = helion.kernel(_padded_range_row_stats, static_shapes=True)
    var_and_std = helion.kernel(_padded_range_var_and_std, static_shapes=True)
    for width in (5, 200):
        device = helion._testing.DEVICE
        for x in (
            torch.zeros(16, width, device=device),
            -1.0 - torch.rand(16, width, device=device),
            torch.randn(16, width, device=device) + 1.0,
        ):
            _, (mean, var, top, total) = helion._testing.code_and_output(kernel, (x,))
            # torch.var and torch.std divide by the range's length less one.
            _, (unbiased, std) = helion._testing.code_and_output(var_and_std, (x,))
            torch.testing.assert_close(unbiased, x.var(-1), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(std, x.std(-1), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(mean, (x + 1.0).mean(-1), rtol=1e-4, atol=1e-4)
            torch.testing.assert_close(
                var, x.var(-1, correction=0, keepdim=True), rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(top, x.amax(-1))
            torch.testing.assert_close(
                total, (x * 2.0 + 1.0).sum(-1), rtol=1e-4, atol=1e-4
            )


def _scatter_through_loaded_columns(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for i in hl.grid(m):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        hl.store(out, [i, cols], hl.load(x, [i, cols]) * 2.0)
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_an_index_loaded_through_a_padded_range_masks_its_lanes() -> None:
    # ``cols`` has 128 lanes, and its 28 past the range load 0: unmasked,
    # the gather reads ``x[i, 0]`` there and the store writes it into
    # ``out[i, 0]``, overwriting it whenever ``idx`` does not hold 0.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_scatter_through_loaded_columns, static_shapes=True),
            (
                torch.empty(4, 200),
                torch.zeros(100, dtype=torch.int64),
                torch.empty(4, 200),
            ),
        )
        code = bound.to_code(bound.config_spec.default_config())
    accesses = [
        line
        for line in code.splitlines()
        if ("tl.load(x + " in line or "tl.store(out + " in line)
    ]
    assert len(accesses) == 2, code
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton"])
def test_a_scatter_through_loaded_columns_leaves_the_other_columns() -> None:
    device = helion._testing.DEVICE
    x = torch.randn(4, 200, device=device)
    idx = torch.arange(1, 101, device=device)
    out = torch.full((4, 200), -7.0, device=device)
    expected = out.clone()
    expected[:, idx] = x[:, idx] * 2.0
    _, result = helion._testing.code_and_output(
        helion.kernel(_scatter_through_loaded_columns, static_shapes=True),
        (x, idx, out),
    )
    torch.testing.assert_close(result, expected)


def _scatter_in_a_nested_loop(
    x: torch.Tensor, idx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for _ in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_an_index_loaded_before_a_nested_loop_masks_its_lanes_inside_it() -> None:
    # The loop reads ``cols`` through a placeholder (``cols_copy_0``), which
    # the mask now follows to the load outside the loop.
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_scatter_in_a_nested_loop, static_shapes=True),
            (
                torch.empty(4, 200),
                torch.zeros(100, dtype=torch.int64),
                torch.empty(4, 200),
            ),
        )
        code = bound.to_code(bound.config_spec.default_config())
    accesses = [
        line
        for line in code.splitlines()
        if ("tl.load(x + " in line or "tl.store(out + " in line)
    ]
    assert len(accesses) == 2, code
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), code


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "ref-mode hl.load meshgrids the tile's int32 index with int64 cols"
)
def test_a_scatter_in_a_nested_loop_leaves_the_other_columns() -> None:
    device = helion._testing.DEVICE
    x = torch.randn(4, 200, device=device)
    idx = torch.arange(1, 101, device=device)
    out = torch.full((4, 200), -7.0, device=device)
    expected = out.clone()
    expected[:, idx] = x[:, idx] * 2.0
    _, result = helion._testing.code_and_output(
        helion.kernel(_scatter_in_a_nested_loop, static_shapes=True),
        (x, idx, out),
    )
    torch.testing.assert_close(result, expected)


def _scatter_through_a_carried_index(
    x: torch.Tensor, idx: torch.Tensor, full: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for _ in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
            if full.size(0) == 1:
                cols = cols + 1
            else:
                cols = hl.load(full, [hl.arange(full.size(0))])
    return out


def _scatter_accesses(kernel: helion.Kernel, update: torch.Tensor) -> list[str]:
    """The gather and scatter lines of ``kernel(x, idx, update, out)``,
    rendered for 100 columns of a ``[4, 200]`` ``x``."""
    with cpu_render_target():
        bound = cpu_bind(
            kernel,
            (
                torch.empty(4, 200),
                torch.zeros(100, dtype=torch.int64),
                update,
                torch.empty(4, 200),
            ),
        )
        code = bound.to_code(bound.config_spec.default_config())
    return [
        line
        for line in code.splitlines()
        if ("tl.load(x + " in line or "tl.store(out + " in line)
    ]


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_a_loop_carried_index_keeps_the_padding_its_update_keeps() -> None:
    # A carry keeps its lanes in place: ``cols + 1`` leaves the 28 padding
    # lanes where they were, so the loop's accesses mask them.  Reloaded
    # through a range of 128, the update has no padding, but the first
    # iteration reads the 100 columns: no one mask fits every iteration, so
    # the kernel is refused rather than scatter the padding lanes, which load
    # index 0, into column 0.
    kernel = helion.kernel(_scatter_through_a_carried_index, static_shapes=True)
    accesses = _scatter_accesses(kernel, torch.zeros(1, dtype=torch.int64))
    assert len(accesses) == 2, accesses
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), accesses
    with pytest.raises(
        helion.exc.BackendUnsupported, match="padding a loop or branch changes"
    ):
        _scatter_accesses(kernel, torch.zeros(128, dtype=torch.int64))


def _scatter_before_a_later_loop_carries_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for _g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
        for _t in hl.tile(m):
            cols = cols + 1
    return out


def _scatter_before_an_if_reassigns_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
        if flag[g] != 0:
            cols = cols + 1
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
@pytest.mark.parametrize(
    "fn",
    [_scatter_before_a_later_loop_carries_cols, _scatter_before_an_if_reassigns_cols],
    ids=["later_loop", "if"],
)
def test_a_value_changed_after_a_loop_is_read_only_inside_it(fn) -> None:
    # The first loop only reads ``cols``; a later loop carrying it or an
    # ``if`` reassigning it leaves that loop's accesses masked.
    kernel = helion.kernel(fn, static_shapes=True)
    accesses = _scatter_accesses(kernel, torch.zeros(1, dtype=torch.int32))
    assert len(accesses) == 2, accesses
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), accesses


def _scatter_after_a_loop_updates_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for _g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for _t in hl.tile(m):
            cols = cols + 1
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
    return out


def _scatter_in_nested_loops_carrying_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for _g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            for _u in hl.tile(2):
                hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
                cols = cols + 1
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
@pytest.mark.parametrize(
    "fn",
    [_scatter_after_a_loop_updates_cols, _scatter_in_nested_loops_carrying_cols],
    ids=["after_a_loop", "nested_loops"],
)
def test_an_index_loops_carry_keeps_its_padding(fn) -> None:
    # A loop's result is its body's output, which ``cols + 1`` leaves padded
    # as ``cols``: the accesses mask the 28 padding lanes.
    kernel = helion.kernel(fn, static_shapes=True)
    accesses = _scatter_accesses(kernel, torch.zeros(1, dtype=torch.int32))
    assert len(accesses) == 2, accesses
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), accesses


def _scatter_after_an_if_reassigns_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        if flag[g] != 0:
            cols = cols + 1
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
    return out


def _scatter_while_an_if_updates_cols(
    x: torch.Tensor, idx: torch.Tensor, flag: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
            if flag[g] != 0:
                cols = cols + 1
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
@pytest.mark.parametrize(
    "fn",
    [_scatter_after_an_if_reassigns_cols, _scatter_while_an_if_updates_cols],
    ids=["before_the_loop", "inside_the_loop"],
)
def test_an_index_an_if_reassigns_keeps_its_padding(fn) -> None:
    # An ``if`` result is its branch's output, or the value the other branch
    # passes through: ``cols + 1`` and ``cols`` are padded alike, so the
    # accesses mask the 28 padding lanes.
    kernel = helion.kernel(fn, static_shapes=True)
    accesses = _scatter_accesses(kernel, torch.zeros(1, dtype=torch.int32))
    assert len(accesses) == 2, accesses
    assert all("tl.arange(0, 128) < 100" in line for line in accesses), accesses


@pytest.mark.cuda
@helion._testing.skipIfNotCUDA()
@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager(
    "ref-mode hl.load meshgrids the tile's int32 index with int64 cols"
)
@pytest.mark.parametrize("flag", [0, 1])
def test_a_scatter_through_an_index_an_if_reassigns_leaves_the_other_columns(
    flag: int,
) -> None:
    device = helion._testing.DEVICE
    x = torch.randn(4, 200, device=device)
    idx = torch.arange(1, 101, device=device)
    out = torch.full((4, 200), -7.0, device=device)
    expected = out.clone()
    expected[:, idx + flag] = x[:, idx + flag] * 2.0
    _, result = helion._testing.code_and_output(
        helion.kernel(_scatter_after_an_if_reassigns_cols, static_shapes=True),
        (x, idx, torch.tensor([flag], dtype=torch.int32, device=device), out),
    )
    torch.testing.assert_close(result, expected)


def _scatter_after_an_if_reloads_cols(
    x: torch.Tensor, idx: torch.Tensor, full: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        if full[g] != 0:
            cols = hl.load(full, [hl.arange(full.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
    return out


def _scatter_while_an_if_reloads_cols(
    x: torch.Tensor, idx: torch.Tensor, full: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        for t in hl.tile(m):
            hl.store(out, [t, cols], hl.load(x, [t, cols]) * 2.0)
            if full[g] != 0:
                cols = hl.load(full, [hl.arange(full.size(0))])
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager(
    "checks a compile-time refusal; ref mode does not compile"
)
@pytest.mark.parametrize(
    "fn",
    [_scatter_after_an_if_reloads_cols, _scatter_while_an_if_reloads_cols],
    ids=["before_the_loop", "inside_the_loop"],
)
def test_an_index_an_if_pads_otherwise_is_refused(fn) -> None:
    # One branch reloads ``cols`` over 128 columns, the other keeps the 28
    # padding lanes, which load index 0: no one mask fits both, so the
    # kernel is refused rather than scatter them into column 0.
    kernel = helion.kernel(fn, static_shapes=True)
    with pytest.raises(
        helion.exc.BackendUnsupported, match="padding a loop or branch changes"
    ):
        _scatter_accesses(kernel, torch.zeros(128, dtype=torch.int64))


def _scatter_through_an_index_an_if_offsets(
    x: torch.Tensor, idx: torch.Tensor, jdx: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, _ = x.shape
    for g in hl.grid(1):
        cols = hl.load(idx, [hl.arange(idx.size(0))])
        j = hl.load(jdx, [hl.arange(jdx.size(0))])
        if jdx[g] != 0:
            j = j + cols.sum()
        for t in hl.tile(m):
            hl.store(out, [t, j], hl.load(x, [t, j]) * 2.0)
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_an_if_reading_a_padded_index_leaves_another_index_alone() -> None:
    # The branch reads the padded ``cols``, but ``j`` has 128 lanes and no
    # padding on either path: its accesses stay unmasked, as before this
    # commit, not refused.
    kernel = helion.kernel(_scatter_through_an_index_an_if_offsets, static_shapes=True)
    accesses = _scatter_accesses(kernel, torch.zeros(128, dtype=torch.int64))
    assert len(accesses) == 2, accesses
    assert not any("< 100" in line for line in accesses), accesses


def _mean_plus(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.mean() + b


def _scan_of_a_mean(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in hl.tile(x.size(0)):
        out[i, :] = hl.associative_scan(_mean_plus, x[i, :], dim=1)
    return out


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_a_combine_functions_parameters_are_no_captured_values() -> None:
    # The mean's divisor walk reaches the combine function's parameters,
    # which a loop or branch body's captures are not: Triton renders the
    # scan, and CuTe refuses a mean in a combine function.
    kernel = helion.kernel(_scan_of_a_mean, static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(8, 64),))
        if helion._testing.matchesBackends(["cute"]):
            with pytest.raises(helion.exc.BackendUnsupported, match="combine"):
                bound.to_code(bound.config_spec.default_config())
        else:
            assert "tl.associative_scan" in bound.to_code(
                bound.config_spec.default_config()
            )


def _row_stats_in_a_nested_loop(
    x: torch.Tensor, total: torch.Tensor, mean: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    m, _ = x.shape
    for _ in hl.grid(1):
        row = hl.load(x, [0, hl.arange(200)])
        for i in hl.tile(m):
            total[i] = (row + 1.0).sum(-1) + i.index * 0.0
            mean[i] = row.mean(-1) + i.index * 0.0
    return total, mean


@helion._testing.onlyBackends(["triton", "cute"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_a_row_loaded_before_a_nested_loop_reduces_over_its_range() -> None:
    # The loop reads ``row`` through a placeholder: its sum takes the
    # padding lanes out on Triton, and its mean divides by the range's 200,
    # not the 256 lanes, on both backends.
    kernel = helion.kernel(_row_stats_in_a_nested_loop, static_shapes=True)
    with cpu_render_target():
        bound = cpu_bind(kernel, (torch.empty(4, 200), torch.empty(4), torch.empty(4)))
        code = bound.to_code(bound.config_spec.default_config())
    if helion._testing.matchesBackends(["cute"]):
        assert "cutlass.Float32(200)" in code, code
        assert "cutlass.Float32(256)" not in code, code
    else:
        assert code.count("tl.where(tl.arange(0, 256) < 200") == 2, code
        # The mean's divisor, a constant of its own.
        lines = [line.strip() for line in code.splitlines()]
        assert any(line.endswith(" = 200") for line in lines), code
        assert not any(line.endswith(" = 256") for line in lines), code


def _store_through_a_long_chain_of_gathers(
    x: torch.Tensor, a: torch.Tensor, b: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    m, n = x.shape
    for i in hl.grid(m):
        cols = hl.load(a, [hl.arange(n)])
        for _ in hl.static_range(160):
            cols = (hl.load(a, [cols]) + hl.load(b, [cols])) % n
        hl.store(out, [i, hl.arange(n)], hl.load(x, [i, cols]))
    return out


@helion._testing.onlyBackends(["triton"])
@helion._testing.skipIfRefEager("inspects generated code; ref mode has none")
def test_an_index_through_a_long_chain_of_gathers_compiles() -> None:
    # Each gather nests the walk one load deeper; past 32 loads the padding of
    # ``cols`` is unknown and the last gather keeps the lowering it had
    # before loads were followed, rather than the walk recursing past
    # Python's limit (a RecursionError at 160 gathers).
    with cpu_render_target():
        bound = cpu_bind(
            helion.kernel(_store_through_a_long_chain_of_gathers, static_shapes=True),
            (
                torch.empty(4, 200),
                torch.zeros(200, dtype=torch.int64),
                torch.zeros(200, dtype=torch.int64),
                torch.empty(4, 200),
            ),
        )
        code = bound.to_code(bound.config_spec.default_config())
    assert "tl.store(out + " in code, code
