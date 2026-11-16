from __future__ import annotations

import pytest
import torch

from helion.autotuner.kernel_args import _clone_args
from helion.autotuner.kernel_args import _clone_tensor_leaves

pytestmark = pytest.mark.cpu_only


def _independent_counter():
    state = 0

    def counter():
        nonlocal state
        state += 1
        return state

    return counter


def _cell(callback, name):
    return dict(zip(callback.__code__.co_freevars, callback.__closure__, strict=True))[
        name
    ]


@pytest.mark.parametrize("mode", ["partial", "all", "bank"])
def test_cloned_callbacks_keep_shared_cells_and_transitive_owners(mode):
    x = torch.arange(4)
    untouched = torch.ones(3)
    state = 0
    next_state = 0

    def read():
        return x, state

    def update():
        nonlocal state
        state += next_state + 1
        return untouched, state

    def advance():
        nonlocal next_state
        next_state += 2
        return next_state

    unrelated = _independent_counter()
    assert _cell(read, "state") is _cell(update, "state")
    assert _cell(update, "next_state") is _cell(advance, "next_state")
    # Equal interned contents must not connect otherwise distinct cells.
    assert _cell(read, "state").cell_contents is _cell(unrelated, "state").cell_contents
    assert _cell(read, "state") is not _cell(unrelated, "state")
    args = (x, read, update, advance, unrelated)
    if mode == "bank":
        copied = _clone_tensor_leaves(args)
    else:
        copied = _clone_args(
            args, None, idx_to_clone=(0,) if mode == "partial" else None
        )
    private_x, private_read, private_update, private_advance, private_unrelated = copied
    assert private_x is private_read()[0]
    assert private_x is not x
    assert private_read is not read
    assert private_update is not update
    assert private_advance is not advance
    assert private_unrelated is unrelated
    assert _cell(private_read, "state") is _cell(private_update, "state")
    assert _cell(private_update, "next_state") is _cell(private_advance, "next_state")
    assert _cell(private_read, "state") is not _cell(read, "state")
    assert _cell(private_advance, "next_state") is not _cell(advance, "next_state")

    assert private_advance() == 2
    private_untouched, result = private_update()
    assert result == 3 and private_read()[1] == 3
    assert read()[1] == 0
    assert _cell(advance, "next_state").cell_contents == 0
    assert private_unrelated() == 1
    assert read()[1] == 0
    assert private_read()[1] == 3
    if mode == "partial":
        assert private_untouched is untouched
    else:
        assert private_untouched is not untouched
    torch.testing.assert_close(private_untouched, untouched)
    private_x.add_(5)
    torch.testing.assert_close(x, torch.arange(4))
