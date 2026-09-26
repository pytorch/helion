from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

from test import test_cute_epilogue_fanout as fanout


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("body_fails", [False, True])
def test_cpu_metadata_fixture_preserves_entry_state_and_threads(
    initialized: bool, body_fails: bool
) -> None:
    threads = torch.get_num_threads()
    fixture = contextmanager(cast("Any", fanout.cuda_trace).__wrapped__)
    with patch.object(torch.cuda, "is_initialized", return_value=initialized):
        try:
            with fixture():
                assert torch.get_num_threads() == 1
                assert torch.cuda.is_initialized() is initialized
                with pytest.raises(AssertionError, match="GPU forbidden"):
                    torch.cuda._lazy_init()
                if body_fails:
                    raise RuntimeError("held body failure")
        except RuntimeError as error:
            assert body_fails and str(error) == "held body failure"
        else:
            assert not body_fails
        assert torch.get_num_threads() == threads
        assert torch.cuda.is_initialized() is initialized


def test_cpu_metadata_fixture_rejects_a_new_initialization() -> None:
    threads = torch.get_num_threads()
    state = [False]
    fixture = contextmanager(cast("Any", fanout.cuda_trace).__wrapped__)
    with (
        patch.object(torch.cuda, "is_initialized", side_effect=lambda: state[0]),
        pytest.raises(AssertionError),
        fixture(),
    ):
        state[0] = True
    assert torch.get_num_threads() == threads
