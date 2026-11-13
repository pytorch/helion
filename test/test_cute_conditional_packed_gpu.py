"""Exact native SS packet coverage; full attention is checked in its existing gate."""

from __future__ import annotations

import pytest

from helion._testing.cute_conditional_packed import check_packed_panel
from helion._testing.cute_conditional_packed import validate_packed_source
from helion._testing.cute_fragment_shared_replay import begin_record
from helion._testing.cute_fragment_shared_replay import require_sm100_native
from helion._testing.cute_fragment_shared_replay import run_authored_storage_replay

pytestmark = [pytest.mark.cuda, pytest.mark.slow]


def test_conditional_packed_panel_all_byte_pairs(tmp_path, record_property):
    require_sm100_native()
    receipt = tmp_path / "raw.pt"
    record_property("raw_bits", str(receipt))
    check_packed_panel(receipt)


def test_conditional_packed_attention_full_source_and_replay(tmp_path, record_property):
    output, record = begin_record("authored-packed", tmp_path)
    record_property("raw_bits", str(output / "raw.pt"))
    run_authored_storage_replay(output, record, validate_packed_source)
