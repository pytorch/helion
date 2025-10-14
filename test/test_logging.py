from __future__ import annotations

import inspect
import logging
import os
from typing import Callable
import unittest

import torch
import torch.fx.experimental._config as fx_config

import helion
from helion._testing import DEVICE
from helion._testing import RefEagerTestDisabled
from helion._testing import TestCase
import helion.language as hl


@helion.kernel(autotune_effort="none")
def logging_reduce_rows(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    n = hl.specialize(n)

    m_block = hl.register_block_size(m)

    result = torch.zeros(n, dtype=torch.float32, device=x.device)

    for outer in hl.tile(m, block_size=m_block):
        for inner in hl.tile(outer.begin, outer.end):
            zero_idx = inner.begin - inner.begin
            result[zero_idx] = result[zero_idx]
    return result


def _run_symbol_logging_example() -> None:
    x = torch.randn((128, 5632), device=DEVICE, dtype=torch.float16)
    logging_reduce_rows(x)


def _run_with_symbol_logs(fn: Callable[[], None]) -> str:
    logger = logging.getLogger("torch.fx.experimental.symbolic_shapes")
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # type: ignore[override]
            records.append(record.getMessage())

    handler = _Capture()
    previous_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)

    return "\n".join(records)


class TestLogging(RefEagerTestDisabled, TestCase):
    def test_log_set(self):
        import logging

        from helion._logging._internal import init_logs_from_string

        init_logs_from_string("foo.bar,+fuzz.baz")
        self.assertEqual(
            helion._logging._internal._LOG_REGISTRY.log_levels["foo.bar"],
            logging.INFO,
        )
        self.assertEqual(
            helion._logging._internal._LOG_REGISTRY.log_levels["fuzz.baz"],
            logging.DEBUG,
        )

    def test_kernel_log(self):
        @helion.kernel(
            config=helion.Config(
                block_sizes=[1], num_warps=16, num_stages=8, indexing="pointer"
            )
        )
        def add(x, y):
            x, y = torch.broadcast_tensors(x, y)
            out = torch.empty_like(x)
            for tile in hl.tile(out.size()):
                out[tile] = x[tile] + y[tile]
            return out

        x = torch.randn(4, device=DEVICE)

        with self.assertLogs("helion.runtime.kernel", level="DEBUG") as cm:
            add(x, x)
        self.assertTrue(
            any("INFO:helion.runtime.kernel:Output code:" in msg for msg in cm.output)
        )
        self.assertTrue(
            any("DEBUG:helion.runtime.kernel:Debug string:" in msg for msg in cm.output)
        )

    def test_symbolic_shape_log_includes_kernel_source(self):
        symbol_filter = ",".join(f"u{i}" for i in range(5))
        file_path = os.path.abspath(__file__)
        source_lines, start_line = inspect.getsourcelines(logging_reduce_rows.fn)

        def get_line_no(snippet: str) -> int:
            return start_line + next(
                idx for idx, line in enumerate(source_lines) if snippet in line
            )

        line_for_block = get_line_no("m_block = hl.register_block_size(m)")
        line_for_inner = get_line_no("for inner in hl.tile(outer.begin, outer.end):")
        line_for_zero = get_line_no("zero_idx = inner.begin - inner.begin")

        with fx_config.patch(extended_debug_create_symbol=symbol_filter):
            output = _run_with_symbol_logs(_run_symbol_logging_example)

        lines = output.splitlines()

        self.assertTrue(output, msg="no logs captured for symbolic shapes")
        self.assertIn("create_unbacked_symint", output)
        self.assertIn("Helion kernel stack:", output)
        self.assertTrue(
            any(
                f'  File "{file_path}", line {line_for_block}, in logging_reduce_rows'
                in line
                for line in lines
            ),
            msg="register_block_size location missing",
        )
        self.assertIn("    m_block = hl.register_block_size(m)", output)

        def assert_symbol(symbol: str, lineno: int, snippet: str) -> None:
            marker = f"create_unbacked_symint {symbol}"
            for idx, line in enumerate(lines):
                if marker in line:
                    window = lines[idx : idx + 6]
                    break
            else:
                self.fail(f"missing log for {symbol}")

            expected_file_line = (
                f'  File "{file_path}", line {lineno}, in logging_reduce_rows'
            )
            self.assertTrue(
                any(expected_file_line in line for line in window),
                msg=f"missing file info for {symbol}",
            )
            self.assertTrue(
                any(snippet in line for line in window),
                msg=f"missing source line for {symbol}",
            )

        for sym, line_no, snippet in [
            ("u0", line_for_block, "m_block = hl.register_block_size(m)"),
            ("u1", line_for_inner, "for inner in hl.tile(outer.begin, outer.end):"),
            ("u2", line_for_inner, "for inner in hl.tile(outer.begin, outer.end):"),
            ("u3", line_for_inner, "for inner in hl.tile(outer.begin, outer.end):"),
            ("u4", line_for_zero, "zero_idx = inner.begin - inner.begin"),
        ]:
            assert_symbol(sym, line_no, snippet)


if __name__ == "__main__":
    unittest.main()
