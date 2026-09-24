from __future__ import annotations

import os
import re
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_initialized()
    patch.object(
        torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")
    ).start()

from .test_cute_chained_tcgen05 import _code
from .test_cute_chained_tcgen05 import _compile
from .test_cute_chained_tcgen05 import _inputs
from helion._testing import skipUnlessBackends

pytestmark = skipUnlessBackends(["cute"])


def _assert_tmem_allocation_base(ptx: str) -> None:
    """TMEM A and C must retain the same runtime allocation-base ancestor."""
    definitions = {
        match[2]: (match[1], re.findall(r"%r\d+", match[3]))
        for match in re.finditer(r"(?m)^\s*([a-z0-9_.]+)\s+(%r\d+),\s*([^;]+);", ptx)
    }

    def shared_loads(register: str) -> set[str]:
        pending = [register]
        seen: set[str] = set()
        result: set[str] = set()
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            if current not in definitions:
                continue
            operation, operands = definitions[current]
            if operation.startswith("ld.shared."):
                result.add(current)
            else:
                pending.extend(operands)
        return result

    addresses = re.findall(r"tcgen05\.mma[^\n]*?\[(%r\d+)\],\s*\[(%r\d+)\]", ptx)
    assert addresses, "Expected at least one TMEM-source MMA"
    for destination, operand_a in addresses:
        assert shared_loads(destination) & shared_loads(operand_a), (
            f"TMEM A {operand_a} lost the allocation base used by C {destination}"
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_tmem_address_check_detects_absolute_zero(dynamic: bool) -> None:
    address = "add.s32 %r3, %r1, 8;" if dynamic else "mov.b32 %r3, 8;"
    ptx = (
        "ld.shared.b32 %r1, [allocation];\n"
        "add.s32 %r2, %r1, 128;\n"
        f"{address}\n"
        "tcgen05.mma.cta_group::1.kind::f16 [%r2], [%r3], desc;\n"
    )
    if dynamic:
        _assert_tmem_allocation_base(ptx)
    else:
        with pytest.raises(AssertionError, match="lost the allocation base"):
            _assert_tmem_allocation_base(ptx)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
def test_tcgen_bridge_retains_allocated_base_in_ptx(dtype: str) -> None:
    # Isolate fake compilation from CUDA initialization by earlier GPU tests.
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "CUTE_DSL_ARCH": "sm_103a",
        "CUTE_DSL_KEEP_PTX": "1",
    }
    result = subprocess.run(
        [sys.executable, "-m", __name__, dtype],
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[sys.argv[1]]
    values = _inputs("cpu", dtype)
    ptx = _compile(_code(values, "plain"), values, "plain")
    _assert_tmem_allocation_base(ptx)
    assert not torch.cuda.is_initialized()
