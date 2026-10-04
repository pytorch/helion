from __future__ import annotations

from pathlib import Path
import re
import tempfile
import unittest

from triton._C.libtriton import ir
from triton._C.libtriton import passes

from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRocm
from helion.runtime.triton import ws_registers

_ACC = "!ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>"
_LOAD = f"%0 = ttng.tmem_load %ACC : {_ACC} -> tensor<128x128xf32, #blocked>"
# Default plus 4-, 1- and 2-warp partitions; DEFAULT/EPILOGUE take a TMEM load.
_TTGIR = f"""
#blocked = #ttg.blocked<{{sizePerThread = [1, 128], threadsPerWarp = [32, 1], \
warpsPerCTA = [4, 1], order = [0, 1]}}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {{"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func @_helion_kernel() {{
    %acc = ttng.tmem_alloc : () -> {_ACC}
    ttg.warp_specialize(%acc) attributes {{requestedRegisters = array<i32: REQ>}}
    default {{
      DEFAULT
      ttg.warp_yield
    }}
    partition0(%arg0: {_ACC}) num_warps(4) {{
      EPILOGUE
      ttg.warp_return
    }}
    partition1(%arg0: {_ACC}) num_warps(1) {{
      ttg.warp_return
    }}
    partition2(%arg0: {_ACC}) num_warps(2) {{
      ttg.warp_return
    }} : ({_ACC}) -> ()
    tt.return
  }}
}}
"""


def _ttgir(requested: str, *, epilogue: bool, default: bool) -> str:
    return (
        _TTGIR.replace("REQ", requested)
        .replace("EPILOGUE", _LOAD.replace("ACC", "arg0") if epilogue else "")
        .replace("DEFAULT", _LOAD.replace("ACC", "acc") if default else "")
    )


def _actual_registers(ttgir: str) -> list[int]:
    context = ir.context()
    ir.load_dialects(context)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.ttgir"
        path.write_text(ttgir)
        mod = ir.parse_mlir_module(str(path), context)
    pm = ir.pass_manager(context)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    pm.run(mod, "allocate_warp_groups")
    found = re.search(r"actualRegisters = array<i32: ([\d, ]+)>", str(mod))
    assert found is not None
    return [int(r) for r in found[1].split(",")]


@onlyBackends(["triton"])
@skipIfRocm("warp specialization registers are NVIDIA only")
class TestWsRegisters(TestCase):
    def test_epilogue_partition_takes_default_budget(self) -> None:
        ttgir = _ttgir("88, 24, 88", epilogue=True, default=False)
        self.assertEqual(_actual_registers(ttgir), [328, 88, 88, 88, 88])
        split = ws_registers.split_registers(ttgir)
        self.assertIn("array<i32: 256, 24, 88>", split)
        self.assertEqual(_actual_registers(split), [160, 256, 88, 88, 88])

    def test_default_epilogue_keeps_requests(self) -> None:
        for epilogue in (False, True):
            ttgir = _ttgir("88, 24, 88", epilogue=epilogue, default=True)
            self.assertEqual(ws_registers.split_registers(ttgir), ttgir)

    def test_partition_without_tmem_load_keeps_requests(self) -> None:
        ttgir = _ttgir("88, 24, 88", epilogue=False, default=False)
        self.assertEqual(ws_registers.split_registers(ttgir), ttgir)


if __name__ == "__main__":
    unittest.main()
