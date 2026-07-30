"""Tests for :meth:`helion.runtime.kernel.BoundKernel.to_code` with
:class:`helion.OutputCodeOptions` -- i.e. emitting dependency-free ("standalone")
output code that runs with no ``helion`` runtime dependency (``torch`` + the
backend DSL only)."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
from typing import Any
import unittest

import torch

import helion
from helion._testing import DEVICE
from helion._testing import TestCase
from helion._testing import onlyBackends
from helion._testing import skipIfRefEager
from helion._testing import skipUnlessPallas
import helion.language as hl

_FREE = helion.OutputCodeOptions(allow_helion_deps=False)


@helion.kernel(config=helion.Config(block_sizes=[32, 32]))
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = torch.broadcast_tensors(x, y)
    out = torch.empty(
        x.shape,
        dtype=torch.promote_types(x.dtype, y.dtype),
        device=x.device,
    )
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[64, 64], pid_type="persistent_blocked")
)
def add_persistent(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Persistent grid -> the launch computes its grid from ``get_num_sm``."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[64, 64], indexing="tensor_descriptor"))
def add_tma(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """TMA (tensor-descriptor) indexing -> the launch calls
    ``set_triton_allocator`` for the device-side descriptor allocator."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(config=helion.Config(block_sizes=[32, 32]), static_shapes=False)
def add_dynamic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """static_shapes=False -> the standalone must run at any shape: the emitted
    host wrapper derives the grid + output shape from the runtime input rather than
    baking the sample shape."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[64, 64], pid_type="persistent_blocked")
)
def get_num_sm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """A kernel deliberately named like a launcher helper. The launcher helpers
    live in the private ``_make_helion_runtime`` scope, so this name cannot
    collide with them (the persistent grid still calls the shim's get_num_sm)."""
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


def _write(tmp: str, code: str) -> str:
    path = Path(tmp) / "generated.py"
    path.write_text(code)
    return str(path)


def _run_add_no_helion(code: str, entrypoint: str, shape: tuple[int, int]) -> None:
    """Run an **add** standalone in a subprocess where importing ``helion`` is
    blocked, proving zero Helion runtime dependency and the right result. Specific
    to add kernels -- it hardcodes the ``x + y`` reference."""
    m, n = shape
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, code)
        script = textwrap.dedent(f"""
            import sys
            # Poison the helion package so any leftover dependency fails loudly.
            sys.modules["helion"] = None
            import importlib.util
            import torch

            spec = importlib.util.spec_from_file_location("_standalone", {path!r})
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            torch.manual_seed(0)
            x = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
            y = torch.randn([{m}, {n}], device="{DEVICE}", dtype=torch.bfloat16)
            out = mod.{entrypoint}(x, y)
            torch.testing.assert_close(out, x + y)
            print("STANDALONE_OK")
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        if "STANDALONE_OK" not in result.stdout:
            raise AssertionError(
                f"standalone run failed:\nstdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )


def _import_code(code: str, name: str, tmp: str) -> Any:
    """Import generated standalone source as a module (registered in
    ``sys.modules`` before ``exec_module`` so any inlined ``@dataclass`` can
    resolve ``cls.__module__``)."""
    import importlib.util

    path = _write(tmp, code)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return mod


@onlyBackends(["triton"])
@skipIfRefEager("to_code compiles real Triton code; not meaningful in ref-eager")
class TestToCodeTriton(TestCase):
    def test_default_still_has_helion_deps(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        # options=None (default) keeps the original to_code behavior, which still
        # depends on helion at runtime (here: the launcher import).
        code = add.bind((x, y)).to_code()
        self.assertIn("from helion.runtime import default_launcher", code)

    def test_allow_helion_deps_false_has_no_helion_import(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        code = add.bind((x, y)).to_code(options=_FREE)
        # The helion *package* is never imported. The body DOES contain
        # ``helion.runtime.*`` references, but they resolve against the inlined
        # local ``helion`` shim (see the shim banner), not the real package.
        self.assertNotIn("import helion", code)
        self.assertNotIn("from helion", code)
        self.assertIn("def default_launcher(", code)
        self.assertIn(
            "helion = types.SimpleNamespace(runtime=_make_helion_runtime())", code
        )
        self.assertIn("_default_launcher = helion.runtime.default_launcher", code)
        self.assertIn("def add(", code)

    def test_standalone_runs_without_helion(self) -> None:
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add.bind((x, y)).to_code(options=_FREE)
        _run_add_no_helion(code, "add", (128, 128))

    def test_to_triton_code_alias_unchanged(self) -> None:
        x = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([64, 64], device=DEVICE, dtype=torch.bfloat16)
        bound = add.bind((x, y))
        # Back-compat: to_triton_code is an alias for to_code's default behavior.
        self.assertEqual(bound.to_triton_code(), bound.to_code())

    def test_persistent_kernel_get_num_sm_via_shim(self) -> None:
        """A persistent kernel calls ``helion.runtime.get_num_sm`` to size its
        grid. The body is emitted verbatim and the call is satisfied by the inlined
        ``helion.runtime`` shim, so it runs with no real helion."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_persistent.bind((x, y)).to_code(options=_FREE)
        self.assertNotIn("import helion", code)
        self.assertIn("helion.runtime.get_num_sm(", code)  # verbatim body call
        self.assertIn("get_num_sm=get_num_sm", code)  # shim re-export
        _run_add_no_helion(code, "add_persistent", (128, 128))

    def test_tma_kernel_set_triton_allocator_via_shim(self) -> None:
        """A tensor-descriptor (TMA) kernel calls
        ``helion.runtime.set_triton_allocator``; the shim satisfies it."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_tma.bind((x, y)).to_code(options=_FREE)
        self.assertNotIn("import helion", code)
        self.assertIn("helion.runtime.set_triton_allocator(", code)  # verbatim call
        self.assertIn("set_triton_allocator=set_triton_allocator", code)
        _run_add_no_helion(code, "add_tma", (128, 128))

    def test_dynamic_shapes_run_at_other_shapes(self) -> None:
        """A static_shapes=False kernel's standalone runs correctly at shapes other
        than the one it was bound with (grid + output shape derived from input)."""
        x = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.zeros([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = add_dynamic.bind((x, y)).to_code(options=_FREE)
        for shape in ((128, 128), (256, 64), (64, 256)):
            _run_add_no_helion(code, "add_dynamic", shape)

    def test_kernel_named_like_runtime_helper(self) -> None:
        """A kernel literally named ``get_num_sm`` exports safely: the launcher
        helpers are isolated in a private scope, so the kernel name can't shadow
        them (this would previously have produced two module-level defs)."""
        x = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        y = torch.randn([128, 128], device=DEVICE, dtype=torch.bfloat16)
        code = get_num_sm.bind((x, y)).to_code(options=_FREE)
        _run_add_no_helion(code, "get_num_sm", (128, 128))


@helion.kernel(backend="pallas", static_shapes=True)
def pallas_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for tile in hl.tile(out.size()):
        out[tile] = x[tile] + y[tile]
    return out


@helion.kernel(
    config=helion.Config(block_sizes=[8]), static_shapes=False, backend="pallas"
)
def pallas_rows_times_two(x: torch.Tensor) -> torch.Tensor:
    """static_shapes=False Pallas (torch-tensor): the standalone must run at any
    leading dim (grid + output shape derived from the runtime input)."""
    n, _d = x.shape
    out = torch.empty_like(x)
    for tile in hl.tile(n):
        out[tile, :] = x[tile, :] * 2.0
    return out


def _pallas_to_code(
    kernel: Any, args: tuple[object, ...], options: helion.OutputCodeOptions
) -> str:
    """``to_code`` with an explicit config. These kernels are never run/autotuned in
    the test, so ``to_code(config=None)`` has no implicit config to resolve; use the
    kernel's own config when it declares one, else the backend default."""
    bound = kernel.bind(args)
    configs = bound.kernel.configs
    config = configs[0] if len(configs) == 1 else bound.config_spec.default_config()
    return bound.to_code(config, options=options)


@skipUnlessPallas("Pallas to_code test requires the Pallas backend / TPU")
@skipIfRefEager("to_code compiles real kernels; not meaningful in ref-eager")
class TestToCodePallas(TestCase):
    def test_pallas_torch_standalone_runs(self) -> None:
        x = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        y = torch.randn([256, 256], device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_add, (x, y), _FREE)
        # The helion package is never imported; the dependency-free Pallas launcher
        # is inlined into a local ``helion.runtime`` shim instead.
        self.assertNotIn("import helion", code)
        self.assertNotIn("from helion", code)
        self.assertIn("def default_pallas_launcher(", code)
        self.assertIn(
            "_default_pallas_launcher = helion.runtime.default_pallas_launcher", code
        )
        self.assertIn("def pallas_add(", code)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_add_standalone_test"
            mod = _import_code(code, name, tmp)
            try:
                torch.testing.assert_close(mod.pallas_add(x, y), x + y)
            finally:
                sys.modules.pop(name, None)

    def test_pallas_torch_dynamic_shapes(self) -> None:
        """A static_shapes=False Pallas (torch) standalone runs at other shapes."""
        d = 128
        x0 = torch.zeros(512, d, device=DEVICE, dtype=torch.float32)
        code = _pallas_to_code(pallas_rows_times_two, (x0,), _FREE)
        with tempfile.TemporaryDirectory() as tmp:
            name = "pallas_rt2_test"
            mod = _import_code(code, name, tmp)
            try:
                for t in (512, 128, 256):
                    x = torch.randn(t, d, device=DEVICE, dtype=torch.float32)
                    out = mod.pallas_rows_times_two(x)
                    self.assertEqual(tuple(out.shape), (t, d))
                    torch.testing.assert_close(out, x * 2.0)
            finally:
                sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
