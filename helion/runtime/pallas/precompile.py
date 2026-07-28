"""Pallas-specific precompilation.

Two standalone flavors:

- torch-tensor entrypoint: inline the dependency-free Pallas launcher
  (:mod:`helion.runtime.pallas.launcher`), via the backend-neutral default flow.
- ``jax_fn=True`` pure-JAX entrypoint: inline the *jax-only slice* of that
  launcher (the transitive code closure of ``_pallas_jax_call`` -- the real
  ``pl.kernel`` compile core) and emit a jax-native wrapper, so the standalone
  reuses the exact runtime launch logic with only ``jax`` at runtime.

The generic assembly helpers live in the backend-neutral orchestrator
(:mod:`helion.runtime.precompile`).
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

from ..precompile import BackendPrecompiler
from ..precompile import _dedupe_preserve_order
from ..precompile import _load_launcher_source
from ..precompile import _split_module_source
from ..precompile import _write_standalone

if TYPE_CHECKING:
    from ..precompile import PrecompilationInput
    from ..precompile import PrecompilationResult


# Launcher kwargs that mark a kernel using a Pallas feature the pure-JAX
# standalone doesn't emit yet (scratch/VMEM buffers, HBM pass-through, SMEM,
# dynamic-shape padding, in-place aliasing, compact-worklist, matmul dot_general).
_JAX_UNSUPPORTED_KWARGS = (
    "_scratch_shapes",
    "_hbm_arg_indices",
    "_smem_arg_indices",
    "_ds_pad_dims",
    "_inplace_indices",
    "_compact_build_worklist",
    "_matmul_dot_general",
)

# Dtypes the Pallas launcher rejects and that JAX would mishandle under x32
# (int64/uint64 silently narrow to 32-bit; float64 is unsupported on TPU).
_JAX_UNSUPPORTED_DTYPES = frozenset({torch.int64, torch.uint64, torch.float64})


class PallasPrecompiler(BackendPrecompiler):
    """Precompiler for the Pallas backend (torch-tensor and ``jax_fn`` modes)."""

    launcher_module = "helion.runtime.pallas.launcher"
    launcher_symbol = "default_pallas_launcher"
    launcher_alias = "_default_pallas_launcher"
    deps = "torch + jax"
    helion_call_rewrites = ()

    def precompile(
        self,
        input: PrecompilationInput,  # noqa: A002
        args: tuple[object, ...],
    ) -> PrecompilationResult:
        if input.jax_fn:
            return _precompile_pallas_jax(input, args)
        return super().precompile(input, args)


def _torch_dtype_to_jnp_name(dtype: torch.dtype) -> str:
    """``torch.float32`` -> ``"jnp.float32"`` (``torch.bool`` -> ``"jnp.bool_"``)."""
    name = str(dtype).rsplit(".", 1)[-1]
    if name == "bool":
        name = "bool_"
    return f"jnp.{name}"


# Cap on inlining a host-wrapper-created constant *tensor* launch arg by value.
# Lifted module scalars (``torch.tensor([_NEG])``) are tiny; a large constant
# tensor is unexpected here and would bloat the standalone, so reject it clearly.
_MAX_EMBED_CONST_ELEMS = 256


def _embed_jax_const(value: object) -> str:
    """Python source reconstructing a host-wrapper-created constant launch arg as a
    JAX value: a lifted scalar-constant tensor -> ``jnp.array(...)``; a
    specialization scalar -> its int/float/bool literal. These are baked into the
    standalone, whose entrypoint takes only the user's tensor inputs."""
    if isinstance(value, torch.Tensor):
        if value.numel() > _MAX_EMBED_CONST_ELEMS:
            raise NotImplementedError(
                "helion.precompile(jax_fn=True) cannot inline a constant tensor "
                f"launch arg with {value.numel()} elements (limit "
                f"{_MAX_EMBED_CONST_ELEMS})"
            )
        values = value.detach().cpu().tolist()
        return f"jnp.array({values!r}, dtype={_torch_dtype_to_jnp_name(value.dtype)})"
    if isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    raise NotImplementedError(
        "helion.precompile(jax_fn=True) does not support a launch arg of type "
        f"{type(value).__name__!r}"
    )


# Distinct scale factors for the second shape probe (see ``_precompile_pallas_jax``):
# one per symbolic input dim, distinct so each launch value maps unambiguously to
# the dim it tracks.
_PROBE_FACTORS = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def _scaled_probe_args(
    args: tuple[object, ...],
    fake_args: list[object],
    sym_factor: dict[str, int],
) -> list[object]:
    """Second-probe args: each user tensor with its symbolic dims scaled by that
    dim's per-symbol factor (dims sharing a symbol scale together); non-tensor
    args and concrete (specialized) dims are left unchanged."""
    probe: list[object] = []
    for i, a in enumerate(args):
        if not isinstance(a, torch.Tensor):
            probe.append(a)
            continue
        fake_shape = cast("torch.Tensor", fake_args[i]).shape
        new_shape: list[int] = []
        for d in range(a.dim()):
            fsz = fake_shape[d]
            factor = 1
            if isinstance(fsz, torch.SymInt) and fsz.node.expr.is_symbol:
                factor = sym_factor.get(str(fsz.node.expr), 1)
            new_shape.append(int(a.shape[d]) * factor)
        probe.append(torch.empty(new_shape, dtype=a.dtype, device=a.device))
    return probe


def _match_input_dim(
    v0: int,
    v1: int,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str | None:
    """``inputs[k].shape[d]`` for a value that scaled ``v0 -> v1`` across the two
    probes; ``None`` if unchanged (a constant). Raises if it changed but matches no
    input dim (a value we can't derive, rather than silently baking it wrong)."""
    if v0 == v1:
        return None
    for k, (s0, s1) in enumerate(zip(in_shapes0, in_shapes1, strict=True)):
        for d in range(len(s0)):
            if s0[d] == v0 and s1[d] == v1:
                return f"inputs[{k}].shape[{d}]"
    raise NotImplementedError(
        "helion.precompile(jax_fn=True) cannot derive a dynamic launch value "
        f"({v0} -> {v1}) from the input shapes"
    )


def _grid_axis_expr(
    g0: int,
    g1: int,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str:
    """Python expression for one grid axis: a constant literal if it didn't change,
    else ``cdiv(inputs[k].shape[d], block)`` for the input dim it tracks (the block
    is recovered from the sample: ``block = dim / grid`` when the sample dim is a
    whole number of blocks)."""
    if g0 == g1:
        return repr(g0)
    for k, (s0, s1) in enumerate(zip(in_shapes0, in_shapes1, strict=True)):
        for d in range(len(s0)):
            a0, a1 = s0[d], s1[d]
            if a0 == a1 or g0 <= 0 or a0 % g0 != 0:
                continue
            block = a0 // g0
            if block >= 1 and -(-a1 // block) == g1:
                if block == 1:
                    return f"inputs[{k}].shape[{d}]"
                return f"(inputs[{k}].shape[{d}] + {block - 1}) // {block}"
    raise NotImplementedError(
        f"helion.precompile(jax_fn=True) cannot derive grid axis ({g0} -> {g1}) "
        "from the input shapes"
    )


def _const_slot_expr(
    v0: object,
    v1: object,
    in_shapes0: list[list[int]],
    in_shapes1: list[list[int]],
) -> str:
    """Expression filling a host-wrapper-created launch slot: baked by value if it
    stayed constant across the two probes, else input-derived. Covers lifted module
    scalars / specialization ints (constant) and shape-derived scalars such as a
    reduction's row-count ``torch.tensor([t])`` (a ``(1,)`` tensor that tracks a
    runtime dim)."""
    if isinstance(v0, torch.Tensor):
        vals0 = v0.detach().cpu().reshape(-1).tolist()
        vals1 = cast("torch.Tensor", v1).detach().cpu().reshape(-1).tolist()
        if vals0 == vals1:
            return _embed_jax_const(v0)
        if len(vals0) != 1:
            raise NotImplementedError(
                "helion.precompile(jax_fn=True) cannot derive a multi-element "
                "dynamic constant tensor launch arg"
            )
        expr = _match_input_dim(vals0[0], vals1[0], in_shapes0, in_shapes1)
        return f"jnp.array([{expr}], dtype={_torch_dtype_to_jnp_name(v0.dtype)})"
    if v0 == v1:
        return _embed_jax_const(v0)
    return cast(
        "str",
        _match_input_dim(cast("int", v0), cast("int", v1), in_shapes0, in_shapes1),
    )


def _precompile_pallas_jax(
    input: PrecompilationInput,  # noqa: A002
    args: tuple[object, ...],
) -> PrecompilationResult:
    """Emit a pure-JAX standalone: a ``jax.Array`` entrypoint that runs the kernel
    via the real ``pl.kernel`` compile core (``_pallas_jax_call``) -- the *same*
    launch path the jax_fn runtime uses -- with no torch / helion dependency.

    Captures the launch metadata (grid, per-tensor block specs, output shape/dtype,
    input/output arg positions) by running the compiled host wrapper with a
    capturing launcher, then inlines the JAX-only slice of the Pallas launcher and
    emits a jax-native wrapper that calls ``_pallas_jax_call``. Constants the host
    wrapper creates -- lifted module scalars and specialization ints -- are baked
    into the standalone. Kernels using advanced Pallas features (scratch/pipeline/
    SMEM/ds-pad/in-place/compact-worklist/matmul-dot-general) or int64/uint64/
    float64 args are not supported yet and raise ``NotImplementedError``.
    """
    kernel = input.kernel
    bound = kernel.bind(args)
    bound.ensure_config_exists(args)
    config = bound._implicit_config()
    assert config is not None, "ensure_config_exists did not resolve a config"

    generated = bound.to_triton_code(config)
    compiled = bound.compile_config(config)

    # Capture the launch metadata by running the host wrapper with a launcher
    # that records its arguments instead of executing the kernel.
    captured: dict[str, Any] = {}

    def _capture(
        pallas_kernel: object, grid: object, *launch_args: object, **kw: object
    ) -> object:
        captured["grid"] = tuple(int(g) for g in cast("Any", grid))
        captured["args"] = launch_args
        captured["kwargs"] = kw
        out_indices = cast("list[int]", kw.get("_output_indices") or [])
        # Mirror the real launcher's return convention so the host wrapper's
        # ``a, b = _launcher(...)`` unpack (multi-output kernels) succeeds: a
        # tuple for >1 outputs, the bare tensor for one, None for zero.
        if len(out_indices) > 1:
            return tuple(launch_args[i] for i in out_indices)
        return launch_args[out_indices[0]] if out_indices else None

    compiled(*args, _launcher=_capture)

    kw = captured["kwargs"]
    for name in _JAX_UNSUPPORTED_KWARGS:
        if kw.get(name):
            raise NotImplementedError(
                f"helion.precompile(jax_fn=True) does not support kernels using "
                f"{name!r} yet (kernel {kernel.name!r})"
            )

    launch_args = cast("tuple[object, ...]", captured["args"])
    output_indices = list(cast("list[int]", kw.get("_output_indices") or []))
    block_spec_info = cast("list[Any] | None", kw.get("_block_spec_info"))
    if block_spec_info is None:
        # Emitted only when codegen resolved a grid/tiling; its absence means a
        # no-tiling / degenerate-grid kernel the launch core can't map.
        raise NotImplementedError(
            "helion.precompile(jax_fn=True) does not support kernels without a "
            "resolved block spec (no-tiling / degenerate grid) yet"
        )
    for a in launch_args:
        if isinstance(a, torch.Tensor) and a.dtype in _JAX_UNSUPPORTED_DTYPES:
            raise NotImplementedError(
                f"helion.precompile(jax_fn=True) does not support {a.dtype} tensors "
                "(unsupported on TPU / narrowed by JAX x32)"
            )
    # The host wrapper passes the kernel's own arguments first, then the values it
    # creates itself: output buffers, lifted module-scalar constants (e.g.
    # ``torch.tensor([_NEG])``), specialization scalars (e.g. a reduction dim size
    # as a plain int), and shape-derived scalars (e.g. a reduction's row-count
    # ``torch.tensor([t])``). The standalone entrypoint takes only the user inputs;
    # every other launch arg is reconstructed inline.
    n_user = len(args)
    user_positions = [p for p in range(n_user) if p not in output_indices]
    const_positions = [
        p
        for p in range(len(launch_args))
        if p not in user_positions and p not in output_indices
    ]
    out_dtypes = [
        _torch_dtype_to_jnp_name(cast("torch.Tensor", launch_args[p]).dtype)
        for p in output_indices
    ]
    interpret = bool(kw.get("_pallas_interpret") or False)

    # Derive the grid, output shapes, and shape-derived scalar launch args from the
    # RUNTIME input shapes so a single standalone is correct at every dynamic shape.
    # One trace can't tell a value that happens to equal the sample size from one
    # that tracks an input dim -- and a materialized row-count ``torch.tensor([t])``
    # even specializes that dim during tracing -- so probe a SECOND shape (each
    # symbolic input dim scaled by a distinct factor) and compare: a launch value
    # that moved tracks the input dim it moved with (derive it); one that stayed is
    # a genuine constant (bake it). Static kernels have no symbolic dims, so every
    # value stays -> all baked (identical standalone as before).
    grid0 = cast("tuple[int, ...]", captured["grid"])
    in_shapes0 = [
        [int(s) for s in cast("torch.Tensor", launch_args[p]).shape]
        for p in user_positions
    ]

    sym_dims: dict[str, list[tuple[int, int]]] = {}
    for i, fake in enumerate(bound.fake_args):
        shape = getattr(fake, "shape", None)
        if shape is None:
            continue
        for d, size in enumerate(shape):
            if isinstance(size, torch.SymInt) and size.node.expr.is_symbol:
                sym_dims.setdefault(str(size.node.expr), []).append((i, d))

    if sym_dims:
        sym_factor = {sym: _PROBE_FACTORS[k] for k, sym in enumerate(sorted(sym_dims))}
        probe_cap: dict[str, Any] = {}

        def _probe(pk: object, grid: object, *pa: object, **pkw: object) -> object:
            probe_cap["grid"] = tuple(int(g) for g in cast("Any", grid))
            probe_cap["args"] = pa
            poi = cast("list[int]", pkw.get("_output_indices") or [])
            if len(poi) > 1:
                return tuple(pa[i] for i in poi)
            return pa[poi[0]] if poi else None

        probe_args = _scaled_probe_args(args, bound.fake_args, sym_factor)
        compiled(*probe_args, _launcher=_probe)
        grid1 = cast("tuple[int, ...]", probe_cap["grid"])
        launch1 = cast("tuple[object, ...]", probe_cap["args"])
        in_shapes1 = [
            [int(s) for s in cast("torch.Tensor", launch1[p]).shape]
            for p in user_positions
        ]
    else:
        grid1, launch1, in_shapes1 = grid0, launch_args, in_shapes0

    grid_exprs = [
        _grid_axis_expr(g0, g1, in_shapes0, in_shapes1)
        for g0, g1 in zip(grid0, grid1, strict=True)
    ]
    out_shape_exprs: list[list[str]] = []
    for p in output_indices:
        sh0 = [int(s) for s in cast("torch.Tensor", launch_args[p]).shape]
        sh1 = [int(s) for s in cast("torch.Tensor", launch1[p]).shape]
        out_shape_exprs.append(
            [
                _match_input_dim(a, b, in_shapes0, in_shapes1) or repr(a)
                for a, b in zip(sh0, sh1, strict=True)
            ]
        )
    const_slots = {
        p: _const_slot_expr(launch_args[p], launch1[p], in_shapes0, in_shapes1)
        for p in const_positions
    }

    entrypoint = input.entrypoint_name or kernel.name
    source = _build_pallas_jax_standalone(
        generated,
        kernel.name,
        entrypoint,
        grid_exprs=grid_exprs,
        output_indices=output_indices,
        user_positions=user_positions,
        const_slots=const_slots,
        block_spec_info=cast("list[Any]", block_spec_info),
        out_shape_exprs=out_shape_exprs,
        out_dtypes=out_dtypes,
        interpret=interpret,
        n_args=len(launch_args),
    )
    return _write_standalone(input, entrypoint, source)


def _device_kernel_body(generated: str, kernel_name: str) -> str:
    """The generated module minus imports and the host wrapper -- i.e. the
    ``_helion_<name>`` device kernel(s) and any module-level constants."""
    _, body = _split_module_source(generated)
    marker = f"def {kernel_name}("
    idx = body.find(f"\n{marker}")
    if body.startswith(marker):
        idx = 0
    if idx > 0:
        body = body[:idx]
    body = body.rstrip("\n")
    if "import helion" in body or "from helion" in body:
        raise NotImplementedError(
            f"cannot precompile {kernel_name!r} for jax_fn: the device kernel "
            "references helion (an in-kernel helper is not inlined yet)"
        )
    # The standalone imports no torch; a device body that references torch in
    # code (not just in ``# src[...]`` provenance comments, which is why this is
    # AST-based) would NameError at runtime.
    if _references_name(body, "torch"):
        raise NotImplementedError(
            f"cannot precompile {kernel_name!r} for jax_fn: the device kernel "
            "references torch (only jax-native device code is supported)"
        )
    return body


def _references_name(src: str, name: str) -> bool:
    """True if ``src`` references ``name`` in code (comments/strings ignored).

    ``src`` is generated kernel code and must be valid Python; a parse failure is
    a codegen bug and is left to raise rather than silently swallowed (masking it
    would let an unparseable body slip into the standalone).
    """
    return any(
        isinstance(node, ast.Name) and node.id == name
        for node in ast.walk(ast.parse(src))
    )


def _is_torch_import(imp: str) -> bool:
    """True if ``imp`` is an ``import torch`` / ``from torch ...`` statement."""
    return imp == "import torch" or imp.startswith(
        ("import torch.", "import torch ", "from torch.", "from torch ")
    )


def _stmt_def_names(node: ast.stmt) -> list[str]:
    """Top-level names a statement binds (function/class/assignment targets)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _code_name_refs(node: ast.AST) -> set[str]:
    """Names referenced in a node's *code* (calls, attribute bases, values) --
    ignoring type annotations, which stay lazy strings under
    ``from __future__ import annotations`` and never execute at runtime."""
    refs: set[str] = set()

    def visit(n: ast.AST) -> None:
        if isinstance(n, ast.Name):
            refs.add(n.id)
        for field, value in ast.iter_fields(n):
            if field in ("annotation", "returns"):
                continue
            if isinstance(value, ast.AST):
                visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        visit(item)

    visit(node)
    return refs


def _launcher_jax_slice() -> tuple[list[str], str]:
    """Return ``(imports, body)`` of the JAX-only slice of the Pallas launcher.

    Keeps exactly the transitive *code* closure of ``_pallas_jax_call`` -- the
    shared compile core (``_pallas_compile_jit_fn`` / block specs / ``pl.kernel``
    / the compact variant) -- and drops everything else (the torch launcher,
    JaxCallable dispatch, torch<->jax conversions, ``import torch``). This reuses
    the exact runtime launch logic; torch names left in kept functions' *type
    annotations* are lazy strings that never execute.
    """
    imports, body = _load_launcher_source("helion.runtime.pallas.launcher")
    imports = [
        imp for imp in imports if "helion" not in imp and not _is_torch_import(imp)
    ]
    tree = ast.parse(body)
    defs: dict[str, ast.stmt] = {}
    for node in tree.body:
        for name in _stmt_def_names(node):
            defs[name] = node

    keep: set[str] = set()
    queue = ["_pallas_jax_call"]
    while queue:
        name = queue.pop()
        if name in keep or name not in defs:
            continue
        keep.add(name)
        queue.extend(_code_name_refs(defs[name]))

    kept_body = [
        node
        for node in tree.body
        if any(name in keep for name in _stmt_def_names(node))
    ]
    sliced = ast.unparse(ast.Module(body=kept_body, type_ignores=[]))
    # Torch in annotations is fine (lazy strings); torch in *code* is a bug.
    if "torch" in _code_name_refs(ast.parse(sliced)):
        raise AssertionError(
            "jax_fn launcher slice unexpectedly references torch in code; the "
            "compile core reachable from _pallas_jax_call must stay torch-free."
        )
    return imports, sliced


def _build_pallas_jax_standalone(
    generated: str,
    kernel_name: str,
    entrypoint_name: str,
    *,
    grid_exprs: list[str],
    output_indices: list[int],
    user_positions: list[int],
    const_slots: dict[int, str],
    block_spec_info: list[Any],
    out_shape_exprs: list[list[str]],
    out_dtypes: list[str],
    interpret: bool,
    n_args: int,
) -> str:
    """Assemble the pure-JAX standalone: inlined JAX-only launcher slice + the
    generated device kernel + a jax-native wrapper that drives ``_pallas_jax_call``
    (the real ``pl.kernel`` compile core).

    The wrapper derives the grid, output shapes, and shape-derived scalar args from
    the runtime input shapes (``grid_exprs`` / ``out_shape_exprs`` / ``const_slots``
    reference ``inputs[i].shape[d]``), so one standalone is correct at every dynamic
    shape. Static dims come through as literals, so a static kernel bakes plain
    constants."""
    device_body = _device_kernel_body(generated, kernel_name)
    device_kernel = f"_helion_{kernel_name}"

    launcher_imports, launcher_body = _launcher_jax_slice()

    # Preserve the generated module's imports (minus helion/torch): the device
    # body references e.g. ``lax`` for dtype casts. get_needed_import_lines only
    # emits used imports, so a cast-free kernel won't carry ``import jax.lax``.
    gen_imports, _ = _split_module_source(generated)
    gen_imports = [
        imp for imp in gen_imports if "helion" not in imp and not _is_torch_import(imp)
    ]
    imports = _dedupe_preserve_order(
        [
            "import jax",
            "import jax.numpy as jnp",
            "from jax.experimental import pallas as pl",
            *gen_imports,
            *launcher_imports,
        ]
    )

    # Host-wrapper-created constants (lifted module scalars, specialization ints)
    # are filled into their slots by value; the entrypoint takes only user inputs.
    const_lines = [
        f"    slots[{p}] = {expr}" for p, expr in sorted(const_slots.items())
    ]
    if const_lines:
        const_lines = [
            "    # constants baked in from the original host wrapper:",
            *const_lines,
        ]

    out_lines = [
        f"    slots[{pos}] = jnp.empty("
        f"({', '.join(out_shape_exprs[oi])},), {out_dtypes[oi]})"
        for oi, pos in enumerate(output_indices)
    ]

    parts = [
        f"# Auto-generated by helion.precompile (jax_fn) for kernel '{kernel_name}'.",
        "# Standalone: depends only on jax, no helion/torch at runtime.",
        "from __future__ import annotations",
        "",
        *imports,
        "",
        "# --- inlined Pallas launcher (jax-only slice of runtime/pallas/launcher.py) ---",
        launcher_body,
        "",
        "# --- device kernel (generated) ---",
        device_body,
        "",
        "# --- pure-JAX launch (reuses the real pl.kernel compile core) ---",
        f"_BLOCK_SPEC_INFO = {block_spec_info!r}",
        f"_OUTPUT_INDICES = {output_indices!r}",
        f"_USER_POSITIONS = {user_positions!r}",
        f"_INTERPRET = {interpret!r}",
        f"_N_ARGS = {n_args}",
        "",
        f"def {entrypoint_name}(*inputs):",
        "    # Grid + output shapes are derived from the runtime input shapes, so a",
        "    # single standalone is correct at every (dynamic) input shape.",
        f"    _grid = ({', '.join(grid_exprs)},)",
        "    slots = [None] * _N_ARGS",
        "    for pos, inp in zip(_USER_POSITIONS, inputs):",
        "        slots[pos] = inp",
        *out_lines,
        *const_lines,
        "    results = _pallas_jax_call(",
        f"        {device_kernel},",
        "        _grid,",
        "        tuple(slots),",
        "        output_indices=_OUTPUT_INDICES,",
        "        inplace_indices=[],",
        "        block_spec_info=_BLOCK_SPEC_INFO,",
        "        scratch_shapes=None,",
        "        hbm_arg_indices=None,",
        "        smem_arg_indices=None,",
        "        interpret=_INTERPRET,",
        "        compact=None,",
        "    )",
        "    return results[0] if len(results) == 1 else tuple(results)",
        "",
    ]
    return "\n".join(parts)
