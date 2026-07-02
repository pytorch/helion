"""Pure-JAX export path for Helion-generated Pallas kernels.

The torch-tensor launchers (``default_pallas_*_launcher``) compile a
``pl.pallas_call`` and wrap it in a ``JaxCallable`` so that torch tensor
inputs/outputs cross the torch<->JAX boundary.  This module reuses the
same compile step (``_pallas_compile_jit_fn``) but skips the JaxCallable
wrap so the kernel can be invoked from inside a ``jax.jit`` with JAX
arrays directly — exposed via ``Kernel.jax_fn``.

The Helion-generated wrapper is torch-flavoured (it calls
``q.size(-2)``, ``q.reshape(...)``, ``torch.empty_like(...)``).  To run
that wrapper unchanged on JAX inputs, we wrap each input in a
``_JaxExportTensor`` — a ``torch.Tensor`` subclass whose storage is
``device='meta'`` (so torch operations only touch shape/dtype) but
which carries the underlying JAX array on the side.  Operations the
wrapper performs that *return a tensor we will later read back into
JAX* — ``reshape`` / ``view`` / ``empty_like`` / ``F.pad`` — must be
intercepted via ``__torch_function__`` so the JAX side stays in sync.
Pure-shape introspection (``.size()``, ``.shape``, ``.ndim``,
``.dim()``) is left to the meta storage (it carries shape).  Anything
else falls through to torch's meta backend, which will raise from
``func`` itself if real storage is needed; if a future Helion
codegen change starts emitting a new tensor-returning op in the
host wrapper, that op needs adding to the intercept list here.

The export launcher recognises these adapters, extracts the JAX
arrays, drives the Pallas call, and re-wraps the JAX outputs as
adapters so the wrapper's trailing ``out.view(...)`` propagates the
JAX side correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import torch

if TYPE_CHECKING:
    from collections.abc import Callable

    from . import _BlockSpecInfo
    from . import _PallasLoopKind
    from .kernel import Kernel


_TORCH_TO_JNP_DTYPE: dict[torch.dtype, object] | None = None
_JNP_TO_TORCH_DTYPE: dict[object, torch.dtype] | None = None


def _build_dtype_maps() -> tuple[dict[torch.dtype, object], dict[object, torch.dtype]]:
    """Return cached torch<->jnp dtype maps, building them on first use.

    Built lazily so importing this module never imports JAX (which is
    optional and TPU-only on most checkouts).
    """
    global _TORCH_TO_JNP_DTYPE, _JNP_TO_TORCH_DTYPE
    if _TORCH_TO_JNP_DTYPE is not None and _JNP_TO_TORCH_DTYPE is not None:
        return _TORCH_TO_JNP_DTYPE, _JNP_TO_TORCH_DTYPE
    import jax.numpy as jnp

    pairs: list[tuple[torch.dtype, object]] = [
        (torch.float32, jnp.float32.dtype),
        (torch.float64, jnp.float64.dtype),
        (torch.float16, jnp.float16.dtype),
        (torch.bfloat16, jnp.bfloat16.dtype),
        (torch.int8, jnp.int8.dtype),
        (torch.int16, jnp.int16.dtype),
        (torch.int32, jnp.int32.dtype),
        (torch.int64, jnp.int64.dtype),
        (torch.uint8, jnp.uint8.dtype),
        (torch.bool, jnp.bool_.dtype),
    ]
    _TORCH_TO_JNP_DTYPE = dict(pairs)
    _JNP_TO_TORCH_DTYPE = {j: t for t, j in pairs}
    return _TORCH_TO_JNP_DTYPE, _JNP_TO_TORCH_DTYPE


def _jnp_to_torch_dtype(jnp_dtype: object) -> torch.dtype:
    _, j2t = _build_dtype_maps()
    return j2t[jnp_dtype]


class _JaxExportTensor(torch.Tensor):
    """A ``torch.Tensor`` subclass that carries a JAX array on the side.

    The underlying storage is ``device='meta'`` so all torch operations
    invoked by the Helion-generated wrapper see correct shape/dtype but
    perform no real compute on the torch side.  ``__torch_function__``
    intercepts the small set of operations the wrapper actually uses
    (``reshape``/``view``/``empty_like``/``F.pad``) and mirrors them on
    the JAX array so the JAX side stays in sync.

    Use ``_JaxExportTensor.from_jax(arr, device=...)`` to construct one;
    use ``adapter._jax_arr`` (or :func:`_unwrap_jax` on a result) to
    recover the underlying JAX array.
    """

    _jax_arr: object
    _declared_device: torch.device

    @staticmethod
    def from_jax(jax_arr: object, *, device: torch.device) -> _JaxExportTensor:
        torch_dtype = _jnp_to_torch_dtype(jax_arr.dtype)  # type: ignore[union-attr]
        meta = torch.empty(
            tuple(int(s) for s in jax_arr.shape),  # type: ignore[union-attr]
            dtype=torch_dtype,
            device="meta",
        )
        out = torch.Tensor._make_subclass(_JaxExportTensor, meta, require_grad=False)
        out._jax_arr = jax_arr
        out._declared_device = device  # pyrefly: ignore[read-only]
        return out

    @property
    def device(self) -> torch.device:  # pyrefly: ignore[bad-override]
        # Helion's ``_find_device``/specialization machinery reads this to
        # decide which backend to compile for; the real meta storage
        # would otherwise route compilation through the meta device.
        return self._declared_device

    @classmethod
    def __torch_function__(
        cls,
        func: object,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if kwargs is None:
            kwargs = {}

        # ``torch.empty_like(adapter, device='meta')`` shows up in
        # generated wrappers when the kernel allocates an output-only
        # tensor.  Allocate a JAX-side empty with matching shape/dtype
        # so the export launcher can use the JAX array as the output
        # placeholder.
        if func is torch.empty_like:
            template = args[0] if args else kwargs.get("input")
            assert isinstance(template, _JaxExportTensor)
            import jax.numpy as jnp

            new_jax = jnp.empty(
                tuple(int(s) for s in template._jax_arr.shape),  # type: ignore[union-attr]
                dtype=template._jax_arr.dtype,  # type: ignore[union-attr]
            )
            return cls.from_jax(new_jax, device=template._declared_device)

        # ``q_view = q_in.reshape([...])`` and ``out.view(...)`` are
        # the wrapper's only shape-mutating operations.  Re-wrap the
        # JAX array under the new shape so subsequent attribute reads
        # (``.size(...)``, ``.shape``) on the result stay in sync.
        if func is torch.Tensor.reshape or func is torch.Tensor.view:
            t = args[0]
            assert isinstance(t, _JaxExportTensor)
            new_shape = _normalize_shape(args[1:], kwargs)
            import jax.numpy as jnp

            new_jax = jnp.reshape(t._jax_arr, list(new_shape))  # type: ignore[arg-type]
            return cls.from_jax(new_jax, device=t._declared_device)

        # ``_pallas_apply_ds_padding`` calls ``torch.nn.functional.pad``
        # on adapter inputs to bring tile-misaligned tensors up to a
        # block multiple before the pallas_call.  Mirror the same
        # zero-padding on the JAX side so the JAX array passed to the
        # kernel matches the adapter's reported shape.
        if func is torch.nn.functional.pad:
            t = args[0]
            assert isinstance(t, _JaxExportTensor)
            pad_widths = args[1] if len(args) > 1 else kwargs["pad"]
            mode = kwargs.get("mode", "constant")
            value = kwargs.get("value")
            import jax.numpy as jnp

            # ``F.pad`` walks dims back-to-front in pairs of
            # (left_pad, right_pad).  ``jnp.pad`` takes a list of
            # ``(before, after)`` pairs in dim order — invert.
            pairs: list[tuple[int, int]] = [(0, 0)] * t.ndim
            n_pairs = len(pad_widths) // 2  # type: ignore[arg-type]
            for i in range(n_pairs):
                left = int(pad_widths[2 * i])  # type: ignore[index]
                right = int(pad_widths[2 * i + 1])  # type: ignore[index]
                pairs[t.ndim - 1 - i] = (left, right)
            assert mode == "constant", (
                f"_JaxExportTensor only supports F.pad mode='constant', got {mode!r}"
            )
            new_jax = jnp.pad(
                t._jax_arr,  # type: ignore[arg-type]
                pairs,
                mode="constant",
                constant_values=0 if value is None else value,
            )
            return cls.from_jax(new_jax, device=t._declared_device)

        # Catch-all: any op the wrapper might invoke that we haven't
        # special-cased above.  Shape-only ops (``.size()``, ``.shape``,
        # ``.ndim``, ``.dim()``) work on the meta storage; ops that
        # need real data raise from torch's meta backend.  If a new
        # tensor-returning op appears here whose result the wrapper
        # later passes back into the kernel, it must be added to the
        # intercept list above so the JAX side stays in sync.
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)  # type: ignore[operator]


def _normalize_shape(
    rest: tuple[object, ...], kwargs: dict[str, object]
) -> tuple[object, ...]:
    """Normalise the shape arg of ``Tensor.reshape``/``Tensor.view``.

    Accepts both ``t.reshape(2, 3)`` and ``t.reshape([2, 3])`` /
    ``t.reshape((2, 3))`` calling conventions.
    """
    if "shape" in kwargs:
        candidate = kwargs["shape"]
        if isinstance(candidate, (list, tuple)):
            return tuple(candidate)
        return (candidate,)
    if len(rest) == 1 and isinstance(rest[0], (list, tuple)):
        return tuple(rest[0])
    return rest


def _unwrap_jax(obj: object) -> object:
    """Recursively unwrap ``_JaxExportTensor`` instances to JAX arrays."""
    if isinstance(obj, _JaxExportTensor):
        return obj._jax_arr
    if isinstance(obj, tuple):
        return tuple(_unwrap_jax(x) for x in obj)
    if isinstance(obj, list):
        return [_unwrap_jax(x) for x in obj]
    return obj


def _device_for_jax_export() -> torch.device:
    """Pick the torch device used to drive Helion compilation in jax_fn.

    Helion's specialization key includes the input device type.  For
    the JAX-export path we don't actually run any torch ops on real
    storage, so we just need a device that lines up with the active
    backend.  Prefer ``tpu`` when ``torch_tpu`` is loaded; otherwise
    fall back to CPU (this matches the interpret-mode test
    environment).
    """
    if hasattr(torch, "tpu"):
        try:
            if torch.tpu.is_available():
                return torch.device("tpu", 0)
        except Exception:
            pass
    return torch.device("cpu")


def default_pallas_jax_launcher(
    pallas_kernel: object,
    grid: tuple[int, ...],
    *args: object,
    _output_indices: list[int] | None = None,
    _inplace_indices: list[int] | None = None,
    _block_spec_info: _BlockSpecInfo | None = None,
    _scratch_shapes: list[object] | None = None,
    _pipeline_arg_indices: list[int] | None = None,
    _ds_pad_dims: list[tuple[int, int, int, int]] | None = None,
    _smem_arg_indices: list[int] | None = None,
    _pallas_interpret: bool | None = None,
    _pallas_arbitrary_grid_dims: tuple[int, ...] | None = None,
    _kind: _PallasLoopKind | None = None,
    **kwargs: object,
) -> object:
    """Pallas launcher used when running a Helion kernel inside ``jax.jit``.

    Each tensor argument in ``args`` is a ``_JaxExportTensor`` whose
    underlying ``_jax_arr`` is either a concrete JAX array or a JAX
    tracer.  This launcher pulls the JAX side out, calls the shared
    ``_pallas_compile_jit_fn`` to build a fresh ``pl.pallas_call``
    (no JaxCallable, no torch<->JAX bridge), invokes it on the JAX
    inputs, and re-wraps the output(s) as ``_JaxExportTensor`` so the
    Helion wrapper's trailing reshape/view operations stay traceable.
    """
    from . import _pallas_apply_ds_padding
    from . import _pallas_compile_jit_fn
    from . import _pallas_output_only_descriptors
    from . import _pallas_padded_output_dims_by_arg
    from . import _pallas_slice_to_orig
    from . import _PallasLoopKind as _LoopKind
    from .settings import is_pallas_interpret

    interpret = (
        _pallas_interpret if _pallas_interpret is not None else is_pallas_interpret()
    )

    if _kind is None:
        _kind = _LoopKind.UNROLL
    output_indices = _output_indices if _output_indices is not None else []

    # Capture original shapes BEFORE padding so output-only tensors can
    # be sliced back after the pallas_call.  ``_pallas_apply_ds_padding``
    # calls ``F.pad`` on each entry, which the adapter intercepts and
    # mirrors on the JAX side, so no JAX-side duplicate is needed here.
    orig_shapes: dict[int, tuple[int, ...]] = {}
    if _ds_pad_dims:
        for arg_idx, _, _, _ in _ds_pad_dims:
            if arg_idx in orig_shapes:
                continue
            a = args[arg_idx]
            if isinstance(a, _JaxExportTensor):
                orig_shapes[arg_idx] = tuple(int(s) for s in a._jax_arr.shape)  # type: ignore[union-attr]

        args, _ = _pallas_apply_ds_padding(args, output_indices, _ds_pad_dims)

    device = next(
        (a._declared_device for a in args if isinstance(a, _JaxExportTensor)),
        _device_for_jax_export(),
    )

    if _kind is _LoopKind.COMPACT_WORKLIST:
        from . import _pallas_compile_compact_jit_fn

        result = _pallas_compile_compact_jit_fn(
            pallas_kernel,
            args,
            _output_indices=output_indices,
            _inplace_indices=_inplace_indices,
            _block_spec_info=_block_spec_info,
            _scratch_shapes=_scratch_shapes,
            _smem_arg_indices=_smem_arg_indices,
            _pipeline_arg_indices=_pipeline_arg_indices,
            build_worklist=cast("Any", kwargs["_compact_build_worklist"]),
            offset_arg_indices=cast(
                "Any", kwargs.get("_compact_offset_arg_indices") or []
            ),
            metadata_fields=cast("Any", kwargs.get("_compact_metadata_fields") or []),
            owner_ref_pos=cast("Any", kwargs.get("_compact_owner_ref_pos", 0)),
            num_scalar_prefetch=cast(
                "Any", kwargs.get("_compact_num_scalar_prefetch", 0)
            ),
            aligned_arg_indices=cast(
                "Any", kwargs.get("_compact_aligned_arg_indices") or []
            ),
            tile_start_ref_pos=cast(
                "Any", kwargs.get("_compact_tile_start_ref_pos", 1)
            ),
            compact_block=cast("Any", kwargs.get("_compact_block", 1)),
            interpret=interpret,
        )
    else:
        result = _pallas_compile_jit_fn(
            pallas_kernel,
            grid,
            args,
            kind=_kind,
            _output_indices=output_indices,
            _inplace_indices=_inplace_indices,
            _block_spec_info=_block_spec_info,
            _smem_arg_indices=_smem_arg_indices,
            _scratch_shapes=_scratch_shapes,
            _pipeline_arg_indices=_pipeline_arg_indices,
            _matmul_dot_general=None,
            interpret=interpret,
            _pallas_arbitrary_grid_dims=_pallas_arbitrary_grid_dims,
        )

    jax_inputs = [
        cast("_JaxExportTensor", args[i])._jax_arr for i in result.tensor_arg_indices
    ]
    jax_results = result.jit_fn(*jax_inputs)  # type: ignore[operator]
    if not isinstance(jax_results, (tuple, list)):
        jax_results = (jax_results,)

    # Same descriptor list the torch fast-path uses (see
    # ``_LauncherFastPath.output_only_descriptors``).  In-place positions
    # would normally alias back into a torch tensor on the torch path — but
    # JAX has no in-place mutation, so when every output is in-place we
    # surface them as fresh JAX values instead of returning ``None``.
    descriptors = _pallas_output_only_descriptors(
        output_indices, result.arg_to_tensor_pos
    )
    if not descriptors:
        descriptors = tuple(enumerate(output_indices))

    output_results: list[object] = [jax_results[out_idx] for out_idx, _ in descriptors]
    output_orig_pos: list[int] = [orig_pos for _, orig_pos in descriptors]

    # Slice padded output results back to their original shapes via the
    # same ``arg → padded dims`` grouping the torch fast-path uses
    # (see ``_LauncherFastPath.padded_output_dims_by_arg``); we just
    # slice on JAX arrays via the shared ``_pallas_slice_to_orig`` helper.
    if _ds_pad_dims and orig_shapes:
        padded_dims_by_arg = _pallas_padded_output_dims_by_arg(
            _ds_pad_dims, set(orig_shapes.keys())
        )
        for i, orig_pos in enumerate(output_orig_pos):
            dims = padded_dims_by_arg.get(orig_pos)
            orig_shape = orig_shapes.get(orig_pos)
            if dims and orig_shape is not None:
                output_results[i] = _pallas_slice_to_orig(
                    cast(
                        "torch.Tensor", output_results[i]
                    ),  # JAX arrays index identically
                    dims,
                    cast("torch.Size", orig_shape),
                )

    if len(output_results) == 1:
        return _JaxExportTensor.from_jax(output_results[0], device=device)
    return tuple(_JaxExportTensor.from_jax(r, device=device) for r in output_results)


def make_jax_fn(kernel: Kernel) -> Callable[..., Any]:
    """Build the callable returned by ``Kernel.jax_fn``.

    The returned callable can be invoked with JAX arrays / tracers
    inside ``jax.jit`` and returns JAX arrays.  Internally it drives
    the same Helion-generated wrapper that the torch-tensor launchers
    use; only the launcher kwarg changes.
    """

    def _runtime_call(*args: object) -> object:
        from . import _PallasLoopKind as _LoopKind

        device = _device_for_jax_export()
        adapter_args: list[object] = []
        for a in args:
            if (
                hasattr(a, "shape")
                and hasattr(a, "dtype")
                and not isinstance(a, torch.Tensor)
            ):
                adapter_args.append(_JaxExportTensor.from_jax(a, device=device))
            else:
                adapter_args.append(a)

        # Compile the helion kernel for these (adapter) arg shapes.  We
        # go through the same Kernel.bind/compile path used by the
        # torch entrypoint so specialization, autotune, etc. all run
        # exactly once per shape signature — the adapter is a
        # torch.Tensor subclass so this just works.
        bound = kernel.bind(tuple(adapter_args))
        bound.ensure_config_exists(adapter_args)
        compiled = bound._run
        assert compiled is not None

        # Resolve the JAX-export launcher to use for the kernel's
        # configured loop type.  ``_run`` is the generated wrapper
        # which forwards ``_launcher`` via kwargs.
        config = bound._config
        assert config is not None
        kind = _LoopKind(cast("str", config.config.get("pallas_loop_type", "unroll")))

        def _launcher(
            pallas_kernel: object,
            grid: tuple[int, ...],
            *launch_args: object,
            **launch_kwargs: object,
        ) -> object:
            return default_pallas_jax_launcher(
                pallas_kernel,
                grid,
                *launch_args,
                _kind=kind,
                **cast("dict[str, Any]", launch_kwargs),
            )

        result = compiled(*adapter_args, _launcher=_launcher)
        return _unwrap_jax(result)

    return _runtime_call
