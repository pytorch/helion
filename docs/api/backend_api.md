# Backend API (experimental)

`helion.backend_api` is the interface for codegen backends that live outside
the Helion repository. A backend subclasses `helion.backend_api.Backend`
and imports Helion internals only from this module.

```{warning}
This interface is experimental. Out-of-tree backends should pin one Helion
minor version (e.g. `helion>=1.Y,<1.(Y+1)`).
```

## Packaging a backend plugin

Declare the backend class in the `helion.backends` entry-point group of the
plugin's `pyproject.toml`, using the backend's name as the entry-point name,
and declare the version of this API it targets on the class:

```toml
[project.entry-points."helion.backends"]
mybackend = "my_pkg.backend:MyBackend"
```

```python
class MyBackend(helion.backend_api.Backend):
    HELION_BACKEND_API_VERSION = 1

    @property
    def name(self) -> str:
        return "mybackend"
```

- Installed plugins are listed by `helion.backend_api.list_backends()` (so
  `HELION_BACKEND` can name them), but a plugin is imported only when its
  backend is first used, e.g. `HELION_BACKEND=mybackend` or
  `@helion.kernel(backend="mybackend")`. Spawned autotuning workers load it
  the same way. `helion.backend_api.load_backend_plugins()` loads all of
  them eagerly.
- Per-op `@codegen` handlers go in `my_pkg/_codegen_modules.py` (the package
  of the backend class's module); Helion imports it right after registering
  the backend.
- Helion instantiates backend classes more than once (name discovery, plugin
  validation, and again per compile), so `__init__` must be cheap and free of
  side effects.
- Using a plugin raises `ValueError` if it fails to import, is not a `Backend`
  subclass, lacks or mismatches `HELION_BACKEND_API_VERSION`, names its backend
  differently from its entry point, or replaces another backend while
  importing. Its registry entry and the `@codegen` handlers defined in its own
  package are then removed. A plugin cannot overwrite an existing handler.
  The failure is remembered for the process lifetime, so fixing the package in
  place requires a fresh process to retry. The rollback covers the registry
  and `@codegen` registrations only: other import side effects (threads
  started, files written, modules cached in `sys.modules`) are not undone.
- A lookup from another thread while a plugin is loading waits for that load
  to finish. A plugin whose import looks up a plugin that is (directly or via
  another thread) waiting on it sees it as not yet available, instead of
  deadlocking. Any other wait on another thread's load is bounded (60s): on
  timeout the lookup raises `RuntimeError` (a likely circular plugin import or
  lookup), which is not remembered, so a later lookup retries. A retry does
  not re-run the plugin's already-imported module, so keep `@codegen`
  handlers in `_codegen_modules`, which Helion reloads.
- Entry points named after a built-in backend are ignored with a warning. A
  backend name provided by more than one installed distribution is an error:
  using it raises `ValueError` naming every provider, and none is imported,
  unless a class was already registered under that name by hand, which wins.
- Set `HELION_DISABLE_BACKEND_PLUGINS=1` to ignore installed plugins. It
  prevents *discovery* of plugins not yet loaded in the current process;
  plugins already registered stay usable. An unrecognized value is an error.

## What is covered

`BACKEND_API_VERSION` covers exactly the following. Anything else, including
other attributes of the classes listed here, may change without notice.

1. **Importability.** Every name in `helion.backend_api.__all__` stays
   importable from `helion.backend_api`, except `CuteVectorWidthSpec` (see
   *What is not covered* below), which is exported for convenience and may
   be renamed or removed without a bump. `register_compiler_backend` is
   idempotent for the same class and raises `ValueError` if a different
   class is registered under an already-registered name (including
   built-in names).
2. **`Backend` hooks.** The names and signatures of these `Backend` members:
   `name`, `dtype_str`, `acc_type`, `function_decorator`, `constexpr_type`,
   `default_launcher_name`, `library_imports` (the abstract members);
   `codegen_name`, `state`, `validate_environment`;
   `pre_inductor_lowering`, `inductor_op_overrides`;
   `customize_ast`, `pre_codegen`, `create_loop_strategy`,
   `create_reduction_strategy`;
   `program_id_expr`, `grid_index_expr`, `loop_index_expr`, `arange_expr`,
   `range_str`, `lane_index_expr`, `lane_offset_expr`,
   `thread_in_tile_mask_expr`, `scalar_load_expr`, `inline_constexpr`,
   `scalar_arg_preamble`, `kernel_prologue` (called once per `generate_ast`
   pass, after body codegen; its statements go after the scalar-argument
   preamble and before the kernel preamble and body; its `fn` argument is
   passed only so the backend can tell which device function this is, none of
   its attributes are covered by `BACKEND_API_VERSION`, and the backend must
   not modify it);
   `cast_expr`, `cast_scalar_ast`;
   `where_expr`, `minimum_expr`, `zeros_expr`, `full_expr`, `reshape_expr`,
   `broadcast_to_expr`, `expands_broadcast_dims`;
   `reduction_expr`, `reduction_combine_expr`, `reduction_index_expr`,
   `reduction_index_zero_expr`, `wrap_reduction_accumulator`,
   `looped_reduction_thread_count`, `register_reduction_loop_config_slots`,
   `max_reduction_threads`, `max_reduction_loop`,
   `reduction_block_size_is_inlined_constexpr`,
   `launches_surplus_tile_threads`;
   `launcher_keyword_args`;
   `supports_config_key`, `tunable_fragments`, `supports_precompile`,
   `adjust_block_size_constraints`, `autotune`,
   `get_do_bench`, `get_interleaved_bench`.
   The autotuner times candidates with `get_do_bench()` (single config) and
   `get_interleaved_bench()` (candidates compared side by side); returning
   `None` keeps Helion's GPU-event timers. A returned timer is called like
   `helion.autotuner.benchmarking.do_bench` (`fn`, keyword `warmup` and
   `rep` in ms, `return_mode`, `process_group_name`, and for some timers
   `l2_flush_mode`) or `interleaved_bench` (`fns`, keyword `repeat`,
   `desc`, `max_total_ms`, ...), and returns milliseconds (`do_bench` may
   return a tuple; the first value is used). Accept keyword arguments you
   do not use, because the set Helion passes can grow.
3. **Strategy subclassing.** The names and signatures of
   `TileStrategy.user_size`, `TileStrategy.codegen_grid`,
   `TileStrategy.codegen_device_loop`, `TileStrategy.compact_shape`,
   `TileStrategy.codegen_preamble` and `ReductionStrategy.codegen_reduction`.
4. **Op objects.** `reduce_op` and `mask_to_op` are the API functions that
   backends attach `@codegen` handlers to. Public ops are attached through
   `helion.language` (e.g. `hl.load`, `hl.store`, `hl.subscript`).
   The `codegen` decorator itself is covered as
   `codegen(original_fn, backend)` used as `@codegen(op, "backend_name")`.
5. **Sympy helper.** `symint_sympy_expr(x)` returns the `sympy.Expr` behind a
   `torch.SymInt`; its signature (takes a SymInt, returns `sympy.Expr`) is
   covered, which is all codegen sites rely on.
6. **Plugin contract.** The `helion.backends` entry-point group, the rule that
   an entry point is named after its backend, the
   `HELION_BACKEND_API_VERSION` class attribute, and the
   `<package>._codegen_modules` convention, as described above.

## What is not covered

The contents of `exc`, the behavior of `TritonBackend` beyond being
subclassable, what `InductorOpOverrides` aliases (torch's private
`OpsHandler[Any]`; only the name is covered), and `CuteVectorWidthSpec`
(provisional; it may be renamed to a backend-neutral name).
`CuteVectorWidthSpec` is in `__all__` for convenience and is exempt from
rule 1 above.

## Version policy

| Change | Action |
|---|---|
| Add an optional hook with a safe default | No bump |
| Make a hook required | Bump |
| Remove, rename, or change the signature of a covered member | Bump |
| Change the documented call order of a covered hook | Bump |
| Change anything not covered | No bump |

```{eval-rst}
.. py:data:: helion.backend_api.BACKEND_API_VERSION
   :type: int
   :value: 1

   Version of the covered interface above.
```
