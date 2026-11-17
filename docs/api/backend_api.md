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
  `@helion.kernel(backend="mybackend")`. A fresh process (e.g. a spawned
  worker) loads it the same way on its first lookup.
  `helion.backend_api.load_backend_plugins()` loads all of them eagerly.
- Per-op `@codegen` handlers go in `my_pkg/_codegen_modules.py` (the package
  of the backend class's module); Helion imports it right after registering
  the backend. These modules run twice: once when the plugin loads, and again
  on the backend's first use, when Helion's codegen repair reloads
  `_codegen_modules` and each of the package's modules it imports. Keep them
  free of import-time side effects, and do not rely on `isinstance` checks
  against classes they define (the reload creates new class objects).
- A plugin that registers itself (`register_compiler_backend` at import) and
  is imported directly before any lookup is used as registered: the
  entry-point checks below are skipped and its `_codegen_modules` is imported
  only when its codegen is first needed. Let Helion load it through the
  entry point instead.
- Helion instantiates backend classes more than once (plugin validation,
  registration, and again per compile), so `__init__` must be cheap and free
  of side effects.
- Using a plugin raises `ValueError` if it fails to import, is not a `Backend`
  subclass, lacks or mismatches `HELION_BACKEND_API_VERSION`, names its
  backend differently from its entry point, or replaces another backend while
  importing. Every backend name registered during the failed load is then
  unregistered (names registered before it stay); nothing else is undone
  (`@codegen` handlers, modules in `sys.modules`, other import side effects).
  A plugin cannot overwrite an existing handler (unless Python runs with
  `-O`). The failure, including an interrupted load (e.g. Ctrl-C), is
  remembered for the process lifetime, so a fresh process is required to
  retry.
- Plugins load one at a time. Do not look up your own backend while your
  modules are being imported (e.g. defining kernels for it at import time):
  during Helion's load the lookup sees the backend as not yet available, and
  if your module is imported directly while another thread loads the plugin,
  the two threads can deadlock.
- Entry points named after a built-in backend are ignored with a warning. A
  backend name provided by more than one installed distribution is an error:
  using it raises `ValueError` naming every provider, and none is imported,
  unless a class was already registered under that name by hand, which wins.
- Set `HELION_DISABLE_BACKEND_PLUGINS=1` to ignore installed plugins. It is
  read once, when plugins are first discovered (the first `list_backends()`,
  or the first use of a backend name that is not registered); set it before
  that. Plugins already registered stay usable. An unrecognized value is an
  error.

## What is covered

`BACKEND_API_VERSION` covers exactly the following. Anything else, including
other attributes of the classes listed here, may change without notice.

1. **Importability.** Every name in `helion.backend_api.__all__` stays
   importable from `helion.backend_api`. `register_compiler_backend` is
   idempotent for the same class and raises `ValueError` if a different
   class is registered under an already-registered name (including
   built-in names). The exception is a class with the same module and
   qualified name, as a reloaded module (`importlib.reload`) or a re-run
   factory defines: it replaces the earlier one (so a factory that builds two
   different backends under one name replaces the first silently).
2. **`Backend` hooks.** The names and signatures of these `Backend` members
   (the list is also `helion.backend_api.COVERED_BACKEND_HOOKS`):
   `name`, `dtype_str`, `acc_type`, `function_decorator`, `constexpr_type`,
   `default_launcher_name`, `library_imports` (the abstract members);
   `codegen_name`, `state`, `validate_environment`;
   `pre_inductor_lowering`, `inductor_op_overrides`;
   `customize_ast`, `pre_codegen`, `create_loop_strategy`,
   `create_reduction_strategy`;
   `program_id_expr`, `grid_index_expr`, `loop_index_expr`, `arange_expr`,
   `range_str`, `lane_index_expr`, `lane_offset_expr`,
   `thread_in_tile_mask_expr`, `scalar_load_expr`, `inline_constexpr`,
   `scalar_arg_preamble`, `finalize_device_body` (called once per device
   function after its body is generated and before variable renaming;
   returns the body and the kernel parameters to append. Of its
   `DeviceBodyContext`, `device_function`, `args` and `body_start` (the index
   of the first traced-body statement; the statements before it are the
   `scalar_arg_preamble` statements, then the device function's own preamble,
   e.g. persistent program-id setup) are covered);
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
   `supports_config_key`, `config_spec_extension`, `tunable_fragments`,
   `supports_precompile`, `adjust_block_size_constraints`, `autotune`,
   `get_do_bench`, `get_interleaved_bench`.

   `config_spec_extension` returns a `BackendSearchSurface` (normally one
   module-level instance) whose `config_keys` are the backend's own autotuner
   keys and whose `debug_keys` are keys `normalize` accepts but the search
   does not try. Other backends reject these keys unless they list them
   explicitly, and while the backend is registered no kernel can use their
   names (or the names without a trailing `s`) for `hl.register_tunable`, so
   prefix them with the plugin's name. A plugin key is searched when
   `tunable_fragments` gives it a fragment; as for built-in tunables,
   `normalize` fills in its default but does not check a given value.
   `tunable_fragments` may return Helion's built-in backend tunables and the
   backend's own surface keys; declaring a gated generic key such as
   `xcd_remap` on the surface does not add it. `supports_config_key`
   overrides must defer to `super()` for these keys (an override that accepts
   a fixed list of keys must list them), or `ConfigSpec` raises; for a key
   supported only on some targets, `tunable_fragments` must return its
   fragment only on those targets.
   Registering raises `ValueError` if either set holds a generic key such as
   `num_warps`. Of `BackendSearchSurface`, `config_keys`, `debug_keys` and
   `fingerprint_extra` are covered; its other members (e.g. `fields`) are
   not.

   The surface's `fingerprint_extra(config_spec)` returns extra
   identity for the autotune cache (e.g. the plugin's package version): return
   a value whose `repr` is the same in every process and host, or `None` (the
   default, which leaves cache keys unchanged). A changed value makes cached
   configs miss, so an upgraded plugin re-tunes; it also stops older results
   from seeding the new search.

   The autotuner times candidates with `get_do_bench()` (single config) and
   `get_interleaved_bench()` (candidates compared side by side); returning
   `None` keeps Helion's GPU-event timers. A returned timer is called like
   `helion.autotuner.benchmarking.do_bench` (`fn`, keyword `warmup` and
   `rep` in ms, `return_mode`, `process_group_name`, and for some timers
   `l2_flush_mode`) or `interleaved_bench` (`fns`, keyword `repeat`,
   `desc`, `max_total_ms`, ...), and returns milliseconds: a `float` from
   the single-config timer, a list from the interleaved one. Accept keyword
   arguments you do not use, because the set Helion passes can grow. With
   a custom `get_do_bench`, candidates are timed in the autotuning process
   (not in a separate benchmark process) and `autotune_benchmark_input_banks`
   must stay 1; `autotune_benchmark_fn`, when set, takes the place of
   `get_interleaved_bench`.
3. **Strategy subclassing.** The names and signatures of
   `TileStrategy.user_size`, `TileStrategy.codegen_grid`,
   `TileStrategy.codegen_device_loop`, `TileStrategy.codegen_preamble` and
   `ReductionStrategy.codegen_reduction`
   (also `helion.backend_api.COVERED_STRATEGY_METHODS`).
4. **Op objects.** `reduce_op` and `mask_to_op` are the API functions that
   backends attach `@codegen` handlers to. Public ops are attached through
   `helion.language` (e.g. `hl.load`, `hl.store`, `hl.subscript`).
   The `codegen` decorator itself is covered as
   `codegen(original_fn, backend)` used as `@codegen(op, "backend_name")`.
5. **Sympy helper.** `symint_sympy_expr(x)` returns the `sympy.Expr` behind a
   `torch.SymInt`; its signature (takes a SymInt, returns `sympy.Expr`) is
   covered, which is all codegen sites rely on.
6. **Per-function state.** `DeviceFunction.backend_codegen_state`, a dict for
   codegen state shared across a backend's op handlers while one device
   function is generated. Use your backend's name as the key; the value is
   yours.
7. **Plugin contract.** The `helion.backends` entry-point group, the rule that
   an entry point is named after its backend, the
   `HELION_BACKEND_API_VERSION` class attribute, and the
   `<package>._codegen_modules` convention, as described above.

## What is not covered

The contents of `exc`, the behavior of `TritonBackend` beyond being
subclassable, and what `InductorOpOverrides` aliases (torch's private
`OpsHandler[Any]`; only the name is covered).

## Version policy

| Change | Action |
|---|---|
| Add an optional hook with a safe default | No bump |
| Add a covered attribute or exported name | No bump |
| Make a hook required | Bump |
| Remove, rename, or change the signature of a covered member | Bump |
| Change anything not covered | No bump |

```{eval-rst}
.. py:data:: helion.backend_api.BACKEND_API_VERSION
   :type: int
   :value: 1

   Version of the covered interface above.
```
