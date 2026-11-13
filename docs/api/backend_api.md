# Backend API (experimental)

`helion.backend_api` is the interface for codegen backends that live outside
the Helion repository. A backend subclasses `helion.backend_api.Backend`
and imports Helion internals only from this module.

```{warning}
This interface is experimental. Out-of-tree backends should pin one Helion
minor version (e.g. `helion>=1.Y,<1.(Y+1)`).
```

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
   `supports_config_key`, `tunable_fragments`, `supports_precompile`,
   `adjust_block_size_constraints`, `autotune`,
   `get_do_bench`, `get_interleaved_bench`.
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
   `TileStrategy.codegen_device_loop`, `TileStrategy.compact_shape`,
   `TileStrategy.codegen_preamble` and `ReductionStrategy.codegen_reduction`
   (also `helion.backend_api.COVERED_STRATEGY_METHODS`).
4. **Op objects.** `reduce_op` and `mask_to_op` are the API functions that
   backends attach `@codegen` handlers to. Public ops are attached through
   `helion.language` (e.g. `hl.load`, `hl.store`, `hl.subscript`).
   The `codegen` decorator itself is covered as
   `codegen(original_fn, backend)` used as `@codegen(op, "backend_name")`.
5. **Sympy helper.** `symint_sympy_expr(x)` returns the `sympy.Expr` behind a
   `torch.SymInt`; its signature (takes a SymInt, returns `sympy.Expr`) is
   covered, which is all codegen sites rely on.

## What is not covered

The contents of `exc`, the behavior of `TritonBackend` beyond being
subclassable, and what `InductorOpOverrides` aliases (torch's private
`OpsHandler[Any]`; only the name is covered).

## Version policy

| Change | Action |
|---|---|
| Add an optional hook with a safe default | No bump |
| Make a hook required | Bump |
| Remove, rename, or change the signature of a covered member | Bump |
| Change anything not covered | No bump |

```{eval-rst}
.. py:data:: helion.backend_api.BACKEND_API_VERSION
   :type: int
   :value: 1

   Version of the covered interface above.
```
