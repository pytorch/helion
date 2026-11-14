# The CuTe backend

`helion/_compiler/cute` compiles Helion kernels to the CuTe DSL (the CUTLASS
Python DSL); `helion/runtime/cute` launches what it generates.  The design is
*a collection of special-purpose and general-purpose code emitters that compose
together*: the general-purpose emitters render one site of a kernel (a load, a
store, a reduction, a lane loop, a PTX primitive) without knowing which kernel
they are in; the special-purpose emitters own a whole root loop or a whole
kernel (a tcgen05 GEMM pipeline, a flash-attention body, an MMA dataflow, a
recurrence) and are written over the general-purpose ones.  The backend-neutral
compiler reaches the package only through the `Backend` hooks of
`helion/_compiler/backend.py`.  Everything named below exists on the tree;
grep for it.

## 1. Map

| path | role |
|---|---|
| `backend.py` | `CuteBackend` (the hook implementations) with the module-level MMA / attention loop detectors before the class; the per-thread tile planner it answers is core's `helion/_compiler/per_thread_planning.py` |
| `emitter_protocols.py` (one level up, core) | `RootEmitter`, `KernelGenerator`, `RetryPolicy`, `EmittedBody`: the contracts between the pipeline and any backend's emitters |
| `emitters/` | `ROOT_EMITTERS` (the special-purpose root registry) beside the general-purpose helpers `memory_access`, `lane_packets`, `persistent_item_loop`, `role_item_loop` |
| `kernel_generators.py` | `KERNEL_GENERATORS` (whole-kernel generators) and `CuteRegeneration` (the retry policy) |
| `ir_passes.py`, `search_surface.py` | the device-IR stage passes and the knob registration that runs with them |
| `passes/` | `pipeline.py` (the device-body pass tables and `PassContext`), `pdl.py`, `launch_attributes.py`, `hoist_chunk_recurrence.py` |
| `lane_loops/` | the per-thread element loops: markers, placement, split, barriers, owner guards, `nest_check.py` (the nest check and barrier pass of the per-thread strategies), `root_passes.py` |
| `lane_cluster_split.py`, `tile_packing.py` | the `cute_cluster_n` split of a lane-looped axis and the packed grid block, answered to the per-thread strategies through hooks |
| `analysis/` | `match_*` / `prove_*` readers of the device IR returning frozen facts |
| `state.py`, `device_state.py` | the typed objects behind the core `backend_state` slots |
| `autotune/` | `CuteSearchSurface`, the config-key tables, the flash / mma_dag surfaces, `heuristics/`, `llm_profile.py` (the LLM autotuner's prompt prose) |
| `tcgen05/`, `tcgen05_config/`, `cute_mma.py`, `strategies.py` | the tcgen05 (Blackwell tensor-core) GEMM family |
| `cute_flash*.py`, `flash_*.py`, `attention_plan.py`, `flex_*match.py` | the flash-attention families |
| `cute_mma_dag*.py` | the MMA dataflow (`mma_dag`) family |
| `chunk_*`, `gdn_recurrence*`, `*rank1_recurrence*`, `direct_affine_*` | the recurrence families |
| `selection/` | top-k, row top-k, radix select |
| `computed_fragment.py`, `fragment_*.py` | CTA-local fragment programs |
| `device/`, `*_runtime.py`, `*_helpers.py` | device helpers traced into kernels (`ptx.py` primitive rows, `sync.py`, the family runtimes) |
| `runtime_hooks.py`, `launch_shape.py`, `standalone.py` | the host-side hooks, the structural launch-shape check, the standalone exporter |
| `target_facts.py`, `_codegen_modules.py` | the target model and the per-CTA SMEM probes, the op-handler modules registered through `@_decorators.codegen(op, "cute")` |
| `helion/runtime/cute/` | `launcher.py`, `host_wrappers/`, the launch caches and descriptors, the compile cache |

## 2. The compile pipeline, hook by hook

`KernelCompiler.compile` (`kernel_compiler.py`) runs parse -> unroll ->
`Backend.customize_ast` -> type propagation -> config-spec finalization ->
`lower_to_device_ir`; `BoundKernel` then calls `generate_ast` per config and
the launcher runs the module.  The CuTe backend plugs in at these points:

1. **Host rewrites** -- `CuteBackend.customize_ast`: `online_to_3pass`
   (`rewrite_online_to_3pass`), `materialized_fission`
   (`apply_materialized_fission`, when the bound kernel recorded a fission
   plan), `full_slice_matmul` (`stripmine_full_slice_matmuls`),
   `segmented_matmul` (`normalize_segmented_matmuls`),
   `flatten_nested_reductions`, and last `stage_aligned_operands`
   (`stage_unaligned_matmul_operands`).
2. **Device IR** -- `decomp_table_overrides` keeps `aten.fmin` / `aten.fmax`;
   `rewrite_device_ir` is called at the 14 positions of `DeviceIRStage`
   (`backend.py`, core) and dispatches through the table in `ir_passes.py`
   (`select_random_stream_pass`, `canonicalize_traced_graphs_pass`,
   `refresh_scaled_contraction_blocks_pass`, `promote_partitioned_outputs_pass`,
   `cse_device_graphs_pass`, `mark_rolled_graph_nodes_pass`, and the
   `search_surface` registrations); `register_search_surface`
   (`search_surface.register_search_surface`) detects the flash, gated-flash
   and tcgen05 GEMM shapes and opens their knob families on the `ConfigSpec`.
3. **Strategies** -- `create_loop_strategy` -> core's
   `per_thread_planning.plan_per_thread_loop_strategy` (one element per
   thread, shared with Metal; CuTe answers its `per_thread_mma_candidate_loop`,
   `per_thread_matmul_loop_has_mma_path`, `per_thread_mma_mode` and
   `per_thread_root_thread_counts` hooks, and the attention / MMA detectors
   `_detect_attention_mma_loop`, `_detect_attention_bwd_mma_loop`,
   `_detect_flex_attention_loop`, `_detect_flex_attention_bwd_loop`,
   `_detect_specialized_mma_loop` record the flash matches and the
   specialized-MMA plan on `cute_state`); `create_program_ids` ->
   `tcgen05.scheduler_pids.Tcgen05PersistentProgramIDs` for persistent pid
   types on a tcgen05 body; `create_reduction_strategy` + `reduction_emitter`
   -> `reduction_emitter.reduction_emitter_for`.  The per-thread strategies
   of `tile_strategy.py` (shared with Metal) take their CuTe behaviour from
   the thread-layout hooks of `Backend`: `plans_launch_thread_axes`,
   `thread_axis_limit`, `loop_var_prefixes`, `packed_grid_block`
   (`tile_packing.py`) and the other layout predicates;
   `record_launch_thread_extent` / `record_grid_thread_extent` /
   `record_vloop_wrapper` (the launch facts on `cute_state.lanes`);
   `apply_lane_cluster_split` -> `lane_cluster_split.apply_cluster_split`;
   `check_lane_loop_nest` and `add_lane_nest_barriers` ->
   `lane_loops/nest_check.py` (the nest exactness check and redistribution
   of `placement.py`, then the barriers of `barriers.py`);
   `demote_grid_tile_vec_stores` / `demote_loop_tile_vec_stores` ->
   `memory_ops.demote_reordered_tile_vec_stores`.
4. **generate_ast** (`generate_ast.py`, core) -- `retry_policy()` wraps every
   pass (`CuteRegeneration`: five signals, each fires at most once, so a call
   generates a kernel at most three times); `kernel_generators()` is tried
   first (the `KernelGenerator`s, first match wins); otherwise `pre_codegen`
   runs the planner cascade (section 3.2), `GenerateAST.visit_For` tries
   `root_emitters()` for each root (in the order `test/test_cute_root_emitters.py`
   pins) before the ordinary
   lowering, and `lower_owned_lane_loop` ->
   `lane_loops.root_passes.run_root_lane_passes` runs the root-exit lane
   pipeline on the rendered body.
5. **Device body** (`DeviceFunction.codegen_function_def`) --
   `finalize_device_body` -> `passes.pdl.prepare_device_body`,
   `passes.pipeline.run_device_body_passes` (section 4),
   `passes.pdl.place_late_wait`; `finalize_kernel_def` ->
   `run_kernel_def_passes`, `pdl.insert_spread_pad`,
   `pdl_ops.validate_explicit_launch`, `launch_attributes.module_statements`,
   and `project_launch_block`. Core code then splices explicit
   `hl.pdl_*` entry/exit markers after every body rewrite, preserving source
   order. Explicit markers govern the launch protocol and disable automatic
   `cute_pdl` placement and spread padding. Whole-kernel/native launch paths
   that cannot preserve that protocol reject the configuration. Explicit
   markers restrict tcgen05 search to single-CTA launches; clustered families
   are conservatively screened, including promoted compiler defaults that
   bypass autotuning. Some single-CTA members of those families are also
   excluded until their native protocols can be distinguished. The
   structural fission planners' pure-source gates reject marker-containing
   programs, so enabling those settings retains the ordinary route.
6. **Host launch** -- `transform_launch_grid`, `build_launcher_args`,
   `output_only_names`, and `launcher_keyword_args`, which still recovers the
   SIMT launch block from the rendered kernel text; `launch_shape.py` compares
   the structural launch shape with it in check mode (`check_launch_shape`).
7. **Runtime hooks** (`helion/runtime/kernel.py`) -- `structural_policy_type`,
   `on_bind`, `host_structural_facts`, `before_codegen`, `after_codegen`,
   `after_compile` forward to `runtime_hooks.py` (`plan_bound_kernel`,
   `record_host_structural_facts`, `begin_codegen`, `finish_codegen`,
   `register_compiled`); `default_launcher_name` returns the import alias
   `_default_cute_launcher`, which `library_imports` binds to
   `helion.runtime.default_cute_launcher` (re-exported from
   `helion.runtime.cute.launcher`; section 7).

## 3. How the emitters compose

### 3.1 General-purpose emitters

A general-purpose emitter is a function or class over a *site* -- an
`AccessSite` (`emitters/memory_access.py`), a `ReduceCall` (`reduce_call.py`),
a `LaneLoopContext` (`_compiler/lane_loop_context.py`), a fragment context -- and never
reads the config; the caller that owns the site passes what it decided.

| module | emits |
|---|---|
| `memory_ops.py` + `emitters/memory_access.py` | loads, stores and packet forms; the index / address / region / mask builders the atomic and scan lowerings reuse |
| `emitters/lane_packets.py` (`LanePacketEmitter`) | register-tile vector packet loads and stores |
| `emitters/persistent_item_loop.py`, `emitters/role_item_loop.py` (`RoleItemLoop`) | the work-item loop of a warp role, as text (flash) and as AST (tcgen05 scheduler roles) |
| `reduction_emitter.py` (`CuteReductionEmitter`), `reduce_call.py`, `reduce_ops.py`, `argreduce.py` | tile / warp / block reductions and argreductions behind `ReductionStrategy` |
| `atomic_ops.py`, `scan_ops.py` (`_CuteScanEmitter`), `cute_reshape.py`, `view_ops.py`, `tile_ops.py`, `aten_lowering.py`, `matmul_ops.py` / `matmul_fallback.py` | the `@_decorators.codegen(op, "cute")` handlers of the language ops (registered by `_codegen_modules.py`) |
| `cute_epilogue.py`, `fragment_epilogue.py`, `scalar_epilogue.py` | pointwise epilogue chains over fragments and scalars |
| `indexing.py`, `packet_access.py`, `pipeline_smem.py` (`SmemArena`), `scalar_recipe.py` | index algebra, the memory-access call grammar, SMEM pricing, scalar staging recipes |
| `lane_loops/` | the lane loops and their passes (`markers`, `placement`, `split`, `barriers`, `owner_guards`, `statements`, `nest_check`) |
| `device/ptx.py` (`ptx_op` rows), `device/sync.py`, the `*_runtime` / `*_helpers` modules | device helpers traced into kernels; the PTX wrappers are one table row each |

### 3.2 Special-purpose emitters

A special-purpose family is a planner plus an emitter.  The planner runs in
`CuteBackend.pre_codegen` (or, for flash and the specialized MMA plan, in
`create_loop_strategy`) and records its plan in `cute_state.root_plans.<slot>`;
the emitter is a `PlannedRootEmitter(name, slot, module, entry, failure,
passes_plan, clears_slot)` whose `entry` is resolved on its module at call time
(so `mock.patch` on the module intercepts it), or one of the hand-written
`RootEmitter`s (`AttentionFlashRootEmitter`, `ComputedFragmentRootEmitter`,
`GatedRootEmitter`).  `RootEmitter.try_emit` is the dispatch step: `plan` says
whether the family owns the root, `emit` renders it, `decline` applies the
family's own refusal (which exception, whether the slot is cleared, or falling
through).  Emitters that return a body return it through `EmittedBody`.

The `pre_codegen` cascade stops at the first planner that claims the kernel:
radix selection -> top-k -> register region -> direct affine scan (when
`cute_affine_scan_schedule` asks for it) -> register sub-blocks -> block scaled
-> chunk prepare -> chunk recurrence -> gdn recurrence -> mma_dag -> single-token
rank-1 -> split rank-1 -> fixed-token rank-1; a kernel nobody claims gets
`annotate_view_subtiles`, `annotate_literal_dims` and `execution_plans.plan_execution`
(the thread-axis plans).  `ROOT_EMITTERS` is tried in this order:
`required_computed_fragment` (a full uint8 chain with no scalar fallback),
`register_subblocks`, `radix_selection`, the gated `computed_fragment` (when the
root asks for fragments), `block_scaled`, `topk`, `register_region`,
`chunk_prepare`, `chunk_recurrence`, `gdn_recurrence`, `mma_dag`,
`single_token_rank1`, `split_single_token_rank1`, `fixed_token_rank1`,
`warp_mma_gemm`, `attention_flash`, `computed_fragment`.  The order is pinned by
`test/test_cute_root_emitters.py`.

| family | planner | emitter | host side |
|---|---|---|---|
| tcgen05 GEMM (not a root emitter: a matmul lowering under the ordinary root) | `tcgen05/search_plan.py` at device-IR time; `cute_mma.plan_tcgen05_pipeline`; `tcgen05_config/` (`CuteTcgen05Config` state record with `facts`, `surface`, `repair`, `seeds`, `smem`, `knobs`); `strategies.py` | `cute_mma.codegen_cute_mma` / `codegen_cute_mma_dot`; `tcgen05/scheduler_pids.py` (the role-local persistent schedulers, one `SchedulerFamily` per plan from `tcgen05/scheduler_family.select_scheduler_family`); `tcgen05/store_epilogue.py`; `tcgen05/universal_mma.py`; `tcgen05/aux_producer.py`; `epilogue_fanout.py` | `host_wrappers/tcgen05.py`, `host_wrappers/grouped_static.py` |
| flash forward: pipeline families `ws_overlap`, `ws_2cta`, `fa4` (+ `fa4_deep_1cta`, `fa4_2cta_causal`, `fa4_tma_4d`, `fa4_local_tma`, `fa4_clc`, ...), `fa4_alt` and `row_mma` -- the keys of `flash_knobs.FLASH_PIPELINE_FAMILY_FLAGS`, `row_mma` under the name `flash_knobs.ROW_MMA_FAMILY`; flex | `analysis/attention_match.py`, `flex_match.py`, `flash_schedule.py`, `flash_policy.py`, `attention_plan.py` | `cute_flash.codegen_attention_flash` -> `emit_flash_fa4_device_body` (over `FlashBodyContext`), `emit_flash_ws_device_body`, `emit_flash_device_body` (the legacy single-warpgroup body); `cute_flash_alt.py`, `cute_flash_row_mma.py`, `flash_template_runs.py`, `flash_row_epilogue.py`, `flash_score_mod.py` | `host_wrappers/flash.py`, `runtime/cute/flash_flex.py`, `flash_document.py` |
| flash backward: `cute_flash_bwd_row_mma.FLASH_BWD_FAMILIES` (`fa4`, `row_mma`); flex | `cute_flash_bwd.py` matchers, `flex_bwd_match.py` | `cute_flash_bwd.codegen_attention_flash_bwd`, `cute_flash_bwd_flex*.py`, `cute_flash_bwd_row_mma.py`, `cute_flash_bwd_common.py` | as above |
| gated attention | `cute_flash_gated.py` | `cute_flash_gated.codegen_gated_attention_flash`, `cute_flash_gated_row_mma.py` | as above |
| mma_dag (warp and tcgen05 modes) | `cute_mma_dag.plan_mma_dag` (`MmaDagPlan`), `cute_mma_dag_seeds.py` | `cute_mma_dag_codegen.codegen_mma_dag` (`WarpMmaDagEmitter`), `cute_mma_dag_tcgen05.py` (`Tcgen05MmaDagEmitter`), `cute_mma_dag_tcgen05_regions.py`, `cute_mma_dag_emitter_base.py` | `host_wrappers/mma_dag.py` |
| warp_mma GEMM (tiny tiles) | `create_loop_strategy` | `cute_warp_mma_gemm.py` | -- |
| collective / SIMT / gathered GEMM | `collective_matmul.py`, `simt_gemm.py`, `gathered_mma.py` | device-body prologue passes `lower_collective_matmul`, `lower_simt_gemm`, `extend_gathered_mma_tma_launch` | `host_wrappers/gathered.py` |
| recurrences | `chunk_prepare.py`, `chunk_recurrence.py`, `gdn_recurrence.py`, `single_token_rank1_recurrence.py`, `split_single_token_rank1_recurrence.py`, `fixed_token_rank1_recurrence.py` (`analysis/rank1_recurrence.py`), `direct_affine_candidate.py` / `direct_affine_plan.py` | the same modules plus `chunk_recurrence_sm100.py`, `chunk_recurrence_dv4_sm100.py`, `gdn_recurrence_sm100.py`; `passes/hoist_chunk_recurrence.py` | `host_wrappers/recurrences.py` |
| selection | `selection/topk.py`, `row_topk.py`, `radix_selection.py`, `ordered_selection.py` | `selection/topk_codegen.py`, `row_topk_codegen.py`, `radix_selection_codegen.py`, `selection_lowering.py` (the `RegisterProgramEmitter` bridge) | -- |
| computed fragments, register sub-blocks / region / program, block scaled | `computed_fragment.prefer_computed_fragment_root`, `cute_register_subblocks.py`, `register_region.py`, `block_scaled_mma.py` | `computed_fragment.codegen_computed_fragment_root` + `fragment_*.py`; `register_program.py`; `block_scaled_*.py`, `mxfp8_*.py` | `host_wrappers/block_scaled.py` |
| host-sum tails | `host_sums.py` | `host_single_sum.py`, `host_paired_sum.py`, `host_full_sum.py`, `host_remainder_fills.py` | `runtime/cute/single_sum.py`, `paired_sum.py`, `full_sum.py` |

MMA dataflow operand recipes may read a typed 2D grouped auxiliary tensor at
`floor(global_axis / positive_group)`. The matcher proves its physical capacity;
INT8/UINT8 auxiliaries require this recipe. Grouped values use scalar addressing
and stay outside affine vector rings, prefetch tables, and raw carry seeds.

`analysis/separable_index_map.py` proves explicit storage offsets whose logical
axes use independent quotient/remainder folds. It preserves the physical tensor
and checks each original integer intermediate before simplifying the map.
`Access.index_map` is an explicit consumer contract; it has no affine logical
strides. The initial consumer is the paged attention transport for original
`[request, query, grouped head, dimension]` Q/O and optional rank-three LSE. The
transport proves the request/head relation, nonoverlapping output strides, and
masked query tails; `_paged_attention_io.py` renders those addresses without packing.
Ordinary MMA, packet loads, auxiliary rings, and TMA routes reject these maps.

The explicit paged-attention schedules honor `load_eviction_policies` on their
four TMA inputs: raw K, K scales, raw V and V scales. Their original load slots
survive normalization and materialized-phase projection; other slots retain
the owning launch's rules. TMA bypasses L1, so L1-only `first`/`last` are inert.
L2 priorities canonicalize to `streaming` (evict first) and `l2_last` (evict
last), with the empty policy preserving the default. Generic seeds try all
three uniform choices, and search can vary the four inputs independently.

### 3.3 Whole-kernel generators and regeneration

`KERNEL_GENERATORS` (`kernel_generators.py`): `grouped_prefix_adaptive_n`,
`packed_operand`, `row_resident`, `grouped_warp_tf32`, `flat_grouped_native`,
`bounded_cache`, `split_k_cluster`, `split_k_workspace`, `materialized_fission`,
`ordered_phases`.  Each `applies` to a config, carries a `TransformPolicy`
saying which specializing arguments of `generate_ast` it cannot honor, and
re-enters `generate_ast` for the kernels it composes.  `CuteRegeneration`
regenerates after `RegisterTileUnsupported`, `VloopSinkNotApplied`,
`FragmentMmaNotApplied`, `RingMmaNotApplied` or `LaneUnrollNotApplied`, with
the knob that raised switched off.

### 3.4 Where composition is not yet complete

The fragment, selection, recurrence and atomic families are written over the
general-purpose layer.  The two largest stacks compose least: the flash bodies
render most of their text in `cute_flash.py` over their own device
helpers (`_flash_runtime.py`, `_flash_gemm_ptx.py`, `_flash_alt_runtime.py`,
`_flash_row_mma_runtime.py`), sharing only `persistent_item_loop`,
`flash_template_runs` and the row epilogue; `mma_dag` carries its own
`emit_*` stack over `_mma_dag_runtime.py` / `_mma_dag_tcgen05_runtime.py`; the
tcgen05 pipeline (`cute_mma.py`) composes `pipeline_smem`,
`cute_epilogue`, `fragment_epilogue` and `role_item_loop` but is one module.

## 4. The device-body pass pipeline

`passes/pipeline.py` owns the order.  `PassContext` is the frozen record of
everything the passes read besides the body, built once per render
(`build_pass_context`).  `DEVICE_BODY_PROLOGUE` lowers the collective
matmuls and decides the launch shape; `freeze_pass_context` seals it;
`DEVICE_BODY_PASSES` (from `finalize_shared_reduce_groups` to
`hoist_exact_invariant_divisions`) rewrite the body; `KERNEL_DEF_PASSES` run
after `ast_rename` over the `FunctionDef`; `test/test_cute_pass_pipeline.py`
pins the three tables' order.  Every entry is a
`DeviceBodyPass(name, gate, run, requires, provides)`; `requires` / `provides`
are checked once at import.  The pass bodies live in their own modules and are
imported at call time, so a test that patches a pass on its module still
intercepts the render.  The order is a compatibility invariant: reordering is
a render change.  The root-exit lane pipeline (`lane_loops/root_passes.py`)
runs earlier, from `lower_owned_lane_loop`, and is ordered the same way.

## 5. State

Core objects carry one opaque `backend_state` slot each, filled by
`Backend.new_env_state` / `new_codegen_state` / `new_device_state(config)`.
`state.py` defines what CuTe puts there: `CuteEnvState` (per
`CompileEnvironment`), `CuteCodegenState` (per `GenerateAST` pass) and the four
namespaces of `CuteDeviceFunctionState` (`device_state.py`): `tcgen05`
(`CuteTcgen05State`), `flash` (`CuteFlashState`), `lanes` (`CuteLaneState`,
where `LaneLoopContext` attaches) and `root_plans` (`CuteRootPlans`, whose
`ROOT_PLAN_SLOTS` the `PlannedRootEmitter`s read; each slot is written by one
planner).  `state.py` imports nothing from the emitters.  The flat names every field had
before the regrouping are gone: every reader names the namespace.  The
`cute_*` / `_cute_*` names the core objects (`CompileEnvironment`,
`GenerateAST`, `DeviceFunction`) carried before the slot existed are gone as
well: readers call `cute_env_state(env)`, `cute_codegen_state(codegen)` and
the device-state accessor (`DeviceFunction.cute_state`); core forwards no
flat names.  The
core defaults of the `new_*_state` hooks are `None`: only a CuTe compile
carries these objects, and core reaches them through `Backend` hooks
(`synthetic_thread_axis_sizes`,
`consume_root_lane_loop_suppression`, `reorder_post_loop_statements`,
`roots_fissioned`, `finalize_host_statements`, ...) or inside this package.

## 6. The autotune surface boundary

`Backend.config_spec_extension()` returns `autotune.search_surface.CUTE_SEARCH_SURFACE`,
a `CuteSearchSurface(BackendSearchSurface)`: the config keys, the flat-field
table, the normalization phases (called by `ConfigSpec` at the fixed positions
listed in `helion/autotuner/search_surface.py`) and the compiler seeds.
`autotune/config_keys.py` declares the key tables once (the core
`VALID_KEYS` / `BACKEND_SPECIFIC_KEYS` / `DEBUG_KEYS` are the union over the
registered backends' surfaces, `backend_registry.all_search_surfaces()`);
`flash_surface.py` and `mma_dag_surface.py` hold the two families' fragments
and normalizers; `config_generation_ext.py` (`CuteConfigGenerationExtension`,
returned by `CuteSearchSurface.config_generation_extension` and stored as
`ConfigGeneration.extension`) holds the SIMT launch-thread repairs, the flash
structural population and every `flash_*` coverage accessor;
`flash_search.py` (`FlashCoverageDesign`) is the structural coverage design
the generic `ConfigGeneration` delegates to.  Knob tables are owned by their
readers: `flash_knobs.py` (`FLASH_KNOBS`), `autotune/fragment_knobs.py`
(`FragmentKnob` rows), `tcgen05_config/knobs.py`, `tcgen05/knobs.py`,
`cute_mma_dag_seeds.py`; seeds in `tcgen05_config/seeds.py` and the surfaces.
`Backend.autotuner_heuristics()` returns `autotune.heuristics.CUTE_HEURISTICS`,
sorted by each heuristic's `priority`; the order is part of a seed list's
identity.  `register_seed_facts`, `postprocess_seed_configs` and
`register_coverage_groups` live beside it; the core
`helion/_compiler/autotuner_heuristics/` package runs them by backend.  The
structural settings a bound kernel records beside its config are
`runtime/cute/structural_policy.CuteStructuralPolicy`
(`CUTE_STRUCTURAL_SETTINGS`), selected through `structural_policy_type`.
Three hooks answer the autotuner itself: `llm_prompt_profile()` returns
`autotune.llm_profile.CuteLlmPromptProfile` (the CuTe lines of the LLM
autotuner's prompts, over `helion/autotuner/llm/profile.py`), `rng_streams()`
lists the `cute_rng_stream` settings the RNG lowering accepts (`auto`,
`word0`, `philox4`; `Settings` defaults to the first), and
`final_rebenchmark_top_k()` / `final_rebenchmark_isolated()` set the final
rebenchmark's finalist count (32) and isolated timing.  Rule:
an emitter reads `config` only through its family's knob table; a
general-purpose emitter does not read it at all.

## 7. The runtime

`helion/runtime/cute/launcher.py` (`default_cute_launcher`) builds the
`@cute.jit` host wrapper for a generated `@cute.kernel`, compiles it and
launches it.  The compiler hands it wrapper plans (`_helion_cute_wrapper_plans`,
one `dict` per plan with a `kind`); `_append_cute_wrapper_plan` dispatches on
the kind through `host_wrappers.HOST_WRAPPER_EMITTERS`, one `HostWrapperEmitter`
per kind, declared in the runtime-side modules `block_scaled`, `flash`,
`gathered`, `grouped_static`, `mma_dag`, `recurrences`, `tcgen05`.  Leaf modules
of the launcher: `compile_cache.py` (`CompiledCuteLauncher`, `disk_cache_key`),
`source_dependencies.py` (the helper sources the key fingerprints, including
`_COMMON_DEPENDENCIES`), `launch_descriptor.py` (`CuteLaunchDescriptor`, the
plan-derived constants computed once per plan list), `launch_caches.py`
(`LaunchArgCacheEntry`, `CudaGraphResources`, `cute_cuda_graph`),
`fast_relaunch.py` (`CuteFastRelaunch`), `arg_signature.py`, `tensor_maps.py`,
`occupancy.py`, `sass_check.py`, `plan_facts.py`.  `standalone_launcher.py`
runs kernels exported by `helion/_compiler/cute/standalone.py`
(`Backend.build_standalone_code`) without Helion.

## 8. Device helpers and byte-pinned paths

`device/__init__.py` states the device-module rules: importable without CUDA,
one top-level binding per helper, no `from __future__ import annotations`
beside a `@cute.struct`, and helper names are part of the rendered source and
of the disk-cache key.  Two sets of module paths are byte-pinned:

* the modules `CuteBackend.library_imports` writes into generated import lines
  (`atomic_helpers`, `aux_copy_layout`, `clc_helpers`, `cluster_helpers`,
  `copysign`, `epilogue_helpers`, `exact_division`, `_flash_row_mma_runtime`,
  `flex_score_mod_runtime`, `fold_helpers`, `grid_barrier`,
  `inline_asm_helpers`, `integer_helpers`, `join_cast`, `l2_policy`,
  `masked_packet_load`, `_mma_dag_runtime`, `packet_store_helpers`,
  `_pdl_spread_runtime`, `quantized_helpers`, `readonly_packet_load_helpers`,
  `reduce_helpers`, `remainder_fill_runtime`, `resident_reduction_runtime`,
  `single_token_rank1_recurrence`, `split_k_exchange`, `tensor_copy_helpers`,
  `thread_budget`, `vec_utils`, `warp_atomic_runtime`, `_warp_mma_runtime`,
  and `helion.runtime.cute.{single,paired,full}_sum`), plus the
  `helion._compiler.cute.<module>` strings emitters compose in f-strings;
* the paths `helion/runtime/cute/source_dependencies.py` hashes into the
  disk-cache key (`_COMMON_DEPENDENCIES` and the per-kind tables).

Renaming a module of the first set changes every render that imports it; a
body edit to a module of the second set changes the cache key, not the render.
Classify a module by grep before moving it.

## 9. Testing seams

* `helion/_testing/`: `cpu_render_target` / `patch_target_facts` (render on
  CPU against a `TargetFacts` model from `cute/target_facts.py`),
  `patch_cute_mma_support`, `onlyBackends` / `skipUnlessBackends` /
  `skipUnlessCuteAvailable`, `code_and_output`, `check_example`, and
  `AssertExpectedJournal` (journals: `<test>.expected` for Triton,
  `<test>.expected_cute` under `HELION_BACKEND=cute`; regenerate with
  `EXPECTTEST_ACCEPT=1`).
* `test/conftest.py`: the `test/test_cute_*.py` modules are collected only
  when `HELION_BACKEND=cute` (`pytest_ignore_collect`: the CuTe lane); the
  `cpu_render` marker runs a test under `cpu_render_target`.
* Patch seams: the defining module (there is no re-export facade to patch),
  `passes.pipeline.PassContext`, and the call-time imports of the pass and
  emitter tables.
* Registry pins: `test/test_cute_root_emitters.py`,
  `test/test_cute_pass_pipeline.py`, `test/test_backend_capabilities.py` (the
  no-op defaults of every hook), `test/test_backend_registry.py`,
  `test/test_cute_device_state.py`, `test/test_cute_device_helper_names.py`,
  `test/test_cute_device_modules.py`, `test/test_cute_launch_shape.py`.
* Render identity: the refactor's gate is a CPU render dump of
  `BoundKernel.to_code()` for every test on a base tree and a tip tree,
  compared file by file (a pytest plugin that records every render or its
  exception, keyed by test id; the benchmark campaign's recorded winners
  were re-rendered and re-timed the same way).  Two move rules follow:
  retarget every `mock.patch` string in the same commit as the move, and
  treat the generated import strings of section 8 as byte-pinned.

## 10. The backend-isolation rule and what a new backend implements

`scripts/check_backend_isolation.py` (run by `lint.sh` and pre-commit) fails
when the backend-neutral core -- `helion/_compiler/*.py`,
`helion/_compiler/autotuner_heuristics/`, `helion/autotuner/`,
`helion/language/`, `helion/runtime/*.py` -- or a sibling backend package
compares a backend name with `"cute"`, raises a literal
`BackendUnsupported("cute", ...)` or imports `helion._compiler.cute` or
`helion.runtime.cute`; the `ALLOWLIST` block records the sites that remain,
per file and kind, and is regenerated with `--update` (never edited by hand).
A backend-specific decision belongs behind a `Backend` hook with a no-op
default.

`Backend` (`helion/_compiler/backend.py`) has seven abstract members --
`name`, `dtype_str`, `acc_type`, `function_decorator`, `constexpr_type`,
`default_launcher_name`, `library_imports` -- and the printers that raise
`BackendUnsupported` until overridden (`program_id_expr`, `cast_expr`,
`arange_expr`, `grid_index_expr`, `loop_index_expr`, `scalar_load_expr`,
`grid_barrier_stmt`, `where_expr`, `minimum_expr`, `arange_index_expr`,
`zeros_expr`, `full_expr`, `reshape_expr`, `broadcast_to_expr`,
`reduction_index_expr`, `reduction_index_zero_expr`,
`next_power_of_2_host_expr`, `lane_index_expr`, `lane_offset_expr`,
`thread_index_expr`, `reduction_expr`, `argreduce_result_expr`,
`argreduce_loop_update_statements`, `inductor_op_overrides`), plus three
`NotImplementedError` hooks (`dependency_free_launcher_info`,
`capture_jax_launch_metadata`, `build_jax_fn_code`).  Every other method has a
default describing the Triton-style tile backend; the class docstring lists
the facets the section comments mark.  A new backend registers through
`backend_registry.register_compiler_backend`, lists its op handlers in its own
`_codegen_modules.py` (loaded by `backend_registry.import_backend_codegen`),
and contributes emitters through `root_emitters()`, `kernel_generators()` and
`retry_policy()` (`emitter_protocols.py`), passes through `rewrite_device_ir`
/ `finalize_device_body` / `finalize_kernel_def` / `lower_owned_lane_loop`,
state through `new_*_state`, a per-thread layout (if it has one) through
the thread-layout hooks (`plans_launch_thread_axes` ..
`demote_loop_tile_vec_stores`; section 2 item 3), and its search surface
through `config_spec_extension()` and `autotuner_heuristics()`.

## 11. Role vocabulary

Module docstrings carry a `Role:` line using one of these words:

* **analysis** -- reads the device IR or the rendered AST and returns frozen facts; no codegen, no config.
* **planner** -- decides under one config whether a family owns a root or a kernel and records its plan.
* **general-purpose emitter** -- renders one site; never reads the config.
* **special-purpose emitter** -- owns a root or a kernel; composes the general-purpose emitters.
* **kernel generator** -- an entry of `KERNEL_GENERATORS`.
* **device-body pass** -- an AST rewrite over the rendered body (`passes/`, `lane_loops/root_passes.py`).
* **device-IR pass** -- a rewrite at a `DeviceIRStage` (`ir_passes.py`).
* **host rewrite** -- a `customize_ast` rewrite of the user's kernel.
* **device helper** -- Python traced into generated kernels (section 8).
* **host wrapper** -- a `HostWrapperEmitter` (`helion/runtime/cute/host_wrappers/`).
* **knob table** -- the config keys and choices a family owns.
* **heuristic** -- autotuner-side guidance: a seed heuristic in `autotune/heuristics/`, or the prompt prose of `autotune/llm_profile.py`.
* **state** -- an object behind a `backend_state` slot.
* **testing seam** -- a facade or patch point tests rely on.
* **hook surface** -- the `Backend` implementation (`backend.py`), one module per backend.

## 12. What is not finished

* **Isolation residue.** `check_backend_isolation.py --strict` passes at this
  tip; every allowlisted reference is an import or a `TYPE_CHECKING` import
  (the `ALLOWLIST` block in the script is the inventory, regenerated with
  `--update`; `--verbose` lists the sites).  What remains: the
  structural-policy imports of the runtime half of the package
  (`helion.runtime.cute`) in `helion/runtime` and the autotuner caches, the
  lane-loop forms and `cute_*` spec types that `tile_strategy.py`,
  `config_spec.py` and the surrogate pattern search import from this package,
  type-only imports in core modules, the registry's eager import of
  `CuteBackend`, and the loop-nesting helpers and execution plan `metal/`
  imports from this package.  Core no longer reads
  `cute_state`: the lane contexts the shared per-thread strategies register
  are core state (`DeviceFunction.lane_contexts`, `register_lane`), the
  `new_*_state` defaults are `None` (section 5), and the register-tile gate
  and shared-memory budget the reduction strategies call are
  `thread_budget.register_tile_shape` / `shared_memory_budget_bytes`.
* **Decision-gated retirements.** The legacy single-warpgroup flash body
  (`emit_flash_device_body`, `ws_overlap` with `s_stage == 1`) is the only
  body at `num_kv == 1` and stays; the `ws_overlap` family stays (it wins the
  small forward shapes).  The scheduler families `GROUPED_STATIC_SPECIALIZED`,
  `GROUPED_RUNTIME_DIRECT` and the full/edge store split keep their per-family
  payloads in `scheduler_pids.py`; the skeleton (`RoleItemLoop`,
  `select_scheduler_family`) is shared, the payloads are not tables.
* **Autotuner extension objects.** The flash / CuTe methods of
  `ConfigGeneration` and the surrogate pattern search are not behind an
  extension object; `VALID_KEYS` in `config_spec.py` is not derived from the
  surfaces.  `autotune/heuristics/cute.py` still reads
  `spec._cute_tcgen05_config` directly.
* **Launch shape, composition, layout.** `launcher_keyword_args` regexes are
  the source of truth and `launch_shape.py` only checks; section 3.4; the
  `cute_*` prefixes and `*_codegen` module names are historical;
  `single_token_rank1_recurrence.py` is both a root emitter and a byte-pinned
  device-helper module.
* **Known test gaps.** `test/test_cute_tcgen05_split_k_search.py::test_sampled_splits_render`
  is unseeded; the rank-1 recurrence kernels do not bind on CPU, so the CPU
  render dump never covers them.
