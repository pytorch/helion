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
| `smem_footprint.py` | the shared memory a rendered kernel allocates (its reduce helpers' buffers and unrolled copies included), the per-block check, the `cute_pdl_spread` pad's input |
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
   `record_vloop_wrapper` / `record_flattened_vloop_wrapper` (the launch
   facts on `cute_state.lanes`);
   `apply_lane_cluster_split` -> `lane_cluster_split.apply_cluster_split`;
   `check_lane_loop_nest` and `add_lane_nest_barriers` ->
   `lane_loops/nest_check.py` (the nest exactness check and redistribution
   of `placement.py`, then the barriers of `barriers.py`);
   `demote_grid_tile_vec_stores` / `demote_loop_tile_vec_stores` ->
   `memory_ops.demote_reordered_tile_vec_stores`.
4. **generate_ast** (`generate_ast.py`, core) -- `retry_policy()` wraps every
   pass (`CuteRegeneration`: six signals, each fires at most once per call,
   section 3.3); `kernel_generators()` is tried
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
   `pdl_ops.validate_explicit_launch`,
   `smem_footprint.check_smem_budget` (a kernel past the per-block shared
   memory maximum is `BackendUnsupported`, not a libNVVM failure),
   `launch_attributes.module_statements`, and `project_launch_block`. Core
   code then splices explicit
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
| `reduction_emitter.py` (`CuteReductionEmitter`), `reduce_call.py`, `reduce_routing.py`, `reduce_ops.py`, `argreduce.py` | tile / warp / block reductions and argreductions behind `ReductionStrategy` |
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

A kernel with one CTA-local fragment root can check a scalar Boolean it computed,
for example `torch._assert_async(counts.sum() <= capacity, "capacity exceeded")`.
The message is optional. One thread checks the predicate in the same launch;
a false value reports a CUDA device assertion when the stream synchronizes.
Bare host tensors and non-scalar or non-Boolean predicates are rejected.
Scalar indices on reads and stores must be proved nonnegative after normalizing
a literal such as `x[-1]`. The proof recognizes direct grid coordinates and
tile origins, but not an arbitrary displacement such as `out[i + 1]`.
Unmasked reads also require proof that scalar and tensor indices stay below
the dimension size.
A tile slice must have a nonnegative origin. For an unmasked read, the entire
logical interval must also fit the indexed dimension: clipping could change
a reduction's result. Masked reads and stores do not require this upper bound.
Direct grid coordinates, tile origins and tile IDs use the loop's proved range.
A checked fragment also requires statically proved Int32 grid and nested loop
bounds, a positive step, and a representable final induction update. Tensor-loaded
or runtime-symbolic bounds do not qualify even when their sample values fit.
The flattened logical grid and any scheduler padding must fit Int32. An XYZ
launch must also fit the CUDA limit of 65,535 blocks on each of Y and Z.
Each CTA must own one logical tile: configurations that pack tiles or
replace the fragment with a specialized kernel cannot issue this check.

Earlier CuTe versions could silently drop the message-less form. Such kernels
now raise a device assertion on false. If the predicate or its computation
cannot be lowered (for example, two independent root loops, a bare host
scalar, or a reduction over unsupported dimensions), compilation can now fail
where it previously succeeded. A checked kernel can also use the ordinary
fragment in place of a faster specialized schedule that cannot issue the check.

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

A kernel with the pair proof runs the paired schedule without a config: its
first compiler seed, promoted to the autotune-off default, is the paired body
with V issued two tiles behind K, `streaming` hints on the four TMA inputs
and, where proved, the query TMA map and the `early` probability schedule,
under the schedule's own defaults for retirement, dependent launch and key
clip. The other seeds follow in the planner's order. Kernels without the
proof default to the first seed of the ordinary MMA ranking, or to the per-key
`attention` schedule where folded storage admits the attention schedules
alone. A kernel is without it when its count of KV heads or of leading items
is odd (one KV head among them), when its denominator is reduced rather than
kept per key, when it has an ordered phase or more than one leading axis,
when a bound differs between the two heads of a pair, or when the device
offers a CTA less than 184,320 bytes (180 KiB) of shared memory.

The schedule's default dependent-launch policy is the guarded capture
overlap with the `loader_tail` release (docs/deployment_autotuning.md,
"Capture overlap of paired attention kernels"), and where that release is
placed depends on the grid. On a launch that fits one wave of resident CTAs
it is the loader's tail, and `auto` is the entry of every CTA. On a launch
of one unit per CTA that is known to take more than one wave both render the
entry of every CTA (`passes.pdl._entry_release_past_one_wave`), which is
what `cute_pdl_release="entry"` renders; a generic kernel past one wave, and
a grid the host sizes at runtime, release from the CTAs of the last wave.
The dependent grid launches once every CTA of this grid has released or
exited. A paired CTA holds its SM for its whole life, so with a release at
entry in every CTA the dependents launch as the last CTA enters: every CTA
of the grid is resident or done by then, and the dependents take SMs only as
this grid's CTAs leave them. A release from the last wave alone launches them
when the first CTAs in particular have exited, which is never earlier and
waits for the slowest of them. Measured
on a B200 (148 SMs) with 200 units of a 17-page window (two waves), the next
captured call started 26.6 to 27.1 us after the launch under the last-wave
release and 25.0 to 25.2 under the entry release, with 96 SMs idle in
between, and the entry release read 0.4 to 0.8 us less per call at two calls
a sample (four cases; 0.5 less on 320 units in three waves). So a saved
config that names `auto` or `loader_tail` on such a grid compiles to the
entry release; `cute_pdl_release="last_wave"` names the release from the
last wave alone, with the capture overlap or without it, and no search
samples it.

The paired schedule's `cute_mma_dag_key_clip` shifts the TMA boxes of a
boundary tile past keys that no row of the tile sees. Those keys then lie
outside the descriptor's key extent: the hardware does not fetch them from
their own addresses and fills them with zeros. That is not a load without
memory traffic for them: in the large allocation class, which a descriptor
over a whole cache has, a box that leaves its tensor still issues requests
for the out-of-bounds elements, inside the 128 KiB after the descriptor's
address (`tensormap_alloc_class`). `leading` clips the keys before every
row's window, `trailing` those after every row's causal bound, and `both` the
larger side of each tile. Shifts are multiples of four keys, keeping the packed
scale box 16-byte aligned, and the softmax masks the zero-filled rows with the
same shift. The knob exists only under `fast_math`, because a shifted tile is
not bit-exact: it drops the NaN a nonfinite hidden V would propagate, can turn
a `-0` accumulator into `+0`, and moves its keys to other lanes. The per-key
denominator's row sum then associates differently (normalization and LSE move
in their last FP32 bits), and the PV MMA adds the tile's products in another
order, so each FP32 output accumulator rounds differently. That last change is
absolute, not a scaling of the row: an element whose terms nearly cancel can
change by many of its own BF16 ulps and in sign, so the clip has no bound in
output ulps. Windowed kernels default to `leading`, causal-only kernels to
`off`; value-metadata transport, which copies unshifted scale words, keeps
whole tiles. Naming a
clip the kernel does not offer, or any clip but `off` together with value
metadata, is an `InvalidConfig`; on every body but the paired one a clip the
kernel offers is pinned to `off`, like the other paired knobs. The default
follows those options wherever a config leaves the clip out, also in a config
that is flattened before it is normalized, and an `autotune_config_overrides`
entry that selects a transport a search candidate's clip cannot run
re-derives that clip instead of rejecting the candidate
(`rederive_overridden_key_clip`).

The paired schedule's `cute_mma_dag_partition` chooses how its units (two
adjacent leading items: one request with two heads) are divided among CTAs.
`item`, the default and the meaning of a missing key, launches one CTA per
unit. `whole` launches `min(units, SMs)` persistent CTAs, with the SM count of
the launch device (less `persistent_reserved_sms`) read at each launch, and
balances them by work, not by unit count. A unit that streams `n` pages has
`n + 1` slots, a header and one per page; with `T` slots in all, the first
`G = min(grid, T)` CTAs are active and CTA `c` runs, as one page stream, the
units whose header slot lies in `[c * T // G, (c + 1) * T // G)`. Every unit
has one owner; a CTA whose share lies inside one long unit runs nothing. The
sums are exact and computed on the device at every launch
(`_paged_attention_partition.py`). The threads of the scanning warps, one
unit per thread up to 640 units and `ceil(units / 640)` beyond, each sum the
slots of their units into two small tables in shared memory
(`partition_scan`) and arrive at named barrier 7. Only the loader's warp
waits there, at the head of its role and behind its register quota, which it
holds while the others scan: it finds the CTA's range from the tables with
two divisions (`partition_plan`), its lane zero writes the plan (the first
unit, the end of the units and the pages of the stream) to shared memory,
the warp arrives at the setup rendezvous, barrier 6, without waiting, as it
does with one unit per CTA, and the stream starts from the plan in the
loader's registers. The quota is the control warpgroup's: its four warps
take it from one instruction behind the scan, ahead of the rendezvous,
because `setmaxnreg` has every warp of a warpgroup execute the same
instruction and the loader needs its quota before the other three have a
role. No other warp decodes: every other role waits at the
rendezvous and reads the three plan words behind it. (Decoding in every warp
behind the rendezvous, with the loader waiting there too, measured 0.1 to
0.3 us per call slower on a B200.) Barrier 7 counts the scanning warps and
the loader's warp once. The loader's warp, warp 17, is itself a scanning
warp only when the scan takes eighteen warps or more, which is more than
`544 * c` units at `c` units per thread: 545 to 640 units, 1,089 to 1,280,
1,633 to 1,920. Everywhere else it only waits: up to 544 units, and again
from 641, where two units per thread take eleven warps. The loader then
requests the units' pages back to back and, ahead of each unit's pages,
its queries into one of two query stages (one when shared memory is
short); only the queries of the CTA's first unit with pages are the QK
role's own request, as with one unit per CTA, so that the stream starts
with pages; the conversion warps and the PV issuer follow the stream; a
compute group stores a unit's outputs and resets its panels and row state
between two units while the other roles already stage the next unit. The
row owners reset the state and one thread records the unit for the group,
ahead of the group barrier that publishes both. The compute groups meet at
`bar.sync` barriers; per the PTX ISA (9.5, section 9.7.15.1) a barrier
instruction "causes executing thread to wait for all non-exited threads
from its warp", so a block that some lanes run ahead of a barrier needs no
warp synchronization of its own. The setup rendezvous and the retirement
keep the non-aligned form because their participants reach them at
different instructions. Each unit gets the instructions and state a CTA of
its own would, so results are bit-identical to `item`, and a key clip
applies at each unit's own first and last page, never between two units.
`whole` needs the query TMA, no value-metadata skip, a static unit count and
fewer than 2**31 slots (`units * (page table pages + 1)`); an illegal value is
an `InvalidConfig`, and `fix_invalid` repairs it to `item` without touching
another knob. The launch is one wave, so the dependent-launch placements
treat it as such: the default guarded overlap releases from each loader's
tail, four pages before the end of its CTA's whole stream. The roles of a
persistent CTA evaluate a unit's loop bounds when they reach it. Up to 640
units, and when the bounds read nothing but the unit's one Int32 length (the
generated `..._unit_bounds_scanned` helper exists), the scan keeps every
length it loaded in a third table and the roles evaluate from that word, so
no role waits for a global load of a length ahead of its first page;
otherwise each role loads the length again. Either way the scan and the
roles must see the same lengths, so `lengths` (or whatever the bounds read)
must not change while the kernel runs: a concurrent writer can then hang the
launch, where `item` reads each length once at entry. No seed selects `whole`
and the search does not draw it. Whole units are a coarse grain: with equal
lengths and a unit count that is not a multiple of the CTA count some CTAs
run one unit more and run on alone. On a B200, 200 requests of a 2047-key
window take 49.0 us against 40.5 us for `item` under the default overlap,
whose next call refills the SMs a partial wave leaves idle. Nor is `whole`
the same as `item` when there are no more units than CTAs: the slot rule
gives each unit a CTA of its own only for equal lengths. With ragged
lengths the CTAs whose share lies inside a long unit idle while short
units share a CTA (128 units of 0 to 17 pages on 128 CTAs, by the slot
model: about 30 CTAs without a unit, up to five units on one, and the
busiest CTA streams about 22 pages where no unit has more than 17).
Results stay bit-identical; no identity maps `whole` to `item` there.

`balanced` launches the same grid and gives a CTA the page slots of its
range themselves, an equal share of the pages whatever the units' lengths.
Slot `header + 1 + j` of a unit with `n` pages is its loop step `n - 1 - j`,
so a unit that crosses a range boundary runs in parts on consecutive CTAs,
the lowest CTA its last steps; a unit without pages runs where its header
is. The loader, the QK role and the compute groups clip each unit to the
CTA's slots (`unit_segment`); page coordinates, masks and a key clip keep
the unit's own steps, so no part boundary is a window edge. A part runs the
recurrence from an empty state. Its compute groups then exchange the part's
unnormalized panel and each row's maximum and denominator with the unit's
other parts through a workspace: `helion.runtime.cute.paged_partition` owns
one per device, stream and geometry, the record count and size (for the
sixteen streams that launched most recently, and one per captured graph),
the generated host function passes it as one more tensor, and it holds, per
resident CTA, two arrival counters and four partial slots (83,584 bytes at
40 query columns: 12.4 MB on 148 SMs, whatever the batch). The exchange
never waits for another CTA:

- A part stores its partial into its CTA's slot and arrives with an atomic
  add on the unit's counter, the one of the unit's lowest CTA; the part
  whose arrival is the last merges. The lowest part, which runs the unit's
  last steps as its CTA's last pages and so normally finishes last, first
  reads the counter once and merges without arriving if every other part
  has arrived. The merger resets the counter: all counters are zero
  whenever no launch that uses the workspace is in flight.
- A CTA runs at most one part that is not its unit's lowest, the first
  thing it runs. An arrival takes about as long as a page while the CTA's
  memory traffic is busy, so the compute group does not make that one
  itself: it stores the partial, posts it to the CTA's spare warp and goes
  on. The spare warp joins the group's stores at a named barrier (8 and 9,
  one per head), arrives for them, and tells the group at the end of its
  walk (barriers 10 and 11) whether that arrival was the last; only then,
  which takes the unit's lowest part to finish first, the group merges
  that unit after its other work.
- One thread of a group decides what its part does, and the group's next
  barrier publishes the decision. The decision waits for nothing
  (`split_arrival`: one look at the unit's counter and, under a skew, a
  short walk over CTAs that reads no memory), as no part of the exchange
  waits for another CTA. Keep it so: a single lane that waited there for
  another CTA never stored its unit, and a source test keeps the block
  free of waits. The spare warp and the groups meet at
  barriers of the non-aligned form (`barrier.cta.sync` and `.arrive`),
  which they reach at different instructions; one of the warp's lanes does
  wait, for the launch's dependencies, ahead of the warp's arrival.
- Whichever part merges folds all parts in source order, the unit's first
  steps first, each term one fused multiply-add onto the sum so far: with
  `M` the largest of the parts' row maxima `m`, the unit's denominator and
  panel are the sums of `l * b**(m - M)` and `O * b**(m - M)`. The bits do
  not depend on the order in which the parts finish. The common merge, of a
  unit in two parts by its lowest part, starts the other part's loads
  inside the part's own reduction, so they overlap the wait for its last
  PV, and folds its own panel onto the other's eight columns at a time: no
  path holds two panels in registers, which the softmax's register quota
  has no room for.
- Every access to the workspace follows the launch's dependency wait in the
  thread that makes it: a part's epilogue starts with the wait and the
  spare warp waits before its arrival. The alias plan lists the workspace
  as stored and not as read ahead of that wait, so the capture proof still
  lets a captured launch read its inputs while its predecessor runs, and
  no other argument may overlap the workspace.
- The stores into the workspace carry cache hints of their own, which
  `cute_store_policy` (the outputs' option) does not reach. A partial's
  hint is a trade, which the measurements below decide. In the common
  merge, of a unit in two parts by its lowest part, the other part's
  partial waits for the merge while the CTAs stream the rest of their
  shares through L2: a CTA runs a part that is not its unit's lowest
  first and a lowest part last (at 320 requests of one length on 148
  SMs, where a share is more than two units, the wait is more than half
  of a CTA's walk). The evict-last hint protects the partial's lines for
  the merge's loads, and with them megabytes of L2 against the K/V
  stream; under the evict-first hint the stream may displace them, and
  those loads may miss L2. Against that, one merge uses a partial (only the
  panel's quads are read once on every path: the general merge reads
  every part, its own included, and the rows' pairs twice, and the spare
  warp's look reads the pairs as well): its stores (`store_part`, a
  thread's quads of the panel and a row owner's pair) carry the L2
  evict-first hint (`st.global.L2::cache_hint` under one
  `createpolicy.fractional.L2::evict_first` word a thread, made ahead of
  its stores). A counter's reset (`_count_arrival`) carries the evict-last
  hint, and an arrival is an atomic add, which takes no hint.

On the stream's side, a launch rewrites much of the workspace (at 320
requests on 148 SMs 256 of its 592 slots, 4.3 MB at 32 query columns), and
lines protected from eviction are L2 taken from the K/V stream of every
CTA, at a price that depends on where the workspace lies among the
process's other allocations and on what the process ran before. Measured on
a B200 with the searched configs of a 2047-key window at 320 requests (the
spare warp's look, outputs under `l2_evict_last`), us per call by the legs'
means, evict-first against evict-last in batches of rotated or mirrored
role order, at two calls a sample and, where two figures stand, at two and
at twelve:

    sample                                      draft, 32 columns   verify, 40 columns
    eval harness, the case alone                -0.47 +- 0.07       -0.05 +- 0.05
    eval harness, twelve cases a process        -0.22 +- 0.11       -0.45 +- 0.18
    a tool of its shape, its own allocations    -0.25 / -0.25       -0.23 / +0.08
    that tool, the harness's allocations        -0.48 / -0.17       -0.03 / +0.04
    that tool, behind the draft kernel                              -0.64 / -0.10

The harness clones its inputs ahead of the first launch, which moves a
captured graph's workspace from a 32 MiB boundary to 6 MiB past one: at two
calls a sample the evict-last draft kernel read 0.38 +- 0.10 slower there
than behind the tool's own allocations, the evict-first one 0.15 +- 0.13
and -0.02 +- 0.11 in two batches. The verify kernel reads the same alone
and less behind other balanced kernels of its process, as protected lines
that outlive their kernel's launches would have it. In the harness the
share of the draft kernel's samples at the handwritten kernel's timer step
rose from 2 to 39% (the handwritten kernel's own: 58%), and among twelve
cases in a process the verify kernel's median fell from 56 steps to the
handwritten kernel's 55. No case of that class lost a step. Without a
window a partial waits longer for its reader, hundreds of microseconds at
200 requests: of twelve such cases, all at the handwritten kernel's level
before and after, the three verify kernels at 320 requests read 0.7 to 1.7
less on calls of 640 to 670 us, and one of the three at 200 requests 1.4
+- 0.3 more on 760. Results are bit-identical: a hint moves no value.

`cute_mma_dag_partition_skew`, in pages, makes the shares of a balanced
launch unequal. With a skew of `k` the range of CTA `c` begins at slot
`c * T // G + k * c * (G - c) // (G - 1)`, so the shares fall linearly from
the mean plus `k` for the first CTA, which the launch dispatches first, to
the mean minus `k` for the last, and the CTAs end one after another instead
of together. Zero, the default and the meaning of a missing key, is equal
shares. A skew costs its own length at the end of the launch, where the
first CTA runs on alone, so it pays only as a small part of the share; at
the mean share it would give the last CTAs empty or reversed ranges, on
which a launch hangs. The device therefore applies `min(k, (T // G) // 32)`,
none on a single CTA, whatever the lengths are: a launch of short units
applies less than the config names, down to none, and no range is ever
empty. A config may name only the skews that some launch of the bind can
apply in full (`partition_skew_choices`: up to `units * (pages + 1) //
min(units, SMs) // 32`), where `pages` is the most a unit walks: the page
table's extent, and under a static sliding window the pages that hold a key
one of the unit's rows sees (`window_page_bound`: `(window + positions +
126) // 128 + 1`, 18 for a 2047-key window). The positions are counted
from the rows and their group, `(rows - 1) // group + 1`, not read off the
constant the source subtracts from a request's length, which places the
first row. Only the direct-I/O layout and the head-wide rescale predicate
tie the rows to `query_len * group`, that constant times the group;
elsewhere it may be any offset, zero included. The bound applies where
the mask has a static window and the page table's key coordinates fit an
Int32; without a window, on a longer table, and under a window that holds
more pages than the table, the extent decides as before. It is the mask's
bound, not the loop's: a source whose loop visits more pages than its rows
see is offered fewer skews than its launches could apply. A windowed bind
therefore offers 0 and 1 to 320 units on 148 SMs and no skew to 200 units,
where the page table's extent alone offered skews up to 16, each a kernel a
search compiled and timed that ran the schedule of a smaller skew.
A skew changes which units are split and where, so results move
within the rounding described next. Finding the CTAs of a split unit under
a skew walks down from the equal shares' owner, one CTA and two divisions a
step in one thread (`_slot_cta`); the bound keeps that to two steps on 148
CTAs. On a B200 (148 CTAs, time against no skew): without a window, over
six cases each, 8 pages take 0.2% off 320 requests of 211 pages (shares of
458 slots) and 0.4% off 200 requests of 391 (529), and 16 pages applied
whole added 0.3% and 0.1%; on a 2047-key window (shares of 39 and 25 slots)
8 pages applied whole added 7% and 10% (63.6 against 59.4 us and 45.3
against 41.0), which is where the bound comes from. With it those launches
apply one page and none, and a config that names 8 reads 59.7 and 41.1 us,
within the measurement's spread of no skew. On a windowed bind such a
config is no longer one of its configs: an `InvalidConfig` that names the
bind's choices, repaired to equal shares.

Units that run in one CTA keep `item`'s bits. A unit in parts differs by the
rounding of a reassociated sum: its parts round their probabilities to BF16
against their own maxima and the merge adds a few FP32 operations. On the
partition tests' random data that is at most 0.6% of the unit's largest
output and 2e-6 in the LSE. That is a reassociation only `fast_math`
licenses, so the choice exists only under it, and only for the normalized
epilogue (the raw-statistics one has no merge). The same license covers a
permutation of whole tiles that every row sees in full
(`cute_mma_dag_page_order="bucketed"`, below): in real arithmetic the
recurrence does not depend on the order of such tiles, and in floating point
another order rounds each probability to BF16 against another running
maximum and adds the tiles' products and denominators to the FP32
accumulators in another sequence, as a unit in parts does at its part
boundaries; no tile with a mask or a key clip is moved. It also covers a
walk of a unit's tiles newest first (`cute_mma_dag_page_order="reversed"`,
below), which moves every tile, each with the mask and the key clip of its
own page. Which units are split
follows from the lengths and the launch's CTA count, so results are
reproducible on one device and differ in those bits between devices of
different SM counts. The inputs must not change while a launch runs, as for
`whole`. Here a changed length (CTAs that read different lengths plan
different splits), like a launch that never completes, can also leave a
counter of the workspace at a count `L` above zero, and nothing zeroes it
again. Every later launch that splits a unit at that counter, in `n` parts,
then goes wrong at the unit. With `L < n` a part merges without the last `L`
parts: it folds what their slots hold then, an earlier launch's partials or
stores still in flight, and the late parts count on from its reset and leave
`L` behind again (less only by an arrival that falls between a merger's
read of the counter and its reset), so wrong values can persist across
launches of the same split; when the early merger is the lowest part, a part
that arrives between its look at the counter and its reset merges as well,
and two parts store the unit. With `L >= n` nothing merges, the unit's
outputs keep the previous contents of their memory, and the counter grows
with every launch.
Captured launches adopt a workspace that an earlier eager launch zeroed, so
the graph gets no fill; a capture without one zeroes the counters inside
the graph, once ahead of its first launch. The eager launch zeroes that
spare on a stream the runtime keeps for these fills and waits for the fill
alone, not for the work queued on the stream it launches on. CUDA creates
that stream for the runtime (`cuStreamCreate`, one per device, kept for the
process): a stream of PyTorch's pool is handed to other code too, and a
thread capturing a graph on it would get the fill as a node of its graph and
lose its capture to the wait. CUDA refuses a thread in its default capture
mode the spare's allocation and that wait while any thread captures a graph
in the global mode, the default of `torch.cuda.graph`, and the refused call
invalidates that capture. The launch therefore makes the spare with its
thread in CUDA's `THREAD_LOCAL` mode, pushed and popped around the spare
(`cuThreadExchangeStreamCaptureMode`), in which the captures of other
threads do not bar it: beside a default-mode `torch.cuda.graph` that another
thread holds open, an eager launch makes its spare and the capture replays
right (measured on a B200, where the wait was refused and the capture lost
without the exchange). A capture of the launching thread bars those calls in
either mode, so no spare is made while that thread's current stream captures
on any device, nor while a `cute_cuda_graph` capture is underway in the
process, and none of the latter begins while one is made: the launch asks,
fills and waits under the runtime's capture gate
(`launch_caches.outside_captures`), which the startup of such a capture
waits for and which the launch itself only tries. Not seen is a raw
`torch.cuda.graph` of the launching thread on a stream that is no longer
current on its device: by CUDA's documentation of the capture modes (no run
has this case) a thread that launches eagerly there has to capture in the
`relaxed` mode. A capture that finds no spare zeroes
its counters inside its graph. A graph's
workspace lives as long as the graph. One captured through `cute_cuda_graph`
releases it at its reset. A raw `torch.cuda.graph` does not expose its owner
to the launcher, so its workspace is leased on the graph being captured, as
a CUDA user object (`helion.runtime.cute.graph_lease`): CUDA ends the lease
when the graph is destroyed and its replays have completed, and the next
launch that captures, or that makes an eager workspace or a spare, returns
the workspaces whose leases have ended to the allocator. Until such a launch
a destroyed graph's workspace stays allocated, so a program that captures
again and again holds the workspaces of the graphs it has alive at once, not
one per capture. What a live graph holds is a whole workspace per stream it
captured on, for as long as the graph exists: one record per resident CTA,
12.4 MB on 148 SMs at 40 query columns and 9.9 MB at 32, whatever the batch.
A server that keeps one captured graph per batch size therefore holds one
workspace per graph (about 0.6 GB for fifty graphs at 40 columns), beside
those of its eager streams.

The search weighs `item` against `balanced` wherever a bind offers both
(`ConfigSpec.cute_mma_dag_partition`). A step onto the partition from a
config without the query TMA, or with the value-metadata transport, takes
the first and drops the second in the same step
(`CuteConfigGenerationExtension._partition_neighbor_updates`): normalization
repairs a partition without them to `item` and never changes another knob,
so the step alone would be an alias of where it started. An
`autotune_config_overrides` entry that fixes a partition, or a skew other
than zero, is no step: every candidate on the paired schedule takes the same
prerequisites under it (`take_overridden_partition_transport`), where
without them the override kept only the few paired seeds that already had
the query TMA, or under a skew alone the balanced ones. A partition named
in a config that selects another schedule is an `InvalidConfig` that says
so: a key a config leaves out takes its own default, not the kernel's
default config. The seeds name `balanced` only where the static unit count
exceeds the bind's SM count: the leading paired family once more on the
balanced partition, behind every other seed, and for a kernel without a
window whose shares take it, in a search that samples the skew, a second
time with a skew of 8 pages, which releases its dependents at entry where
the benchmark observes the release
(over twelve cases at 320 and 200 requests without a window, handwritten /
Helion by the means of 3 x 20 samples, in two runs: with the four configs
side by side 0.987 for `balanced`, 0.991 with the skew, 0.990 with the entry
release and 0.993 with both; with three of them again 0.988 for `balanced`,
0.990 with the skew and 0.994 with both). With every unit on an SM of its
own the partition only adds its plan and its merges. The skew is searched
only where candidates are timed over a cycle of input banks
(`autotune_benchmark_input_banks` above one), the benchmark that resolves
a few tenths of a percent, and a step onto a skew takes the balanced
partition with it. The autotune-off default stays `item`. A bound kernel
is specialized on the SM count (`config_num_sm`): the partitions a bind
offers, the width of its plan, its skews and the seeds' choice follow it.

Measured on a B200 (148 SMs) with the SDG eval harness, the kernel without a
config against the same with `balanced`, us per call (means of the first
draft case of each arm):

| requests, keys | handwritten | `item` | `balanced` |
| --- | --- | --- | --- |
| 320, 2047-key window | 54.5 | 64.1 | 59.3 |
| 200, 2047-key window | 35.7 | 40.0 | 41.1 |
| 320, 211 pages | 641 | 699 | 653 |
| 200, 391 pages | 741 | 769 | 757 |

Over the harness's six cases of each row `balanced` takes 6.3% less time
than `item` at 320 requests on the window and 7.1% less on 211 pages, 2.5%
less at 200 requests of 391 pages (1.5% by the medians: `item`'s latency
steps up within a run there), and 2.2% more at 200 requests on the window.
The exchange is what the window pays: a timing-only probe that ran the
same ranges without it read 57.4 and 39.3 us, about 2 us less, and at 200
requests the equal shares alone gain less than that. With 128 requests, an
SM each, `balanced` reads 26.1 us against 25.2.

The paired schedule's `cute_mma_dag_register_quotas` names the `setmaxnreg`
count of each warp role as `compute/K conversion/V conversion/control`
registers per thread. The default, `144/72/72/48`, is the split the body has
always used: a config that leaves the option out, or names the default,
renders the same source. The other choices are `136/80/80/48`,
`136/72/72/48` and `120/96/96/48`. A split changes no result. It changes how
the DSL's backend and its assembler lay out the roles' unchanged code, and
with that the page period of a stream that is bound by the CTA's own latency
rather than by memory: measured on a B200, by up to 7 % of such a call over
all-zero K and V pages and by 2 to 4 % over random ones. That outcome is not
a property of the quotas: lowering the compute quota alone speeds up the
conversion warps, a larger conversion quota can be slower, the gain does not
add to that of other template changes, a split that is the fastest for one
kernel can be slower than the default for another, and all of it can flip
with any edit of the template. So the option is a searched one, in every
timing regime (it is no cache-residency effect), with one seed per split on
the leading paired seed, and no rule derives a split: the choices are
measured splits.
`setmaxnreg` moves registers between the warps of one CTA, whose pool is what
its threads launched with: 640 threads at 96 registers, the largest count at
which they share an SM's 65,536, which is 61,440
(`paged_attention_registers.py`). Every choice fits that pool, and the
template asserts it on the quotas it is rendered with, because an increase
the pool cannot grant never returns; a value that is not one of the choices
is an `InvalidConfig`. On every body but the paired one the option is pinned
to the default, like the other paired knobs. That includes a config that
names a split and nothing else: it does not select the paired schedule, which
the kernel's own default config does, so it is normalized to the default
split without an error. Name the split in a config that selects the paired
schedule.
A work partition keeps the roles and their warps, so every split runs under
`item`, `whole` and `balanced`: a persistent CTA's walking compute groups take
the compute quota, its conversion warps theirs, and its loader, its issuers
and the spare warp the control quota. A partition therefore pins no split, in
a config, in a search step onto it or under an override, as it pins nothing
but the transport it needs; the balanced seeds run the default split, and the
search crosses the two options.

The paired body's K and V conversions have forms of their own on the packed
scale tile (`packed_scale_tile_layout`: 128 dimensions in groups of 32 whose
four scale bytes are one word of the staged scale tile, and a source
postscale of at least `2**-6`). Only the paired body asks for them
(`packed_tile_forms=True`); the single-head pipelines and every other layout
keep the code they had. A native conversion (S2F6 and an exponent
correction) is exact for a scale byte of at most 247, and the full-domain
conversion in FP32 covers every byte. The V conversion runs the native arm
for a whole tile, its sixteen stores, and votes on the tile's scales behind
them (`_stage_voted_s8_transpose_tiles`): where the vote fails, the warp
waits for those stores and the full-domain arm stores the tile again, so the
guard is on no tile's path to its first store. Each warp votes on its own
dimension group. K's rows and V's lanes correct the exponent once, a row or
a lane: the scale word plus `0x01010101 * (6 + postscale)`
(`_corrected_scale_word`). And the per-tile loops of the compute groups,
both conversions and both issuers count in Int32 where the bind bounds the
stream below `2**31` pages (`stream_page_bound`, the page table's static
extent times the units a CTA may stream; `paired_stream_page_bound`), and in
Int64 as the source's loop does elsewhere. No result changes: on a B200 the
ten SDG eval shapes store the bits of the kernel without the forms under two
input generators, events at every tile position and scale bytes out of the
native range included, where a kernel without the guard differs.

A work item's first tile is always an event, on the output panel of zeros
the item starts with, so the rescale of that panel (load, multiply by the
rows' alpha, store) is left out at step zero of a range (`_compute_pages`:
`maximum_changed & (step > 0)`) and every later event keeps it. Both
drivers zero the panel ahead of each range they hand the recurrence, a part
of a split unit included. No result changes.

The paired schedule's `cute_mma_dag_boundary_prefetch=K` gives the control
warp that is idle with one unit per CTA and no value-metadata transport a
role: it requests the first K pages of K, both heads with their packed
scales, of a unit that has not started into L2 (`cp.async.bulk.prefetch.L2`
byte ranges, two a page; `_paged_attention_prefetch.py`). Zero, the default
and the meaning of a missing key, is off and renders the kernel without the
role; a config may name 8, 12 or 4 pages. A prefetch returns no data, so
results are bit-identical with any value. Which unit a CTA serves follows
from the grid, whose CTAs the device dispatches in index order onto SMs as
they free up (`paged_attention_prefetch.boundary_prefetch_form`):

- *Wave ahead*, more units than SMs and at most two and a half waves of
  them: unit `u` serves unit `u + SMs`, the unit that takes over its SM, and
  requests when its own loader has issued its last request, lagging V tiles
  included (one more pipeline barrier, at which the loader arrives and the
  warp waits).
- *Siblings*, fewer units than SMs with at most a fifth of the SMs left
  idle: the launch's first `H` CTAs start on the SMs the launch ahead left
  idle while the others wait for its SMs, and lane `l` of unit `u < H`
  requests for unit `u + H * (l + 1)` at entry, on at most six lanes a warp.
  `H` is not a property of the kernel: the host installs it above the
  capture guard's wait mode (`capture_pdl_cuda.head_start`), the launch
  device's SMs less the grid of the launch ahead when that launch is of the
  same kernel, and zero behind any other launch, for a launch the guard did
  not overlap and for every eager launch. The form therefore needs the
  capture contract with the `loader_tail` release, and requests nothing
  outside back-to-back captured calls of one kernel.

The role keeps its warp's register quota, the control quota, so it runs
under every register split. Only some lanes of the warp serve a unit, and
one of them waits. The role synchronizes nothing itself: where the body
ends with its all-thread join (late retirement, or a release at exit), the
join waits for the requests, since a barrier instruction "causes executing
thread to wait for all non-exited threads from its warp" (PTX ISA 9.5,
section 9.7.15.1), and elsewhere no role waits for them.

Everything else has no role: a grid that exactly fills the device, leaves
more than a fifth of the SMs idle or holds more than two and a half waves,
another schedule, the value-metadata transport, a persistent partition
(the balanced one has a prefetch of its own, with its own option, below), a
K layout whose pair of heads is not one byte range per page, an unknown SM
count and a kernel with other roots. A config that names pages there is an
`InvalidConfig`, repaired to none, where the schedule it names has no role
for them: the paired schedule in one of those cases, or another
`cute_mma_dag_pass_warps` (`"attention"`, or none named: unlike a register
split, which is pinned to its default there, pages named without the paired
schedule are rejected). Where a config names the paired schedule and runs
another body all the same, the pages are dropped without an error, as every
paired knob is pinned there: under `cute_mma_dag="off"`, and where
normalization repairs the schedule itself (`cute_mma_dag_warps=8`, which no
attention schedule runs, is repaired to `cute_mma_dag_pass_warps="off"`).
The role reads only what the launch reads anyway, in tensors the capture
proof lists as loaded: the served unit's bounds, through the statements of
the kernel's own scalar prelude that they depend on as a function of the
unit (`_unit_bounds_helper`; they load the unit's length and, in a kernel
that takes its page range from a tensor, that tensor), and that unit's first
page ids, inside its own range and the page table. A request is formed only
for a page id inside the pool.

The requests pay only where their pages are still in L2 when the unit loads
them, which is a property of the calls around a launch and not of the
kernel, so the default is off and nothing promotes it. What was measured, on
a B200 (148 SMs) with the SDG eval harness and with ladders of its 17-page
window, per call against the kernel without the role:

- It helps back-to-back calls of a short window that rotate over two or
  three inputs whose requests have equal lengths: 8 pages take 0.85 to 1.4
  us off a call of 26 to 67 us on a 2047-key window at the harness's two or
  three calls per sample.
- Past one wave most of that is a gain per timed sample of calls and not
  per call: 200 and 320 requests read 0.9 and 1.4 us less per call at two
  calls a sample, and at twelve 0.2 and 0.3 us less with 8 pages and 0.8 us
  more with 12 pages at 320. On one wave it is per call: 128 requests keep
  0.7 us at twelve calls.
- With requests of unequal length (4 to 17 pages) most of it is gone: 8
  pages keep at most 0.3 us, and 12 pages cost about 1 us at 296 and 320
  requests.
- On one wave it loses where more inputs rotate than L2 holds: 128 requests
  read 0.7 to 0.9 us less per call over three inputs and 0.3 to 0.5 us more
  over six and twelve. That is with the default's load eviction policies: a
  config without them read 0.7 us more over twelve inputs, of which its 8
  pages won 0.5 to 0.6 us back. Past one wave, 160 and 296 units kept their
  gain over twelve inputs.
- The two limits on the grid are where 8 pages lose even over two or three
  inputs. One wave: 90 to 108 units read 0.3 to 0.5 us more, 112 and 116
  units no change, 119 to 144 units 0.1 to 0.9 us less. More waves, in long
  runs of calls: 320 to 380 units read 0.1 to 0.8 us less, 400 units about
  1 us more and 443 units 0.3 more.
- It buys nothing on long ranges, where a unit's first pages are a small
  part of its traffic: without a window, 8 and 12 pages moved calls of 380
  to 780 us by less than 0.15% at 128 and 200 requests and added 0.3% to
  0.6% at 320.

On a one-wave grid a lone or an eager call requests nothing.

So the knob is searched only where candidates are timed over such a cycle of
input banks (`autotune_benchmark_input_banks` above one,
`ConfigSpec.cute_mma_dag_boundary_prefetch_searched`), with off always among
the choices, and there the leading paired family follows the other seeds
with 8 and with 12 pages; every other search and the autotune-off default
keep it off. A step onto pages takes what the role needs in the same step
(`CuteConfigGenerationExtension._prefetch_neighbor_updates`), and an
`autotune_config_overrides` entry is reconciled with a candidate's pages the
same way (`reconcile_overridden_boundary_prefetch`). Neither names a
register split: pages and splits are crossed by the search. A config the
kernel selects without a search (a cache hit, AOT tuning data) under
overrides that decline the capture policy loses its pages with that policy
on a grid of fewer units than SMs, where the role needs it. A search
resolves the knob only where it moves a call by more than the benchmark's
timer step per call (2.048 us a sample: 0.7 us at three calls, 1 us at
two): a smaller effect, such as the cost over many inputs, reads as a tie
between candidates with and without pages.

`cute_mma_dag_boundary_prefetch_cold=N` extends the sibling form to the
launches without a head start. Zero, the default and the meaning of a
missing key, is off and renders the kernel without the option; a config may
name 2, beside a boundary prefetch on a grid whose form is the sibling one.
Anywhere else a depth is an `InvalidConfig`, repaired to none, and it goes
with the pages wherever a step, an override or a repair takes them. A
launch whose installed head start is zero (the first captured call, a call
behind another kernel, an eager call) then lets its first `SMs - units`
CTAs serve as those of an overlapped launch would, the waiting units' first
`N` pages only (`prefetch_boundary_units(cold_depth=N)`); an overlapped
launch serves its `K` pages as before. What holds in each wait mode:

- A cold launch in the early mode has waited at the kernel's entry, so the
  role's reads and requests are behind the launch's dependencies.
- A cold launch in the before-writes mode (behind a launch of another
  kernel that the guard proved it may overlap) reads the served units'
  bounds and page ids and issues its requests ahead of its dependency wait,
  as an overlapped launch's role does. These are tensors the capture proof
  lists as loaded, a launch that stores one of them is a dependency the
  guard answers with the early mode, and a prefetch reads nothing back.

It pays only where the call sequence returns to the same pages within a few
launches. Measured on a B200 with the SDG eval harness (128 requests of a
17-page window, us per call at three calls a sample against the same config
without the depth):

- Over the eval's three input banks: 0.23 to 0.38 less at 27k-bs512 verify
  position 0, and 0.04 to 0.06 less there at twelve calls a sample. At
  position 512 the depth alone read 0.14 less at 27k-bs512 verify and 0.07
  less at 50k-bs512 verify; the ranges measured there, 0.12 to 0.18 and
  0.06 to 0.10 less, are of the depth together with the first-tile skip
  above, which alone read 0.03 to 0.07 less there.
- Over six banks: nothing (0.04 and 0.07 more).
- On the harness's random data (every page nonzero), measured with the
  first-tile skip as well: 0.04 to 0.16 less, and at twelve calls a sample
  between 0.07 more and 0.12 less.
- Deeper cold depths lose: 7 pages 0.18 more, 12 pages 1.21 more.
- The requests of the timed launch itself cost about 0.2 us; the gain is
  made by the launch three calls earlier, which requested the pages this
  one loads.

So the depth is sampled where the sibling prefetch is (bank cycles) with
off leading its choices, and on a windowed kernel every seed that
prefetches follows once more with two pages, behind every other seed. A
step onto a depth from a config without pages takes the first page count
(8) and what the pages need, and an `autotune_config_overrides` entry that
names a depth gives every paired candidate the same: its own pages where it
prefetches, 8 where it does not (`reconcile_overridden_boundary_prefetch`).
The candidates of the other schedules, which have no prefetch, are
rejected under such an entry, as under one that names pages.

`cute_mma_dag_boundary_prefetch_ahead` is the balanced partition's own
prefetch, the range-ahead prefetch, and its value is a distance in CTAs.
Zero, the default and the meaning of a missing key, is off and renders the
kernel without the role; a config may name 48 or 96 on the balanced
partition of a grid of more units than SMs (`range_ahead_distances`: a
distance names a CTA of the grid, and a grid of more than 608 units, whose
spare warp scans lengths, has none). It is an option of its own beside
`cute_mma_dag_boundary_prefetch`, whose values are pages of a unit on a
grid of one unit per CTA: a config has at most one of the two, and a
distance on another partition is an `InvalidConfig`, repaired to none, as
pages are on a persistent one.

The spare control warp of CTA `c` requests, at the CTA's start, the first
three K pages and five V pages of the stream of CTA `c + D`, both heads
with their packed scales (`_prefetch_range_ahead`). The warp waits at the plan
scan's barrier beside the loader, finds the launch's slots, active CTAs and
applied skew from the scan's tables, arrives at the setup rendezvous
without waiting there and reads nothing the rendezvous publishes. The
served range is cut and its first unit located as that CTA's own loader
finds them, and a step's page comes from the body's one walk
(`attention_page_coordinates`), so under every page order a lane requests
the tile the served loader requests at that step. The role reads the page
table only, one id a lane inside the table's extent; the served unit's
length is the scan's word in shared memory. A request is formed only for an
id inside its operand's pool, and a CTA within the distance of the grid's
end, or of the launch's active CTAs, serves nobody. In the before-writes
mode those reads come ahead of the dependency wait, like every loader's: a
page table that the launch ahead stores is a dependency the capture guard
answers with the early mode, and a device test replays that pair with the
guard's decision overridden as its failing control. The whole-units
partition has no range-ahead prefetch: the role was measured and tested on
the balanced partition alone. The search has the distance as a field of
its own wherever the bind offers one, sampled where the page counts are (a
benchmark that cycles input banks); a step onto a distance takes the
balanced partition and its transport, and an `autotune_config_overrides`
entry that names one gives every paired candidate the same
(`boundary_prefetch_ahead_needs`,
`reconcile_overridden_boundary_prefetch_ahead`).

The paired schedule's `cute_mma_dag_release_lead`, in pages, is the lead of
its loader's tail release (`cute_pdl_release="loader_tail"`): the loader
releases the launch's dependents when its stream has that many pages left.
Four, the default and the meaning of a missing key, is the lead the release
has always had and renders the kernel without the option; a config may name
8 or 16. The release moves no data, so results are bit-identical with any
lead. A stream shorter than the lead releases at its first page
(`_tail_release_step` clamps the step at zero), which is never later than
the default lead releases the same stream; the default keeps its own test,
which a stream of fewer than four pages meets only behind its loop. On a
persistent partition the stream is the CTA's whole stream of units.

The lead is the release's: a config may name one only where its resolved
dependent-launch policy is the guarded capture overlap with the `loader_tail`
release, and a lead beside any other policy, on another schedule or on a
kernel that does not take that policy is an `InvalidConfig`, repaired to the
default. Where a config names the paired schedule and runs another body all
the same (`cute_mma_dag="off"`, a repaired schedule), the lead is dropped
without an error, as every paired knob is pinned there. Past one wave, and
on a grid the host sizes at runtime, a `loader_tail` config renders the
automatic release (`passes.pdl.loader_tail_release_applies`: on a grid of
one unit per CTA known to take more than one wave, the entry of every CTA);
the config keeps its lead there as it keeps the release's name, and renders
the kernel it renders without one.

What it changes is when the next captured call may start. The dependent
launch follows the last CTA's release by 4 to 8 us, so with four pages (4.8
us of a memory-bound stream) the next call of a long stream enters as this
one ends, and SMs that freed up earlier sit idle; with a longer lead its
CTAs take them as they free up. A short stream that releases that early
holds its successor's CTAs resident beside its own instead. Measured on a
B200 (148 SMs) with the SDG eval harness over two input banks, us per call
against four pages:

- 128 requests without a window, on one case of 211 pages and one of 391:
  8 and 16 pages read 1.0 and 2.0 us less at two calls per sample (32 and
  64 pages nothing more), 1.9 / 2.4 and 4.25 / 5.4 us less at six, and 16
  pages 5.7 and 6.2 us less at twelve calls over six banks. Over the twelve
  cases of the class at two calls per sample, in two batches: 16 pages read
  1.0 us less in the first (none on three cases, 2.0 on three; handwritten /
  Helion 0.9860 to 0.9886) and 0.8 us less in the second, which also
  measured the ring and the page order (2.05 less to 0.5 more, one case
  slower; 0.9857 to 0.9875), with two of the twelve cases at 0.99 in each.
- 128 requests of a 17-page window, twelve cases at three calls per sample:
  8 pages read 1.0 us more (none to 1.4) and 16 pages 2.5 us more (2.0 to
  3.4; handwritten / Helion 0.988, 0.950 and 0.896).
- 128 requests without a window over random K and V, where the CTA and not
  memory bounds the stream: 16 pages read 1.5 to 1.6 us more.

So the lead is no default and nothing promotes it. It is sampled where the
release itself is, in a search whose benchmark times back-to-back captured
launches (`autotune_benchmark_input_banks` above one,
`ConfigSpec.cute_pdl_capture_overlap_searched`), with four pages always
among the choices; there the leading paired family of a kernel without a
window, on a grid of one unit per CTA that fits one wave, follows the other
seeds with 16 pages. Every other search and the autotune-off default keep
four, and no seed names a lead for a windowed kernel. A step onto a lead
takes the release it belongs to in the same step
(`CuteConfigGenerationExtension._release_lead_neighbor_updates`), an
`autotune_config_overrides` entry is reconciled with a candidate's lead as
with a boundary prefetch (`reconcile_overridden_release_lead`), and a config
the kernel selects without a search loses its lead with a policy that
overrides decline. Where a lead renders nothing (one unit per CTA on a grid
past one wave, whose `loader_tail` config renders the automatic release), a
search does not sample it: every value is the same kernel, so its rows of
such configs are one row (`CuteConfigGenerationExtension._repair_release_lead`)
and a draw or a step onto a lead there is an alias of a config the search
has, which takes no slot; the step takes no release either. The persistent
partitions of such a kernel launch one wave of CTAs and keep the lead as a
searched option, a kernel without them draws four pages alone, and an
override keeps the lead it names. Like the prefetch, a search resolves the
lead only where it moves a call by more than the benchmark's timer step per
call.

The paired schedule's `cute_mma_dag_value_ring` is the depth of its raw V
ring: the pages of raw V the loader stages per head. Two, the default and
the meaning of a missing key, is the depth the body has always had and
renders the kernel without the option; a config may name 3. The loader
requests K, K, V, V of a page in strict sequence, so a V request that waits
for its ring slot also holds back the next page's K requests, and one more
staged page per head decouples them. The depth changes no result, only when
a raw tile is requested: the loader and the V conversion warps take the
ring's slots and barriers in turns of the depth (`_ring_slot`,
`_ring_phase`), under every V lag. The raw K ring keeps two pages; a third
measured nothing.

One more page per head is two more raw tiles with their scales and two more
pairs of ring barriers: 33,792 bytes of the CTA's shared memory and, in a
block of 32 barriers, 256 for its second round. A CTA whose shared memory
the device does not grant never launches, so the depth is admitted against
the device. The body's shared memory is one table
(`paged_attention_shared_memory.py`), from which the template allocates;
the config contract replays that table with the depth, for one unit per CTA
or with the query stages and plan tables of the config's persistent
partition, against the shared memory the device grants one CTA
(`value_ring_reason`, `paged_attention_value_ring_choices`), and a depth
that does not fit is an `InvalidConfig` that says what it would take,
repaired to the default. On a B200 (232,448 bytes) one unit per CTA holds
three pages at every panel width (the contract admits the depth from
214,936 bytes at 40 query columns and from 206,232 at 32, the table with
every optional part), and so does a persistent CTA of up to 32 query columns
(at most 226,136 bytes with two query stages); at 40 columns its two query
stages leave no room (233,912 bytes or more). The render repeats the check:
the body is
given the capacity its depth was admitted on and asserts the table it
allocates against it. The value-metadata transport, whose actor walks the
ring at its default depth, admits no other, and neither does a config that
names another schedule (`cute_mma_dag_pass_warps="attention"`, or none): a
depth there is an `InvalidConfig`, repaired to the default, as pages of a
boundary prefetch are and unlike a register split, which is pinned. Where a
config names the paired schedule and runs another body all the same
(`cute_mma_dag="off"`, a repaired schedule), the depth is dropped without an
error, as every paired knob is pinned there. A render without a device
has no capacity to check and admits every depth, as it admits the schedule.

The depth is a searched option in every timing regime, since it changes the
loader's own schedule, wherever the device holds it with one unit per CTA,
and a search samples it with one unit per CTA alone: a persistent CTA that
holds it is slower with it (below). A step onto the depth drops the
value-metadata transport and takes one unit per CTA in the same step,
whatever room the config's partition has
(`CuteConfigGenerationExtension._value_ring_neighbor_updates`), and a
search's rows on a persistent partition hold the default depth
(`_repair_value_ring`): a draw, a step onto the partition, a seed or a
cached config that names a depth there is that partition's row without it.
The contract keeps the depth wherever the CTA holds it, for a config outside
a search and for an `autotune_config_overrides` entry, which is reconciled
with a candidate's depth by the contract's rule: the candidate takes one
unit per CTA only where its partition has no room
(`reconcile_overridden_value_ring`). What a search can find alone
is limited, because the depth interacts with two other knobs. Measured on a
B200 with the SDG eval harness, 128 requests without a window over two input
banks, us per call:

- Alone it is within the measurement's noise: over eight cases 1.4 us less
  at two calls per sample (none on one case, 3.0 on another) and 0.5 us
  less at twelve, with one case 0.8 us slower. Beside the longer lead of
  the tail release (`cute_mma_dag_release_lead=16`) it reads 2.0 us less
  at twelve calls, and beside a page order that shortens the stream's
  period 2.6. Over the twelve cases of the class at two calls per sample,
  handwritten / Helion read 0.9860 without either, 0.9886 with the lead
  (1.0 us less) and 0.9942 with the lead, the depth and V issued with its K
  (3.2 to 3.4 us less, 1.1 to 6.1 by case).
- It needs the V requests in step with K's (`cute_mma_dag_v_lag=0`): on a
  case whose config has V three tiles behind, the pair of lead and depth
  gained 1.0 us where it gained 3.1 with no lag, and the kernel without
  the pair cannot tell the two lags apart (264.19 against 264.22 us).
- It loses where the CTA and not memory bounds the stream: 1.9 us more over
  random K and V. A 17-page window does not move (twelve cases of 128
  requests at three calls per sample: none on ten, 0.7 us either way on
  two).
- It loses on a persistent partition where the device holds it: the six
  `balanced` configs of 200 and 320 requests at 32 query columns read 4.3
  to 12.8 us more per call (of 647 to 762 us) with three pages per head. A
  cold search of one of those kernels that sampled the depth there spent
  467 of its 576 attention configs on it and returned none of them, which
  is why a search samples the depth with one unit per CTA alone.

So no seed names the depth alone. Where a search samples the lead and seeds
it (a kernel without a window, a grid of one wave, a benchmark over bank
cycles), the seed with 16 pages is followed by its counterpart with three
pages per head and V issued with its K; everywhere else the search starts
from two pages and has the depth as one of its moves.

`cute_mma_dag_value_ring_tiles` counts that ring in tiles for a persistent
CTA of a kernel with a static sliding window. Zero, the default and the
meaning of a missing key, is the ring in pages and renders the kernel
without the option; a config may name 5, beside the default depth in pages
on the whole-units or the balanced partition. One unit per CTA, a kernel
without such a window, another depth in pages and the value-metadata
transport have no ring in tiles: an `InvalidConfig`, repaired to none.
Tile `2 * step + head` of the CTA's stream takes slot `tile % 5`, and the
phase of the slot's barriers is `(tile // 5) & 1` (`_ring_tile`,
`_ring_slot`, `_ring_phase`), in Int32, for which the bind's stream bound
(units times the page table's extent) must be below `2**30` pages. Half a
page per head more than the default is one more raw tile with its scales
and barriers. The contract's replay of the shared-memory table takes the
tile count (`ValueRingTiles`) with the partition's query stages and plan
tables: on a B200 (232,448 bytes) five tiles fit a persistent CTA of 40
query columns with two query stages (222,552 bytes) where a third page per
head does not (238,936), and the render asserts the table it allocates
against the capacity the count was admitted on. The count changes no
result. It is a field of the search in every timing regime where the device
holds it; a step onto it takes a persistent partition, and an override is
reconciled the same way (`reconcile_overridden_value_ring_tiles`). Such a
move can leave another knob of the same candidate without the partition
that runs or holds it, and the candidate then loses that knob, as a row
of the search does when normalization repairs it
(`drop_displaced_stream_knobs`): a tile count or a range-ahead distance
that a fixed page count moved to one unit per CTA, and, on the partition a
fixed tile count or distance moved a candidate to, its pages of one unit
per CTA, a page order that partition does not run (the bucketed order, and
the reversed walk on a grid of one wave) and a ring in pages deeper than
its CTA holds. So under an `autotune_config_overrides` entry that names a
tile count or a distance every paired candidate is on a persistent
partition with it: one with one unit per CTA takes the balanced partition
and its query TMA, and under a tile count gives up a deeper ring in pages,
beside whose default alone a ring is counted in tiles. Only an option the
entry itself names is never dropped: a candidate that cannot hold it is
rejected.

The range-ahead prefetch and the ring in tiles were measured together with
the newest-first walk, on a B200 with the SDG eval harness (320 requests of
a 2047-key window, the frozen harness at two calls a sample, us per call
against the tuned config of the balanced partition): the walk alone 0.58
less, the walk and the prefetch 0.51 less, the walk and the ring 0.68 less,
all three 1.16 less, one timer step, in 11 of 12 legs. With the walk and
the ring, a distance of 48 CTAs read 1.11 less, 96 CTAs 1.45 less and one
CTA 0.29 more. No single step from the walk shows the gain, so on a
windowed kernel past one wave, where a search samples the prefetch, the
balanced seed in reversed order (clipped page eight pages from a unit's
end) follows once per distance with the prefetch and five tiles, and each
once more under the `l2_evict_last` store hint. These seeds and the cold
depth's exist only where the search samples the boundary prefetch, a
benchmark that cycles more than one input bank
(`autotune_benchmark_input_banks`): with one bank no seed names a distance,
a tile count or a cold depth. At twelve calls a sample
the walk's forms were slower than the source order at 296 and 320 requests
(all three 0.28 more at 296). Without a window the ring alone cost 4.96 us
at 320 requests of 211 pages, which is why it is a windowed kernel's.

The softmax tile loop of every paired kernel runs in warp-uniform form where
the bind bounds its trip count. A compute group takes the loop's trip count,
its barrier phase and the unit's first step, page range and length from
words every thread of the group loads from shared memory (a persistent
CTA's plan and the group's record) or from the kernel's inputs. They are
one value for the group, which the assembler cannot know: for a balanced
CTA it keeps the loop's two try-waits inside convergence regions (`BSSY` /
`BSYNC`) and yields at the loop's top, and the loop reloads the unit's first
step from the group's record at every tile. Under the bound that lets the
per-tile loops count in Int32 (`stream_page_bound`) `_compute_pages` passes
those values through `cute.arch.make_warp_uniform` at the unit's head and
keeps the first step in that value. A tile's key base then comes from
the one mapping every role takes its tiles from,
`attention_page_coordinates`, walked in Int32 (the source's loop and the
reversed walk with every lead of its clipped page; a source reflection
keeps its own dtype and the Int64 mapping), where the keys of the page
table's extent fit an Int32 as well: a page's index times its 128 keys, so
a table of fewer than 2**24 pages, which the steps' bound (the units times
that extent) does not imply. Above it the product stays in Int64, and a
table of dynamic extent states neither bound and keeps the loop that marks
nothing. Every value is the integer it was, so results are bit-identical,
and there is no option.

What the compiled loop loses depends on its recurrence and on the assembler
(`softmax_loop_forms` in `helion/_testing/cute_paged_stream.py` counts it
in a disassembly). The counts here are measurements with one toolkit, the
assembler of CUDA 13.3 (`Cuda compilation tools, release 13.3, V13.3.27`),
and another may leave fewer. The tests ask for what does not depend on it:
no yield in the loop, and fewer convergence regions and fewer try-waits
inside one than the same kernel has without the bind's bound; these counts
are their ceilings. The per-key recurrence has two try-waits a tile, and
the loop of a balanced CTA is left with no `YIELD`, `BSSY` or `BSYNC`. The
recurrence over reference scores under `early_tmem`, which the kernels of
the parity work run, votes twice a tile and has three try-waits: the QK
and, on either side of the vote's branch, the previous PV. Three of its
tuned kernels (a 2047-key window, position 0), compiled without a device,
without the form and with it:

| loop of | instructions | `YIELD` | `BSSY` | `WARPSYNC` | `BRA.DIV` |
|---|---|---|---|---|---|
| 320 requests, draft | 605 to 580 | 1 to 0 | 6 to 2 | 6 to 4 | 3 to 1 |
| 320 requests, verify | 714 to 699 | 1 to 0 | 6 to 1 | 5 to 3 | 3 to 1 |
| 128 requests, verify (one unit per CTA) | 692 to 678 | 0 | 3 to 2 | 3 | 1 |

In the two 320-request kernels the yield and the regions around the QK wait
and around the previous-PV wait behind an event are gone; the kernel of one
unit per CTA had neither. The previous-PV wait of a tile without an event
keeps one region of the two it had in the draft kernel and in the
128-request one: it stands under the vote's branch, and the vote is not
marked (next paragraph). The region around the group barrier of an event
stays in all three. The test fixture's loop over reference scores has the
draft kernel's two regions at 32 query columns and only the barrier's at
eight, where the same kernels without the bound have six, with all three
waits inside one.

The result of the maximum vote is not marked, although a barrier's
reduction gives every thread of the group the same value. Marked too, the
loop also loses the warp synchronizations of the vote's branch, and that
kernel measured worse: nothing at 320 requests and 0.2 to 0.4 us more with
one unit per CTA (200 requests, verify position 0: 0.19 more for the vote's
mark alone, 0.37 more with the rest), for a reason that is not understood.
Measured on a B200 with the SDG eval harness (a 2047-key window, us per
call against the tuned kernel of the same batch at two | twelve calls a
sample, hand-edited snapshots of the same changes):

- 320 requests on the balanced partition: draft position 0 0.34 less | 0.44
  less, verify position 0 0.50 less | 0.97 less; in the frozen harness 0.3
  to 1.1 less at each of the six 320-request cases.
- One unit per CTA: 200 requests, verify position 0, level (0.02 less |
  0.03 less), and the six 200-request cases of the frozen harness keep
  their timer steps; 128 requests, verify, 0.17 less | 0.15 less at
  position 0 and 0.19 less | 0.21 less at position 512.

`cute_mma_dag_partition_look` says which threads of a balanced CTA look at
the counter of the split unit it runs the lowest part of. That part runs the
unit's last steps as the CTA's last pages and merges the unit if every other
part has arrived. With `group`, the default and the meaning of a missing
key, one thread of the compute group looks behind the part's pages
(`split_arrival`), and for a unit in two parts the row owners then load the
other part's row statistics ahead of the reduction's second barrier: on the
eval's 320-request kernel that barrier comes 1.7 to 1.9 us behind the last
page, against 0.5 in a whole unit, on the CTA's exit, where a microsecond
of delay was measured to cost the call 1.03 | 0.66 us (two | twelve calls a
sample). With `spare` the CTA's spare warp looks (`look_parts`):

- When the lowest part starts, one thread of the group notes the unit's
  other parts in the group's record (`WALK_LOOK`) and the group arrives,
  without waiting, at a named barrier (LOOK, 12 and 13 for the two heads).
- The spare warp, behind the posts it arrives for (`post_parts`), waits
  there for both groups. Lanes 0 and 1 wait for the launch's dependencies
  and look once each: the group's own relaxed load of the counter and, if
  it shows every other part, its fence and reset (`_count_arrival` without
  an arrival). The lane leaves the answer in the group's record: the
  three words one thread of the group leaves there where the group looks,
  which are the part's role (to merge, as a pair for a unit in two parts,
  or to publish), this CTA as the unit's lowest, and the unit's parts
  (`WALK_ROLE`, `WALK_LOWEST`, `WALK_PARTS`). The role is the whole
  answer: where it is to merge as a pair, in the record of a group that
  asked, the warp's lanes, behind a dependency wait of their own, load the
  other part's (maximum, denominator) pairs into shared memory. Then the
  warp arrives at a second named barrier (MEETING, 14 and 15).
- The group meets the warp there as the first barrier of the lowest part's
  reduction (`_reduce_rows`): the barrier that publishes the row sums is
  the meeting, of the group's 128 threads and the warp's 32, for a part
  the warp looked for and the group's own for every other range, by a
  condition marked warp-uniform. No thread of the group decides or waits
  behind that part's pages; every thread reads the role behind the
  meeting, and `pair_partial` reads the staged pairs. The part's first
  access to the workspace waits for the launch's dependencies where it is
  made: the other part's loads of a pair, in `pair_partial` (of the
  columns the staged panel does not hold, next item), and otherwise the
  wait ahead of the stores. One thread still notes a part to post
  (`split_arrival` with `posts`), which reads the plan and no workspace.
- For that complete unit in two parts the warp also stages the other
  part's panel, behind its answer (`stage_panels`), in the raw K ring. The
  ring is the stream's until its last K tile is converted, so each of the
  four K conversion warps arrives, once, at a barrier of the pipeline's
  block behind its share of that tile (`_convert_keys`), and the warp's
  lane 0 waits for that barrier's one phase, suspended. A slot's own
  barriers do not say this: they alternate with the slot's uses, and a
  wait on their parity from a role that is not in step with the loader
  passes early. The lowest part is the CTA's last range (its unit goes on
  in the next CTA), so nothing is requested into the ring behind that
  tile. Behind the launch's dependency wait the lane then issues one bulk
  copy per head, the panel's first 32 columns (16,384 bytes), or all of a
  narrower panel (8 columns: 4,096 bytes), a quad of columns per thread as
  the workspace holds it (`partition_look_panel_columns`), from the next
  CTA's record into the head's raw K tile, completing on a second barrier
  of the block. The tile is the head's of the ring's position behind the
  stream's last step (`stage_tile`, from the plan's pages, which the warp
  and the group's view of the stage share): the two tiles the stream would
  take next. The group reads its own O panel with one load, as a whole
  unit does, waits for that barrier's one phase, which is complete unless
  the warp was slow, and folds element by element with the other part's
  value from the tile (`merge_pair_staged`): the fused operations of
  `merge_pair_other` and `merge_pair_own` on their operands in their
  order, so the same bits. At 40 query columns the last eight columns are
  loaded and weighted as where the group looks, which loads the two panels
  in turns, eight columns at a time. Both barriers lie in the padding of
  the barrier block, which a persistent CTA's query barriers make 64 slots
  long: the stage adds no byte to the table.
- A group that met the warp there heard the verdict of its own post, which
  the warp answers before it looks, and skips the meeting at the end of its
  walk. The warp's 32 arrivals at that meeting's barrier (POSTED, 10 and
  11) then stay unmatched in a CTA that ran a lowest part, which is
  harmless: nothing waits on that barrier, and the CTA does not use it
  again in its life. A group whose first range is no part to post says so
  when that range starts, so that the warp can look before the walk ends; a
  group that runs no lowest part asks for no look at the end of its walk.
- The meeting at the walk's end also joined the group's warps ahead of the
  late merge of a posted unit (`merge_parts` behind the walk, when the
  posted part arrived last). The merge's row owners write the unit's
  weights into the group's scratch rows, and its maximum, denominator and
  normalization into the row state, ahead of the merge's first barrier; a
  warp that is still in the lowest part's epilogue reads those rows (the
  pair's weights for its fold, behind its wait for the staged panel, the
  weights of a fold of more parts, and every row's normalization for its
  stores). So where the spare warp looks the late merge starts with
  a barrier of the group's 128 threads, on that path alone, and then waits
  for the dependencies itself. Without the barrier a warp that was slow in
  its epilogue folded and normalized its remaining rows with the posted
  unit's values: its 32 output columns of those rows were wrong.
- The group's conditions around the exchange (a range that closes its
  unit, the lowest part, a split range, the meeting) pass through
  `make_warp_uniform` (`_group_uniform`): each is one value for the group,
  and left to the assembler the named barriers under them cost the kernel
  its warp-uniform code behind them (measured on a snapshot that also
  marked the vote: 58 warp synchronizations in the verify kernel against
  12, and 0.4 to 0.8 us at verify in steady state).

The decision is the one the group makes, at an earlier time: merge if every
other part had arrived at the look, else publish and arrive, and the last
arrival merges. Whichever part merges folds the parts in source order, so
the bits are the same with either value and on every launch, and nobody
waits for another CTA. The staged pairs are a row of the body's
shared-memory table (`look_stage`, `4 * query_columns` Float32: 136, 520 and
648 bytes with its alignment at 8, 32 and 40 query columns), and the
contract replays the table with it against the device beside the config's
raw V ring (`partition_look_reason`). Beside the default ring every device
that has the balanced partition holds it (its largest layouts with the
stage, 185,312 and 205,792 bytes, are inside the partition's budgets of
188,416 and 208,896); on a B200 a balanced CTA of 40 query columns with two
query stages holds it beside a ring of five tiles (223,200 of 232,448 bytes
at 640 units) and not beside a third page per head (239,584). Where the
partition and the grid have the look and only this replay refuses it, the
refusal says that the stage does not fit beside the config's ring, and names
`cute_mma_dag_value_ring` and `cute_mma_dag_value_ring_tiles` with their
values, and the bytes. The render asserts the table it allocates against
the capacity the look was admitted on.

The look is offered on the balanced partition of a grid known to hold more
units than the device has SMs; elsewhere there is no field and `spare` is an
`InvalidConfig`, repaired to the group's look. It is a field of the search
in every timing regime: a step onto `spare` takes the balanced partition
with its transport, an override is reconciled the same way
(`reconcile_overridden_partition_look`), and a candidate that another
override moves off the balanced partition, or whose overridden ring leaves
the stage no room, loses its look (`drop_displaced_stream_knobs`). It is a
choice per bind and must not become a default. Measured on a B200 with the
SDG eval harness (a 2047-key window, us per call at two | twelve calls a
sample, hand-edited snapshots, each figure against the tuned kernel of its
batch, which had neither the look nor the warp-uniform softmax loop):

- 320 requests, the loop alone and with the look: draft position 0 0.34
  less | 0.44 less and 0.75 less | 0.60 less, one timer step of the frozen
  harness (55 to 54, the handwritten kernel 53); verify position 0 0.50
  less | 0.97 less and 0.91 less | 1.00 less. With the conditions marked
  the look has no penalty in steady state there.
- At positions 512 and 1023 the frozen harness reads the look 0.2 to 0.4 us
  behind the loop alone in three of the four 320-request cases.
- It loses on a balanced grid with short look margins: at 160 requests,
  whose lowest parts start one or two pages behind their neighbour's post
  (five to seven at 200 requests, eighteen to twenty at 320), the look
  alone costs 0.74 more | 0.48 more, a timer step: the early look finds the
  neighbour less often (87% against 92%), and every CTA pays two more
  meetings in a life of 19 pages.
- Not measured on the snapshots: a search; the look on full attention; the
  look with the marks on an L2-hot stream (without them 0.60 more | 0.77
  more at 320 requests, draft, two legs).

So a search times it case by case. The seeds give it the start it was
measured from: on a windowed kernel past one wave each range-ahead seed
(the reversed walk with a distance and a ring of five tiles, plain and
under the store hint) follows once more with the look, behind every other
seed, where the device holds the stage beside the seed's ring.

The paired schedule's `cute_mma_dag_page_order` is the order in which a CTA
streams a unit's pages. `source`, the default and the meaning of a missing
key, is the order of the source's loop and renders the kernel without the
option. `bucketed` streams the unit's interior pages in the order of their
physical page's bucket, one of 32 equal ranges of the page pool (the page id
without its low bits, `page_order_shift` of the K tensor's page extent), and
within a bucket in source order; the unit's first four pages and its last
twelve keep their places. Every CTA of a launch then reads one region of the
pool at a time, in step with the others. That shortens the page period of a
memory-bound stream (1.199 to 1.187 us per page on a B200) and needs the CTAs
in step: the same buckets rotated per request gained nothing. It is not L2
residency between calls, since the gain holds over six input banks.

What is reordered is decided on the device, by the template's own test
(`_paged_attention_page_order.py`). The CTA's idle control warp, the one a
boundary prefetch uses behind it, takes the stream steps behind the head and
ahead of the tail, at most 416 of them, evaluates
`logical_attention_tile_is_interior` for each with the step's own
coordinates, and requires the steps that pass to be one run of at least 32:
it reduces their number, the first and the last over its lanes, and the run
holds exactly when the span from the first to the last has that number of
steps. A unit whose interior candidates are fewer, or not one run, is
streamed in source order. So every step of the run is a tile without a mask
and without a key clip: the loader requests it at shift zero and the softmax
treats it as any interior tile, and each step keeps its logical coordinates
whichever page of the run it streams. The warp sorts the run's page ids with
a stable counting sort, a function of the ids alone (shared-memory atomics
for the bucket counts, a prefix sum over the lanes, then one round of 32 ids
at a time, each lane taking its bucket's next position plus the number of
lower lanes in the bucket's word), leaves the order and the run in shared
memory (1,928 bytes) and arrives, every lane behind its own stores, at one
more pipeline barrier. The loader streams the head in source order, waits
there, and takes the run's pages from the order; a unit of no more pages than
the head has no step that could take the order, and its loader does not wait
for the warp. The barrier is a pipeline barrier of this role alone; the role
uses no named barrier and keeps the control warpgroup's register quota. The
warp synchronizes (`bar.warp.sync`) between the steps of its sort and nowhere
else: the lanes share the counters, the words and the positions with no other
barrier between one lane's store and another lane's load, which is what that
instruction orders.

The order permutes whole tiles, so it changes results within the rounding
the fast-math paragraph above describes, and only for units it reorders: a
unit streamed in source order has the source order's bits. Two launches over
the same inputs have the same bits, since the sort is stable. On random K
and V a reordered unit's outputs moved by at most 0.7% of the unit's largest
output and its LSE by 2e-6 (the GPU tests bound both). On the SDG eval
harness's inputs, eight cases of 128 requests, no output bit changed over
its timing data; over random K and V the relative RMS against the source
order was 1.8e-3 to 2.0e-3 with three in ten BF16 outputs changed, none by
more than one ulp of the output's top binade (7.8e-3), the LSE moved by at
most 4e-6, and the error against the FP32 reference was the same in both
orders (relative RMS 1.9e-3 to 2.0e-3 and 1.9e-3 to 2.1e-3). There is no
bound in output ulps for an element whose terms nearly cancel, as for the
key clip.

A kernel is offered the order under `fast_math`, on the paired schedule's
`item` partition without the value-metadata transport, where some unit can
have enough interior pages for the sort to pay
(`paged_attention_page_order.page_order_offered`): the page table's extent
less the head and the tail, and under a window the whole pages of the keys
every row sees, must reach 32. The order gains 0.012 to 0.015 us per
reordered page and a sort the stream waits for costs it 0.4 to 0.7 us, and
three reordered pages of a 17-page window cost 1.4 to 2.05 us: a 2047-key
window, with at most sixteen interior pages, is not offered the order.
Elsewhere `bucketed` is an `InvalidConfig`, repaired to `source`, also in
a config that names another schedule or none; where a config names the
paired schedule and runs another body all the same, the order is dropped
without an error, like every paired knob there. The head and the tail are
constants. The head is what the loader streams while the warp sorts: over
twelve cases of 128 requests without a window, with the longer lead and the
deeper ring, a head of two pages read 0.9978 (the stream waits for the
sort), four pages 0.9999 and eight 0.9989, the same as four in the mean of
the samples. The tail keeps the newest pages, whose scores are the largest,
out of mid-stream: with the order's prototype 2 pages read 0.9897 with the
two cases whose newest pages hold large scores 1 to 3 us slower than the
source order, 6 pages 0.9913 and 12 pages 0.9932, and no measured class has
another tail win. A persistent partition has no order
yet: its CTAs stream several units or parts of them, so the sort would run
per unit ahead of the loader, on the spare warp that already arrives for the
compute groups' parts, with a handoff per unit; a host-side reordering sized
that at 0.2 points on 200 and 320 requests, all of it at 200.

The order is a searched option in every timing regime wherever a kernel is
offered it, with a seed for kernels without a window on a grid of one wave:
the leading paired family in bucketed order behind the other seeds, and
where a search seeds the longer lead of the tail release, the last of those
seeds once more in bucketed order: the one with the deeper raw V ring where
the device holds that ring, the lead alone otherwise. Where that seed names
the ring and the search also samples and seeds a boundary prefetch, it
follows once more with a prefetch of four pages, the order's head: a
prefetch requests a unit's first
pages in source order, which behind the head a bucketed stream no longer
takes first. Of four cold searches of 128 requests none timed the three
knobs with a four-page prefetch; set by hand, that form read one timer step
below the returned config on two of the cases (on the other two the
hand-set config, one with four pages and one without a prefetch, read one
step above it). A step onto the order takes
one unit per CTA and drops the value-metadata transport in the same step
(`CuteConfigGenerationExtension._page_order_neighbor_updates`), and an
`autotune_config_overrides` entry is reconciled with a candidate's order the
same way (`reconcile_overridden_page_order`). Measured on a B200 with the
SDG eval harness, twelve cases of 128 requests of 211 and 391 pages over two
input banks at two calls per sample, handwritten / Helion. Two batches have
the order alone: 0.9856 and 0.9857 in source order and 0.9925 and 0.9922 in
bucketed order (2.6 and 2.5 us less per call, 1.6 to 1.9 at 211 pages and
3.4 at 391). The later of them also has the lead of 16 pages, the ring of
three pages and V issued with its K, in both orders:
0.9944 in source order and 0.9986 in bucketed order (3.4 and 5.1 us less
than without the three). The earlier one, the batch that chose the head, has the
three in bucketed order only, at 0.9999 (5.5 us less); in source order the
three read 0.9942 in each of two earlier batches (3.2 and 3.4 us less).
Where the CTA and not memory bounds the stream the order loses (1.1 us more
over random K and V).

`reversed` walks a work item's pages newest first: step `s` of a walk over
the pages `[first, last)` takes page `last - 1 - s`, where the source's loop
takes `first + s`. It is for a kernel with a sliding window, whose every
work item streams the window's few pages and whose newest pages hold the
scores a decode step's rows are largest on. Walked first, they set the
running maximum early, and the older tiles raise it less often: with the
SDG kernel's rescale threshold, 3.4 to 4.0 rescales per request and head
where the source order has 5.1 to 5.5, on inputs whose newest five pages of
seventeen hold random keys (device counters of a prototype). Nothing is
sorted and no warp has a role. The walk is another mapping from a step to
its logical page and key base, and the template has one such mapping
(`attention_page_coordinates`): the loader takes the page it requests and
its key clip from it, the compute groups the softmax's masks, and a
boundary prefetch the pages it requests ahead. The body hands all of them
the same walk, so a step is the same tile to each, and every tile keeps the
logical coordinates of its own page: a tile at the window's edge is clipped
and masked as in source order, at another step.

`cute_mma_dag_clipped_page_lead`, in pages, places the walk's oldest page,
the one a sliding window clips (of its keys only those inside the window
are visible, and under a leading key clip only those are loaded). The walk
takes that page when it has the lead's pages left, the page itself
included, and every newer page behind that step one step later. Zero, the
default and the meaning of a missing key, leaves the page where the
reversed order has it, last, and renders the walk without the option; a
config may name 8. A walk of `n` pages takes the page at step
`max(n - lead, 0)`: a work item of no more pages than the lead takes it
first and the others newest first behind it (the step clamps at zero, as
the tail release's does for a stream shorter than its lead). The lead names
a place in the reversed walk and nothing beside another order, so
normalization drops it there without an error, as it drops the default; a
value that is neither 0 nor 8 is an `InvalidConfig` beside any order.

The walk permutes the source's iterations, so it changes results within the
rounding the fast-math paragraph above describes, in the bucketed order's
terms: in real arithmetic the recurrence does not depend on the order of
its tiles, and in floating point another order rounds each probability to
BF16 against another running maximum and adds the tiles' products and
denominators to the FP32 accumulators in another sequence. Which tiles
raise the running maximum, and so which tiles rescale the accumulators,
changes with it; that is what the walk is for. Unlike the bucketed order it
moves every tile, the masked and the clipped ones too, and every work item
of more than one page (of more than two under a lead): only a work item
whose walk is the source's own has the source order's bits. Two launches
over the same inputs have the same bits. With the page last the kernel has
the bits of a kernel whose source iterates its pages in reverse, which the
GPU tests assert on the fixture; against the same kernel in source order a
request's outputs moved by at most 0.72% of the request's largest output
and its LSE by 1.9e-6 on the tests' fixtures (they bound both at the
bucketed order's 1.6% and 1e-5, and either order within 5% and 0.002 of the
FP32 reference). There is no bound in output ulps for an element
whose terms nearly cancel.

A kernel is offered the walk with the pair proof, under `fast_math`, where
its mask has a sliding window and its source loop visits the pages oldest
first (`PagedAttentionScheduleFacts.reversed_order_missing` says what a
kernel lacks). The window is the mask's fact,
`PagedAttentionScheduleFacts.windowed`: the proved visibility predicate has
a lower bound, `key_position >= query_position - window`
(`LogicalAttentionMask.window_left`). It is no fact about the loop, whose
bounds stay the source's expressions: the page the lead places is the
range's first page, which is the window's clipped page wherever the source
begins its loop at the window's edge, and a tile that no row sees, in a
range wider than the window, keeps its mask at whatever step it has. A
source whose loop index is already the reflection `begin + end - 1 -
iteration` walks newest first in source order and is not offered the
option. A config may name the walk on the paired schedule without the
value-metadata transport: with one unit per CTA, and on a persistent
partition (`whole`, `balanced`) of a grid known to hold more units than the
device has SMs (`paged_attention_reversed_order_partition`). There every
unit of a CTA's stream is walked so, and the parts of a unit that several
CTAs share keep their steps and take the pages the unit's walk has at them.
On a persistent partition of a smaller grid, whose CTAs run about one unit
each, and beside that transport's actor the walk was not examined.
Everything else is an `InvalidConfig` that names the requirement the kernel
or the config does not meet, repaired to `source`, also in a config that
names another schedule or none; a lead beside any order but the walk's is
dropped. Where a config names the paired schedule and runs another body all
the same, both options are dropped without an error, as every paired knob is
pinned there.

Both options are searched in every timing regime wherever a kernel is
offered the walk. A step onto the walk drops the value-metadata transport
and, where the bind's persistent partitions do not run the walk, takes one
unit per CTA, as the bucketed order's step always does
(`CuteConfigGenerationExtension._page_order_neighbor_updates`,
`page_order_needs`); on a grid of more units than SMs it keeps the
partition it starts from. A step onto a lead takes the walk in the same
step, since from source order the lead alone is an alias of where it
started. An `autotune_config_overrides`
entry that names the walk is reconciled as one that names the bucketed
order is (`reconcile_overridden_page_order`); one that names a lead and no
order applies to the candidates in reversed order and names nothing for the
others. A deeper raw V ring stays unsampled on persistent partitions, with
the walk as without. The seeds are two forms, on the leading paired family
and, where a search samples and seeds a boundary prefetch, on each of those
seeds, behind every other seed: the page 8 pages ahead of the walk's end
beside three pages of raw V per head (where the device holds them), and the
page last beside the default ring. Both keep the family's V lag of two:
with V issued with its K the first form lost its gains. On a grid of more
units than SMs one more seed follows them, the leading `balanced` seed in
reversed order with the page 8 pages ahead of a unit's end (below).

Which form gains is a property of the case, not of the kernel. Measured on a
B200 (148 SMs) in the SDG eval harness, the class's cases in one process,
with the handwritten kernel, PR103, and the searched configs in every batch.
Those configs come from searches that sampled the boundary prefetch and
none of the lead of the tail release, the ring depth and the bucketed
order, so they name none of the three, and the two forms are those configs
with the walk, its lead and the ring. A step is one 2.048 us step of the
sample timer, and the
figures are the legs' medians in steps (0.683 us per call at three calls a
sample, 1.024 at two). 128 requests of a 2047-key window, 17 pages, over
three input banks at three calls a sample, two arms of six cases each
(draft and verify at three decode positions, whose newest one, five and
nine pages hold random keys):

    case                 PR103   source   lead 8, ring 3   page last, ring 2
    27k draft, 1 page    33-34   34       33-34            34-35
    27k draft, 5 pages   34      35-36    34               35
    27k draft, 9 pages   37-38   36       36               36
    27k verify, 1 page   34-35   35       34-35            35-36
    27k verify, 5 pages  35      37       38               36-37
    27k verify, 9 pages  39      39       40               39
    50k draft, 1 page    34      34-35    33-34            34
    50k draft, 5 pages   34      36       35               35
    50k draft, 9 pages   37-38   36       37               36-37
    50k verify, 1 page   34-35   35-36    36               35-36
    50k verify, 5 pages  35-36   38       38               37
    50k verify, 9 pages  39-40   39-40    40               40

A range says the legs read both steps. The first form (twelve legs) takes
three draft cases one step down, to PR103's step or below it on two of them
(handwritten / Helion by the pooled medians 0.9710 to 0.9995 and 0.9984 to
1.0298; the third 0.9449 to 0.9719), keeps a fourth with 0.47 us less in
the mean, and loses a step on five cases, four of them in verify, where
the deeper ring costs one. The second (four legs of one batch) takes the
two verify cases two steps behind one step down (0.9455 to 0.9718 and
0.9474 to 0.9730) and one draft case (0.9449 to 0.9719), and loses a step
on two cases. With the best of the source order and the two forms per case,
nine of the twelve cases are at or above 0.99 where the source order has
eight, and the class's geomean is 1.0022 against 0.9902. Leads of 0, 3, 5
and 12 beside the ring were never better than 8 on the draft cases (means
of 23.47, 23.38, 23.22 and 23.32 us against 22.98 on one of them, where
only 8 reached the lower step), and with the ring and V issued with its K
the first form lost its two draft steps (35 timer steps against 34 and 34
against 33). 200 requests of the same window, six cases over two banks at two
calls a sample, on configs with a 12-page prefetch: the walk with a lead of
5, with or without the deeper ring, or with a lead of 8 or 12 beside that
ring, takes one verify case one of its two steps down (40 to 39, 0.9500 to
0.9744; the ring with the page last does not, and neither a lead of 8 nor
the page last was measured beside the default ring there), every form loses
a step on one draft case where the source order is already a step behind
PR103 (37 against 36 and 35), and the other four cases keep their steps. So
no default and no promotion: the search times the forms on the case's own
inputs, at its benchmark's call count.

Cold full searches of the SDG kernel, each from a fresh seed in the eval's
timing regime, timed all six seeds in reversed order. On the 27k draft case
with five pages of random keys (three input banks) they were the first four,
the seventh and the eighth of the 74 timed members of the initial
population: 23.21, 23.89, 23.89 and 23.99 us on the two prefetch seeds,
whose source-order forms read 24.58, and 24.93 and 25.26 on the leading
family, which read 25.85. The search (541 configs, 1,171 s) returned the
first form on the 8-page prefetch, with another register split. On the 50k
verify case at that position they were the first three, the sixth, the
eighth and the ninth of 59 (25.47 to 26.63 us against 25.94 and 26.80), and
the search (501 configs, 1,108 s) returned the second form with V three
tiles behind K; its best reading of the first form was 24.85 us against
24.57. In one comparison of the class in the eval's layout (three rounds,
PR103 at both ends of each), one role ran the configs that cold searches of
the same twelve cases return when the walk is not among their options, on a
compiler without it, and another the same configs with these two cases'
replaced by the two above: the draft case read PR103's 34 steps in three of
three legs where the config without the walk read 35 (handwritten / Helion
0.9714 to 1.0000), and the verify case 36, 36 and 37 against PR103's 36
where the config without the walk read 38 (0.9474 to 0.9987); the ten other
cases kept their readings (pooled medians within 0.31%). On 320 requests of
the same window the walk with one unit per CTA is no contender: the seeds
read 62.5 to 64.2 us there (their source-order forms 63.5 to 64.6) against
the 59.4 us of the `balanced` config in source order that the search
returned, and 17 of that search's 632 timed configs named the walk.

On such a grid the walk belongs to the persistent partitions. A persistent
CTA streams several units, and in source order each of them ends on the
tile of its newest page, which raises the running maximum and rescales
right ahead of the unit's epilogue; the walk makes that tile the unit's
first, which takes the rescale arm anyway. Measured on a B200 (148 SMs) on
`balanced` configs with a 2047-key window over prompts whose K and V are
zero (the SDG eval's data, on which a unit's newest page alone raises the
maximum), us per call against the source order at 2 / 12 calls a sample:

    units   page last       lead 3          lead 8
    296     -0.65 / -0.43   -1.14 / -0.39   -1.03 / -0.84
    320     -0.15 / -0.34   -0.49 / -0.60   -0.75 / -0.76

An independent repeat read every cell within 0.13 us of these. The gain
is the data's: over pages that are all nonzero a lead of 8 read 0.5 /
0.1 us more at 320 units, and in the SDG eval harness (the six cases of
320 requests over two banks at two calls a sample, steps of 1.024 us, each
case's searched `balanced` config in source order and with a lead of 8) it
gains at some decode positions and loses a step at others:

    case            PR103   source   lead 8
    draft, 0        53      56       56 (0.72 us less)
    draft, 512      56      58       57
    draft, 1023     63      61       62
    verify, 0       55      58       58
    verify, 512     60      61       61-62
    verify, 1023    68      67       68

Handwritten / Helion over the six, by geomean: 0.9811 in source order,
0.9779 with the lead on every case, 0.9844 with the better of the two per
case. So no default takes the walk on a persistent partition and nothing
promotes it: a search times it per case, from one seed, since it does not
get there one knob at a time (the order alone, with the page last, read
0.15 us of that kernel's 0.75 at 320 units). `whole` runs the walk by the
same rule and was not timed under it. With no more units than SMs a
persistent CTA runs about one unit, the form one unit per CTA has, and the
walk was not measured on a persistent partition there: such a bind offers
it with one unit per CTA alone.

The paired schedule makes its output stores itself: each compute thread
stores its column of every query row of the O panel, one element a row, and
the row owners a row's LSE (`_compute_store`). The generic packet flushes'
`cute_store_policy` finds no packet to rewrite there, and the body honours
the option itself (`store_output_rows`), in a one-unit CTA, in a persistent
CTA's compute groups between two units, and in the part that merges a split
unit:

- `streaming` stores every O element with the streaming cache operator
  (`st.global.cs`, evict-first);
- `l2_evict_last` stores it under the L2 evict-last cache hint
  (`st.global.L2::cache_hint`), with one policy word a thread makes ahead of
  its stores (`createpolicy`), the form the balanced partition's workspace
  stores use under hints of their own (the option does not reach them: a
  part's partial is stored evict-first whatever it says). The word is made
  at run time, so on this schedule the choice does not need the fixed
  descriptor that gates it for the packet flushes
  (`cute_store_policy_choices`, `_normalize_paired_store_policy`): a kernel
  with the paired schedule is offered all three policies on every target,
  and a config of it on another body takes `default` for `l2_evict_last`
  where the descriptor is missing, as before.

Either way the same value goes to the same address, as many stores in the
same order. A BF16 element is rounded to nearest even inside the store's
instruction pair, as the plain store's conversion rounds it, and the
raw-statistics epilogue's fp32 panel is stored as it is, so results are
bit-identical under all three policies. `default` renders the plain stores;
the per-row stores (a row's LSE, the raw-statistics epilogue's maxima and
denominators) are plain under every policy, and the other attention bodies
store as before. The option was already a searched field of these kernels,
with one kernel behind its values; now a search times three.

The final selection's tie rule for streaming stores covers these stores.
Among finalists its windows cannot tell apart,
`CuteBackend.finalist_preference` ranks a kernel that renders streaming
stores three ranks back: the kernel's own timing cannot see what an
evict-first output costs the kernel that reads it next. The rule reads the
rendered module (`packet_flush_names.streaming_store_rendered`), where a
packet flush with the policy is a call of its streaming helper. The paired
body calls none of those helpers, since its stores are operators of its own
template, and is known by the `store_policy='streaming'` of the kernel's
call of the body. A paired finalist under `streaming` therefore ranks three
behind the same config under `default`; one under `l2_evict_last` ranks as
under `default`, by its release alone.

What the policies are for: the price of an output write is the first touch
of its lines. Measured on a B200 (148 SMs) in the SDG eval harness on a
2047-key window, four legs a role, us per call by the legs' means: a store
into a line that another launch has just put into L2 costs 64 to 80% less,
twelve rotating input banks instead of two double the cost of the output
writes, and neither the stores' width nor their alignment nor writing lines
whole changes it. The handwritten kernel pays the same price in its early
zeroing. `l2_evict_last` keeps an output's lines in L2 across the K/V stream
of the following calls; against the plain stores it read

    requests              calls a sample   us per call
    128 (one unit a CTA)  3 / 12           -0.05 / -0.27
    200 draft             2 / 12           -0.15 / -0.30
    200 verify            2 / 12           -0.13 / -0.27
    320 (balanced)        2 / 12           -1.21 / -0.95

and with twelve rotating banks nothing at 200 requests and -0.56 at 320.
With each case's searched config (24 window cases over the harness's two or
three banks, handwritten kernel in every batch) no case lost a timer step
and two gained one (320 requests, 57 to 56 steps, handwritten / Helion
0.9300 to 0.9463; 128 requests, 35 to 34, 0.9732 to 1.0000); twelve cases
without a window read within noise. `streaming` read 0.3 to 0.5 less at 320
requests and nothing with one unit per CTA (on 320 requests, `balanced`
configs, six cases over two banks at two calls a sample: 0.37, 0.26 and
0.22 less on the three draft cases and 0.17, 0.06 and 0.22 on the three
verify cases); `l2_evict_last` was the better one in every cell. So the
leading family of a kernel with a window is seeded once more under
`l2_evict_last`, with one unit per CTA and, where the units outnumber the
SMs, on the balanced partition in source order and in reversed order (the
walk there was measured on configs that store under the hint); the search
times all three per case.

### 3.3 Whole-kernel generators and regeneration

`KERNEL_GENERATORS` (`kernel_generators.py`): `grouped_prefix_adaptive_n`,
`packed_operand`, `row_resident`, `grouped_warp_tf32`, `flat_grouped_native`,
`bounded_cache`, `split_k_cluster`, `split_k_workspace`, `materialized_fission`,
`ordered_phases`.  Each `applies` to a config, carries a `TransformPolicy`
saying which specializing arguments of `generate_ast` it cannot honor, and
re-enters `generate_ast` for the kernels it composes.  `CuteRegeneration`
regenerates after `RegisterTileUnsupported`, `VloopSinkNotApplied`,
`FragmentMmaNotApplied`, `RingMmaNotApplied` or `LaneUnrollNotApplied`, with
the knob that raised switched off, and after `VloopSinkLayoutRefused` (codegen
refused the tile-first reduction layout `cute_vloop_sink` planned) with the
knob still on and the default layout.  `materialized_fission` (and
`ordered_phases` through it) regenerates a stage after the two vector-loop
signals itself, from graphs it rebuilds.

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
(`build_pass_context`).  `DEVICE_BODY_PROLOGUE` issues each atomic on the
leader thread of the launch axes claimed after it, lowers the collective
matmuls and decides the launch shape; `freeze_pass_context` seals it;
`DEVICE_BODY_PASSES` (from `finalize_shared_reduce_groups` to `jam_grid_lanes`)
rewrite the body; `KERNEL_DEF_PASSES` run
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

One knob is validated and never proposed: `tcgen05_grouped_d_shift`.  The
bounded D shift stores every tile of a group through one static descriptor
based M rows below the output.  Nearly every shape gives that descriptor the
large allocation class, for which PTX ISA 9.5 leaves the store of a group's
partial last tile undefined unless the 128 KiB from the descriptor's address
are accessible, and they begin below the output, in memory the kernel does
not own.  Measured on one B200 only, a store through this geometry writes
the in-bounds bytes and touches no other page.  So no candidate of a search
stores through the shift: no compiler seed carries it, neither it nor the
knob of the paired prefix recipes, which are legal only on it, is a search
coordinate (`tcgen05_config.surface.D_SHIFT_PROPOSAL_KEYS`), and an
LLM-guided search benchmarks a reply that names one without it
(`ConfigSpec.search_proposal`).  The M,N worklist tiles stay seeds, without
the shift.  A config that names the shift, passed to the kernel or read from
the autotune cache, is accepted, normalized and compiled as before.  The
docstring of `grouped_d_shift.d_shift_searched` has the rule, the
measurement, what the gate covers and what lifts it; `KNOWN_GAPS.md` has the
entry for what remains: grouped-d-shift-store-below-output (a config that
names the shift).  The descriptors a kernel rewrites per group take the
small class (section 8).

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
(`Backend.build_standalone_code`) without Helion; its `_standalone_compile`
is also how Helion's launcher compiles (`compile_cache.compile_with_options`),
so the two parse compile options alike.

## 8. Device helpers and byte-pinned paths

`device/__init__.py` states the device-module rules: importable without CUDA,
one top-level binding per helper, no `from __future__ import annotations`
beside a `@cute.struct`, and helper names are part of the rendered source and
of the disk-cache key.  Two sets of module paths are byte-pinned:

* the modules `CuteBackend.library_imports` writes into generated import lines
  (`atomic_helpers`, `aux_copy_layout`, `clc_helpers`, `cluster_helpers`,
  `copysign`, `element_policy_loads`, `epilogue_helpers`, `exact_division`,
  `_flash_row_mma_runtime`,
  `flex_score_mod_runtime`, `fold_helpers`, `grid_barrier`,
  `inline_asm_helpers`, `integer_helpers`, `join_cast`, `l2_policy`,
  `masked_packet_load`, `_mma_dag_runtime`, `packet_store_helpers`,
  `_pdl_spread_runtime`, `quantized_helpers`, `readonly_packet_load_helpers`,
  `reduce_helpers`, `remainder_fill_runtime`, `resident_reduction_runtime`,
  `single_token_rank1_recurrence`, `split_k_exchange`, `tensor_copy_helpers`,
  `tensormap_alloc_class`, `thread_budget`, `vec_utils`, `warp_atomic_runtime`,
  `_warp_mma_runtime`, and `helion.runtime.cute.{single,paired,full}_sum`),
  plus the `helion._compiler.cute.<module>` strings emitters compose in
  f-strings;
* the paths `helion/runtime/cute/source_dependencies.py` hashes into the
  disk-cache key (`_COMMON_DEPENDENCIES` and the per-kind tables).

Renaming a module of the first set changes every render that imports it; a
body edit to a module of the second set changes the cache key, not the render.
Classify a module by grep before moving it.

One helper edits a structure the driver owns:
`tensormap_alloc_class.select_small_alloc_class` (and its
`_every_lane` form, which the flat grouped kernel uses) clears the
allocation-class selector of a TensorMap template a kernel rewrites per
group, so that no group keeps the large class of the operand it is cut
from.  NVIDIA's documentation does not say which bit that is; the code of
CuTe DSL 4.8.1 does: its experimental `tensormap_replace` clears it on every
device, without asking the driver, when it replaces an extent or a stride.
That is not the rewrite the kernels use (`update_tensormap`, which leaves
the class as encoded), so the helper clears the same bit, as that primitive
does on every device: `ALLOC_CLASS_QWORD` and `ALLOC_CLASS_LARGE`, which
`test/test_cute_grouped_descriptors.py` holds to the DSL's constants.  That
a cleared bit is the small class on the hardware was measured on a B200;
the page-touch tests (`helion._testing.cuda_page_touch`) check it on the
device they run on.

The helper covers the descriptors a kernel rewrites.  One static descriptor
is outside it and outside what PTX ISA 9.5 defines: the store descriptor of
the bounded D shift (`tcgen05_grouped_d_shift`), which the CuTe DSL encodes
in its compiled host code, where no helper of this module reaches it.
Section 6 describes it, what was measured, and the gate that keeps the
autotuner from selecting it; a config that names the option still runs on
that descriptor.  The descriptor itself is unchanged, and how to replace it
is open.

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
