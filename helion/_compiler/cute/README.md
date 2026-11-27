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
none on a single CTA, whatever the lengths are: a launch of short units, or
of a window's few pages out of a long page table, applies less than the
config names, down to none, and no range is ever empty. A config may name
only the skews that some launch of the bind can apply in full
(`partition_skew_choices`: up to `units * (pages + 1) // min(units, SMs) //
32`). A skew changes which units are split and where, so results move
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
within the measurement's spread of no skew.

Units that run in one CTA keep `item`'s bits. A unit in parts differs by the
rounding of a reassociated sum: its parts round their probabilities to BF16
against their own maxima and the merge adds a few FP32 operations. On the
partition tests' random data that is at most 0.6% of the unit's largest
output and 2e-6 in the LSE. That is a reassociation only `fast_math`
licenses, so the choice exists only under it, and only for the normalized
epilogue (the raw-statistics one has no merge). Which units are split
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
