# Commits by yf225 (yfeng.us@gmail.com) on main branch

Total commits: 237

| Commit Hash | Commit Message |
|-------------|----------------|
| 28cc903 | Allow using `hl.specialize` to specialize on tensor strides (#1215) |
| a173211 | Avoid broadcasting for non-consecutive tensor indexers (#1254) |
| 249872f | Fix RNG codegen for constant (specialized) dimensions (#1253) |
| 29639a9 | Clean up distributed examples path refs (#1241) |
| b1a76bd | Move distributed examples to `examples/distributed/` (#1240) |
| ab8c4f9 | Make `hl.triton_kernel` support output_like=None without being DCE'd (#1237) |
| c47581f | Make `hl.triton_kernel` support global var and recursive kernel call (#1234) |
| 58cbc67 | Reduce log volume by moving output code logging behind HELION_PRINT_OUTPUT_CODE=1 (#1233) |
| 7caeaa2 | Move jagged_dense_bmm expected code to the right place (#1232) |
| 72d09a7 | [CI] Fix NVSHMEM env vars and re-enable distributed CI job (#1201) |
| a2cb804 | [Interpret Mode] Fix `hl.load` with multiple 1D tensor indices (#1227) |
| 96f169f | [Interpret Mode] Fix `hl.store` automatic dtype conversion (#1226) |
| 6ff9480 | [Interpret Mode] Raise error if `hl.store` is used with duplicate indices (#1221) |
| 850f096 | [Interpret Mode] Re-enable block-size dependent tests (#1212) |
| a5497a2 | Fix memory leak when Triton compile error occurs (#1217) |
| 737d84a | Fix `static_shapes` setting in test_dot.py (#1220) |
| 8ada5e0 | Fix unit test breakage due to upstream change (#1219) |
| 4cd2e65 | Fix linter errors (#1218) |
| 7aada66 | [Autotuner] Add `autotune_benchmark_fn` setting (#1199) |
| a2f5ed1 | [Interpret Mode] Support custom block size (#1194) |
| 9b6dfde | [CI] Skip all failing distributed tests (#1206) |
| 1f2593c | Print repro code on autotune success (#1203) |
| c24dd2c | [CI] Skip failing distributed test (#1196) |
| 1f880ea | Add 2d and 3d indirect indexing support (#593) |
| 3efc015 | [Ref Mode] Fix error message (#1175) |
| 1d52e7b | Fix lint errors in local dev env (#1174) |
| 4cd42b5 | Fix min hoisting bug (#1157) |
| 73cba18 | Fix specialize + reshape use case (#1146) |
| ce25c5f | Add user-customizable autotune_baseline_atol / rtol settings (#1136) |
| b074c9e | [CI] Skip TestBreakpoint in ref-eager CI job (#1141) |
| 793be09 | [Benchmark CI] Print generated Triton code for the best config (#1142) |
| 605e152 | Fix CI to surface errors correctly, fix all existing errors (#1138) |
| a30ce01 | Support tuple indexing by hl.static_range iterator (#1134) |
| 6a98dc9 | [CI] Fix fbcode test_breakpoint error (#1132) |
| 913f7c7 | [CI] Fail the distributed CI job if any unit test fails (#1125) |
| 51580b4 | Remove `@helion.jit` usage and advise use of `@helion.kernel` (#1116) |
| b5430fe | Add warning to discourage use of `acc += lhs @ rhs` pattern (#1111) |
| d44009f | Support tensor.T for transpose (#1110) |
| 8a23df1 | Add distributed CI job (4xH100) and example unit tests (#1106) |
| 19b6cf9 | Fix no libdw.so issue on AMD CI (#1107) |
| e9e4957 | Add extra line before repro log; update repro log tests (#1102) |
| f05043e | Fix builtin min / max handling in device loop (#1085) |
| ec12380 | [Autotuner] Better error message for default config error (#1092) |
| 157be88 | Fix GRPO loss example unit tests (#1079) |
| b1d6da6 | Use HELION_PRINT_REPRO=1 to print repro when device IR lowering or Triton codegen error (#1078) |
| c864ed4 | Raise user error if device-loop is empty after DCE (#1074) |
| 788c8cf | [Unblock internal] Fix log capture issue on internal tests (#1076) |
| a0ba4d2 | Make HELION_PRINT_REPRO=1 take effect in more error cases (#1066) |
| 0c49062 | Support torch.rand / torch.rand_like with dynamic tile sizes (#1057) |
| bfa223a | Add `HELION_PRINT_REPRO=1` to print Helion kernel repro script to console (#1049) |
| 0bafd91 | Add `settings.autotune_baseline_fn` to allow passing in custom baseline function to autotuner (#1054) |
| 8e535e3 | [Fix upcoming CI error] Set current node in inductor lowering (#1052) |
| 182d16f | [CI] Fix debug_str() to be compatible with latest PyTorch nightly (#1050) |
| 5be73dd | Fix FlattenedTileStrategy to handle unit-sized block dimensions (#1048) |
| 79d57b7 | Fix layernorm bwd unit test (#1047) |
| 7db09b7 | Fix matmul output dtype to match PyTorch eager behavior (#1044) |
| 8926c5c | Fix dtype mismatch error in se_block example (#1040) |
| f213d44 | Disable autotuner progress bar in fbcode unit test (#1035) |
| 3227746 | Default config: reduce block_size further to avoid shared mem OOM (#1034) |
| ab532b0 | Default config: reduce block_size and num_stages to avoid shared mem OOM (#1033) |
| 09b9b45 | [Benchmark CI] Set welford num_inputs to 6 to avoid timeout (#1032) |
| c314ed2 | [Benchmark] Update welford torch.compile function name (#1029) |
| d39c50a | Support `breakpoint()` in device code when interpret mode is on (#1020) |
| 38675f9 | [CI] Fix AMD journal check errors (#1016) |
| 4e470c0 | Raise informative error when `hl.dot` with 3D inputs have batch dim mismatch (#1012) |
| 9a30bd1 | Always raise `FailedToUnpackTile` when `for tile_m, tile_d in hl.tile(m, d)` is used (#1009) |
| 2ff186c | Better error message for calling Helion kernel from another kernel (#1008) |
| d7e69f9 | Layer Norm bwd kernel to support large B*M case used by internal (#973) |
| fc69870 | Add skipIfA10G decorator (#982) |
| 37c3e3f | [Autotuner] Fix fork-based autotuner to avoid re-initializing CUDA context in subprocess (#981) |
| 1ab4208 | [Benchmark] Fix tritonbench auto-installation (#980) |
| 944e7a8 | [Benchmark CI] Set atol and rtol to 1e-2 (#976) |
| cfe3d9b | [Benchmark CI] Allow specifying custom args to benchmark runner via UI (#974) |
| f8bad83 | [Benchmark CI] Allow specifying custom env vars via UI (#972) |
| d4d122b | Register tile symbol origin, to support `tile + offset` use case in blackwell attention (#939) |
| b77301f | [Benchmark CI] Use `--non-square` flag for gemm (#938) |
| b40750f | Support `hl.arange()` with non-power-of-2 input (#862) |
| fe33e3b | Suggest use of `@helion.kernel(index_dtype=torch.int64)` if index offset is out of bound for int32 (#850) |
| 7efa2b0 | Set HELION_DEV_LOW_VRAM=1 on a10g CI machines (#923) |
| 811be91 | Remove `triton_helpers.*` usage in lifted device function arguments (#849) |
| 0a4a7fa | Avoid setting default `--input-sample-mode` to `equally-spaced-k` (#922) |
| 6ad53ad | Print error message for torch.chunk / torch.unbind to redirect users to hl.split (#921) |
| f370fa5 | [Benchmark CI] Use triton_tutorial_matmul for triton matmul baseline (#911) |
| 4162db7 | [Benchmark CI] Reduce num_inputs for grouped_gemm and gemm benchmarks (#903) |
| d01d37a | [Benchmark CI] grouped_gemm: include input preproc in timing measurement; update gemm backend name mapping (#898) |
| 649eaa8 | [Benchmark CI] Use regular matmul instead of split-k (#884) |
| b49c626 | [Autotuner] Run CUDA synchronize before / after candidate func call, to surface CUDA errors sooner (#872) |
| c16969b | Print Triton code when error for easier debugging (#874) |
| d963181 | [Benchmark CI] Use equally-spaced-k mode to sample input shapes (#861) |
| 86c54e5 | Remove legacy `register_inductor_lowering` code (#864) |
| 05fb47d | [Benchmark CI] Use fewer num_inputs for flash_attention to avoid timeout (#857) |
| 0361edf | Decrease `num_stages` default from 3 to 2, to avoid shared memory OOM (#841) |
| b2cae6b | Remove hardcoded `block_size=1` usage in attention kernel example (#843) |
| fd039fd | Error message for boolean masking or torch.nonzero (#687) |
| 4675e6a | Remove `hl.register_reduction_dim` API (#834) |
| 6581aac | Update input shapes for example kernels (#845) |
| 55d6aa0 | Pad to next power of 2 for hl.specialize'ed shape value used in device tensor creation (#804) |
| ebbd2c4 | Apply simplification to range indexing in order to reuse block size symbols (#809) |
| 2d54358 | Set range_num_stages <= 1 if using tensor_descriptor, to avoid CUDA misaligned address error (#792) |
| 01261f5 | Better error message for augmented assignment (e.g. +=) on host tensor without subscript (#807) |
| b03bb06 | Print Helion kernel source line in symbolic shape debugging (#771) |
| 2a39f4f | Fix jagged_layer_norm linter error (#770) |
| c7fa936 | Support torch.matmul with 3D inputs (#715) |
| 06782cc | [Bug fix] Preserve masks on reduction inputs that depend on reduction outputs; fix layer_norm accuracy check failure (#722) |
| 2ec0416 | [Benchmark CI] Make benchmark runner respect custom CLI args (#723) |
| 38b9967 | Skip default config printing if in ref eager mode (#721) |
| c934dad | [Benchmark CI] Print config that causes tritonbench accuracy check failure (#716) |
| 308f00a | [Benchmark CI] Skip last input shape for rms_norm-bwd (#712) |
| e885cf0 | [Benchmark CI] Set tolerance values that match autotuner setting (#710) |
| 7980bdb | [Benchmark CI] Run fewer inputs for layer_norm-bwd to avoid job timeout (#709) |
| 7d5aacf | [Benchmark CI] Run `rms_norm-bwd` and `layer_norm-bwd` kernels (#708) |
| 8b97243 | Revert "[Benchmark CI] Add `--list-kernels-for-benchmark-ci` to benchmark runner" (#707) |
| b1b4046 | Revert "[Benchmark CI] Run rms_norm-bwd and layer_norm-bwd kernels" (#706) |
| b113ca2 | [Benchmark CI] Run rms_norm-bwd and layer_norm-bwd kernels (#705) |
| ebeb4e9 | [Benchmark CI] Add `--list-kernels-for-benchmark-ci` to benchmark runner (#703) |
| 4d5f729 | [Benchmark CI] Allow customized mapping into tritonbench impls (#700) |
| 19a7442 | [Benchmark CI] Change DEFAULT_NUM_INPUTS to MAX_NUM_INPUTS (#702) |
| e9511ec | [Benchmark CI] fix layer_norm mapping bug (#701) |
| 65c2cd6 | [Benchmark CI] Customizable num_inputs for each kernel (#699) |
| a10bbc2 | [Autotune] Skip Triton shared memory OOM (#684) |
| 52eb173 | Print bad default config if compute baseline fails (#688) |
| 9849925 | [Benchmark CI] Use equally-spaced K input shapes (#689) |
| fc9a005 | [Benchmark CI] use --op instead of --kernel for better tritonbench compat (#694) |
| b1fb170 | [Benchmark CI] Use do_bench cudagraph to avoid profiler failure; select specific kernel impls to run (#682) |
| 213c6f9 | [CI] Fix missing setuptools (#680) |
| e3d4609 | Fix missing block size constexpr assignment in host code (#678) |
| ebdd636 | [Benchmark CI] Run one kernel per gpu to maximize successful kernel reporting (#681) |
| 00e53b0 | [Autotune] Allow skipping Triton compilation error (#679) |
| f791da0 | rms_norm: get weight from function args (#664) |
| b15b026 | Fix misaligned address error for matmul (#662) |
| 07b1182 | Log autotune random seed for easier repro (#661) |
| 5ccf6f4 | [Autotune] Filter bad config with accuracy check (#655) |
| 1701f8d | Raise better error when `hl.atomic_*` is used on device tensor (#658) |
| 94b0650 | Fix `register_block_size` codegen (#659) |
| eff83b1 | Print `static_shapes` settings value along with config for accurate repro (#649) |
| 53141ca | Skip test_autotune_random_seed_from_settings on rocm (#651) |
| 025b6ce | Fix lint related to welford and also local_cache (#646) |
| 0bb3719 | Add `HELION_AUTOTUNE_RANDOM_SEED` env var and `autotune_random_seed` setting (#644) |
| 8cf9e61 | Avoid skipping CUDA errors that crashes the CUDA context (#645) |
| 013bdec | [Easy][Benchmark CI] Exit job on any exception, for easier error catching (#643) |
| 61854cf | int4_gemm: remove use_default_config=True (#639) |
| 6aa7ddd | [Example] grouped_gemm kernel example and tritonbench integration (#620) |
| 8442721 | Always clear inductor cache before benchmark (#608) |
| 029df79 | [Benchmark CI] Print input shapes and surface problematic Helion config (#626) |
| 6299e33 | Support using block size var outside of hl.tile loop (#619) |
| ad46fb3 | [Ref Mode] Fix hl.store for complex mask pattern (#621) |
| 62bbcaf | Adjust tolerance for test_rms_norm_bwd_dx (#628) |
| b6af661 | Set requires_grad=True for rms_norm backward inputs (#629) |
| 5902317 | [Example] int4_gemm kernel example and tritonbench integration (#613) |
| b06d4fb | Add small dim size (<16) support to hl.dot and torch.addmm; Always prefer using `tl.dot(acc=...)` for addmm / baddbmm (#564) |
| d2e5f4a | Add layer_norm backward kernels (#588) |
| 65995f5 | Allow passing tritonbench operator instance into kernel benchmark wrapper; Always return lambda for timing measurement (#596) |
| ead8a63 | hl.atomic_add: support 1D tensor as index (#587) |
| bbc2be4 | [Benchmark] Remove hardcoded num_inputs for rms_norm kernel (#581) |
| 67c2392 | [Benchmark] Fix layer_norm accuracy issue (#580) |
| 87ccb4d | Add `skipIfLowVRAM` or `use_default_config=False` to specific unit tests to enable local testing (#574) |
| a61bd17 | [Fix CI] Convert tiles to sizes for all torch.* functions (#563) |
| 84d7430 | torch.rand_like and torch.randn_like support (#530) |
| 0f3e2d5 | Add torch.stack support (#524) |
| 7c1b5e7 | Fix local runs for test_triton_repro tests (#539) |
| 849959b | Pass int arg instead of dummy tensor into example kernels (#538) |
| 08de077 | Prevent naming conflicts in expr_from_string placeholder replacement (#519) |
| a716de4 | Lower symbolic slices to hl.arange (#518) |
| 50ab952 | Support symbolic range with multiples of block-size as length (#509) |
| 2cae9ed | Sort config keys alphabetically in `__str__` (#505) |
| 4158abf | Fix reshape + sum case (#504) |
| db41224 | Support reshape with block_size expressions (#495) |
| 3713241 | Add helion prefix to Triton kernel name (#486) |
| 543ec8e | [Benchmark] Add try-catch for tritonbench import path (#487) |
| b9342f6 | [Benchmark] Allow passing kwargs; Set static_shape = True for better benchmark perf (#465) |
| 6c56694 | Fix tritonbench integration issue (#463) |
| 0a19c36 | Re-enable associative_scan tests in ref eager mode (#443) |
| ea8c7fd | [Benchmark] Avoid using _run in TritonBench integration (#444) |
| 4718678 | Fix strided slice support for static slices (e.g., buf[::2]) (#426) |
| a252cca | Fix scalar tensor broadcasting in type propagation (#425) |
| bc62cf2 | Fix scalar value assignment to tensor slices (#424) |
| 11c9b1d | Skip associative_scan tests in ref eager mode (#433) |
| aabe823 | Add extensive setter/getter unit tests for indexed tensor; fix bugs discovered by new tests (#422) |
| 642836c | [Benchmark] Fix arg parsing issue in tritonbench integration (#417) |
| b403f2e | [Ref Mode] Expand ref eager mode support to more hl.* APIs (e.g. load / store / scan / reduce) (#410) |
| 4c0f7f3 | [Ref Mode] PyTorch reference mode (eager only) (#339) |
| 24d453d | [Benchmark] Move per-operator settings from example file to benchmarks/run.py (#403) |
| 8727491 | [Benchmark] Enable CSV output; clean up benchmark hot path (#398) |
| 45e9600 | Relax tolerance for test_input_float16_acc_float16_dynamic_shape (#399) |
| 19f23b2 | [Benchmark] Support kernel variants; setup matmul tritonbench integration (#380) |
| 8fd5a4b | [Examples] Add matmul variants with bias support and tests (#379) |
| 3c3c64a | Fix test_inline_asm_packed expected output due to upstream PyTorch change (#385) |
| 2a9aa70 | Add hl.dot() API; Use hl.dot instead of torch.matmul for FP8 GEMM ops in Helion kernel (#356) |
| 6c5c4ca | [Benchmark] Allow running a specific shard of input via --input-shard M/N cli arg (#377) |
| 462fc00 | Set MAX_JOBS=4 for tritonbench build to avoid OOM (#376) |
| 6db2105 | Skip accuracy check for test_moe_matmul_ogs (#333) |
| 7d01817 | [Benchmark] Add fp8_attention to tritonbench integration (#319) |
| 3175785 | Add fp8_attention example and unit test (#318) |
| d884774 | [Benchmark] Add cross_entropy to tritonbench integration (#321) |
| deb7c8a | Add cross_entropy example and unit test (#320) |
| 7c8a560 | Add HELION_DEV_LOW_VRAM env var for low GPU memory machines (#325) |
| 11b49c6 | Fix variable scoping in nested loops for multi-pass kernels (#324) |
| 8f5068c | [Benchmark] Add softmax tritonbench integration (#286) |
| 47878bf | [Benchmark] Add attention tritonbench integration (#284) |
| c90a4ef | [Benchmark] Allow using 'python benchmarks/run.py' to run all kernels (#280) |
| 6d54c8f | [Benchmark] Fix tritonbench integration due to upstream changes (#278) |
| 2fa2b31 | [Benchmark] Add fp8_gemm to TritonBench integration (#268) |
| d503788 | Add fp8_gemm example and test (#267) |
| cbabb06 | [Benchmark] Add jagged_mean tritonbench integration (#264) |
| e429c84 | Add jagged_mean example (#263) |
| 110df59 | Fix pyright errors in type_propagation.py (#266) |
| 3e61e24 | Rename benchmark folder (#258) |
| 9f7158a | [Benchmark] Add sum to TritonBench integration (#257) |
| 573fc23 | Add sum example and test (#256) |
| 7c64ae1 | [Benchmark] Add rms_norm benchmark (#253) |
| da9cf12 | Add rms_norm example and test (#252) |
| 3ff927d | [Benchmark] Add vector_exp benchmark (#249) |
| 8567309 | [Benchmark] Add embedding benchmark (#248) |
| 7aa722b | [Benchmark] Add initial TritonBench integration and vector_add benchmark example (#247) |
| 46b617d | Codegen `if tl.sum(one_elem_tensor):` instead of `if one_elem_tensor` (#158) |
| b64bf00 | Add support for print(prefix_str, *tensors) (#140) |
| 0e047af | Try enable test_moe_matmul_ogs on CI (#147) |
| f7ed720 | Add hl.register_reduction_dim(); add support for matmul+layernorm example (#80) |
| 80510e0 | Re-enable unit test for moe_matmul_ogs example; skip in fbcode (#123) |
| 13e6e7e | Temporarily disable unit test for moe_matmul_ogs example (#120) |
| 49a6c06 | Add main() to moe_matmul_ogs (#118) |
| 1aa5f03 | MoE matmul example  (#110) |
| 43f1a4c | Fix test_matmul_tensor_descriptor unit test (#65) |
| 9858b0c | Add `hl.grid(...)` support (#59) |
| 6497bea | Support python test/test_X.py command for all unit test files (#60) |
| 4165a69 | Minor fix on the matmul example dim order |
| 4ab101e | dense bmm support |
| a1e9c99 | Make BlockSizeOrigin host_str return 1 for block_size=1 case |
| bd3d984 | Add HELION_USE_DEFAULT_CONFIG env var to force use default config (#37) |
| 46b96f9 | Fix TensorDescriptor handling in _find_device (#35) |
| 57b631d | Add ../pytorch-nightly to Pyre optional_search_path (#36) |
| 160c168 | Support Python 3.10; Run lint in CI (#7) |
| de2a3e5 | Adjust rtol/atol for test_sum_keepdims (#14) |
| 6f33398 | [CI] Use A10G (g5.4xlarge) machine type (#4) |
| 1ca73b3 | Allow direct running of add.py example (#6) |
| 1e70ccd | Add CI workflow (#2) |
| fcc1908 | Minor fix to test file name (#1) |

---

## Commit Analysis by Category

### Summary

| Category | Count | Percentage |
|----------|-------|------------|
| Benchmark Infrastructure | 55 | 23.2% |
| CI/Build/Testing | 36 | 15.2% |
| Shape/Stride Specialization | 4 | 1.7% |
| Indexing/View/Symbolic Reasoning | 16 | 6.8% |
| New APIs | 13 | 5.5% |
| Other Bug Fixes & Improvements | 39 | 16.5% |
| Examples | 22 | 9.3% |
| Autotuner | 15 | 6.3% |
| Interpret/Ref Mode | 13 | 5.5% |
| Debugging | 20 | 8.4% |
| Code Cleanup/Refactoring | 5 | 2.1% |

---

### 1. Benchmark Infrastructure (55 commits)

Commits related to TritonBench integration, benchmark CI, and performance measurement.

| Commit | Message |
|--------|---------|
| 793be09 | [Benchmark CI] Print generated Triton code for the best config (#1142) |
| 09b9b45 | [Benchmark CI] Set welford num_inputs to 6 to avoid timeout (#1032) |
| c314ed2 | [Benchmark] Update welford torch.compile function name (#1029) |
| b77301f | [Benchmark CI] Use `--non-square` flag for gemm (#938) |
| f370fa5 | [Benchmark CI] Use triton_tutorial_matmul for triton matmul baseline (#911) |
| 4162db7 | [Benchmark CI] Reduce num_inputs for grouped_gemm and gemm benchmarks (#903) |
| d01d37a | [Benchmark CI] grouped_gemm: include input preproc in timing measurement (#898) |
| 649eaa8 | [Benchmark CI] Use regular matmul instead of split-k (#884) |
| d963181 | [Benchmark CI] Use equally-spaced-k mode to sample input shapes (#861) |
| 05fb47d | [Benchmark CI] Use fewer num_inputs for flash_attention to avoid timeout (#857) |
| 2ec0416 | [Benchmark CI] Make benchmark runner respect custom CLI args (#723) |
| c934dad | [Benchmark CI] Print config that causes tritonbench accuracy check failure (#716) |
| 308f00a | [Benchmark CI] Skip last input shape for rms_norm-bwd (#712) |
| e885cf0 | [Benchmark CI] Set tolerance values that match autotuner setting (#710) |
| 7980bdb | [Benchmark CI] Run fewer inputs for layer_norm-bwd to avoid job timeout (#709) |
| 7d5aacf | [Benchmark CI] Run `rms_norm-bwd` and `layer_norm-bwd` kernels (#708) |
| 8b97243 | Revert "[Benchmark CI] Add `--list-kernels-for-benchmark-ci`" (#707) |
| b1b4046 | Revert "[Benchmark CI] Run rms_norm-bwd and layer_norm-bwd kernels" (#706) |
| b113ca2 | [Benchmark CI] Run rms_norm-bwd and layer_norm-bwd kernels (#705) |
| ebeb4e9 | [Benchmark CI] Add `--list-kernels-for-benchmark-ci` (#703) |
| 4d5f729 | [Benchmark CI] Allow customized mapping into tritonbench impls (#700) |
| 19a7442 | [Benchmark CI] Change DEFAULT_NUM_INPUTS to MAX_NUM_INPUTS (#702) |
| e9511ec | [Benchmark CI] fix layer_norm mapping bug (#701) |
| 65c2cd6 | [Benchmark CI] Customizable num_inputs for each kernel (#699) |
| 9849925 | [Benchmark CI] Use equally-spaced K input shapes (#689) |
| fc9a005 | [Benchmark CI] use --op instead of --kernel (#694) |
| b1fb170 | [Benchmark CI] Use do_bench cudagraph to avoid profiler failure (#682) |
| ebdd636 | [Benchmark CI] Run one kernel per gpu (#681) |
| 013bdec | [Benchmark CI] Exit job on any exception (#643) |
| 029df79 | [Benchmark CI] Print input shapes and surface problematic config (#626) |
| 8442721 | Always clear inductor cache before benchmark (#608) |
| 65995f5 | Allow passing tritonbench operator instance into kernel benchmark wrapper (#596) |
| bbc2be4 | [Benchmark] Remove hardcoded num_inputs for rms_norm kernel (#581) |
| 67c2392 | [Benchmark] Fix layer_norm accuracy issue (#580) |
| 543ec8e | [Benchmark] Add try-catch for tritonbench import path (#487) |
| b9342f6 | [Benchmark] Allow passing kwargs; Set static_shape = True (#465) |
| 6c56694 | Fix tritonbench integration issue (#463) |
| ea8c7fd | [Benchmark] Avoid using _run in TritonBench integration (#444) |
| 642836c | [Benchmark] Fix arg parsing issue in tritonbench integration (#417) |
| 24d453d | [Benchmark] Move per-operator settings from example file (#403) |
| 8727491 | [Benchmark] Enable CSV output; clean up benchmark hot path (#398) |
| 19f23b2 | [Benchmark] Support kernel variants; setup matmul tritonbench integration (#380) |
| 6c5c4ca | [Benchmark] Allow running a specific shard of input (#377) |
| 462fc00 | Set MAX_JOBS=4 for tritonbench build to avoid OOM (#376) |
| 7d01817 | [Benchmark] Add fp8_attention to tritonbench integration (#319) |
| d884774 | [Benchmark] Add cross_entropy to tritonbench integration (#321) |
| 8f5068c | [Benchmark] Add softmax tritonbench integration (#286) |
| 47878bf | [Benchmark] Add attention tritonbench integration (#284) |
| c90a4ef | [Benchmark] Allow using 'python benchmarks/run.py' to run all kernels (#280) |
| 6d54c8f | [Benchmark] Fix tritonbench integration due to upstream changes (#278) |
| 2fa2b31 | [Benchmark] Add fp8_gemm to TritonBench integration (#268) |
| cbabb06 | [Benchmark] Add jagged_mean tritonbench integration (#264) |
| 3e61e24 | Rename benchmark folder (#258) |
| 9f7158a | [Benchmark] Add sum to TritonBench integration (#257) |
| 7c64ae1 | [Benchmark] Add rms_norm benchmark (#253) |
| 3ff927d | [Benchmark] Add vector_exp benchmark (#249) |
| 8567309 | [Benchmark] Add embedding benchmark (#248) |
| 7aa722b | [Benchmark] Add initial TritonBench integration and vector_add (#247) |
| 1ab4208 | [Benchmark] Fix tritonbench auto-installation (#980) |

---

### 2. CI/Build/Testing (36 commits)

Commits related to CI infrastructure, build system, and test improvements.

| Commit | Message |
|--------|---------|
| 72d09a7 | [CI] Fix NVSHMEM env vars and re-enable distributed CI job (#1201) |
| 9b6dfde | [CI] Skip all failing distributed tests (#1206) |
| c24dd2c | [CI] Skip failing distributed test (#1196) |
| b074c9e | [CI] Skip TestBreakpoint in ref-eager CI job (#1141) |
| 605e152 | Fix CI to surface errors correctly, fix all existing errors (#1138) |
| 6a98dc9 | [CI] Fix fbcode test_breakpoint error (#1132) |
| 913f7c7 | [CI] Fail the distributed CI job if any unit test fails (#1125) |
| 8a23df1 | Add distributed CI job (4xH100) and example unit tests (#1106) |
| 19b6cf9 | Fix no libdw.so issue on AMD CI (#1107) |
| 788c8cf | [Unblock internal] Fix log capture issue on internal tests (#1076) |
| 8e535e3 | [Fix upcoming CI error] Set current node in inductor lowering (#1052) |
| 182d16f | [CI] Fix debug_str() to be compatible with latest PyTorch nightly (#1050) |
| 38675f9 | [CI] Fix AMD journal check errors (#1016) |
| fc69870 | Add skipIfA10G decorator (#982) |
| 944e7a8 | [Benchmark CI] Set atol and rtol to 1e-2 (#976) |
| cfe3d9b | [Benchmark CI] Allow specifying custom args to benchmark runner (#974) |
| f8bad83 | [Benchmark CI] Allow specifying custom env vars via UI (#972) |
| 7efa2b0 | Set HELION_DEV_LOW_VRAM=1 on a10g CI machines (#923) |
| 0a4a7fa | Avoid setting default `--input-sample-mode` to `equally-spaced-k` (#922) |
| 213c6f9 | [CI] Fix missing setuptools (#680) |
| 53141ca | Skip test_autotune_random_seed_from_settings on rocm (#651) |
| 62bbcaf | Adjust tolerance for test_rms_norm_bwd_dx (#628) |
| b6af661 | Set requires_grad=True for rms_norm backward inputs (#629) |
| 87ccb4d | Add `skipIfLowVRAM` or `use_default_config=False` to specific tests (#574) |
| a61bd17 | [Fix CI] Convert tiles to sizes for all torch.* functions (#563) |
| 7c1b5e7 | Fix local runs for test_triton_repro tests (#539) |
| aabe823 | Add extensive setter/getter unit tests for indexed tensor (#422) |
| 45e9600 | Relax tolerance for test_input_float16_acc_float16_dynamic_shape (#399) |
| 3c3c64a | Fix test_inline_asm_packed expected output (#385) |
| 6db2105 | Skip accuracy check for test_moe_matmul_ogs (#333) |
| 0e047af | Try enable test_moe_matmul_ogs on CI (#147) |
| 80510e0 | Re-enable unit test for moe_matmul_ogs example (#123) |
| 13e6e7e | Temporarily disable unit test for moe_matmul_ogs example (#120) |
| 6497bea | Support python test/test_X.py command for all unit test files (#60) |
| 160c168 | Support Python 3.10; Run lint in CI (#7) |
| 6f33398 | [CI] Use A10G (g5.4xlarge) machine type (#4) |
| 1e70ccd | Add CI workflow (#2) |
| 8ada5e0 | Fix unit test breakage due to upstream change (#1219) |
| 4cd2e65 | Fix linter errors (#1218) |
| 1d52e7b | Fix lint errors in local dev env (#1174) |
| 157be88 | Fix GRPO loss example unit tests (#1079) |
| 79d57b7 | Fix layernorm bwd unit test (#1047) |
| 8926c5c | Fix dtype mismatch error in se_block example (#1040) |
| 025b6ce | Fix lint related to welford and also local_cache (#646) |
| 110df59 | Fix pyright errors in type_propagation.py (#266) |
| 43f1a4c | Fix test_matmul_tensor_descriptor unit test (#65) |
| 2a39f4f | Fix jagged_layer_norm linter error (#770) |
| de2a3e5 | Adjust rtol/atol for test_sum_keepdims (#14) |
| fcc1908 | Minor fix to test file name (#1) |
| 4165a69 | Minor fix on the matmul example dim order |
| f791da0 | rms_norm: get weight from function args (#664) |
| 57b631d | Add ../pytorch-nightly to Pyre optional_search_path (#36) |


---

### 3. Shape/Stride Specialization (4 commits)

Commits related to shape and stride specialization features.

| Commit | Message |
|--------|---------|
| 28cc903 | Allow using `hl.specialize` to specialize on tensor strides (#1215) |
| 249872f | Fix RNG codegen for constant (specialized) dimensions (#1253) |
| 73cba18 | Fix specialize + reshape use case (#1146) |
| 55d6aa0 | Pad to next power of 2 for hl.specialize'ed shape value (#804) |

---

### 4. Indexing/View/Symbolic Reasoning (16 commits)

Commits related to indexing, views, slicing, broadcasting, and symbolic reasoning enhancements.

| Commit | Message |
|--------|---------|
| a173211 | Avoid broadcasting for non-consecutive tensor indexers (#1254) |
| 1f880ea | Add 2d and 3d indirect indexing support (#593) |
| a30ce01 | Support tuple indexing by hl.static_range iterator (#1134) |
| d44009f | Support tensor.T for transpose (#1110) |
| d4d122b | Register tile symbol origin, to support `tile + offset` use case (#939) |
| b40750f | Support `hl.arange()` with non-power-of-2 input (#862) |
| 6299e33 | Support using block size var outside of hl.tile loop (#619) |
| ead8a63 | hl.atomic_add: support 1D tensor as index (#587) |
| a716de4 | Lower symbolic slices to hl.arange (#518) |
| 50ab952 | Support symbolic range with multiples of block-size as length (#509) |
| db41224 | Support reshape with block_size expressions (#495) |
| 4158abf | Fix reshape + sum case (#504) |
| 4718678 | Fix strided slice support for static slices (#426) |
| a252cca | Fix scalar tensor broadcasting in type propagation (#425) |
| bc62cf2 | Fix scalar value assignment to tensor slices (#424) |
| ebbd2c4 | Apply simplification to range indexing to reuse block size symbols (#809) |

---

### 5. New APIs (13 commits)

Commits adding new APIs or extending existing API capabilities.

| Commit | Message |
|--------|---------|
| ab8c4f9 | Make `hl.triton_kernel` support output_like=None without being DCE'd (#1237) |
| c47581f | Make `hl.triton_kernel` support global var and recursive kernel call (#1234) |
| 0c49062 | Support torch.rand / torch.rand_like with dynamic tile sizes (#1057) |
| b06d4fb | Add small dim size (<16) support to hl.dot and torch.addmm (#564) |
| 84d7430 | torch.rand_like and torch.randn_like support (#530) |
| 0f3e2d5 | Add torch.stack support (#524) |
| c7fa936 | Support torch.matmul with 3D inputs (#715) |
| 2a9aa70 | Add hl.dot() API; Use hl.dot instead of torch.matmul for FP8 GEMM (#356) |
| f7ed720 | Add hl.register_reduction_dim(); add support for matmul+layernorm (#80) |
| 9858b0c | Add `hl.grid(...)` support (#59) |
| 4ab101e | dense bmm support |
| b64bf00 | Add support for print(prefix_str, *tensors) (#140) |

---

### 6. Other Bug Fixes & Improvements (39 commits)

Other bug fixes, linter fixes, test fixes, and miscellaneous improvements.

| Commit | Message |
|--------|---------|
| a5497a2 | Fix memory leak when Triton compile error occurs (#1217) |
| 737d84a | Fix `static_shapes` setting in test_dot.py (#1220) |
| 4cd42b5 | Fix min hoisting bug (#1157) |
| 5be73dd | Fix FlattenedTileStrategy to handle unit-sized block dimensions (#1048) |
| 7db09b7 | Fix matmul output dtype to match PyTorch eager behavior (#1044) |
| f05043e | Fix builtin min / max handling in device loop (#1085) |
| 06782cc | [Bug fix] Preserve masks on reduction inputs (#722) |
| e3d4609 | Fix missing block size constexpr assignment in host code (#678) |
| b15b026 | Fix misaligned address error for matmul (#662) |
| 94b0650 | Fix `register_block_size` codegen (#659) |
| 8cf9e61 | Avoid skipping CUDA errors that crashes the CUDA context (#645) |
| 11b49c6 | Fix variable scoping in nested loops for multi-pass kernels (#324) |
| 46b96f9 | Fix TensorDescriptor handling in _find_device (#35) |
| 2d54358 | Set range_num_stages <= 1 if using tensor_descriptor (#792) |
| 0361edf | Decrease `num_stages` default from 3 to 2, to avoid shared memory OOM (#841) |
| 3227746 | Default config: reduce block_size further to avoid shared mem OOM (#1034) |
| ab532b0 | Default config: reduce block_size and num_stages to avoid shared mem OOM (#1033) |
| a1e9c99 | Make BlockSizeOrigin host_str return 1 for block_size=1 case |
| 08de077 | Prevent naming conflicts in expr_from_string placeholder replacement (#519) |
| 2cae9ed | Sort config keys alphabetically in `__str__` (#505) |
| 3713241 | Add helion prefix to Triton kernel name (#486) |
| 811be91 | Remove `triton_helpers.*` usage in lifted device function arguments (#849) |
| 46b617d | Codegen `if tl.sum(one_elem_tensor):` instead of `if one_elem_tensor` (#158) |

---

### 7. Examples (22 commits)

Commits adding or updating kernel examples.

| Commit | Message |
|--------|---------|
| 29639a9 | Clean up distributed examples path refs (#1241) |
| b1a76bd | Move distributed examples to `examples/distributed/` (#1240) |
| 7caeaa2 | Move jagged_dense_bmm expected code to the right place (#1232) |
| d7e69f9 | Layer Norm bwd kernel to support large B*M case (#973) |
| 6581aac | Update input shapes for example kernels (#845) |
| b2cae6b | Remove hardcoded `block_size=1` usage in attention kernel example (#843) |
| 61854cf | int4_gemm: remove use_default_config=True (#639) |
| 6aa7ddd | [Example] grouped_gemm kernel example and tritonbench integration (#620) |
| 5902317 | [Example] int4_gemm kernel example and tritonbench integration (#613) |
| d2e5f4a | Add layer_norm backward kernels (#588) |
| 849959b | Pass int arg instead of dummy tensor into example kernels (#538) |
| 8fd5a4b | [Examples] Add matmul variants with bias support and tests (#379) |
| 3175785 | Add fp8_attention example and unit test (#318) |
| deb7c8a | Add cross_entropy example and unit test (#320) |
| d503788 | Add fp8_gemm example and test (#267) |
| e429c84 | Add jagged_mean example (#263) |
| 573fc23 | Add sum example and test (#256) |
| da9cf12 | Add rms_norm example and test (#252) |
| 49a6c06 | Add main() to moe_matmul_ogs (#118) |
| 1aa5f03 | MoE matmul example (#110) |
| 1ca73b3 | Allow direct running of add.py example (#6) |

---

### 8. Autotuner (15 commits)

Commits related to autotuning infrastructure.

| Commit | Message |
|--------|---------|
| 7aada66 | [Autotuner] Add `autotune_benchmark_fn` setting (#1199) |
| 1f2593c | Print repro code on autotune success (#1203) |
| ce25c5f | Add user-customizable autotune_baseline_atol / rtol settings (#1136) |
| ec12380 | [Autotuner] Better error message for default config error (#1092) |
| 0bafd91 | Add `settings.autotune_baseline_fn` to allow custom baseline function (#1054) |
| f213d44 | Disable autotuner progress bar in fbcode unit test (#1035) |
| b49c626 | [Autotuner] Run CUDA synchronize before / after candidate func call (#872) |
| a10bbc2 | [Autotune] Skip Triton shared memory OOM (#684) |
| 00e53b0 | [Autotune] Allow skipping Triton compilation error (#679) |
| 07b1182 | Log autotune random seed for easier repro (#661) |
| 5ccf6f4 | [Autotune] Filter bad config with accuracy check (#655) |
| eff83b1 | Print `static_shapes` settings value along with config (#649) |
| 37c3e3f | [Autotuner] Fix fork-based autotuner to avoid re-initializing CUDA context (#981) |


---

### 9. Interpret/Ref Mode (13 commits)

Commits related to interpret mode and PyTorch reference mode.

| Commit | Message |
|--------|---------|
| a2cb804 | [Interpret Mode] Fix `hl.load` with multiple 1D tensor indices (#1227) |
| 96f169f | [Interpret Mode] Fix `hl.store` automatic dtype conversion (#1226) |
| 6ff9480 | [Interpret Mode] Raise error if `hl.store` is used with duplicate indices (#1221) |
| 850f096 | [Interpret Mode] Re-enable block-size dependent tests (#1212) |
| a2f5ed1 | [Interpret Mode] Support custom block size (#1194) |
| 3efc015 | [Ref Mode] Fix error message (#1175) |
| d39c50a | Support `breakpoint()` in device code when interpret mode is on (#1020) |
| 38b9967 | Skip default config printing if in ref eager mode (#721) |
| ad46fb3 | [Ref Mode] Fix hl.store for complex mask pattern (#621) |
| 0a19c36 | Re-enable associative_scan tests in ref eager mode (#443) |
| 11c9b1d | Skip associative_scan tests in ref eager mode (#433) |
| b403f2e | [Ref Mode] Expand ref eager mode support to more hl.* APIs (#410) |
| 4c0f7f3 | [Ref Mode] PyTorch reference mode (eager only) (#339) |

---

### 10. Debugging (20 commits)

Commits improving debugging capabilities, adding env vars, and enhancing error messages.

| Commit | Message |
|--------|---------|
| 58cbc67 | Reduce log volume by moving output code logging behind HELION_PRINT_OUTPUT_CODE=1 (#1233) |
| b1d6da6 | Use HELION_PRINT_REPRO=1 to print repro when device IR lowering error (#1078) |
| c864ed4 | Raise user error if device-loop is empty after DCE (#1074) |
| a0ba4d2 | Make HELION_PRINT_REPRO=1 take effect in more error cases (#1066) |
| bfa223a | Add `HELION_PRINT_REPRO=1` to print Helion kernel repro script (#1049) |
| 4e470c0 | Raise informative error when `hl.dot` with 3D inputs have batch dim mismatch (#1012) |
| 9a30bd1 | Always raise `FailedToUnpackTile` when `for tile_m, tile_d in hl.tile(m, d)` is used (#1009) |
| 2ff186c | Better error message for calling Helion kernel from another kernel (#1008) |
| fe33e3b | Suggest use of `@helion.kernel(index_dtype=torch.int64)` if index offset is out of bound (#850) |
| 6ad53ad | Print error message for torch.chunk / torch.unbind to redirect users to hl.split (#921) |
| c16969b | Print Triton code when error for easier debugging (#874) |
| fd039fd | Error message for boolean masking or torch.nonzero (#687) |
| 52eb173 | Print bad default config if compute baseline fails (#688) |
| 0bb3719 | Add `HELION_AUTOTUNE_RANDOM_SEED` env var and `autotune_random_seed` setting (#644) |
| 1701f8d | Raise better error when `hl.atomic_*` is used on device tensor (#658) |
| 7c8a560 | Add HELION_DEV_LOW_VRAM env var for low GPU memory machines (#325) |
| bd3d984 | Add HELION_USE_DEFAULT_CONFIG env var to force use default config (#37) |
| 01261f5 | Better error message for augmented assignment on host tensor without subscript (#807) |
| b03bb06 | Print Helion kernel source line in symbolic shape debugging (#771) |
| e9e4957 | Add extra line before repro log; update repro log tests (#1102) |
| b5430fe | Add warning to discourage use of `acc += lhs @ rhs` pattern (#1111) |

---

### 11. Code Cleanup/Refactoring (5 commits)

Commits focused on code cleanup or API deprecation.

| Commit | Message |
|--------|---------|
| 51580b4 | Remove `@helion.jit` usage and advise use of `@helion.kernel` (#1116) |
| 86c54e5 | Remove legacy `register_inductor_lowering` code (#864) |
| 4675e6a | Remove `hl.register_reduction_dim` API (#834) |
