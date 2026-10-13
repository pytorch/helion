# K3 native MXFP8 hillclimb

Worktree: /home/shangdiy/helion-native-mxfp8-20261013T185629Z
Base: origin/main 279e851e. GPU2, GB300. 24 existing K3 cases.
User requested native support and measurement using cute-hillclimb.
No commits or pushes are authorized for this new work.

## Target

General CuTe backend lowering of full-K hl.dot_scaled(E4M3,E8M0) to native
Blackwell MXFP8. Preserve public language semantics, operand liveness, graph
replay and numerical accuracy. Every benchmark variant must match the best
measured baseline within 1%, with geomean ratio >=1.

## Ranked backlog

1. Prove a single-root full-K scaled dot and lower using raw FP8 TMA plus
   fused two-operand E8M0 scale packing. Reuse generic native block-scaled
   producer/MMA/epilogue pipeline, preserving NVFP4 behavior.
2. Admit and seed format-specific K/stage geometry; normalize invalid SMEM
   combinations and ensure cold full autotuning discovers them.
3. Reduce tiny-M padding/epilogue costs, direct output stores and scale prep.
4. Compare worst variant against DeepGEMM structure and transplant missing
   mechanisms through general backend passes/knobs.
5. Complete broad verification and independent code review.

## Iteration log

- Setup: native runtime format extension implemented by side-task agent;
  CPU-only fake tensor compilation passes tiny, irregular, and two-CTA cases.
  No GPU correctness/performance claims yet; goal ratios unmeasured.

- Iteration1: native whole-root MXFP8 support and zero-copy singleton TMA
  coordinate repair pass initial GPU checks. Cold full shape1 (409 configs,
  465.8s) verified 9.936us native vs14.720us DeepGEMM and17.776us Triton;
  artifact cold_shape01_v1.json. Goal 1/24 measured/pass, others pending.
  Initial search included per-call host gaps; final verification used graphs.
  The experiment extended custom benchmark callbacks to initial candidates.
  That extension was later removed from the final stack at the user's request;
  its measurements remain tied to the original experiment source snapshots.

- Iteration2: larger-M cold search shape21 continues against full FlashInfer
  CuTe tactic search plus DeepGEMM/Triton/PyTorch; fixed campaign ninja PATH.
  Review fixes developed in independent prototype at identical base279e851e:
  full domain proof, raw operand alignment staging/live replay, helper hashes,
  benchmark typing/PATH and optional unsupported-tactic receipts. Initial
  safety tests59CPU/12CUDA pass. Proof-scoped source-config canonicalization
  preserves all64 native geometries and scalar/user tunable choices; CPU31
  and CUDA9 equivalence tests pass. Prototype lint0errors. No new speedup
  claim yet; goal1/24 measured. Other K19 session currently occupies0/1/3;
  full verification waits for campaign exit rather than launching in gaps.

- Final cold shape23: ratio1.250233 vsdeepgemm; artifactcold_shape23_final_v1.json; isolated GPU2, six mirrored rounds.

- Final cold shape22: ratio1.429245 vsdeepgemm; artifactcold_shape22_final_v1.json; isolated GPU2, six mirrored rounds.

- Final cold shape21: ratio1.311741 vsdeepgemm; artifactcold_shape21_final_v1.json; isolated GPU2, six mirrored rounds.

- Timing phase: remaining cold full searches keep their full search effort and six mirrored rounds. Final sampling uses20ms warmup/100ms repeat; near-goal or unstable cases receive a longer pinned verification. Existing search is allowed to finish unchanged; only supervisor is transferred. Compiler and benchmark sources remain frozen.

- Final cold shape0: ratio1.112903 vstriton; artifactcold_shape00_final_v1.json; isolated GPU2, six mirrored rounds.

- Final cold shape2: ratio1.304962 vsdeepgemm; artifactcold_shape02_final_v1.json; isolated GPU2, six mirrored rounds.

- Final cold shape3: ratio1.380117 vsdeepgemm; artifactcold_shape03_final_v1.json; isolated GPU2, six mirrored rounds.

- Final cold shape4: ratio1.251748 vsdeepgemm; artifactcold_shape04_final_v1.json; isolated GPU2, six mirrored rounds.

- Panel update20261013T222000Z: cases04–07 nowM8 at user request. Isolated library reference dispatch guard/benchmark shapes extended from7 to8 to retain the published decode/BF16/patch algorithm; existing JSON AOT configs use normal nearest-shape selection. Original library checkout remains untouched. OldM7 artifacts preserved inbefore_m8_20261013T222000Z.

- Final cold shape4: ratio1.271028 vsdeepgemm; artifactcold_shape04_final_v2.json; isolated GPU2, six mirrored rounds.

- Additional lead tested: uint32 scale packing passed46CPU/12CUDA tests but measured15.424us(width1 anchor),15.456us(width4),15.424us(width1 anchor) onGPU0 with identical geometry. No speedup withinnoise; prototype excluded from clean history. Raw probes remain in the portability worktree.

- Final cold shape5: ratio1.462783 vsdeepgemm; artifactcold_shape05_final_v2.json; isolated GPU2, six mirrored rounds.

- Final pinned verification shape6: ratio1.089189 vstriton; artifactcold_shape06_final_v1_long_verify.json; isolated GPU2, six mirrored rounds; admissioncold_shape06_final_v1_long_verify.admission.json.

- Corrected supervisor cache scope: its explicit startup HELION_SKIP_CACHE=1 caused AOT baselines to retune in unfinished cases7/8/12. Those runs are excluded. New library harness forces evaluate/cache-enabled public AOT calls and scopes bypass to native full search only. All previously accepted11rows were initialized by the legacy baseline-before-bypass protocol and remain valid.

- Final cold shape8: ratio1.265295 vsdeepgemm; artifactcold_shape08_final_v2_gpu0.json; isolated GPU0, six mirrored rounds; admissioncold_shape08_final_v2_gpu0.admission.json.

- Final cold shape12: ratio1.269321 vsdeepgemm; artifactcold_shape12_final_v2_gpu1.json; isolated GPU1, six mirrored rounds; admissioncold_shape12_final_v2_gpu1.admission.json.

- Final cold shape9: ratio1.478689 vsdeepgemm; artifactcold_shape09_final_v1_gpu0.json; isolated GPU0, six mirrored rounds; admissioncold_shape09_final_v1_gpu0.admission.json.

- Final cold shape7: ratio1.369048 vsdeepgemm; artifactcold_shape07_final_v2_gpu2.json; isolated GPU2, six mirrored rounds; admissioncold_shape07_final_v2_gpu2.admission.json.

- Final cold shape10: ratio1.175790 vstriton; artifactcold_shape10_final_v1_gpu0.json; isolated GPU0, six mirrored rounds; admissioncold_shape10_final_v1_gpu0.admission.json.

- Final cold shape11: ratio1.463942 vsdeepgemm; artifactcold_shape11_final_v1_gpu0.json; isolated GPU0, six mirrored rounds; admissioncold_shape11_final_v1_gpu0.admission.json.

- Final cold shape16: ratio1.273148 vsdeepgemm; artifactcold_shape16_final_v1_gpu2.json; isolated GPU2, six mirrored rounds; admissioncold_shape16_final_v1_gpu2.admission.json.

- Final pinned verification shape13: ratio1.136691 vstriton; artifactcold_shape13_final_v1_gpu1_long_verify_v1.json; isolated GPU1, six mirrored rounds; admissioncold_shape13_final_v1_gpu1_long_verify_v1.admission.json.

- Final cold shape14: ratio1.052770 vstriton; artifactcold_shape14_final_v1_gpu1.json; isolated GPU1, six mirrored rounds; admissioncold_shape14_final_v1_gpu1.admission.json.

- Final cold shape17: ratio1.059211 vstriton; artifactcold_shape17_final_v1_gpu2.json; isolated GPU2, six mirrored rounds; admissioncold_shape17_final_v1_gpu2.admission.json.

- Final cold shape15: ratio1.490964 vsdeepgemm; artifactcold_shape15_final_v1_gpu1.json; isolated GPU1, six mirrored rounds; admissioncold_shape15_final_v1_gpu1.admission.json.

- Final pinned verification shape18: ratio0.953488 vstriton; artifactcold_shape18_final_v1_gpu2_long_verify_v1.json; isolated GPU2, six mirrored rounds; admissioncold_shape18_final_v1_gpu2_long_verify_v1.admission.json.

- Final cold shape19: ratio1.470060 vsdeepgemm; artifactcold_shape19_final_v1_gpu2.json; isolated GPU2, six mirrored rounds; admissioncold_shape19_final_v1_gpu2.admission.json.

- Optimization iteration: final24-case checker23PASS/1FAIL, GM1.2764; worstcase18M128N2048K4096 ratio0.953488. GPU0 SF4probe anchors12.320us / width4 12.8005us / anchor12.320us; reject width4. Testing coalesced per-thread scale batching onGPU2; investigate generalMXFP8 BK512/stages2 andN64 capability in separate frozen worktrees. No probe numbers enter goal.

- BK512 cold full iteration shape18: ratio0.955959 vstriton; artifactcold_shape18_bk512_final_v1_gpu0_long_verify_v1.json; isolatedGPU0; sixmirroredrounds; newMXFP8-onlyK512choice andstages2seeds.

- N64 cold full iteration case18: native11.136us vsTriton11.776us, ratio1.057471; artifactcold_shape18_n64_final_v1_gpu1.json; isolatedGPU1 with sixmirroredrounds, freshseed/fullsearch/cachebypass. AutotunerfoundBN64BK256stages4cluster1persistentTrue. Provisionalchecker24PASS; longerpairedverificationandfullgatespending.

- N64 longpaired verification case18: native11.168us vsTriton11.808us, ratio1.057307; authoritativeartifactcold_shape18_n64_final_v1_gpu1_long_verify_v1.json; isolatedGPU1 sixmirroredrounds100mswarmup/500msrepeat. Checker24PASS, GM1.2820. Fullsuite verification ofN64 runningdefaultGPU0/CuTeGPU2.

- Post-goal N64 cold full shape14: ratio1.169591 vstriton; artifactcold_shape14_n64_final_v1_gpu1.json; all6providers, isolatedGPU1, mirroredrounds, longer verification when near1.02.

- Post-goal N64 cold full shape17: ratio1.118056 vstriton; artifactcold_shape17_n64_final_v1_gpu1.json; all6providers, isolatedGPU1, mirroredrounds, longer verification when near1.02.

- Post-goal N64 cold full shape06: ratio1.131653 vstriton; artifactcold_shape06_n64_final_v1_gpu1.json; all6providers, isolatedGPU1, mirroredrounds, longer verification when near1.02.
