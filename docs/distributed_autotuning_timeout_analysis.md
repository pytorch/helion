# Distributed Autotuning Timeout Analysis

## The Core Problem

For distributed kernels using `symm_mem_sync` (symmetric memory synchronization):

1. **All ranks must coordinate at synchronization barriers** - The kernel code includes barriers where all 4 ranks must arrive before any can proceed
2. **Bad configs can cause hangs** - For example, `block_sizes=[1]` may cause some ranks to not reach the barrier, causing all ranks to hang indefinitely
3. **We need to run the kernel in a killable subprocess** - This protects the main process's GPU from getting stuck

## Why Current Approaches Don't Work

### Current Spawn-mode Precompile

Each rank independently spawns an isolated subprocess:
- Rank 0's main process spawns subprocess A
- Rank 1's main process spawns subprocess B
- Rank 2's main process spawns subprocess C
- Rank 3's main process spawns subprocess D

**Problem:** These 4 subprocesses (A, B, C, D) are completely isolated. They cannot communicate with each other. When the kernel tries to do distributed sync:
- Subprocess A waits for B, C, D
- Subprocess B waits for A, C, D
- etc.

But A, B, C, D don't know about each other! They hang immediately for **ALL configs**, not just bad ones.

**Error seen:**
```
RuntimeError: Could not resolve the process group registered under the name 0
```

### Fork-mode Precompile

Fork mode only compiles the Triton kernel, it doesn't actually run the kernel. This means:
- It cannot detect runtime hangs
- The hang happens during kernel execution, not compilation
- Fork mode is useless for detecting distributed kernel hangs

### Gloo-based Timeout Detection (in main process)

The original commit added gloo-based timeout detection in `benchmark_function`:
1. Run kernel in main process
2. Poll CUDA event for completion with 30s timeout
3. Use gloo all_reduce to sync timeout status across ranks
4. If any rank times out, all ranks skip the config

**Problem:** If the kernel hangs, **the main process's GPU is stuck**. Even though we detect the timeout and move on, the GPU has a hung kernel consuming resources. Subsequent work on the same CUDA stream will queue behind the hung kernel and never execute.

## The Correct Solution: Coordinated Subprocess Distributed Group

### Concept

The subprocesses must form their **OWN coordinated distributed group**, separate from the main process group:

```
Main Process Group (ranks 0-3, using NCCL)
    |
    | spawns
    v
Subprocess Group (ranks 0-3, using new distributed init)
    - Forms its own distributed group
    - Sets up its own symmetric memory
    - Runs the kernel
    - Can be killed if it hangs
```

### Implementation Steps

1. **Detect distributed context** in `start_precompile_and_check_for_hangs`
   - Check if `torch.distributed.is_initialized()` and `world_size > 1`

2. **Coordinate with other ranks via gloo before spawning**
   - All ranks must agree to test the same config at the same time
   - Use gloo barrier to synchronize

3. **Create shared temp file store for subprocess group**
   - All ranks agree on a temp file path (e.g., based on config hash)
   - This file store will be used for subprocess distributed init

4. **Spawn subprocess with distributed init info**
   - Pass to subprocess:
     - Serialized kernel function
     - Tensor arguments
     - Rank and world_size
     - File store path for distributed init
     - GPU device to use

5. **Subprocess initialization**
   - Init distributed using the temp file store
   - Set symmetric memory backend to NVSHMEM
   - Enable symmetric memory for the group
   - Rendezvous for symmetric memory buffers
   - Run the kernel

6. **Main process monitors with timeout**
   - Wait for subprocess with timeout (e.g., 60s)
   - If timeout, kill subprocess
   - Killing subprocess cleans up its CUDA context completely

7. **Cleanup**
   - Remove temp file store
   - Continue to next config

### Environment Variable

Add a new precompile mode: `HELION_AUTOTUNE_PRECOMPILE=spawn_dist`

- `fork` - Fork mode, only compiles Triton (default for non-distributed)
- `spawn` - Spawn mode, runs kernel in isolated subprocess (for non-distributed)
- `spawn_dist` - Spawn mode with coordinated distributed group (for distributed kernels)

### Pros of This Approach

1. **Subprocess isolation protects main process GPU** - If subprocess hangs and is killed, main process GPU is unaffected
2. **Clean GPU state** - Killing subprocess destroys its CUDA context completely
3. **Catches hangs early** - Precompile phase, before wasting time on benchmarking
4. **Works for any distributed kernel** - Not limited to specific sync patterns

### Cons / Challenges

1. **Complex implementation** - Requires setting up full distributed + symm_mem in subprocesses
2. **Subprocess rendezvous** - Subprocesses from different parent processes need to coordinate
3. **Symmetric memory setup** - NVSHMEM initialization in subprocesses may have quirks
4. **Performance overhead** - Setting up distributed group for each config test is expensive

### Alternative: Spawn-mode Benchmark

Same concept, but applied to the benchmarking phase instead of precompile:
- Precompile is skipped or done without running kernel
- Benchmarking spawns coordinated subprocesses

This has similar pros/cons but runs later in the pipeline, potentially wasting time on compilation before detecting hangs.

## Summary

| Approach | Detects Hangs? | Protects Main GPU? | Complexity |
|----------|---------------|-------------------|------------|
| Fork precompile | No (only compiles) | N/A | Low |
| Spawn precompile (isolated) | No (all configs hang) | Yes | Medium |
| Gloo timeout (main process) | Yes | No (GPU stuck) | Low |
| **Spawn precompile (coordinated)** | **Yes** | **Yes** | **High** |

**Recommendation:** Implement `spawn_dist` mode with coordinated subprocess distributed group.
