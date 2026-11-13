# Experiment environments

The prepared runtime is
`artifacts/paper-matched-gb300-20261021T212533Z/`. Its
`runtime_manifest.json` identifies the interpreters, archived source trees,
packages, source patches, and validation hashes used by the runs.
`runtime.json` locates this dependency relative to the reproduction folder.

Activate and verify a prepared environment:

```bash
source benchmarks/cute/paper_attention_repro/setup_env.sh forward
# Or: backward, forward_sparse.
```

The optional second argument selects another prepared artifact. Activation
checks its imported packages and source revisions.

| Component | Forward | Backward |
|---|---|---|
| Python | 3.12.14 | 3.12.14 |
| Torch | 2.15.0.dev20260829+cu130 | 2.16.0.dev20261014+cu130 |
| cuDNN | 9.24.0.43 | 9.26.0.51 |
| Triton | 3.8.0+gitb252c7c4 | 3.9.0+gitfc8d31a5 |
| CuTe DSL | 4.7.0 | 4.7.0 |
| FlashAttention 4 | 4.0.0b19 | 4.0.0b19 |
| Quack | 0.6.4 | 0.6.4 |
| CUDA | 13.0 | 13.0 |

`requirements-*.in` specify version inputs; `requirements-*.lock` record
all installed versions. `environment-lock.json` records wheel URLs/hashes,
the Python source/build settings, dependency overrides, and source identities.
The prepared artifact contains the interpreters, installed packages, source
repositories, and setup/validation evidence.

Torch and Triton use the cu130 nightly index; the other packages use PyPI.
Quack 0.6.4 declares CuTe 4.6.2 while the measured stack uses CuTe 4.7.0;
`dependency-overrides.in` records that override. Both environments include
pytest 9.1.1 for the archived Flex example's `helion._testing` import.
Triton's distribution version identifies its pinned Git build.

Forward uses Helion cleanup `a16ffc60` and the exact fused/sparse harnesses
`b530d94` and `f7844a4`. The unavailable original measured forward compiler
revisions are declared in the runtime manifest.

Backward uses measured Helion `61c840f39` and harness `d839679`, including
the preserved SDPA backward timing patch. Saved GB300 configs and fresh
searches select this same source. The latest MLA pass-7 source/configs
remain unavailable.

The environment records the current ARM GB300 driver and native power
settings. The paper measured an x86 B200 at 750 W. Each run captures its
hardware settings.
