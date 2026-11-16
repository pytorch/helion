# SDG paged MXINT8 attention

`kernel.py:run` dispatches stored SDG attention profiles to authored Helion
functions and ordinary `helion.Config` compilation. Each case has a separate
Kernel instance. Dispatch uses host-visible layouts and scalar arguments, never
tensor contents or a random seed. The three source families implement recurrence,
rowwise rescaling and complete-head rescaling. The legacy `rowwise_extended`
profile name selects the same rowwise function. The shared domain is defined
by `profile_contract.ROWWISE_THRESHOLDS`: rowwise tuning now includes T4/T8
in addition to its original T2/T16/T32/T64 choices. Both profile names accept
the six-value domain; selecting an existing value keeps its arithmetic.

The caller supplies strided BF16 queries, page-major INT8 caches with embedded
E8M0 scales, page tables, sequence lengths and an output allocation. Unsupported
layouts report the profile file used by the dispatcher. The kernels explicitly
use the CuTe backend, static shapes and `fast_math=True`. A stored config may
enable `cute_pdl_capture_overlap`, which requires fixed captured graph topology
and pointer parameters.

## Programmatic use

Raw profiles are external autotuner artifacts, intentionally ignored by Git.
Obtain the matching profile JSON from the experiment owner. With the compatible
Helion/CuTe dependencies installed, use the source entry without the private
benchmark harness:

```python
import os

os.environ["SDG_ATTENTION_PROFILES"] = "/path/to/retained-profiles.json"
from benchmarks.cute.sdg_attention_kernel.kernel import profile_names, run

print(profile_names())
output, lse = run(
    q, k, v, k_scale, v_scale, page_table, lengths,
    max_kv_len=max_kv_len, window_left=window_left,
    return_lse=True, out=out,
)
```

These tensors must have one of the profile JSON's exact layouts and backing
storage relationships. Without the environment variable, the source entry reads
`configs.json` alongside `kernel.py`; a missing dataset produces an explicit
error. This example demonstrates the public entry point, not input generation
or a performance claim.

## Export an isolated candidate

Commit compiler and benchmark changes before exporting. The exporter uses Git
archive for both trees and assembles the output atomically; it rejects tracked
source edits, an existing output directory, invalid profile lineage, incomplete
layout records, and duplicate profile names or dispatch signatures.

```bash
python benchmarks/cute/sdg_attention_kernel/export.py \
  --output artifacts/sdg-attention/candidate \
  --profiles /path/to/retained-profiles.json \
  --dependencies /path/to/trusted-candidate/_vendor
```

The dependency directory must contain the compatible CuTe package and supporting
Python dependencies. Its `.pth` files are trusted executable installation data.
The exporter replaces Helion with the committed source and removes the copied
Helion wheel metadata. `provenance.json` records the commit, separate Git tree
IDs for Helion and the benchmark, SHA256 hashes of committed benchmark files,
the raw profile hash, and hashes of every shipped source/dependency file.

A profile's `authored_source_sha256` identifies either a supported historical
source before formatting/packaging, or its current authored family file.
For a new tuning run, obtain the current value with
`profile_contract.source_fingerprint(family)`. Historical values remain lineage
metadata; they are **not** current file hashes. Neither kind certifies a whole
compiler/dependency bundle or proves that tuning took place. The manifest's
independent file hashes describe the committed/shipped bundle and detect changed
source or profile bytes. This local manifest is an integrity record, not a
signature from a third party.

Exported bundles always use their copied `configs.json` and reject
`SDG_ATTENTION_PROFILES`, even if it names that file. Start a fresh isolated
interpreter. Fixed-profile selection explicitly sets `force_autotune=False`,
`autotune_force_persistent=False` and an empty `autotune_config_overrides`.
The first two preserve stored configs despite the corresponding environment
flags; a nonempty effective `HELION_AUTOTUNE_CONFIG_OVERRIDES` mapping (inline
JSON or a JSON file) is rejected, including on repeated kernel access.
Search effort and cache settings do not replace this single supplied config.
Other runtime settings retain Helion's normal behavior.

Use an interpreter without ambient Helion distribution metadata or already imported
Helion/CuTe code. The bootstrap processes the bundled dependency installation
paths, then checks actual module origins before use. It accepts only the
code-free CuTe namespace created by Python startup and binds that namespace to
the vendor before loading its children. File-less extension submodules must be
owned by their registered parent chain ending in a vendored native extension.
It never substitutes an unrelated installed Helion version for the committed
compiler's identity. These origin checks cover Helion and CuTe only; other
supporting libraries already imported by the host may remain active, and bundled
paths take precedence for later imports.

Before importing bundle helpers or dependencies, the trusted exported entry
sets `sys.pycache_prefix` to a fresh private directory for the process lifetime
and removes it on exit. Thus preexisting timestamp bytecode cannot replace later
verified source imports, including lazy imports. The entry itself must be loaded
from trusted source: its code cannot protect an import that happened before it
ran. `-B` alone only disables cache writes and is not this protection.

## Canonical experiment reproduction

Canonical reproduction requires the original experiment's private harness,
profile dataset and reference outputs. Obtain them from the experiment owner;
they are not distributed by this package. Pass the exported directory as the
harness's `--implementation`; unset `SDG_ATTENTION_PROFILES`. Do not prepend
candidate vendor directories to `PYTHONPATH`: the entry loads dependencies only
after the original harness verifies its frozen Python 3.12 runtime.

Preserve that harness's workload, reference outputs, bank counts, validation,
graphs and timing settings. Compare implementations on the same idle GPU with
interleaved measurements. The package runs only the layouts in its supplied
profile dataset; it neither generates inputs nor claims that those configs are
optimal. CPU config/ABI/AST correspondence and native correctness or performance
are separate checks and must identify the exact exported bundle.
