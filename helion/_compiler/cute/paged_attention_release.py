"""What the config contract and the device template share of the paired
paged-attention loader's tail release.

Codegen renders no keyword for the default lead, so the template's own
default has to be the contract's number, and the template imports nothing of
the compiler: both take the number from here. This module imports nothing,
so the template can import it; that makes this file a source dependency of
the paged-attention wrapper.
"""

from __future__ import annotations

# Pages the paired loader's stream has left when its tail release lets the
# launch's dependents in, unless a config names another lead
# (``cute_mma_dag_release_lead``, described at
# ``cute_mma_dag.MMA_DAG_RELEASE_LEAD_KEY``).
MMA_DAG_RELEASE_LEAD_DEFAULT = 4
