# Issue #492: A way to provide auto tuning stop condition

## Metadata
- **State**: OPEN
- **Author**: [v0i0](https://github.com/v0i0)
- **Created**: August 12, 2025 at 23:46 UTC
- **Updated**: August 13, 2025 at 15:17 UTC

## Description

**Is your feature request related to a problem? Please describe.**
I would like a pluggable interface to determine "good enough" performance for a kernel, where auto-tuning stops.
For most kernels, for example, it seems like hitting ~90% of peak bandwidth would be a suitable criterion, but other criteria are imaginable (e.g. hitting a flop rate goal)

**Describe the solution you'd like**
1. Provide an `autotune_stop_condition=xxx` parameter, a function that received information on kernel inputs, and measured time, and returns a bool (True: stop, False: continue).
2. Provide a `PeakBandwidthStopCondition(slack=0.9)` and a `PeakFlopsStopCondition(dtype, slack=?)`

**Describe alternatives you've considered**
Could just autotune to the end, or with a time budget, but early exist seems useful.


## Comments

### Comment 1 by [jansel](https://github.com/jansel)
*Posted on August 13, 2025 at 15:17 UTC*

Custom stop conditions seem like a useful feature.  To implement this you would need to change this line:
https://github.com/pytorch/helion/blob/948030ab0c2438fbfa54b6753e7ef9d162dd325e/helion/autotuner/differential_evolution.py#L98

For PeakBandwidthStopCondition/PeakFlopsStopCondition we would also need to add code to estimate flops.

---
