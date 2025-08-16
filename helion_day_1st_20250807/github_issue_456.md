# Issue #456: Limit block sizes in autotuning because of triton max numel range

## Metadata
- **State**: OPEN
- **Author**: [oulgen](https://github.com/oulgen)
- **Created**: August 07, 2025 at 22:36 UTC
- **Updated**: August 07, 2025 at 22:36 UTC

## Description

ValueError('numel (2097152) exceeds triton maximum tensor numel (1048576)')

Config(block_sizes=[2048, 1024], loop_orders=[[1, 0]], flatten_loops=[False], l2_groupings=[8], range_unroll_factors=[0], range_num_stages=[0], range_multi_buffers=[None], range_flattens=[None], num_warps=32, num_stages=8, indexing='pointer', pid_type='flat', range_warp_specializes=[])

## Comments

*No comments yet.*
