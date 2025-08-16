# Issue #497: Tensor size specialization error

## Metadata
- **State**: OPEN
- **Author**: [mlazos](https://github.com/mlazos)
- **Created**: August 14, 2025 at 02:29 UTC
- **Updated**: August 14, 2025 at 02:30 UTC

## Description

**Describe the bug**
I get an error when using a tensor size in a device allocation. ([full trace](https://gist.github.com/mlazos/2d90b18d3eeb03a3a2e31c45de2ce7a5))

**To Reproduce**
[repro](https://gist.github.com/mlazos/2ab1816e7ae55218601508af7d19a928)

**Expected behavior**
So I think it should be possible to use the size or symintify it. 

**Versions**
PyTorch/Triton/Helion versions and any other relevant library version.
Pytorch b1f43548cad8fc0e30bda250f6e196310fa7a4bc
Helion 334095fbd68555506f46c7adb52db654178fffe3

**Additional context**
Add any other context about the problem here.


## Comments

*No comments yet.*
