# Issue #498: Invalid triton kernel generated

## Metadata
- **State**: CLOSED
- **Author**: [mlazos](https://github.com/mlazos)
- **Created**: August 14, 2025 at 02:50 UTC
- **Updated**: August 15, 2025 at 22:01 UTC
- **Closed**: August 15, 2025 at 22:01 UTC
- **Assignees**: [jansel](https://github.com/jansel)

## Description

**Describe the bug**
Helion generates invalid triton code. [trace](https://gist.github.com/mlazos/2a2bea537e728eebee84ec76951dd65f)
**To Reproduce**
[repro](https://gist.github.com/mlazos/7286727e6e987ad3e2586cef19889c98)

**Expected behavior**
I think we should be able to generate either valid triton code or a clearer error

**Versions**
Pytorch b1f43548cad8fc0e30bda250f6e196310fa7a4bc
Helion 334095fbd68555506f46c7adb52db654178fffe3
Triton/pytorch-triton                3.4.0+gitf7888497

**Additional context**
My code is pretty borked so it's more likely a better error message is the anwer here.

## Comments

*No comments yet.*
