# Issue #496: jagged softmax "index out of bounds" error in ref eager mode

## Metadata
- **State**: OPEN
- **Author**: [pianpwk](https://github.com/pianpwk)
- **Created**: August 13, 2025 at 22:53 UTC
- **Updated**: August 13, 2025 at 22:53 UTC

## Description

Might be a wrong kernel, but I hit this error from the `hl.store` call, running `HELION_INTERPRET=$([[ "ref-eager" == "ref-eager" ]] && echo "1") python test/test_examples.py -k test_jagged_softmax` in https://github.com/pytorch/helion/pull/480
```
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [112,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [113,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [114,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [115,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [116,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [117,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [118,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [119,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [120,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [121,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [122,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [123,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [125,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [104,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [105,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [106,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [107,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [108,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [109,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [110,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [111,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [112,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [113,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [114,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [115,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [116,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [117,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [118,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [119,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [120,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [121,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [122,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [123,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [125,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [126,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [64,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [65,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [66,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [67,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [68,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [69,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [70,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [71,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [72,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [73,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [74,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [75,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [76,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [77,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [78,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [79,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [80,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [81,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [82,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [83,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [84,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [85,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [86,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [87,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [88,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [89,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [90,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [91,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [92,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [93,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [94,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [95,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [0,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [1,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [2,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [3,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [4,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [5,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [6,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [7,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [8,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [9,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [10,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [11,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [12,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [13,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [14,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [15,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [16,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [17,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [18,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [19,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [20,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [21,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [22,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [23,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [24,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [25,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [26,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [27,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [28,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [29,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [30,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [31,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [32,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [33,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [34,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [35,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [36,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [37,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [38,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [39,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [40,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [41,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [42,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [43,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [44,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [45,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [46,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [47,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [48,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [49,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [50,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [51,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [52,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [53,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [54,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [55,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [56,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [57,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [58,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [59,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [60,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [61,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [62,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/data/users/pianpwk/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:113: operator(): block: [127,0,0], thread: [63,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
E
======================================================================
ERROR: test_jagged_softmax (__main__.TestExamples)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 1289, in not_close_error_metas
    pair.compare()
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 740, in compare
    self._compare_values(actual, expected)
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 898, in _compare_values
    compare_fn(
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 1077, in _compare_regular_values_close
    matches = torch.isclose(
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/data/users/pianpwk/helion-fork/helion/test/test_examples.py", line 652, in test_jagged_softmax
    check_example(
  File "/data/users/pianpwk/helion-fork/helion/helion/_testing.py", line 434, in check_example
    torch.testing.assert_close(
  File "/data/users/pianpwk/helion-fork/helion/helion/_testing.py", line 151, in counting_assert_close
    return RefEagerTestBase._original_assert_close_func(*args, **kwargs)  # type: ignore[misc]
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 1567, in assert_close
    error_metas = not_close_error_metas(
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 1296, in not_close_error_metas
    f"Comparing\n\n"
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 407, in __repr__
    body = [
  File "/data/users/pianpwk/pytorch/torch/testing/_comparison.py", line 408, in <listcomp>
    f"    {name}={value!s},"
  File "/data/users/pianpwk/pytorch/torch/_tensor.py", line 568, in __repr__
    return torch._tensor_str._str(self, tensor_contents=tensor_contents)
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 722, in _str
    return _str_intern(self, tensor_contents=tensor_contents)
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 643, in _str_intern
    tensor_str = _tensor_str(self, indent)
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 375, in _tensor_str
    formatter = _Formatter(get_summarized_data(self) if summarize else self)
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 411, in get_summarized_data
    return torch.stack([get_summarized_data(x) for x in (start + end)])
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 411, in <listcomp>
    return torch.stack([get_summarized_data(x) for x in (start + end)])
  File "/data/users/pianpwk/pytorch/torch/_tensor_str.py", line 401, in get_summarized_data
    return torch.cat(
torch.AcceleratorError: CUDA error: device-side assert triggered
Search for `cudaErrorAssert' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


----------------------------------------------------------------------
Ran 1 test in 0.634s

FAILED (errors=1)

```


## Comments

*No comments yet.*
