# Matrix multiplication

Resources:
- https://developer.arm.com/documentation/102467/0201/
- https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
- https://developer.apple.com/documentation/accelerate/blas-library

**Apple M1**, plugged in. `M=N=K=1024`

Kernel name                    | Time (ms) | % of Apple Accelerate
-------------------------------|-----------|----------------------
Apple Accelerate               |      1.94 | 100.00%
Naive matmul                   |    889.92 |   0.22%
Tile matmul v1 (dot product)   |     97.10 |   2.00%
Tile matmul v2 (outer product) |     92.06 |   2.11%
NEON intrinsics                |     84.78 |   2.29%
NEON intrinsics + OpenMP       |     18.61 |  10.42%

Lessons learned:
- NEON has Q-registers (128-bit) and D-registers (64-bit). Intrinsics using Q-registers has suffix `q` e.g. `vld1q_f32()`. Some instructions has `_lane` in the name, meaning the op will only affect 1 element in each register e.g. `vld4q_lane_f32()` means we will load 4 FP32 into the specified lane across 4 registers.
- NEON's dot product and matrix multiply only support INT8 and BF16. For FP32, we have to use fused multiply-accumulate (FMA) instead, which operates directly on FP32x4 registers.
