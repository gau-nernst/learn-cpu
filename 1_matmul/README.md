# Matrix multiplication

Resources:
- https://developer.arm.com/documentation/102467/0201/
- https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
- https://developer.apple.com/documentation/accelerate/blas-library

**Apple M1**, plugged in. `M=N=K=1024`

Kernel name      | Time (ms) | % of Apple Accelerate
-----------------|-----------|----------------------
Apple Accelerate |     11.83 | 100.00%
Naive matmul     |  4,390.78 |   0.27%
Tile matmul      |    531.83 |   2.22%
NEON intrinsics  |    413.67 |   2.86%

Lessons learned:
- NEON has Q-registers (128-bit) and D-registers (64-bit). Intrinsics using Q-registers has suffix `q` e.g. `vld1q_f32()`
- NEON's dot product and matrix multiply only support INT8 and BF16. For FP32, we have to use fused multiply-accumulate (FMA) instead, which operates directly on FP32x4 registers.
