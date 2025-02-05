# Matrix multiplication

Resources:
- https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf (to be read...)
- ARM NEON:
  - https://developer.arm.com/documentation/102467/0201/
  - https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
- Apple:
  - https://developer.apple.com/documentation/accelerate/blas-library
  - https://github.com/corsix/amx
  - https://zhen8838.github.io/2024/04/23/mac-amx_en/
- AVX2 (TODO)

**Apple M1**, plugged in. `M=N=K=1024`

```bash
# single core
clang++ main.cpp -O3 -ffast-math -march=native -std=c++17 -Wall -o main -framework Accelerate -DACCELERATE_NEW_LAPACK && ./main

# with OpenMP
$(brew --prefix llvm)/bin/clang++ main.cpp -O3 -ffast-math -march=native -std=c++17 -Wall -fopenmp -o main -framework Accelerate -DACCELERATE_NEW_LAPACK && ./main

```

Kernel name                    | Time (ms) | % of Apple Accelerate
-------------------------------|-----------|----------------------
Apple Accelerate               |      1.99 | 100.00%
Naive matmul                   |     84.34 |   2.36%
Tile matmul v1 (dot product)   |     27.15 |   7.33%
Tile matmul v2 (outer product) |     26.73 |   7.44%
&emsp; + OpenMP                |      7.93 |  25.09%
NEON intrinsics                |     31.48 |   6.32%

**Ryzen 5600**. `M=N=K=1024`

```bash
pip install mkl-devel

# linker options generated with https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# change $CONDA_PREFIX to wherever your pip installs MKL to.
g++ main.cpp -O3 -ffast-math -march=native -std=c++17 -Wall -o main -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -m64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl && LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./main
```

Kernel name                    | Time (ms) | % of Intel MKL
-------------------------------|-----------|----------------
Intel MKL                      |      3.20 | 100.00%
Naive matmul                   |     82.18 |   3.89%
&emsp; + OpenMP                |      9.93 |  32.23%
Tile matmul v1 (dot product)   |     72.38 |   4.42%
Tile matmul v2 (outer product) |     57.54 |   5.56%
&emsp; + OpenMP                |     11.43 |  28.00%

TODO:
- Investigate OpenMP thread scheduling and cache-friendliness?

Lessons learned:
- NEON has Q-registers (128-bit) and D-registers (64-bit). Intrinsics using Q-registers has suffix `q` e.g. `vld1q_f32()`. Some instructions has `_lane` in the name, meaning the op will only affect 1 element in each register e.g. `vld4q_lane_f32()` means we will load 4 FP32 into the specified lane across 4 registers.
- NEON's dot product and matrix multiply only support INT8 and BF16. For FP32, we have to use fused multiply-accumulate (FMA) instead, which operates directly on FP32x4 registers.
- To enable auto-vectorization, use `-ffast-math`, which enables math ops re-ordering.
- Apple M1 and Ryzen 5600 have very different optimal kernel parameters (tile size). Perhaps with sufficient knowledge about data movements and CPU cache, someone can explain the difference...
