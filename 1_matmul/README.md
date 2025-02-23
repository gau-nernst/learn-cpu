# Matrix multiplication

Resources:
- https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf (to be read...)
- https://siboehm.com/articles/22/Fast-MMM-on-CPU
- ARM NEON:
  - https://developer.arm.com/documentation/102467/0201/
  - https://arm-software.github.io/acle/neon_intrinsics/advsimd.html
- Apple:
  - https://developer.apple.com/documentation/accelerate/blas-library
  - https://github.com/corsix/amx
  - https://zhen8838.github.io/2024/04/23/mac-amx_en/
- AVX2 (TODO)
- https://ppc.cs.aalto.fi/

**Apple M1**, plugged in. `M=N=K=1024`

128KB L1 cache (per core), and 12MB L2 cache (shared)

```bash
# install google/benchmark
brew install google-benchmark

# single core
clang++ main.cc -Wall -std=c++17 -ffast-math -O3 -march=native -lbenchmark -lpthread -o main -framework Accelerate -DACCELERATE_NEW_LAPACK && OMP_NUM_THREADS=1 ./main  --benchmark_counters_tabular=true

# with OpenMP
$(brew --prefix llvm)/bin/clang++ main.cc -Wall -std=c++17 -ffast-math -O3 -march=native -lbenchmark -lpthread -o main -framework Accelerate -DACCELERATE_NEW_LAPACK -fopenmp && ./main --benchmark_counters_tabular=true

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

32KB L1 data cache (per core), 512KB L2 cache (per core), and 32MB L3 cache (shared). 6 physical cores.

```bash
# install google/benchmark
sudo apt install libbenchmark-dev

# install Intel MKL
# linker options generated with https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html
# change $CONDA_PREFIX to wherever your pip installs MKL to.
pip install mkl-devel

# gcc 11.4.0
g++ main.cc -Wall -std=c++17 -ffast-math -O3 -march=native -lbenchmark -lpthread -o main -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -m64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lm -ldl && LD_LIBRARY_PATH=$CONDA_PREFIX/lib OMP_NUM_THREADS=1 ./main --benchmark_counters_tabular=true

# with OpenMP
g++ main.cc -Wall -std=c++17 -ffast-math -O3 -march=native -lbenchmark -lpthread -o main -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -m64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lm -ldl -fopenmp && LD_LIBRARY_PATH=$CONDA_PREFIX/lib ./main --benchmark_counters_tabular=true
```

Kernel name                   | 1-thread (ms) | % of Intel MKL | Multi-thread (ms) | % of Intel MKL
------------------------------|---------------|----------------|-------------------|---------------
Intel MKL                     | 15.9          | 100.00%        | 2.99              | 100.00%
Naive                         | 81.4          |  19.53%        | 8.27              |  36.15%
Register tile (dot product)   | 66.1          |  24.05%        | 10.7              |  27.94%
Register tile (outer product) | 54.7          |  29.07%        | 9.38              |  31.88%
L1 tile                       | 47.9          |  33.19%        | 7.73              |  40.79%
L1 + register tile            | 32.6          |  48.77%        | 5.45              |  54.86%

Lessons learned:
- NEON has Q-registers (128-bit) and D-registers (64-bit). Intrinsics using Q-registers has suffix `q` e.g. `vld1q_f32()`. Some instructions has `_lane` in the name, meaning the op will only affect 1 element in each register e.g. `vld4q_lane_f32()` means we will load 4 FP32 into the specified lane across 4 registers.
- NEON's dot product and matrix multiply only support INT8 and BF16. For FP32, we have to use fused multiply-accumulate (FMA) instead, which operates directly on FP32x4 registers.
- To enable auto-vectorization, use `-ffast-math`, which enables math ops re-ordering.
- Apple M1 and Ryzen 5600 have very different optimal kernel parameters (tile size). Perhaps with sufficient knowledge about data movements and CPU cache, someone can explain the difference...
- OpenMP: `parallel` (start parallel region, create thread pool), `for`, `schedule(static,1)`, `collapse(2)` (collapse 2 for loops), `nowait` (don't wait for sync after for loop), `critical` (only 1 thread execute at a time), `atomic` (use hardware atomic op).
- Using OpenMP may (and will) generate different code. One example is the compiler fail to auto-vectorize when there is OpenMP. Be extra careful.
- If using OpenMP results in more than no. of physical of cores speedup, it means that the code is not taking advantage of ILP well i.e. it does not schedule enough independent instructions. Another way to check is to use `perf` and check for instructions/clock.
