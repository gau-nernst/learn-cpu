#include "matmul.h"
#include <random>
#include <benchmark/benchmark.h>
#include <iostream>

const int alignment = 256;
const int M = 1024;
const int N = 1024;
const int K = 1024;

// const int M = 4096;
// const int N = 4096;
// const int K = 4096;

void randn_(float *A, int N) {
  static std::default_random_engine rng;
  std::normal_distribution<float> dist;
  for (int i = 0; i < N; i++)
    A[i] = dist(rng);
}

double check(const float *A, const float *B, int M, int N) {
  int num_mismatch = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      const int idx = m * N + n;

      // compute error in double precision
      const double a = A[idx], b = B[idx];
      const double r_error = abs(a - b) / a;

      if (r_error > 1e-5) {
        num_mismatch += 1;
        // printf("Mismatch at m=%i, n=%i: A=%.2e, B=%.2e\n", m, n, A[idx], B[idx]);
      }
    }

  return static_cast<double>(num_mismatch) / static_cast<double>(M * N);
}

typedef void MatmulFunc(const float * __restrict A,
                        const float * __restrict B,
                              float * __restrict C,
                        int M, int N, int K);

template <MatmulFunc f>
void BM_matmul(benchmark::State& state) {
  // setup
  float *A, *B, *C, *C_ref;
  if (alignment == 0) {
    A = new float[M * K];
    B = new float[N * K];
    C = new float[M * N];
    C_ref = new float[M * N];
  }
  else {
    // faster in some cases
    A = reinterpret_cast<float *>(aligned_alloc(alignment, M * K * sizeof(float)));
    B = reinterpret_cast<float *>(aligned_alloc(alignment, N * K * sizeof(float)));
    C = reinterpret_cast<float *>(aligned_alloc(alignment, M * N * sizeof(float)));
    C_ref = reinterpret_cast<float *>(aligned_alloc(alignment, M * N * sizeof(float)));
  }
  randn_(A, M * K);
  randn_(B, N * K);
  std::fill(C, C + M * N, 0.0f);
  std::fill(C_ref, C_ref + M * N, 0.0f);

  // correctness check
  naive_matmul(A, B, C_ref, M, N, K);  // ref impl
  f(A, B, C, M, N, K);

  const float mismatch_ratio = check(C_ref, C, M, N);
  state.counters["Mismatch ratio"] = mismatch_ratio;

  // benchmark
  // NOTE: we does not flush cache in-between runs. it will probably affect benchmark results.
  for (auto _ : state) {
    f(A, B, C, M, N, K);

    benchmark::ClobberMemory();
  }

  // cleanup
  if (alignment == 0) {
    delete[] A;
    delete[] B;
    delete[] C;
  }
  else {
    free(A);
    free(B);
    free(C);
  }
}

BENCHMARK(BM_matmul<naive_matmul>)->Unit(benchmark::kMillisecond);
#ifdef BLAS_NAME
BENCHMARK(BM_matmul<blas_matmul>)->Unit(benchmark::kMillisecond);
#endif
#ifdef __APPLE__
BENCHMARK(BM_matmul<register_tile_matmul<1, 4, 4, 1>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul<register_tile_matmul<2, 4, 4, 1>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul<neon_matmul<16, 16, 8>>)->Unit(benchmark::kMillisecond);
#else
BENCHMARK(BM_matmul<register_tile_matmul<1, 4, 4, 4>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul<register_tile_matmul<2, 4, 4, 2>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul<l1_tile_matmul<32, 32, 128>>)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matmul<tile_2level_matmul<16, 16, 128, 4, 4, 2>>)->Unit(benchmark::kMillisecond);
#endif

BENCHMARK_MAIN();
