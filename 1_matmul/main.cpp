#include "matmul.h"
#include <random>
#include <stdio.h>
#include <chrono>

const int alignment = 128;
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

void check(const float *A, const float *B, int M, int N) {
  int num_mismatch = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      const int idx = m * N + n;
      const float r_error = abs((A[idx] - B[idx]) / A[idx]);
      if (r_error > 1e-5) {
        num_mismatch += 1;
        // printf("Mismatch at m=%i, n=%i: A=%.2e, B=%.2e\n", m, n, A[idx], B[idx]);
      }
    }

  const float mismatch_ratio = static_cast<float>(num_mismatch) / static_cast<float>(M * N);
  printf("Mismatch percent: %.2f%%\n", mismatch_ratio * 100);
}

typedef void MatmulFunc(const float * __restrict A,
                        const float * __restrict B,
                              float * __restrict C,
                        int M, int N, int K);

template <MatmulFunc f>
float benchmark(int n = 100) {
  float *A, *B, *C;

  if (alignment == 0) {
    A = new float[M * K];
    B = new float[N * K];
    C = new float[M * N];
  }
  else {
    // faster in some cases
    A = reinterpret_cast<float *>(aligned_alloc(alignment, M * K * sizeof(float)));
    B = reinterpret_cast<float *>(aligned_alloc(alignment, N * K * sizeof(float)));
    C = reinterpret_cast<float *>(aligned_alloc(alignment, M * N * sizeof(float)));
  }

  // re-fill A and B with random data to avoid cache re-use
  randn_(A, M * K);
  randn_(B, N * K);

  // warmup
  for (int i = 0; i < 3; i++)
    f(A, B, C, M, N, K);

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; i++)
    f(A, B, C, M, N, K);
  auto t2 = std::chrono::high_resolution_clock::now();

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

  std::chrono::duration<float, std::milli> duration = t2 - t1;
  return duration.count() / n;
}


int main(int argc, char *argv[]) {
  // Apple M1 has 128KB L1 cache, and 12MB L2 cache
  // Ryzen 5600 has 384KB L1 cache, 3MB L2 cache, and 32MB L3 cache

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

  std::fill(C_ref, C_ref + M * N, 0.0f);
  naive_matmul(A, B, C_ref, M, N, K);  // ref impl

#ifdef BLAS_NAME
  std::fill(C, C + M * N, 0.0f);
  blas_matmul(A, B, C, M, N, K);
  check(C_ref, C, M, N);
#endif

  std::fill(C, C + M * N, 0.0f);
  tile_matmul<1, 4, 4, 1>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

  std::fill(C, C + M * N, 0.0f);
  tile_matmul<2, 4, 4, 1>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

  std::fill(C, C + M * N, 0.0f);
  tile_2level_matmul<8, 8, 4, 4, 2>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

#ifdef __ARM_NEON__
  std::fill(C, C + M * N, 0.0f);
  neon_matmul<16, 16, 8>(A, B, C, M, N, K);
  check(C_ref, C, M, N);
#endif

#ifdef BLAS_NAME
  printf("%s: %.2fms\n", BLAS_NAME, benchmark<blas_matmul>());
#endif
  printf("Naive matmul: %.2fms\n", benchmark<naive_matmul>(5));
#ifdef __APPLE__
  // tune on Apple M1
  printf("Tile matmul v1: %.2fms\n", benchmark<tile_matmul<1, 4, 4, 1>>());
  printf("Tile matmul v2: %.2fms\n", benchmark<tile_matmul<2, 4, 4, 1>>());
  printf("NEON matmul: %.2fms\n", benchmark<neon_matmul<16, 16, 8>>());
#else
  // tune on Ryzen 5600
  printf("Tile matmul v1: %.2fms\n", benchmark<tile_matmul<1, 4, 4, 4>>());
  printf("Tile matmul v2: %.2fms\n", benchmark<tile_matmul<2, 4, 4, 2>>());
  printf("2-level tile matmul: %.2fms\n", benchmark<tile_2level_matmul<16, 8, 4, 4, 2>>());
#endif

  return 0;
}
