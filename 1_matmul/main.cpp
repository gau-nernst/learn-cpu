#include "matmul.h"
#include <random>
#include <stdio.h>
#include <chrono>

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define BLAS_NAME "Apple Accelerate"
#elif __has_include(<mkl_cblas.h>)
#include "mkl_cblas.h"
#define BLAS_NAME "Intel MKL"
#endif


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

template <typename F>
float benchmark(F f, int n = 100) {
  f();  // warmup

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < n; i++)
    f();
  auto t2 = std::chrono::high_resolution_clock::now();

  std::chrono::duration<float, std::milli> duration = t2 - t1;
  return duration.count() / n;
}


int main(int argc, char *argv[]) {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;

  // float *A = new float[M * K];
  // float *B = new float[N * K];
  // float *C_ref = new float[M * N];
  // float *C = new float[M * N];

  // faster in some cases
  const int alignment = 128;
  float *A = reinterpret_cast<float *>(aligned_alloc(alignment, M * K * sizeof(float)));
  float *B = reinterpret_cast<float *>(aligned_alloc(alignment, N * K * sizeof(float)));
  float *C_ref = reinterpret_cast<float *>(aligned_alloc(alignment, M * N * sizeof(float)));
  float *C = reinterpret_cast<float *>(aligned_alloc(alignment, M * N * sizeof(float)));

  randn_(A, M * K);
  randn_(B, N * K);

  std::fill(C_ref, C_ref + M * N, 0.0f);
  naive_matmul(A, B, C_ref, M, N, K);  // ref impl

#ifdef BLAS_NAME
  std::fill(C, C + M * N, 0.0f);
  cblas_sgemm(CblasRowMajor,  // layout
              CblasNoTrans,  // transpose A
              CblasTrans,  // tranpose B
              M, N, K, 1.0f,
              A, M,
              B, N,
              0.0f, C, M);
  check(C_ref, C, M, N);
#endif

  std::fill(C, C + M * N, 0.0f);
  tile_matmul<1, 4, 4, 1>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

  std::fill(C, C + M * N, 0.0f);
  tile_matmul<2, 4, 4, 1>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

  std::fill(C, C + M * N, 0.0f);
  tile_2level_matmul<4, 4, 1, 4, 4, 1>(A, B, C, M, N, K);
  check(C_ref, C, M, N);

#ifdef __ARM_NEON__
  std::fill(C, C + M * N, 0.0f);
  neon_matmul<16, 16, 8>(A, B, C, M, N, K);
  check(C_ref, C, M, N);
#endif

#ifdef BLAS_NAME
  printf("%s: %.2fms\n", BLAS_NAME,
         benchmark([A, B, C]() {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, A, M, B, N,
                        0.0f, C, M);
         }));
#endif
  printf("Naive matmul: %.2fms\n",
         benchmark([A, B, C]() {
          naive_matmul(A, B, C, M, N, K);
         }, 5));

#ifdef __APPLE__
  // tune on Apple M1
  printf("Tile matmul v1: %.2fms\n",
         benchmark([A, B, C]() {
          tile_matmul<1, 4, 4, 1>(A, B, C, M, N, K);
         }));
  printf("Tile matmul v2: %.2fms\n",
         benchmark([A, B, C]() {
          tile_matmul<2, 4, 4, 1>(A, B, C, M, N, K);
         }));
  printf("NEON matmul: %.2fms\n",
         benchmark([A, B, C]() {
          neon_matmul<16, 16, 8>(A, B, C, M, N, K);
         }));

#else
  // tune on Ryzen 5600
  printf("Tile matmul v1: %.2fms\n",
         benchmark([A, B, C]() {
          tile_matmul<1, 4, 4, 4>(A, B, C, M, N, K);
         }));
  printf("Tile matmul v2: %.2fms\n",
         benchmark([A, B, C]() {
          tile_matmul<2, 4, 4, 2>(A, B, C, M, N, K);
         }));
#endif

  return 0;
}
