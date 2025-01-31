#include "matmul.h"
#include <random>
#include <stdio.h>
#include <chrono>

// TODO: only include this on macOS
#include <Accelerate/Accelerate.h>


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

  float *A = new float[M * K];
  float *B = new float[N * K];
  float *C_naive = new float[M * N];
  float *C_tile = new float[M * N];
  float *C_neon = new float[M * N];
  float *C_blas = new float[M * N];

  randn_(A, M * K);
  randn_(B, N * K);

  naive_matmul(A, B, C_naive, M, N, K);  // ref impl
  cblas_sgemm(CblasRowMajor,  // layout
              CblasNoTrans,  // transpose A
              CblasTrans,  // tranpose B
              M, N, K, 1.0f,
              A, M,
              B, N,
              0.0f, C_blas, M);
  tile_matmul<8, 8, 4>(A, B, C_tile, M, N, K);
  neon_matmul(A, B, C_neon, M, N, K);

  check(C_naive, C_blas, M, N);
  check(C_naive, C_tile, M, N);
  check(C_naive, C_neon, M, N);

  printf("Apple Accelerate: %.2fms\n",
         benchmark([A, B, C_blas]() {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K, 1.0f, A, M, B, N,
                        0.0f, C_blas, M);
         }));
  printf("Naive matmul: %.2fms\n",
         benchmark([A, B, C_naive]() {
          naive_matmul(A, B, C_naive, M, N, K);
         }, 5));
  // tune on Apple M1
  printf("Tile matmul: %.2fms\n",
         benchmark([A, B, C_tile]() {
          tile_matmul<8, 8, 4>(A, B, C_tile, M, N, K);
         }));
  printf("Neon matmul: %.2fms\n",
         benchmark([A, B, C_tile]() {
          neon_matmul(A, B, C_tile, M, N, K);
         }));

  return 0;
}
