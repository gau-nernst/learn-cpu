#pragma once

#include "assert.h"
#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define BLAS_NAME "Apple Accelerate"
#elif __has_include(<mkl_cblas.h>)
#include "mkl_cblas.h"
#define BLAS_NAME "Intel MKL"
#endif

#ifdef BLAS_NAME
// wrapper to provide unified signature
void blas_matmul(const float * __restrict A,
                 const float * __restrict B,
                       float * __restrict C,
                 int M, int N, int K) {
  cblas_sgemm(CblasRowMajor,  // layout
              CblasNoTrans,   // transpose A
              CblasTrans,     // transpose B
              M, N, K, 1.0f,
              A, M,
              B, N,
              0.0f, C, M);
}
#endif

void naive_matmul(const float * __restrict A,
                  const float * __restrict B,
                        float * __restrict C,
                  int M, int N, int K) {
  // assume A is row-major
  //        B is column-major
  //        C is row-major
  // loading A and B both utilize memory well (contiguous in memory)
#pragma omp parallel for schedule(static,1)
  for (int im = 0; im < M; im++)
    for (int in = 0; in < N; in++) {
      float acc = 0.0f;
      for (int ik = 0; ik < K; ik++)
        acc += A[im * K + ik] * B[in * K + ik];
      C[im * N + in] = acc;
    }
}

template <int VERSION, int TILE_M, int TILE_N, int TILE_K>
void register_tile_matmul(const float * __restrict A,
                          const float * __restrict B,
                                float * __restrict C,
                          int M, int N, int K) {
  // divide output into tiles of (TILE_M, TILE_N).
  // each output tile (TILE_M, TILE_N) = (TILE_M, K) x (K, TILE_N)
  // can be computed by each thread independently.
#pragma omp parallel for schedule(static,1)
  for (int tile_m = 0; tile_m < M; tile_m += TILE_M) {
    for (int tile_n = 0; tile_n < N; tile_n += TILE_N) {

      // TILE_M and TILE_N must be sufficiently small to hold
      // (TILE_M, TILE_N) accumulator in registers.
      float acc[TILE_M][TILE_N] = {{0.0f}};

      // divide K into tiles of TILE_K
      for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        if constexpr (VERSION == 1) {
          for (int m = 0; m < TILE_M; m++)
            for (int n = 0; n < TILE_N; n++) {
              float A_reg[TILE_K], B_reg[TILE_K];

              // load A and B to registers
              for (int k = 0; k < TILE_K; k++) {
                A_reg[k] = A[(tile_m + m) * K + (tile_k + k)];
                B_reg[k] = B[(tile_n + n) * K + (tile_k + k)];
              }

              // dot product
              for (int k = 0; k < TILE_K; k++)
                acc[m][n] += A_reg[k] * B_reg[k];
            }
        }  // VERSION 1

        else if constexpr (VERSION == 2) {
          float A_reg[TILE_M][TILE_K], B_reg[TILE_K][TILE_N];

          // load A and B to registers
          for (int m = 0; m < TILE_M; m++)
            for (int k = 0; k < TILE_K; k++)
              A_reg[m][k] = A[(tile_m + m) * K + (tile_k + k)];

          for (int n = 0; n < TILE_N; n++)
            for (int k = 0; k < TILE_K; k++)
              B_reg[k][n] = B[(tile_n + n) * K + (tile_k + k)];

          // outer product
          // this has better performance since for each (TILE_M,1) and (TILE_N,1)
          // cached in L1, we can do more math ops (TILE_M x TILE_N to be exact).
          // -> better arithmetic intensity.
          for (int k = 0; k < TILE_K; k++)
            for (int m = 0; m < TILE_M; m++)
              for (int n = 0; n < TILE_N; n++)
                acc[m][n] += A_reg[m][k] * B_reg[k][n];
        }  // VERSION 2

        else {
          assert(false);
        }

      } // TILE_K

      // write output tile (TILE_M, TILE_N)
      for (int m = 0; m < TILE_M; m++)
        for (int n = 0; n < TILE_N; n++)
          C[(tile_m + m) * N + (tile_n + n)] = acc[m][n];
    }  // TILE_N
  }  // TILE_M
}

template <int TILE_M, int TILE_N, int TILE_K>
void l1_tile_matmul(const float * __restrict A,
                    const float * __restrict B,
                          float * __restrict C,
                    int M, int N, int K) {
#pragma omp parallel for collapse(2) schedule(static,1)
  for (int tile_m = 0; tile_m < M; tile_m += TILE_M) {
    for (int tile_n = 0; tile_n < N; tile_n += TILE_N) {
      // we want all tiles A_tile[TILE_M][TILE_K], B_tile[TILE_N][TILE_K],
      // and C_tile[TILE_M][TILE_N] to fit in L1 cache.

      // this kernel is only different from register_tile kernel
      // in the absence of explicitly loading A_reg[TILE_K] and B_reg[TILE_K].

      const float *A_tile = A + tile_m * K;
      const float *B_tile = B + tile_n * K;

      // when TILE_M and TILE_N are large, C_tile will go to L1 cache.
      float C_tile[TILE_M][TILE_N] = {{0.0f}};

      for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {
        for (int m = 0; m < TILE_M; m++)
          for (int n = 0; n < TILE_N; n++)
            for (int k = 0; k < TILE_K; k++)
              C_tile[m][n] += A_tile[m * K + k] * B_tile[n * K + k];

        A_tile += TILE_K;
        B_tile += TILE_K;
      }

      for (int m = 0; m < TILE_M; m++)
        for (int n = 0; n < TILE_N; n++)
          C[(tile_m + m) * N + (tile_n + n)] = C_tile[m][n];
    }
  }
}

template <int TILE_M1, int TILE_N1, int TILE_K1,
          int TILE_M2, int TILE_N2>
void l1_register_tile_matmul(const float * __restrict A,
                             const float * __restrict B,
                                   float * __restrict C,
                             int M, int N, int K) {
  static_assert(TILE_M1 % TILE_M2 == 0);
  static_assert(TILE_N1 % TILE_N2 == 0);

  // TILE_M1, TILE_N1, TILE_K1 are for L1 cache
  // TILE_M2, TILE_N2, TILE_K2 are for register cache

#pragma omp parallel for collapse(2) schedule(static,1)
  for (int tile_m1 = 0; tile_m1 < M; tile_m1 += TILE_M1) {
    for (int tile_n1 = 0; tile_n1 < N; tile_n1 += TILE_N1) {
      // these will go to L1 cache
      const float *A_tile1 = A + tile_m1 * K;
      const float *B_tile1 = B + tile_n1 * K;
      float acc[TILE_M1][TILE_N1] = {{0.0f}};

      for (int tile_k1 = 0; tile_k1 < K; tile_k1 += TILE_K1) {
        // for each of this TILE_K1 iteration, we will load
        // A[TILE_M1][TILE_K1] and B[TILE_N1][TILE_K1] from
        // RAM into L1.

        // NOTE: doing explicit A, B, C registers actually make it slower.
        // it probably affect autovectorization codegen.
        for (int tile_m2 = 0; tile_m2 < TILE_M1; tile_m2 += TILE_M2)
          for (int tile_n2 = 0; tile_n2 < TILE_N1; tile_n2 += TILE_N2)
            // load from L1 to registers.
            // K is outer loop to improve AI
            for (int k = 0; k < TILE_K1; k++)
              for (int m = 0; m < TILE_M2; m++)
                for (int n = 0; n < TILE_N2; n++)
                  acc[tile_m2 + m][tile_n2 + n] += A_tile1[(tile_m2 + m) * K + k] *
                                                   B_tile1[(tile_n2 + n) * K + k];

        A_tile1 += TILE_K1;
        B_tile1 += TILE_K1;
      }  // TILE_K1

      for (int m = 0; m < TILE_M1; m++)
        for (int n = 0; n < TILE_N1; n++)
          C[(tile_m1 + m) * N + (tile_n1 + n)] = acc[m][n];

    }  // TILE_N1
  }  // TILE_M1
}

#ifdef __ARM_NEON__
// https://developer.arm.com/documentation/102467/0201/Example---matrix-multiplication
template <bool transpose_B>
void neon_mma_m4n4k4(const float * __restrict A,
                     const float * __restrict B,
                     int A_row_stride,
                     int B_row_stride,
                     float32x4_t acc[4])
{
  float32x4x4_t A_reg, B_reg;

  // load (4,4) tile of A and B each
  for (int m = 0; m < 4; m++)
    A_reg.val[m] = vld1q_f32(A + m * A_row_stride);

  if constexpr (transpose_B) {
    // can't use for-loop here since `lane` (last argument) is required to be a constant integer
    // (and the compiler can't unroll + infer it to be constant automatically...)
    B_reg = vld4q_lane_f32(B + 0 * B_row_stride, B_reg, 0);
    B_reg = vld4q_lane_f32(B + 1 * B_row_stride, B_reg, 1);
    B_reg = vld4q_lane_f32(B + 2 * B_row_stride, B_reg, 2);
    B_reg = vld4q_lane_f32(B + 3 * B_row_stride, B_reg, 3);
  } else {
    for (int k = 0; k < 4; k++)
      B_reg.val[k] = vld1q_f32(B + k * B_row_stride);
  }

  for (int m = 0; m < 4; m++) {
    // this is iterating along k-dim
    acc[m] = vfmaq_laneq_f32(acc[m], B_reg.val[0], A_reg.val[m], 0);
    acc[m] = vfmaq_laneq_f32(acc[m], B_reg.val[1], A_reg.val[m], 1);
    acc[m] = vfmaq_laneq_f32(acc[m], B_reg.val[2], A_reg.val[m], 2);
    acc[m] = vfmaq_laneq_f32(acc[m], B_reg.val[3], A_reg.val[m], 3);
  }
}

template <int TILE_M, int TILE_N, int TILE_K>
void neon_matmul(const float * __restrict A,
                 const float * __restrict B,
                       float * __restrict C,
                 int M, int N, int K) {
  // NEON registers are 128-bit (16-byte) -> FP32x4
  // TODO: re-investigate this. this is slower than auto-vectorization
  constexpr int MMA_M = 4;
  constexpr int MMA_N = 4;
  constexpr int MMA_K = 4;

  static_assert(TILE_M % MMA_M == 0);
  static_assert(TILE_N % MMA_N == 0);
  static_assert(TILE_K % MMA_K == 0);

#pragma omp parallel for
  for (int tile_m = 0; tile_m < M; tile_m += TILE_M) {
    for (int tile_n = 0; tile_n < N; tile_n += TILE_N) {
      const float *A_tile = A + tile_m * K;
      const float *B_tile = B + tile_n * K;

      float32x4_t acc[TILE_M / MMA_M][TILE_N / 4][4];
      for (int m = 0; m < TILE_M; m += 4)
        for (int n = 0; n < TILE_N; n += 4)
          for (int i = 0; i < 4; i++)
            acc[m / 4][n / 4][i] = vmovq_n_f32(0.0f);  // set to zeros

      for (int tile_k = 0; tile_k < K; tile_k += TILE_K) {

        for (int mma_k = 0; mma_k < TILE_K; mma_k += MMA_K)
          for (int mma_m = 0; mma_m < TILE_M; mma_m += MMA_M)
            for (int mma_n = 0; mma_n < TILE_N; mma_n += MMA_N)
              neon_mma_m4n4k4<true>(A_tile + mma_m * K + mma_k,
                                    B_tile + mma_n * K + mma_k,
                                    K, K,
                                    acc[mma_m / 4][mma_n / 4]);

        A_tile += TILE_K;
        B_tile += TILE_K;
      }

      // write output tile (TILE_M, TILE_N)
      for (int m = 0; m < TILE_M; m++)
        for (int n = 0; n < TILE_N; n += 4)
          vst1q_f32(C + (tile_m + m) * N + (tile_n + n),
                    acc[m / 4][n / 4][m % 4]);

    }  // TILE_N
  }  // TILE_N
}
#endif
