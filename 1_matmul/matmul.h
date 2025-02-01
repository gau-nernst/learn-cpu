#pragma once

#include <arm_neon.h>


void naive_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // assume A is row-major
  //        B is column-major
  //        C is row-major
  for (int im = 0; im < M; im++)
    for (int in = 0; in < N; in++) {
      float acc = 0.0f;
      for (int ik = 0; ik < K; ik++)
        acc += A[im * K + ik] * B[in * K + ik];
      C[im * N + in] = acc;
    }
}

template <int VERSION, int BLOCK_M, int BLOCK_N, int BLOCK_K>
void tile_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int block_m = 0; block_m < M; block_m += BLOCK_M) {
    for (int block_n = 0; block_n < N; block_n += BLOCK_N) {
      float acc[BLOCK_M][BLOCK_N] = {0.0f};

      for (int block_k = 0; block_k < K; block_k += BLOCK_K) {
        if constexpr (VERSION == 1) {
          for (int idx_m = 0; idx_m < BLOCK_M; idx_m++)
            for (int idx_n = 0; idx_n < BLOCK_N; idx_n++) {
              float A_reg[BLOCK_K], B_reg[BLOCK_K];

              // load A and B to registers
              for (int idx_k = 0; idx_k < BLOCK_K; idx_k++) {
                A_reg[idx_k] = A[(block_m + idx_m) * K + (block_k + idx_k)];
                B_reg[idx_k] = B[(block_n + idx_n) * K + (block_k + idx_k)];
              }

              // dot product
              for (int idx_k = 0; idx_k < BLOCK_K; idx_k++)
                acc[idx_m][idx_n] += A_reg[idx_k] * B_reg[idx_k];
            }
        }  // VERSION 0

        else if constexpr (VERSION == 2) {
          float A_reg[BLOCK_M][BLOCK_K], B_reg[BLOCK_K][BLOCK_N];

          // load A and B to registers
          for (int idx_m = 0; idx_m < BLOCK_M; idx_m++)
            for (int idx_k = 0; idx_k < BLOCK_K; idx_k++)
              A_reg[idx_m][idx_k] = A[(block_m + idx_m) * K + (block_k + idx_k)];
          
          for (int idx_n = 0; idx_n < BLOCK_N; idx_n++)
            for (int idx_k = 0; idx_k < BLOCK_K; idx_k++)
              B_reg[idx_k][idx_n] = B[(block_n + idx_n) * K + (block_k + idx_k)];

          // outer product
          for (int idx_k = 0; idx_k < BLOCK_K; idx_k++)
            for (int idx_m = 0; idx_m < BLOCK_M; idx_m++)
              for (int idx_n = 0; idx_n < BLOCK_N; idx_n++)
                acc[idx_m][idx_n] += A_reg[idx_m][idx_k] * B_reg[idx_k][idx_n];
        }  // VERSION 1
        else {
          static_assert(!sizeof(VERSION));
        }

      } // BLOCK_K loop

      // write output tile (BLOCK_M, BLOCK_N)
      for (int idx_m = 0; idx_m < BLOCK_M; idx_m++)
        for (int idx_n = 0; idx_n < BLOCK_N; idx_n++)
          C[(block_m + idx_m) * N + (block_n + idx_n)] = acc[idx_m][idx_n];
    }  // BLOCK_N loop
  }  // BLOCK_M loop
}

// https://developer.arm.com/documentation/102467/0201/Example---matrix-multiplication
template <bool transpose_B>
void neon_mma_m4n4k4(const float *A,
                     const float *B,
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

void neon_matmul(const float *A, const float *B, float *C, int M, int N, int K) {
  // NEON registers are 128-bit (16-byte) -> 4 elements of FP32
  const int BLOCK_M = 4;
  const int BLOCK_N = 4;
  const int BLOCK_K = 4;

  for (int block_m = 0; block_m < M; block_m += BLOCK_M) {
    for (int block_n = 0; block_n < N; block_n += BLOCK_N) {
      // we will do MMA with m4n4k4
      float32x4_t acc[4];
      for (int m = 0; m < 4; m++)
        acc[m] = vmovq_n_f32(0.0f);  // set to zeros

      for (int block_k = 0; block_k < K; block_k += BLOCK_K)
        neon_mma_m4n4k4<true>(A + block_m * K + block_k,
                              B + block_n * K + block_k,
                              K, K, acc);

      // write output tile (BLOCK_M, BLOCK_N)
      for (int m = 0; m < 4; m++)
        vst1q_f32(C + (block_m + m) * N + block_n, acc[m]);
    }  // BLOCK_N loop
  }  // BLOCK_M loop
}
