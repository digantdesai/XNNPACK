// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "immintrin.h"

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

static size_t ukernel_step = 8;

static inline void ukernel_8x8_32bit(__m256 *v0, __m256 *v1, __m256 *v2, __m256 *v3, __m256 *v4, __m256 *v5, __m256 *v6, __m256 *v7){
  __m256 t0 = _mm256_unpacklo_ps(*v0, *v1);
  __m256 t1 = _mm256_unpackhi_ps(*v0, *v1);
  __m256 t2 = _mm256_unpacklo_ps(*v2, *v3);
  __m256 t3 = _mm256_unpackhi_ps(*v2, *v3);
  __m256 t4 = _mm256_unpacklo_ps(*v4, *v5);
  __m256 t5 = _mm256_unpackhi_ps(*v4, *v5);
  __m256 t6 = _mm256_unpacklo_ps(*v6, *v7);
  __m256 t7 = _mm256_unpackhi_ps(*v6, *v7);
  __m256 tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
  __m256 tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
  __m256 tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
  *v0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
  *v1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
  *v2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
  *v3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
  *v4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
  *v5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
  *v6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
  *v7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

static inline void transpose_32bit(const void *input, void *output, size_t ldh, size_t ldw) {
  __m256 v0 = _mm256_loadu_ps((float*)input);
  __m256 v1 = _mm256_loadu_ps((float*)input+ldw);
  __m256 v2 = _mm256_loadu_ps((float*)input+2 * ldw);
  __m256 v3 = _mm256_loadu_ps((float*)input+3 * ldw);
  __m256 v4 = _mm256_loadu_ps((float*)input+4 * ldw);
  __m256 v5 = _mm256_loadu_ps((float*)input+5 * ldw);
  __m256 v6 = _mm256_loadu_ps((float*)input+6 * ldw);
  __m256 v7 = _mm256_loadu_ps((float*)input+7 * ldw);
  ukernel_8x8_32bit(&v0, &v1, &v2, &v3, &v4, &v5, &v6, &v7);
  _mm256_storeu_ps((float*)output, v0);
  _mm256_storeu_ps((float*)output+ldh, v1);
  _mm256_storeu_ps((float*)output+2 * ldh, v2);
  _mm256_storeu_ps((float*)output+3 * ldh, v3);
  _mm256_storeu_ps((float*)output+4 * ldh, v4);
  _mm256_storeu_ps((float*)output+5 * ldh, v5);
  _mm256_storeu_ps((float*)output+6 * ldh, v6);
  _mm256_storeu_ps((float*)output+7 * ldh, v7);
}

void xnn_x32_transpose_ukernel__avx(const void *input, void * output, size_t offset, size_t h, size_t w, size_t h_block_size, size_t w_block_size, const void *params){
  float* in_ptr_i = (float*)input;
  float* out_ptr_i = (float*)output;
  size_t w_size = w_block_size;
  for (; w_size >= ukernel_step; w_size -= ukernel_step) {
    float* in_ptr_j = in_ptr_i;
    float* out_ptr_j = out_ptr_i;
    size_t h_size = h_block_size;
    for (; h_size >= ukernel_step; h_size -= ukernel_step) {
      transpose_32bit(in_ptr_j, out_ptr_j, h, w);
      in_ptr_j += w * ukernel_step;
      out_ptr_j += ukernel_step;
    }
    if XNN_UNLIKELY(h_size != 0) {
      for (; h_size >= 1; --h_size){
        out_ptr_j[0] = in_ptr_j[0];
        out_ptr_j[h] = in_ptr_j[1];
        out_ptr_j[2 * h] = in_ptr_j[2];
        out_ptr_j[3 * h] = in_ptr_j[3];
        out_ptr_j[4 * h] = in_ptr_j[4];
        out_ptr_j[5 * h] = in_ptr_j[5];
        out_ptr_j[6 * h] = in_ptr_j[6];
        out_ptr_j[7 * h] = in_ptr_j[7];
        in_ptr_j += w;
        out_ptr_j += 1;
      }
    }
    in_ptr_i += offset * ukernel_step;
    out_ptr_i += h * ukernel_step;
  }
  if XNN_UNLIKELY(w_size != 0) {
    for (; w_size >= 1; --w_size){
      float* in_ptr_j = in_ptr_i;
      float* out_ptr_j = out_ptr_i;
      size_t h_size = h_block_size;
      for (; h_size >= 1; --h_size) {
        *out_ptr_j = *in_ptr_j;
        in_ptr_j += w;
        out_ptr_j += 1;
      }
      in_ptr_i += offset * 1;
      out_ptr_i += h;
    }
  }
}
