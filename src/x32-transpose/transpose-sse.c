// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "immintrin.h"

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

static size_t ukernel_step = 4;

static inline void transpose_32bit(const void *input, void *output, size_t ldh, size_t ldw) {
  __m128 v0 = _mm_loadu_ps((float*)input);
  __m128 v1 = _mm_loadu_ps((float*)input+ldw);
  __m128 v2 = _mm_loadu_ps((float*)input+2 * ldw);
  __m128 v3 = _mm_loadu_ps((float*)input+3 * ldw);
  _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
  _mm_storeu_ps((float*)output, v0);
  _mm_storeu_ps((float*)output+ldh, v1);
  _mm_storeu_ps((float*)output+2 * ldh, v2);
  _mm_storeu_ps((float*)output+3 * ldh, v3);
}

void xnn_x32_transpose_ukernel__sse(const void *input, void * output, size_t offset, size_t h, size_t w, size_t h_block_size, size_t w_block_size, const void *params){
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
