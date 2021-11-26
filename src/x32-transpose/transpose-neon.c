// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

static size_t ukernel_step = 8;

static inline void ukernel_4x4_32bit(int32x4_t *v0, int32x4_t *v1, int32x4_t *v2, int32x4_t *v3){
  int32x4_t v02_l = vzip1q_s32(*v0, *v2);
  int32x4_t v02_r = vzip2q_s32(*v0, *v2);
  int32x4_t v13_l = vzip1q_s32(*v1, *v3);
  int32x4_t v13_r = vzip2q_s32(*v1, *v3);
  *v0 = vzip1q_s32(v02_l, v13_l);
  *v1 = vzip2q_s32(v02_l, v13_l);
  *v2 = vzip1q_s32(v02_r, v13_r);
  *v3 = vzip2q_s32(v02_r, v13_r);
}

void transpose_32bit_aarch64(void *input, void *output, int h, int w);

static inline void transpose_32bit(void *input, void *output, int h, int w) {
  int32x4_t v0 = vld1q_s32((int*)input);
  int32x4_t v1 = vld1q_s32((int*)input + w);
  int32x4_t v2 = vld1q_s32((int*)input + 2 * w);
  int32x4_t v3 = vld1q_s32((int*)input + 3 * w);
  ukernel_4x4_32bit(&v0, &v1, &v2, &v3);
  vst1q_s32((int*)output, v0);
  vst1q_s32((int*)output + h, v1);
  vst1q_s32((int*)output + 2 * h * 1, v2);
  vst1q_s32((int*)output + 3 * h * 1, v3);
}

void xnn_x32_transpose_ukernel__neon(const void *input, void * output, size_t offset, size_t h, size_t w, size_t h_block_size, size_t w_block_size, const void *params){
  float* in_ptr_i = (float*)input;
  float* out_ptr_i = (float*)output;
  size_t w_size = w_block_size;
  for (; w_size >= ukernel_step; w_size -= ukernel_step) {
    float* in_ptr_j = in_ptr_i;
    float* out_ptr_j = out_ptr_i;
    size_t h_size = h_block_size;
    for (; h_size >= ukernel_step; h_size -= ukernel_step) {
      transpose_32bit_aarch64(in_ptr_j, out_ptr_j, h, w);
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
