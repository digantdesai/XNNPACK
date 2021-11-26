// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

static size_t ukernel_step = 4;

static inline void ukernel_4x4_32bit(v128_t *v0, v128_t *v1, v128_t *v2, v128_t *v3){
  const v128_t tmp0 = wasm_v32x4_shuffle(*v0, *v1, 0, 4, 1, 5);
  const v128_t tmp1 = wasm_v32x4_shuffle(*v2, *v3, 0, 4, 1, 5);
  const v128_t tmp2 = wasm_v32x4_shuffle(*v0, *v1, 2, 6, 3, 7);
  const v128_t tmp3 = wasm_v32x4_shuffle(*v2, *v3, 2, 6, 3, 7);

  *v0 = wasm_v32x4_shuffle(tmp0, tmp1, 0, 1, 4, 5);
  *v1 = wasm_v32x4_shuffle(tmp0, tmp1, 2, 3, 6, 7);
  *v2 = wasm_v32x4_shuffle(tmp2, tmp3, 0, 1, 4, 5);
  *v3 = wasm_v32x4_shuffle(tmp2, tmp3, 2, 3, 6, 7);
}

static inline void transpose_32bit(void *input, void *output, int h, int w) {
  v128_t v0 = wasm_v128_load((float*)input);
  v128_t v1 = wasm_v128_load((float*)input + w);
  v128_t v2 = wasm_v128_load((float*)input + 2 * w);
  v128_t v3 = wasm_v128_load((float*)input + 3 * w);
  ukernel_4x4_32bit(&v0, &v1, &v2, &v3);
  wasm_v128_store((int*)output, v0);
  wasm_v128_store((int*)output + h, v1);
  wasm_v128_store((int*)output + 2 * h, v2);
  wasm_v128_store((int*)output + 3 * h, v3);
}

void xnn_x32_transpose_ukernel__wasmsimd(const void *input, void * output, size_t offset, size_t h, size_t w, size_t h_block_size, size_t w_block_size, const void *params){
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
