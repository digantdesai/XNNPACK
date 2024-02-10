// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>

#include <stdio.h>


void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  printf("Running kernel %s\n", __func__);
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  do {
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    w = (const int32_t*) w + 2;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;

    const float vfilter_output_scale0 = unaligned_indexed_load_f32(w, 0);
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = unaligned_indexed_load_f32(w, 1);
    vout0x1 *= vfilter_output_scale1;

    const float vbias0 = unaligned_indexed_load_f32(w, 2);
    vout0x0 += vbias0;
    const float vbias1 = unaligned_indexed_load_f32(w, 3);
    vout0x1 += vbias1;

    w = (const float*) w + 4;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_qd8_f32_qc8w_bl_gemm_minmax_ukernel_1x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t bl,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  // xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar(mr, nc, kc, a, a_stride, w, c, cm_stride, cn_stride, params, quantization_params);
  // return;

  printf("Running kernel %s: mr=%zu, nc=%zu, kc=%zu, bl=%zu\n", __func__, mr, nc, kc, bl);

  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  size_t nc_block = 0;
  do {
    printf("\nnc_block=%zu, w: %p\n", nc_block++, w);
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;

    printf("w: %p, vksum0=%4d, vksum1:%4d\n", (void*)w, vksum0, vksum1);

    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    w = (const int32_t*) w + 2;

    size_t n_blocks = kc / bl;

    float vout0x0 = 0.0f;
    float vout0x1 = 0.0f;
    for(size_t nb=0; nb<n_blocks; ++nb) {
      printf("\nnb=%zu, w: %p\n", nb, w);
      for(size_t k=0; k < bl; k += sizeof(int8_t)) {
        printf("k=%zu, w: %p\n", k, w);
        const int32_t va0 = (int32_t) *a0++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        printf("w:%p, vb0=%4d, vb1=%4d\n", (void*)w, vb0, vb1);
        w = (const int8_t*) w + 2;

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
      }
      float vf0x0 = (float) vacc0x0;
      float vf0x1 = (float) vacc0x1;

      const float vfilter_output_scale0 = 1.5; // unaligned_indexed_load_f32(w, 0); w = (const float*) w + 1;
      printf("w: %p, scale0=%4.2f\n", (void*)w, vfilter_output_scale0);
      w = (const float*) w + 1;
      vf0x0 *= vfilter_output_scale0;

      const float vfilter_output_scale1 = 1.5; // unaligned_indexed_load_f32(w, 1); w = (const float*) w + 1;
      printf("w: %p, scale1=%4.2f\n", (void*)w, vfilter_output_scale1);
      w = (const float*) w + 1;
      vf0x1 *= vfilter_output_scale1;

      vout0x0 += vf0x0;
      vout0x1 += vf0x1;

      vacc0x0 = 0;
      vacc0x1 = 0;
    }

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;

    // const float vbias0 = unaligned_indexed_load_f32(w, 0);
    // vout0x0 += vbias0;
    // const float vbias1 = unaligned_indexed_load_f32(w, 1);
    // vout0x1 += vbias1;

    // printf("w: %p, bias0=%4.2f, bias1:%4.2f\n", (void*)w, vbias0, vbias1);
    // w = (const float*) w + 2;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
