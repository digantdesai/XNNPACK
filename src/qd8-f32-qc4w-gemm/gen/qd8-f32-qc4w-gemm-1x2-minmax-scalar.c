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


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  kc = round_up_po2(kc, 2);
  do {
    const int32_t vksum0 = unaligned_indexed_load_s32(w, 0);
    const int32_t vksum1 = unaligned_indexed_load_s32(w, 1);
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    w = (const int32_t*) w + 2;

    size_t k = kc;
    for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
      const int32_t va0c0 = (int32_t) a0[0];
      const int32_t va0c1 = (int32_t) a0[1];
      a0 += 2;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      w = (const uint8_t*) w + 2;
      const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
      const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
      const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
      const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);

      vacc0x0 += va0c0 * vb0c0;
      vacc0x1 += va0c0 * vb1c0;
      vacc0x0 += va0c1 * vb0c1;
      vacc0x1 += va0c1 * vb1c1;
    }

    float vout0x0 = (float) math_asr_s32(vacc0x0, 4);
    float vout0x1 = (float) math_asr_s32(vacc0x1, 4);

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

void xnn_qd8_f32_qc4w_bl_gemm_minmax_ukernel_1x2__scalar(
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
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  printf("Running kernel %s: mr=%zu, nc=%zu, kc=%zu, bl=%zu\n", __func__, mr, nc, kc, bl);

  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  size_t nc_block = 0;

  kc = round_up_po2(kc, 2);
  printf("___ kernel ___");
  do {
    printf("\nnc_block=%zu, w: %p\n", nc_block++, w);
    const float vksum0 = unaligned_indexed_load_f32(w, 0);
    const float vksum1 = unaligned_indexed_load_f32(w, 1);
    printf("kernel: kfsum: 0: %f, 1: %f\n", vksum0, vksum1);
    w = (const float*) w + 2;
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;

    float vout0x0 = 0.0f;
    float vout0x1 = 0.0f;

    printf("out0: %f, out1: %f\n", vout0x0, vout0x1);
    printf("izp0: %d, vksum0: %f, vksum0 * izp0: %f\n", vinput_zero_point0, vksum0, (float) vinput_zero_point0 * vksum0);
    printf("izp0: %d, vksum1: %f, vksum1 * izp0: %f\n", vinput_zero_point0, vksum1, (float) vinput_zero_point0 * vksum1);
    vout0x0 += (float) vinput_zero_point0 * vksum0;
    vout0x1 += (float) vinput_zero_point0 * vksum1;

    size_t n_blocks = kc / bl;

    for (size_t nb=0; nb<n_blocks; ++nb){
      int32_t vacc0x0 = 0;
      int32_t vacc0x1 = 0;
      printf("\nnb=%zu, w: %p\n", nb, w);
      for (size_t k=bl; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
        printf("k=%zu, w: %p\n", k, w);
        const int32_t va0c0 = (int32_t) a0[0];
        const int32_t va0c1 = (int32_t) a0[1];
        a0 += 2;
        printf("va0c0: %d, va0c1: %d\n", va0c0, va0c1);

        const uint8_t vbi0 = ((const uint8_t*) w)[0];
        const uint8_t vbi1 = ((const uint8_t*) w)[1];
        printf("w:%p, vbi0=%4d, vbi1=%4d, ", (void*)w, (int32_t)vbi0, (int32_t)vbi1);
        w = (const uint8_t*) w + 2;

        const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
        const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
        const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
        const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);
        printf("vb0c0=%4d, vb0c1=%4d, vb1c0=%4d, vb1c1=%4d\n", vb0c0, vb0c1, vb1c0, vb1c1);

        vacc0x0 += va0c0 * vb0c0;
        vacc0x1 += va0c0 * vb1c0;
        vacc0x0 += va0c1 * vb0c1;
        vacc0x1 += va0c1 * vb1c1;
      }
      printf("int acc0x0: %d, acc0x1: %d\n", vacc0x0, vacc0x1);
      float vf0x0 = vacc0x0;
      float vf0x1 = vacc0x1;

      const float vfilter_output_scale0 = unaligned_indexed_load_f32(w, 0);
      printf("w: %p, scale0=%4.2f\n", (void*)w, vfilter_output_scale0);
      w = (const float*) w + 1;
      vf0x0 *= vfilter_output_scale0;

      const float vfilter_output_scale1 = unaligned_indexed_load_f32(w, 0);
      printf("w: %p, scale1=%4.2f\n", (void*)w, vfilter_output_scale1);
      w = (const float*) w + 1;
      vf0x1 *= vfilter_output_scale1;

      vout0x0 += vf0x0;
      vout0x1 += vf0x1;

      printf("acc0: %d, acc1: %d\n", vacc0x0, vacc0x1);
      vacc0x0 = 0;
      vacc0x1 = 0;
    }

    printf("out0: %f, out1: %f\n", vout0x0, vout0x1);
    vout0x0 /= 16;
    vout0x1 /= 16;
    printf("out0: %f, out1: %f\n", vout0x0, vout0x1);

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    printf("out0: %f, out1: %f, inv_scale: %f\n", vout0x0, vout0x1, vinput_scale0);

    const float vbias0 = unaligned_indexed_load_f32(w, 0);
    vout0x0 += vbias0;
    const float vbias1 = unaligned_indexed_load_f32(w, 1);
    vout0x1 += vbias1;
    printf("w: %p, bias0: %f, bias1: %f\n", w, vbias0, vbias1);
    w = (const float*) w + 2;

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
