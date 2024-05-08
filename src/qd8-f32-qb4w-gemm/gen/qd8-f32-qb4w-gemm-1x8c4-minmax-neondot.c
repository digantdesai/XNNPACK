// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/c4-neondot.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x8c4__neondot(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  float* c0 = c;

  size_t bl = params->scalar.blocksize;
  assert(bl <= kc);
  assert(bl != 0);
  size_t n_blocks = kc / bl;
  const int8x16_t vmask = vmovq_n_s8(INT8_C(0xF0));
  // Loop over groups of 8 columns.
  do {
    // Initialize accumulators with bias. 8 bias values are loaded from the
    // weight matrix, at the start of the group of 8 columns.
    const float32x4_t vinput_zero_point0 = vcvtq_f32_s32(vld1q_dup_s32(&quantization_params[0].zero_point));
    const float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x0123 = vmulq_f32(vksum0123, vinput_zero_point0);
    const float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vout0x4567 = vmulq_f32(vksum4567, vinput_zero_point0);

    for (size_t nb=0; nb < n_blocks; ++nb) {
      int32x4_t vacc0x0123 = vdupq_n_s32(0);
      int32x4_t vacc0x4567 = vdupq_n_s32(0);
    // Inner accumulation loop along the 8 columns.
    size_t k = bl;
    // 2x partial unrolled loop to load 8 bytes at a time.
    while (k >= 8 * sizeof(int8_t)) {
      // Load a 1x8 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 8;

      // Load a 8x8 block of weights.
      const int8x16_t vb01234567x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb01234567x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x0123 = vshlq_n_s8(vb01234567x0123, 4);
      const int8x16_t vb0123x4567 = vshlq_n_s8(vb01234567x4567, 4);
      const int8x16_t vb4567x0123 = vandq_s8(vb01234567x0123, vmask);
      const int8x16_t vb4567x4567 = vandq_s8(vb01234567x4567, vmask);

      // Multiply-accumulate: 1x8 * 8x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb4567x0123, va0x01234567, 1);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb4567x4567, va0x01234567, 1);

      k -= 8 * sizeof(int8_t);
    }
    // Handle up to 4 final positions of `k`
    if XNN_UNLIKELY(k != 0) {
      // Load a 1x4 block of activations.
      const int8x8_t va0x01234567 = vld1_s8(a0); a0 += 4;

      // Load a 4x8 block of weights.
      const int8x16_t vb0123x0123 = vld1q_s8(w); w = (const int8_t*) w + 16;
      const int8x16_t vb0123x4567 = vld1q_s8(w); w = (const int8_t*) w + 16;

      // Multiply-accumulate: 1x4 * 4x8 --> 1x8.
      vacc0x0123 = vdotq_lane_s32(vacc0x0123, vb0123x0123, va0x01234567, 0);
      vacc0x4567 = vdotq_lane_s32(vacc0x4567, vb0123x4567, va0x01234567, 0);
    }
    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;

    float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
    vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
    float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
    vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
    }

    const float32x4_t one_sixteenth = vdupq_n_f32(1/16.0);
    vout0x0123 = vmulq_f32(vout0x0123, one_sixteenth);
    vout0x4567 = vmulq_f32(vout0x4567, one_sixteenth);
    const float32x4_t vinput_scale0 = vld1q_dup_f32(&quantization_params[0].inv_scale);
    vout0x0123 = vmulq_f32(vout0x0123, vinput_scale0);
    vout0x4567 = vmulq_f32(vout0x4567, vinput_scale0);


    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x0123 = vaddq_f32(vbias0123, vout0x0123);
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;
    vout0x4567 = vaddq_f32(vbias4567, vout0x4567);

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      vst1q_f32(c0, vout0x0123);
      vst1q_f32(c0 + 4, vout0x4567);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
     if (nc & 4) {
       vst1q_f32(c0, vout0x0123); c0 += 4;
       vout0x0123 = vout0x4567;
     }
     float32x2_t vout0x01 = vget_low_f32(vout0x0123);
     if (nc & 2) {
       vst1_f32(c0, vout0x01); c0 += 2;
       vout0x01 = vget_high_f32(vout0x0123);
     }
     if (nc & 1) {
       vst1_lane_f32(c0, vout0x01, 0);
     }
      nc = 0;
    }
  } while (nc != 0);
}
