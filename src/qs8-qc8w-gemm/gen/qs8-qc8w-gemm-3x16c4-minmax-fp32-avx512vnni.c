// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/MRx16c4-avx512vnni.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/gemm.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__avx512vnni(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  kc = round_up_po2(kc, 4 * sizeof(int8_t));
  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

  const __m512i vsign_mask =_mm512_set1_epi8(params->fp32_avx512vnni.sign_mask);  // 0x80
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->fp32_avx512vnni.output_max_less_zero_point);
  const __m256i voutput_zero_point = _mm256_load_si256((const __m256i*) params->fp32_avx512vnni.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_avx512vnni.output_min);
  do {
    __m512i vacc0x0x0123456789ABCDEF = _mm512_load_epi32(w);
    __m512i vacc1x0x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc0x1x0123456789ABCDEF = vacc0x0x0123456789ABCDEF;
    __m512i vacc1x1x0123456789ABCDEF = _mm512_setzero_epi32();
    __m512i vacc0x2x0123456789ABCDEF = vacc0x0x0123456789ABCDEF;
    __m512i vacc1x2x0123456789ABCDEF = _mm512_setzero_epi32();
    w = (const int32_t*) w + 16;

    size_t k = kc;
    while (k >= 8 * sizeof(int8_t)) {
      const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
      const __m512i va0x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0 + 4)), vsign_mask);
      a0 += 8;
      const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
      const __m512i va1x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1 + 4)), vsign_mask);
      a1 += 8;
      const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
      const __m512i va2x4567 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2 + 4)), vsign_mask);
      a2 += 8;

      const __m512i vb0123456789ABCDEFx0123 = _mm512_load_si512(w);
      const __m512i vb0123456789ABCDEFx4567 = _mm512_load_si512((const int8_t*) w + 64);

      vacc0x0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEFx0123);
      vacc0x1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEFx0123);
      vacc0x2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEFx0123);
      vacc1x0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0x0123456789ABCDEF, va0x4567, vb0123456789ABCDEFx4567);
      vacc1x1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x1x0123456789ABCDEF, va1x4567, vb0123456789ABCDEFx4567);
      vacc1x2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x2x0123456789ABCDEF, va2x4567, vb0123456789ABCDEFx4567);

      w = (const int8_t*) w + 128;
      k -= 8 * sizeof(int8_t);
    }
    __m512i vacc0x0123456789ABCDEF = _mm512_add_epi32(vacc0x0x0123456789ABCDEF, vacc1x0x0123456789ABCDEF);
    __m512i vacc1x0123456789ABCDEF = _mm512_add_epi32(vacc0x1x0123456789ABCDEF, vacc1x1x0123456789ABCDEF);
    __m512i vacc2x0123456789ABCDEF = _mm512_add_epi32(vacc0x2x0123456789ABCDEF, vacc1x2x0123456789ABCDEF);

    if (k != 0) {
      const __m512i va0x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a0)), vsign_mask);
      a0 += 4;
      const __m512i va1x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a1)), vsign_mask);
      a1 += 4;
      const __m512i va2x0123 = _mm512_xor_epi32(_mm512_set1_epi32((int) unaligned_load_u32(a2)), vsign_mask);
      a2 += 4;

      const __m512i vb0123456789ABCDEF = _mm512_load_si512(w);

      vacc0x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc0x0123456789ABCDEF, va0x0123, vb0123456789ABCDEF);
      vacc1x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc1x0123456789ABCDEF, va1x0123, vb0123456789ABCDEF);
      vacc2x0123456789ABCDEF = _mm512_dpbusd_epi32(vacc2x0123456789ABCDEF, va2x0123, vb0123456789ABCDEF);

      w = (const int8_t*) w + 64;
      k -= 4 * sizeof(int8_t);
    }

    __m512 vscaled0x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc0x0123456789ABCDEF);
    __m512 vscaled1x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc1x0123456789ABCDEF);
    __m512 vscaled2x0123456789ABCDEF = _mm512_cvtepi32_ps(vacc2x0123456789ABCDEF);

    const __m512 vscale012345678ABCDEF = _mm512_load_ps(w);
    w = (const float*) w + 16;
    vscaled0x0123456789ABCDEF = _mm512_mul_ps(vscaled0x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled1x0123456789ABCDEF = _mm512_mul_ps(vscaled1x0123456789ABCDEF, vscale012345678ABCDEF);
    vscaled2x0123456789ABCDEF = _mm512_mul_ps(vscaled2x0123456789ABCDEF, vscale012345678ABCDEF);

    vscaled0x0123456789ABCDEF = _mm512_min_ps(vscaled0x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled1x0123456789ABCDEF = _mm512_min_ps(vscaled1x0123456789ABCDEF, voutput_max_less_zero_point);
    vscaled2x0123456789ABCDEF = _mm512_min_ps(vscaled2x0123456789ABCDEF, voutput_max_less_zero_point);

    vacc0x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled0x0123456789ABCDEF);
    vacc1x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled1x0123456789ABCDEF);
    vacc2x0123456789ABCDEF = _mm512_cvtps_epi32(vscaled2x0123456789ABCDEF);

    __m256i vacc0x012389AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0x0123456789ABCDEF, 1));
    __m256i vacc1x012389AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc1x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc1x0123456789ABCDEF, 1));
    __m256i vacc2x012389AB4567CDEF = _mm256_packs_epi32(_mm512_castsi512_si256(vacc2x0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc2x0123456789ABCDEF, 1));

    vacc0x012389AB4567CDEF = _mm256_adds_epi16(vacc0x012389AB4567CDEF, voutput_zero_point);
    vacc1x012389AB4567CDEF = _mm256_adds_epi16(vacc1x012389AB4567CDEF, voutput_zero_point);
    vacc2x012389AB4567CDEF = _mm256_adds_epi16(vacc2x012389AB4567CDEF, voutput_zero_point);

    const __m128i vout0x012389AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc0x012389AB4567CDEF), _mm256_extracti128_si256(vacc0x012389AB4567CDEF, 1));
    const __m128i vout1x012389AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc1x012389AB4567CDEF), _mm256_extracti128_si256(vacc1x012389AB4567CDEF, 1));
    const __m128i vout2x012389AB4567CDEF = _mm_packs_epi16(_mm256_castsi256_si128(vacc2x012389AB4567CDEF), _mm256_extracti128_si256(vacc2x012389AB4567CDEF, 1));

    __m128i vout0x0123456789ABCDEF = _mm_shuffle_epi32(vout0x012389AB4567CDEF, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i vout1x0123456789ABCDEF = _mm_shuffle_epi32(vout1x012389AB4567CDEF, _MM_SHUFFLE(3, 1, 2, 0));
    __m128i vout2x0123456789ABCDEF = _mm_shuffle_epi32(vout2x012389AB4567CDEF, _MM_SHUFFLE(3, 1, 2, 0));

    vout0x0123456789ABCDEF = _mm_max_epi8(vout0x0123456789ABCDEF, voutput_min);
    vout1x0123456789ABCDEF = _mm_max_epi8(vout1x0123456789ABCDEF, voutput_min);
    vout2x0123456789ABCDEF = _mm_max_epi8(vout2x0123456789ABCDEF, voutput_min);

    if (nc >= 16) {
      _mm_storeu_si128((__m128i*) c0, vout0x0123456789ABCDEF);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      _mm_storeu_si128((__m128i*) c1, vout1x0123456789ABCDEF);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      _mm_storeu_si128((__m128i*) c2, vout2x0123456789ABCDEF);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      nc -= 16;
    } else {
      // Prepare mask for valid 8-bit elements (depends on nc).
      const __mmask16 vmask = _cvtu32_mask16((UINT32_C(1) << nc) - UINT32_C(1));

      _mm_mask_storeu_epi8(c0, vmask, vout0x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c1, vmask, vout1x0123456789ABCDEF);
      _mm_mask_storeu_epi8(c2, vmask, vout2x0123456789ABCDEF);

      nc = 0;
    }
  } while (nc != 0);
}