// Auto-generated file. Do not edit!
//   Template: src/f32-f16-vcvt/scalar-float.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_f16_vcvt_ukernel__scalar_float_x4(
    size_t n,
    const float* input,
    void* output,
    const void* params)
{
  assert(n != 0);
  assert(n % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = (const float*) input;
  const uint32_t vsign_mask = UINT32_C(0x80000000);
  const uint32_t vbias_mask = UINT32_C(0xFF000000);
  const uint32_t vmin_bias = UINT32_C(0x71000000);
  const uint32_t vbase_offset = UINT32_C(0x07800000);
  const uint32_t vexp_mask = UINT32_C(0x00007C00);
  const uint32_t vmantissa_mask = UINT32_C(0x00000FFF);
  const uint16_t vmagic_mask = UINT16_C(0x7E00);
  const float scale_to_inf = fp32_from_bits(UINT32_C(0x77800000));
  const float scale_to_zero = fp32_from_bits(UINT32_C(0x08800000));
  for (; n >= 4 * sizeof(uint16_t); n -= 4 * sizeof(uint16_t)) {
    const float vh0 = i[0];
    const float vh1 = i[1];
    const float vh2 = i[2];
    const float vh3 = i[3];
    i += 4;

    float vbase0 = (fabsf(vh0) * scale_to_inf) * scale_to_zero;
    float vbase1 = (fabsf(vh1) * scale_to_inf) * scale_to_zero;
    float vbase2 = (fabsf(vh2) * scale_to_inf) * scale_to_zero;
    float vbase3 = (fabsf(vh3) * scale_to_inf) * scale_to_zero;

    const uint32_t vw0 = fp32_to_bits(vh0);
    const uint32_t vw1 = fp32_to_bits(vh1);
    const uint32_t vw2 = fp32_to_bits(vh2);
    const uint32_t vw3 = fp32_to_bits(vh3);

    const uint32_t v2w0 = vw0 + vw0;
    const uint32_t v2w1 = vw1 + vw1;
    const uint32_t v2w2 = vw2 + vw2;
    const uint32_t v2w3 = vw3 + vw3;

    const uint32_t vsign0 = vw0 & vsign_mask;
    const uint32_t vsign1 = vw1 & vsign_mask;
    const uint32_t vsign2 = vw2 & vsign_mask;
    const uint32_t vsign3 = vw3 & vsign_mask;

    uint32_t vbias0 = v2w0 & vbias_mask;
    uint32_t vbias1 = v2w1 & vbias_mask;
    uint32_t vbias2 = v2w2 & vbias_mask;
    uint32_t vbias3 = v2w3 & vbias_mask;

    vbias0 = XNN_UNPREDICTABLE(vbias0 < vmin_bias) ? vmin_bias : vbias0;
    vbias1 = XNN_UNPREDICTABLE(vbias1 < vmin_bias) ? vmin_bias : vbias1;
    vbias2 = XNN_UNPREDICTABLE(vbias2 < vmin_bias) ? vmin_bias : vbias2;
    vbias3 = XNN_UNPREDICTABLE(vbias3 < vmin_bias) ? vmin_bias : vbias3;

    vbase0 = fp32_from_bits((vbias0 >> 1) + vbase_offset) + vbase0;
    vbase1 = fp32_from_bits((vbias1 >> 1) + vbase_offset) + vbase1;
    vbase2 = fp32_from_bits((vbias2 >> 1) + vbase_offset) + vbase2;
    vbase3 = fp32_from_bits((vbias3 >> 1) + vbase_offset) + vbase3;

    const uint32_t vbits0 = fp32_to_bits(vbase0);
    const uint32_t vbits1 = fp32_to_bits(vbase1);
    const uint32_t vbits2 = fp32_to_bits(vbase2);
    const uint32_t vbits3 = fp32_to_bits(vbase3);

    const uint32_t vexp_bits0 = (vbits0 >> 13) & vexp_mask;
    const uint32_t vexp_bits1 = (vbits1 >> 13) & vexp_mask;
    const uint32_t vexp_bits2 = (vbits2 >> 13) & vexp_mask;
    const uint32_t vexp_bits3 = (vbits3 >> 13) & vexp_mask;

    const uint32_t vmantissa_bits0 = vbits0 & vmantissa_mask;
    const uint32_t vmantissa_bits1 = vbits1 & vmantissa_mask;
    const uint32_t vmantissa_bits2 = vbits2 & vmantissa_mask;
    const uint32_t vmantissa_bits3 = vbits3 & vmantissa_mask;

    const uint32_t vnonsign0 = vexp_bits0 + vmantissa_bits0;
    const uint32_t vnonsign1 = vexp_bits1 + vmantissa_bits1;
    const uint32_t vnonsign2 = vexp_bits2 + vmantissa_bits2;
    const uint32_t vnonsign3 = vexp_bits3 + vmantissa_bits3;

    const uint16_t vr0 = (vsign0 >> 16) | (XNN_UNPREDICTABLE(v2w0 > vbias_mask) ? vmagic_mask : vnonsign0);
    const uint16_t vr1 = (vsign1 >> 16) | (XNN_UNPREDICTABLE(v2w1 > vbias_mask) ? vmagic_mask : vnonsign1);
    const uint16_t vr2 = (vsign2 >> 16) | (XNN_UNPREDICTABLE(v2w2 > vbias_mask) ? vmagic_mask : vnonsign2);
    const uint16_t vr3 = (vsign3 >> 16) | (XNN_UNPREDICTABLE(v2w3 > vbias_mask) ? vmagic_mask : vnonsign3);

    ((uint16_t*) output)[0] = vr0;
    ((uint16_t*) output)[1] = vr1;
    ((uint16_t*) output)[2] = vr2;
    ((uint16_t*) output)[3] = vr3;

    output = (uint16_t*) output + 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vh = *i++;

      const uint16_t vf = fp16_ieee_from_fp32_value(vh);

      ((uint16_t*) output)[0] = vf;
      output = (uint16_t*) output + 1;

      n -= sizeof(uint16_t);
    } while (n != 0);
  }
}
