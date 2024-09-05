# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: microkernel filename lists for avx2
#
# Auto-generated file. Do not edit!
#   Generator: tools/update-microkernels.py


SET(PROD_AVX2_MICROKERNEL_SRCS
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-1x16-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x16-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-avx2-broadcast.c
  src/f16-pavgpool/f16-pavgpool-9p8x-minmax-avx2-c8.c
  src/f16-pavgpool/f16-pavgpool-9x-minmax-avx2-c8.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40.c
  src/f16-velu/gen/f16-velu-avx2-rr1-p3-u16.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u32.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x16-minmax-avx2-broadcast.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u64.c
  src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u64.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u56.c
  src/f32-vlog/gen/f32-vlog-avx2-rational-3-3-div.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u40.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-1x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-3x8c8-minmax-avx2.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-1x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-4x8c8-minmax-avx2.c
  src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul32.c
  src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u16.c
  src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u16.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-3p16c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-1x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-3x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-1x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-3x8c8-minmax-fp32-avx2.c
  src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-c64.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u64.c
  src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u16.c
  src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u16.c
  src/qs8-vcvt/gen/qs8-vcvt-avx2-u32.c
  src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u32.c
  src/qu8-dwconv/gen/qu8-dwconv-9p16c-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-25p16c-minmax-fp32-avx2-mul32.c
  src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u16.c
  src/qu8-gemm/gen/qu8-gemm-1x8c8-minmax-fp32-avx2.c
  src/qu8-gemm/gen/qu8-gemm-3x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/qu8-igemm-1x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/qu8-igemm-3x8c8-minmax-fp32-avx2.c
  src/qu8-vadd/gen/qu8-vadd-minmax-avx2-mul32-ld64-u16.c
  src/qu8-vaddc/gen/qu8-vaddc-minmax-avx2-mul32-ld64-u16.c
  src/qu8-vcvt/gen/qu8-vcvt-avx2-u32.c
  src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u32.c
  src/s32-f32-vcvt/gen/s32-f32-vcvt-avx2.c
  src/s32-vmul/gen/s32-vmul-avx2.c
  src/s32-vmul/gen/s32-vmulc-avx2.c
  src/x8-lut/gen/x8-lut-avx2-u128.c
  src/x8-transposec/gen/x8-transposec-32x32-reuse-switch-avx2.c
  src/x16-packw/gen/x16-packw-x16-gemm-goi-avx2-u16-prfm.c
  src/x16-transposec/gen/x16-transposec-16x16-reuse-switch-avx2.c)

SET(NON_PROD_AVX2_MICROKERNEL_SRCS
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-1x8-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-3x16-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x8-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-5x8-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-5x16-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-6x8-minmax-avx2-broadcast.c
  src/f16-f32acc-gemm/gen/f16-f32acc-gemm-7x8-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x8-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-3x16-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x8-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x8-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x8-minmax-avx2-broadcast.c
  src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-1x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-1x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-3x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-4x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-4x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-5x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-5x16-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-6x8-minmax-avx2-broadcast.c
  src/f16-gemm/gen/f16-gemm-7x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-1x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-1x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-3x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-4x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-4x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-5x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-5x16-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-6x8-minmax-avx2-broadcast.c
  src/f16-igemm/gen/f16-igemm-7x8-minmax-avx2-broadcast.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc4.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc5.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc3.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc4.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u72-acc3.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u72.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80-acc5.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc2.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc3.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc6.c
  src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96.c
  src/f16-velu/gen/f16-velu-avx2-rr1-p3-u8.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u8.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u16.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u24.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u32.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u40.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u48.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u56.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u64.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u8.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u16.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u24.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u40.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u48.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u56.c
  src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u64.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u8.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u16.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u24.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u32.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u40.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u48.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u56.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u64.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u72.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-div-u80.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u8.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u16.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u24.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u32.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u40.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u48.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u56.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u64.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u72.c
  src/f16-vtanh/gen/f16-vtanh-avx2-expm1minus-rr1-p3h2ts-rcp-u80.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-7x16-minmax-avx2-broadcast.c
  src/f32-qc4w-gemm/gen/f32-qc4w-gemm-8x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-1x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-2x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-2x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-3x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-4x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-5x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-6x16s4-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-7x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-7x16-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x8-minmax-avx2-broadcast.c
  src/f32-qc8w-gemm/gen/f32-qc8w-gemm-8x16-minmax-avx2-broadcast.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u16.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u32.c
  src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u48.c
  src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u16.c
  src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u32.c
  src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u48.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u32-acc2.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u32-acc4.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u32.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u64-acc2.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u64-acc4.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u64.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u72-acc3.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u72.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u80-acc2.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u80-acc5.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u80.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u96-acc2.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u96-acc3.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u96-acc6.c
  src/f32-raddexpminusmax/gen/f32-raddexpminusmax-avx2-p5-u96.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u32-acc2.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u32-acc4.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u32.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64-acc2.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64-acc4.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u72-acc3.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u72.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80-acc2.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80-acc5.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc2.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc3.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc6.c
  src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32-acc4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u32.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64-acc4.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u64.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u72-acc3.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u72.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80-acc5.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u80.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc2.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc3.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96-acc6.c
  src/f32-raddstoreexpminusmax/gen/f32-raddstoreexpminusmax-avx2-rr1-p5-u96.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u8.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u16.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u24.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u32.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u40.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u48.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u64.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u72.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u80.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u8.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u16.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u24.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u32.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u40.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u48.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u56.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u64.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u72.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u80.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u8.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u16.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u24.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u32.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u40.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u48.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u56.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u64.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u72.c
  src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u80.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u8.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u16.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u24.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u32.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u40.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u48.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u56.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u64.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u72.c
  src/f32-velu/gen/f32-velu-avx2-rr1-p6-u80.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u8.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u16.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u24.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u32.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u40.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u48.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u56.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u64.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u72.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u80.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u88.c
  src/f32-vscaleexpminusmax/gen/f32-vscaleexpminusmax-avx2-p5-u96.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u8.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u16.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u24.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u32.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u40.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u48.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u56.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u64.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u72.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u80.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u88.c
  src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u96.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u32.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u48.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u56.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u64.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u72.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u80.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u32.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u40.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u48.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u56.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u64.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u72.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u80.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u8.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u16.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u24.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u32.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u40.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u48.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u56.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u64.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u72.c
  src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u80.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f16-qb4w-gemm/gen/qd8-f16-qb4w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-5x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-6x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-7x8c8-minmax-avx2.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2-madd.c
  src/qd8-f16-qc4w-gemm/gen/qd8-f16-qc4w-gemm-8x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-gemm/gen/qd8-f16-qc8w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-2x8c8-minmax-avx2.c
  src/qd8-f16-qc8w-igemm/gen/qd8-f16-qc8w-igemm-4x8c8-minmax-avx2.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f32-qb4w-gemm/gen/qd8-f32-qb4w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-1x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-4x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-5x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-6x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-7x8c8-minmax-avx2.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2-madd-prfm.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2-madd.c
  src/qd8-f32-qc4w-gemm/gen/qd8-f32-qc4w-gemm-8x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-2x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-gemm/gen/qd8-f32-qc8w-gemm-3x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-2x8c8-minmax-avx2.c
  src/qd8-f32-qc8w-igemm/gen/qd8-f32-qc8w-igemm-3x8c8-minmax-avx2.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-9p8c-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-9p16c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-9p32c-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-25p8c-minmax-fp32-avx2-mul32.c
  src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-25p16c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-dwconv/gen/qs8-dwconv-25p32c-minmax-fp32-avx2-mul32.c
  src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u24.c
  src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u32.c
  src/qs8-f16-vcvt/gen/qs8-f16-vcvt-avx2-u64.c
  src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u8.c
  src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u24.c
  src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-5f5m5l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-6f6m7l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l16c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-8f8m9l32c16s16r-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p8c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p16c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-9p32c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p8c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p16c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-add16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-vpmovsx.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul16-vpunpck.c
  src/qs8-qc8w-dwconv/gen/qs8-qc8w-dwconv-25p32c-minmax-fp32-avx2-mul32.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-2x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-gemm/gen/qs8-qc8w-gemm-4x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-2x8c8-minmax-fp32-avx2.c
  src/qs8-qc8w-igemm/gen/qs8-qc8w-igemm-4x8c8-minmax-fp32-avx2.c
  src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-c32.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u32.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u64-acc2.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u128-acc2.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u128-acc4.c
  src/qs8-rsum/gen/qs8-rsum-avx2-u128.c
  src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u8.c
  src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u24.c
  src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u32.c
  src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u8.c
  src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u24.c
  src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u32.c
  src/qs8-vcvt/gen/qs8-vcvt-avx2-u16.c
  src/qs8-vcvt/gen/qs8-vcvt-avx2-u64.c
  src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u16.c
  src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u64.c
  src/qu8-dwconv/gen/qu8-dwconv-5f5m5l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-5f5m5l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-5f5m5l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-6f6m7l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-6f6m7l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-6f6m7l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-8f8m9l8c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-8f8m9l16c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-8f8m9l32c8s8r-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-9p8c-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-9p32c-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-25p8c-minmax-fp32-avx2-mul32.c
  src/qu8-dwconv/gen/qu8-dwconv-25p32c-minmax-fp32-avx2-mul32.c
  src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u8.c
  src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u24.c
  src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u32.c
  src/qu8-gemm/gen/qu8-gemm-2x8c8-minmax-fp32-avx2.c
  src/qu8-gemm/gen/qu8-gemm-4x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/qu8-igemm-2x8c8-minmax-fp32-avx2.c
  src/qu8-igemm/gen/qu8-igemm-4x8c8-minmax-fp32-avx2.c
  src/qu8-vadd/gen/qu8-vadd-minmax-avx2-mul32-ld64-u8.c
  src/qu8-vaddc/gen/qu8-vaddc-minmax-avx2-mul32-ld64-u8.c
  src/qu8-vcvt/gen/qu8-vcvt-avx2-u16.c
  src/qu8-vcvt/gen/qu8-vcvt-avx2-u64.c
  src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u16.c
  src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u64.c
  src/x8-lut/gen/x8-lut-avx2-u32.c
  src/x8-lut/gen/x8-lut-avx2-u64.c
  src/x8-lut/gen/x8-lut-avx2-u96.c
  src/x8-transposec/gen/x8-transposec-32x32-reuse-mov-avx2.c
  src/x16-packw/gen/x16-packw-x8-gemm-goi-avx2-u16-prfm.c
  src/x16-packw/gen/x16-packw-x8-gemm-goi-avx2-u16.c
  src/x16-packw/gen/x16-packw-x16-gemm-goi-avx2-u16.c
  src/x16-transposec/gen/x16-transposec-16x16-reuse-mov-avx2.c)

SET(ALL_AVX2_MICROKERNEL_SRCS ${PROD_AVX2_MICROKERNEL_SRCS} + ${NON_PROD_AVX2_MICROKERNEL_SRCS})
