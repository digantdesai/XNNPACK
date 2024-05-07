// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qc4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py


#include <fp16/fp16.h>
#include <gtest/gtest.h>

#include <xnnpack/unaligned.h>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>

#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/ppmm.h>
#include "gemm-microkernel-tester.h"

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X2__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X2__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }

      GemmMicrokernelTester()
        .mr(1)
        .nr(2)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X4__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X4__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(1)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X8__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_1X8__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(1)
        .nr(8)
        .kr(kr)
        .sr(1)
        .m(1)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X4__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X4__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X8__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X8__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(8)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, kc) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(4)
        .n(16)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, kc_odd) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc-1; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(4)
        .nr(4)
        .kr(kr)
        .sr(1)
        .m(4)
        .n(15)
        .k(kc-1)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, partial_n) {
  size_t kr = 1;
  for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
    for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
      if (round_up_po2(kc, kr) % bl) {
        continue;
      }
      GemmMicrokernelTester()
        .mr(2)
        .nr(2)
        .kr(kr)
        .sr(1)
        .m(2)
        .n(15)
        .k(kc)
        .bl(bl)
        .b_zero_point(8)
        .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
              xnn_init_f16_qc4w_minmax_scalar_params,
              xnn_pack_qs8_qc4w_gemm_bl_goi_w);
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, partial_m) {
  size_t kr = 1;
  for (size_t m=1; m < 2; m++) {
    for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
      for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
        if (round_up_po2(kc, kr) % bl) {
          continue;
        }
        GemmMicrokernelTester()
          .mr(2)
          .nr(2)
          .kr(kr)
          .sr(1)
          .m(1)
          .n(16)
          .k(kc)
          .bl(bl)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
                xnn_init_f16_qc4w_minmax_scalar_params,
                xnn_pack_qs8_qc4w_gemm_bl_goi_w);
      }
    }
  }
}

TEST(QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, partial_m) {
  size_t kr = 1;
  for (size_t m=1; m < 4; m++) {
    for (size_t kc=8; kc <= 16; kc+=8) { // 8,     16
      for (size_t bl=4; bl<=kc; bl+=4) { // 4, 8,   4, 8, 16
        if (round_up_po2(kc, kr) % bl) {
          continue;
        }
        GemmMicrokernelTester()
          .mr(4)
          .nr(4)
          .kr(kr)
          .sr(1)
          .m(1)
          .n(16)
          .k(kc)
          .bl(bl)
          .b_zero_point(8)
          .Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
                xnn_init_f16_qc4w_minmax_scalar_params,
                xnn_pack_qs8_qc4w_gemm_bl_goi_w);
      }
    }
  }
}