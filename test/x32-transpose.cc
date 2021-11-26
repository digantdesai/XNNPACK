// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/x32-transpose.yaml
//   Generator: tools/generate-transpose-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/transpose.h>
#include "transpose-microkernel-tester.h"


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__AVX_32X32, block_32_32) {
    TEST_REQUIRES_X86_AVX;
    TransposeMicrokernelTester()
      .height(32)
      .width(32)
      .h_block_size(32)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__avx);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__AVX_64X32, block_64_32) {
    TEST_REQUIRES_X86_AVX;
    TransposeMicrokernelTester()
      .height(64)
      .width(32)
      .h_block_size(64)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__avx);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__AVX_18X32, block_18_32) {
    TEST_REQUIRES_X86_AVX;
    TransposeMicrokernelTester()
      .height(18)
      .width(32)
      .h_block_size(18)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__avx);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__AVX_32X18, block_32_18) {
    TEST_REQUIRES_X86_AVX;
    TransposeMicrokernelTester()
      .height(32)
      .width(18)
      .h_block_size(32)
      .w_block_size(18)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__avx);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__AVX_19X31, block_19_31) {
    TEST_REQUIRES_X86_AVX;
    TransposeMicrokernelTester()
      .height(19)
      .width(31)
      .h_block_size(19)
      .w_block_size(31)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__avx);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__SSE_32X32, block_32_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(32)
      .width(32)
      .h_block_size(32)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__SSE_64X32, block_64_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(64)
      .width(32)
      .h_block_size(64)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__SSE_18X32, block_18_32) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(18)
      .width(32)
      .h_block_size(18)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__SSE_32X18, block_32_18) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(32)
      .width(18)
      .h_block_size(32)
      .w_block_size(18)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(X32_TRANSPOSE__SSE_19X31, block_19_31) {
    TEST_REQUIRES_X86_SSE;
    TransposeMicrokernelTester()
      .height(19)
      .width(31)
      .h_block_size(19)
      .w_block_size(31)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__sse);
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__NEON_32X32, block_32_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .height(32)
      .width(32)
      .h_block_size(32)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__NEON_64X32, block_64_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .height(64)
      .width(32)
      .h_block_size(64)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__NEON_18X32, block_18_32) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .height(18)
      .width(32)
      .h_block_size(18)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__NEON_32X18, block_32_18) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .height(32)
      .width(18)
      .h_block_size(32)
      .w_block_size(18)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(X32_TRANSPOSE__NEON_19X31, block_19_31) {
    TEST_REQUIRES_ARM_NEON;
    TransposeMicrokernelTester()
      .height(19)
      .width(31)
      .h_block_size(19)
      .w_block_size(31)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__neon);
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_WASMSIMD
  TEST(X32_TRANSPOSE__WASMSIMD_32X32, block_32_32) {
    TransposeMicrokernelTester()
      .height(32)
      .width(32)
      .h_block_size(32)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(X32_TRANSPOSE__WASMSIMD_64X32, block_64_32) {
    TransposeMicrokernelTester()
      .height(64)
      .width(32)
      .h_block_size(64)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(X32_TRANSPOSE__WASMSIMD_18X32, block_18_32) {
    TransposeMicrokernelTester()
      .height(18)
      .width(32)
      .h_block_size(18)
      .w_block_size(32)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(X32_TRANSPOSE__WASMSIMD_32X18, block_32_18) {
    TransposeMicrokernelTester()
      .height(32)
      .width(18)
      .h_block_size(32)
      .w_block_size(18)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD


#if XNN_ARCH_WASMSIMD
  TEST(X32_TRANSPOSE__WASMSIMD_19X31, block_19_31) {
    TransposeMicrokernelTester()
      .height(19)
      .width(31)
      .h_block_size(19)
      .w_block_size(31)
      .offset(1)
      .Test(xnn_x32_transpose_ukernel__wasmsimd);
  }
#endif  // XNN_ARCH_WASMSIMD
