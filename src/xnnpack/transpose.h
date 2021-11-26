// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(const void* input,          \
                            void* output,               \
                            size_t offset,              \
                            size_t h,                   \
                            size_t w,                   \
                            size_t h_block_size,        \
                            size_t w_block_size,        \
                            const void* params);

DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__avx);
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__sse);
//DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__aarch64);
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__neon);
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__wasmsimd);

#ifdef __cplusplus
}  // extern "C"
#endif
