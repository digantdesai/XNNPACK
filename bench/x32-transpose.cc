// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack/AlignedAllocator.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/transpose.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <vector>

#include "bench/utils.h"
#include <benchmark/benchmark.h>

static void x32_transpose(
    benchmark::State& state, xnn_x32_transpose_ukernel_function transpose,
    size_t ukernel_size,
    benchmark::utils::IsaCheckFunction isa_check = nullptr) {
  if (isa_check && !isa_check(state)) {
    return;
  }

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto i32rng = std::bind(std::uniform_real_distribution<int>(-100, 100),
                          std::ref(rng));

  std::vector<int, AlignedAllocator<int, 64>> x(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(int));
  std::vector<int, AlignedAllocator<int, 64>> y(
      ukernel_size * ukernel_size + XNN_EXTRA_BYTES / sizeof(int));
  std::generate(x.begin(), x.end(), std::ref(i32rng));
  std::fill(y.begin(), y.end(), 0);

  for (auto _ : state) {
    transpose(x.data(), y.data(), 1, ukernel_size, ukernel_size, ukernel_size,
              ukernel_size, nullptr /* params */);
  }
}

#if XNN_ARCH_WASMSIMD
  BENCHMARK_CAPTURE(x32_transpose, neon_x32, xnn_x32_transpose_ukernel__wasmsimd,
                    32, benchmark::utils::CheckNEON)
      ->UseRealTime();
#endif  // XNN_ARCH_WASMSIMD

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  BENCHMARK_CAPTURE(x32_transpose, neon_x32, xnn_x32_transpose_ukernel__neon, 32,
                    benchmark::utils::CheckNEON)
      ->UseRealTime();
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  BENCHMARK_CAPTURE(x32_transpose, sse_x32, xnn_x32_transpose_ukernel__sse, 32)
      ->UseRealTime();
  BENCHMARK_CAPTURE(x32_transpose, avx_x32, xnn_x32_transpose_ukernel__avx, 32,
                    benchmark::utils::CheckAVX)
      ->UseRealTime();
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
