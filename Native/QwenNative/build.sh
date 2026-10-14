#!/bin/sh
# Build the native Qwen engine's development tool into .build/qwen-native.
set -eu
cd "$(dirname "$0")"
out=../../.build/qwen-native
mkdir -p "$out"
{
  echo 'extern const char* const kWhallmQwenKernels = R"WQK('; cat kernels.metal; echo ')WQK";'
  echo 'extern const char* const kWhallmQwenTensorKernels = R"WQK('; cat kernels_tensorops.metal; echo ')WQK";'
} > "$out/kernels.cpp"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Wno-unused-parameter -I. engine.mm cli.mm "$out/kernels.cpp" \
  -framework Metal -framework Foundation -o "$out/wq-cli"
