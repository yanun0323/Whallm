#!/bin/sh
# Build the native DeepSeek V4.1 engine's development tool into .build/deepseek-native.
set -eu
cd "$(dirname "$0")"
out=../../.build/deepseek-native
mkdir -p "$out"
{
  echo 'extern const char* const kWhallmDeepSeekKernels = R"WDK('; cat kernels.metal; echo ')WDK";'
  echo 'extern const char* const kWhallmDeepSeekTensorKernels = R"WDK('; cat kernels_tensorops.metal; echo ')WDK";'
} > "$out/kernels.cpp"
clang++ -std=c++17 -O2 -fobjc-arc -Wall -Wextra -Wno-unused-parameter -I. engine.mm cli.mm "$out/kernels.cpp" \
  -framework Metal -framework Foundation -o "$out/wd-cli"
