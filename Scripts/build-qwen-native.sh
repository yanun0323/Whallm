#!/bin/zsh
# Build the native Qwen engine library (C++ and Metal, no MLX) for the Python runtime.
set -euo pipefail

project_root=${0:A:h:h}
output=${1:-$project_root/.build/native/libWhallmQwenNative.dylib}
source_dir=$project_root/Native/QwenNative
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "${output:h}"

{
  print -r -- 'extern const char* const kWhallmQwenKernels = R"WQK('
  cat "$source_dir/kernels.metal"
  print -r -- ')WQK";'
  print -r -- 'extern const char* const kWhallmQwenTensorKernels = R"WQK('
  cat "$source_dir/kernels_tensorops.metal"
  print -r -- ')WQK";'
} > "$work/kernels.cpp"

xcrun --sdk macosx clang++ \
  -std=c++17 \
  -O2 \
  -fobjc-arc \
  -dynamiclib \
  -arch arm64 \
  -mmacosx-version-min=15.0 \
  -fvisibility=hidden \
  -Wall -Wextra -Wno-unused-parameter \
  -install_name @rpath/libWhallmQwenNative.dylib \
  -I "$source_dir" \
  "$source_dir/engine.mm" "$source_dir/capi.mm" "$work/kernels.cpp" \
  -framework Metal \
  -framework Foundation \
  -o "$output"

print "$output"
