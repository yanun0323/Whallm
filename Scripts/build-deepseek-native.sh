#!/bin/zsh
# Build the native DeepSeek V4.1 engine library (C++ and Metal, no MLX) for the Python runtime.
set -euo pipefail

project_root=${0:A:h:h}
output=${1:-$project_root/.build/native/libWhallmDeepSeekNative.dylib}
source_dir=$project_root/Native/DeepSeekNative
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "${output:h}"

{
  print -r -- 'extern const char* const kWhallmDeepSeekKernels = R"WDK('
  cat "$source_dir/kernels.metal"
  print -r -- ')WDK";'
  print -r -- 'extern const char* const kWhallmDeepSeekTensorKernels = R"WDK('
  cat "$source_dir/kernels_tensorops.metal"
  print -r -- ')WDK";'
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
  -install_name @rpath/libWhallmDeepSeekNative.dylib \
  -I "$source_dir" \
  "$source_dir/engine.mm" "$source_dir/capi.mm" "$work/kernels.cpp" \
  -framework Metal \
  -framework Foundation \
  -o "$output"

print "$output"
