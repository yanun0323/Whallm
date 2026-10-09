#!/bin/zsh
set -euo pipefail

project_root=${0:A:h:h}
output=${1:-$project_root/.build/native/libWhallmANE.dylib}
mkdir -p "${output:h}"

xcrun --sdk macosx clang \
  -O3 \
  -fobjc-arc \
  -fblocks \
  -dynamiclib \
  -arch arm64 \
  -mmacosx-version-min=15.0 \
  -fvisibility=hidden \
  -install_name @rpath/libWhallmANE.dylib \
  "$project_root/Native/ANEBridge/WhallmANE.m" \
  -framework Foundation \
  -framework IOSurface \
  -ldl \
  -o "$output"

print "$output"
