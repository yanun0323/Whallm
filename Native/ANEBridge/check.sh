#!/bin/zsh
set -euo pipefail
root=${0:A:h:h:h}
"$root/Scripts/build-ane-bridge.sh" >/dev/null
mkdir -p "$root/.build/ane-check"
xcrun --sdk macosx clang++ -std=c++17 -O2 -Wall -Wextra \
  "$root/Native/ANEBridge/NativeANE.cpp" "$root/Native/ANEBridge/check.cpp" \
  -o "$root/.build/ane-check/check"
xcrun --sdk macosx clang++ -std=c++17 -O2 -dynamiclib \
  "$root/Native/ANEBridge/mock.cpp" -o "$root/.build/ane-check/libMockANE.dylib"
"$root/.build/ane-check/check" "$root/.build/native/libWhallmANE.dylib" \
  "$root/.build/ane-check/libMockANE.dylib"
