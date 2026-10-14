#!/bin/sh
# Regression check for the native Qwen engine against the installed Swift1.5 model.
# The expected text tokens are the greedy continuation recorded from the MLX Qwen
# engine (2026-10-14, MTP off, BF16 resident weights). Usage: check.sh MODEL_DIR
set -eu
cd "$(dirname "$0")"
model=${1:?usage: check.sh MODEL_DIR}
./build.sh
cli=../../.build/qwen-native/wq-cli
short=760,6511,314,9338,369,11751,13,561,6511,314,6124,369
short_expected="25358 13 561 6511 314 14898 369 21047 13 561 6511 314 17163 369 23327 13 561 6511 314 9564 369 19241 13 561"
long_expected="19 11 220 20 62 16 17 15 1089 4206 25 594 3075 681 198 406 1236 13687 437 86 17 24354 487 12847"
failed=0
check() {  # name, expected, then the wq-cli arguments
  name=$1; expected=$2; shift 2
  got=$("$cli" "$model" "$@" 2>/dev/null | sed -n 's/^greedy //p')
  case "$got" in
    "$expected"*) echo "ok   $name" ;;
    *) echo "FAIL $name"; echo "  expected $expected"; echo "  got      $got"; failed=1 ;;
  esac
}
check "one token per call" "$short_expected" "$short" --chunk 1 --greedy 24
check "three tokens per call" "$short_expected" "$short" --chunk 3 --greedy 24
check "whole prompt in one call" "$short_expected" "$short" --chunk 12 --greedy 24
check "draft layer, 2 drafts" "$short_expected" "$short" --chunk 12 --greedy 25 --mtp 2
check "draft layer, 4 drafts" "$short_expected" "$short" --chunk 12 --greedy 25 --mtp 4
check "3000-token prompt" "$long_expected" @testdata/code_prompt_3000.txt --chunk 3000 --greedy 24
check "3000-token prompt, draft layer" "$long_expected" @testdata/code_prompt_3000.txt --chunk 3000 --greedy 25 --mtp 2
WHALLM_QWEN_NO_SIMDGROUP=1 check "3000-token prompt, fallback kernels" "$long_expected" @testdata/code_prompt_3000.txt --chunk 3000 --greedy 24
check "8-bit resident weights" "$long_expected" @testdata/code_prompt_3000.txt --chunk 3000 --greedy 24 --8bit
# The M5 TensorOps kernels run (slowly) on any Metal 4 GPU, so their wiring is checked here too.
WHALLM_QWEN_TENSOROPS=1 check "3000-token prompt, TensorOps kernels" "$long_expected" @testdata/code_prompt_3000.txt --chunk 3000 --greedy 24

state() {  # name, then the wq-cli arguments of a --state-check run
  name=$1; shift
  if "$cli" "$model" "$@" >/dev/null 2>&1; then echo "ok   $name"; else echo "FAIL $name"; failed=1; fi
}
state_file=$(mktemp -t wq-state)
state "prompt cache: restore, save and load" @testdata/code_prompt_3000.txt --state-check 2500 --greedy 8 --state-file "$state_file"
state "prompt cache with the draft layer" @testdata/code_prompt_3000.txt --state-check 2500 --greedy 9 --mtp 2 --state-file "$state_file"
# A state written with 8-bit weights continues in an 8-bit engine and is refused by a BF16 one.
"$cli" "$model" 760,6511,314,9338,369,11751,13 --chunk 7 --8bit --save-state "$state_file" >/dev/null 2>&1
if "$cli" "$model" 561,6511,314,6124,369 --chunk 5 --8bit --load-state "$state_file" --greedy 2 2>/dev/null | grep -q "^greedy 25358" &&
   ! "$cli" "$model" 561 --load-state "$state_file" >/dev/null 2>&1; then
  echo "ok   prompt cache files keep their settings"
else
  echo "FAIL prompt cache files keep their settings"; failed=1
fi
rm -f "$state_file" "$state_file.other"

# Image input. The image is 320 x 224: red left half, blue top right, white bottom right.
# The MLX engine answers " Red, White, and Blue."; its image encoder runs in BF16, ours in
# FP32 (within 2e-5 of the MLX encoder run in FP32), and the answers part after " Red, White,".
patches=$(mktemp -t wq-patches)
gunzip -c testdata/image_patches_14x20.f32.gz > "$patches"
check "image input" "3605 11 5626 11" @testdata/image_prompt_tokens.txt --image "$patches" --grid 14,20 \
  --vision-catalog ../../Sources/DeepSeekRepack/Resources/QwenVision.json --greedy 4
check "image input, draft layer" "3605 11 5626 11" @testdata/image_prompt_tokens.txt --image "$patches" --grid 14,20 \
  --vision-catalog ../../Sources/DeepSeekRepack/Resources/QwenVision.json --greedy 5 --mtp 2
rm -f "$patches"
exit $failed
