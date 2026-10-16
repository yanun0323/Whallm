#!/bin/sh
# Regression check for the native DeepSeek V4.1 engine against the installed model.
# The expected tokens are the greedy continuations recorded from the MLX V4.1 engine
# (2026-10-16, BF16 activations, Scripts/deepseek_v41_reference.py). The 3,000-token
# prompt is the start of the Code benchmark context; it reads every expert file once
# per pass (about 250 GB). Usage: check.sh MODEL_DIR
set -eu
cd "$(dirname "$0")"
model=${1:?usage: check.sh MODEL_DIR}
./build.sh
cli=../../.build/deepseek-native/wd-cli
work=$(mktemp -d -t wd-check)
trap 'rm -rf "$work"' EXIT
../../.venv/bin/python -I ../../Scripts/deepseek_v41_reference.py "$model" "$work" --token-map-only
map="$work/token_map.i32"
# "<BOS>The capital of France is Paris. The capital of Germany is"
short=0,671,6102,294,8760,344,11111,16,455,6102,294,10322,344
short_expected="17575 16 455 6102 294 14251 344 16235 16 455 6102 294 16603 344 29168 16 455 6102 294 29585 344 76405 16 455"
long_expected="27574 26 47 18 1760 7633 28 764 20 65 15525 14 223 6970 3031 19469 28 1051 36036 3872 361 20319 223 837"
failed=0
check() {  # name, expected, then the wd-cli arguments
  name=$1; expected=$2; shift 2
  got=$("$cli" "$model" "$@" --token-map "$map" 2>/dev/null | sed -n 's/^greedy //p')
  case "$expected" in  # runs with fewer greedy tokens compare a prefix
    "$got"*) if [ -n "$got" ]; then echo "ok   $name"; return; fi ;;
  esac
  echo "FAIL $name"; echo "  expected $expected"; echo "  got      $got"; failed=1
}
check "whole prompt in one call" "$short_expected" "$short" --greedy 24
check "one token per call" "$short_expected" "$short" --chunk 1 --greedy 12
check "three tokens per call" "$short_expected" "$short" --chunk 3 --greedy 12
check "attention two rows at a time" "$short_expected" "$short" --chunk 5 --rows 2 --greedy 12
check "small expert cache" "$short_expected" "$short" --slots 96 --greedy 12
check "3000-token prompt" "$long_expected" @testdata/code_prompt_3000.txt --greedy 24
# The M5 TensorOps kernels run (slowly) on any Metal 4 GPU, so their wiring is checked here
# too. They round activations to BF16, so their logits differ slightly; these tokens hold.
WHALLM_DEEPSEEK_TENSOROPS=1 check "3000-token prompt, TensorOps kernels" "$long_expected" @testdata/code_prompt_3000.txt --greedy 24
# DSpark rounds keep a draft only when it is the model's own greedy choice, so the
# tokens must not change; this also covers verify and commit of rejected drafts.
if [ -f "$model/dspark/common.bin" ]; then
  check "DSpark" "$short_expected" "$short" --greedy 24 --dspark
  check "DSpark, every draft verified" "$short_expected" "$short" --greedy 24 --dspark --threshold 0 --no-backoff
  check "DSpark after a 3000-token prompt" "$long_expected" @testdata/code_prompt_3000.txt --greedy 24 --dspark
else
  echo "skip DSpark (not installed)"
fi

# DSpark verifies up to 8 tokens in one pass; each row must equal a one-token eval, bit for bit.
for n in 2 5 8; do
  if "$cli" "$model" "$short" --verify-check $n --token-map "$map" 2>/dev/null | grep -q "^same"; then echo "ok   verify of $n tokens equals one-token evals"
  else echo "FAIL verify of $n tokens equals one-token evals"; failed=1; fi
done

state() {  # name, then the wd-cli arguments of a --state-check run
  name=$1; shift
  if "$cli" "$model" "$@" --token-map "$map" >/dev/null 2>&1; then echo "ok   $name"; else echo "FAIL $name"; failed=1; fi
}
state "prompt cache: restore, save and load" "$short" --state-check 7 --greedy 8 --state-file "$work/short.state"
state "prompt cache after a long prompt" @testdata/code_prompt_3000.txt --state-check 2400 --greedy 8 --state-file "$work/long.state"
if [ -f "$model/dspark/common.bin" ]; then
  state "prompt cache with DSpark" "$short" --state-check 7 --greedy 12 --dspark --state-file "$work/dspark.state"
fi
exit $failed
