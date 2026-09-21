# Qwen Prefill and Decode speed work

## Scope and baseline

This extends `feat/qwen-flash-optimize` after the App controls at
`113be486422fa88deeeb96fbc29dc19ab7b27414`. It does not change installed-model
bytes, quantization, expert top-k, sampling rules or `master`.

The measurements below are **synthetic component microbenchmarks**, not full-model
Prefill/Decode tokens per second. The runner identifies as **Apple M1 (Virtual)**,
with an **Apple Paravirtual device**, 7 GiB RAM, macOS 15.7.9 and MLX 0.32.2.
They are not measurements on a physical M1, M2 Max, M5 Pro or M5 Ultra.

## Implemented changes

### 1. Vectorized exact N-gram hashes

`qwen_ngram_hash.py` replaces per-token Python EOS scans with array operations.
It reuses the pair hash when constructing triple hashes. Signed int64 overflow,
XOR, remainder, head offsets, EOS boundaries and cross-chunk history retain the
original semantics. No persistent row cache or I/O backend change is involved.
The compatibility names remain exported from `qwen4_exp.py`. This path is always
used and needs no App switch.

### 2. No-gather QSA for the dense region

When `qwen_sparse_sdpa` is enabled and the current KV length is at most the
indexer budget (2048 for this pinned model), all causal keys are already selected.
The runtime now passes the original GQA K/V tensors to SDPA instead of duplicating
them for each four-query group and constructing unused index pools.

Full-prefix calls use the causal mask, cached chunks use absolute-offset masks,
and a one-token last-position decode needs no mask. Raw index state is still
updated, so crossing the sparse threshold and restoring or trimming caches does
not lose history. Above the budget the previous selected-cell path is retained.
This is not a new long-context sparse-attention algorithm.

Fused reductions are not guaranteed bit-identical to the manual path or the old
fused batch layout. The existing opt-in warning about rounding and generated
text remains applicable. Default `qwen_sparse_sdpa=false` is unchanged.

### 3. Singleton routed-expert dispatch

The non-ready `get_many` path skips stable sorting, one-row gathers and unsorting
when there is only one input token and selected experts are unique. Per-expert
QMM shapes remain `[1, hidden]`, and outputs return in router order. Repeated
expert IDs keep the original grouping fallback. Tests and the resident MXFP4
fixture compare bytes exactly.

This does **not** change ready-expert asynchronous SSD reads or batched expert
GEMM. Its synthetic cache contains resident weights: the measured improvement
must not be treated as an SSD or whole-Decode speedup. Do not disable ready
reads solely to obtain the microbenchmark's percentage.

### 4. Single-token hyper-connection fusion

The existing `qwen_compile_tensor_ops` option also compiles small pure tensor
operations around hyper-connection mixing and residual injection for a single
token. Matmuls remain outside those functions; mutable weights, cache and RNG are
not captured. Multi-token inputs keep the previous mixing/injection path with
existing grouped-norm compilation.

This guard follows a negative experiment: enabling the extra fusion for three
tokens raised p50 from 1.4402 to 1.8104 ms in the first runner trial (about 26%).
The guarded trial uses the old operation path for those batches; its minor timing
variation is not credited as a new multi-token optimization. Single-token gains
were modest and varied between trials, so they are not a hardware-wide promise.

The App label is now **Compile Qwen tensor operations**, with English,
Traditional Chinese and Simplified Chinese hints describing the guard. Saved
choices and defaults are unchanged. In particular, the App's existing fresh-Qwen
profile already enabled tensor compilation; standalone RuntimeConfig defaults
it off. The label change does not introduce a new automatically enabled switch.

### 5. Native MTP state-only advance

`MTPModel.advance` executes the normal embedding and decoder-layer path but skips
the final hyper-connection mixer and vocabulary projection when their logits are
unused. MTP prefill and full-acceptance synchronization now call this path.
Actual draft logits, proposal probabilities, target verification, acceptance,
corrected residual sampling and rollback are unchanged.

The full layer output is evaluated along with caches before shared expert slots
can be recycled. Evaluating cache arrays alone would not fence trailing MoE work.
Generic draft fixtures without `advance` retain a compatible fallback.

Native hidden/cache/next-logit parity and fake-drafter token/cursor parity are
tested. No MTP speed multiplier is claimed. This does not unload the shared LM
head weights, and it does not resolve or recharacterize the pre-existing
full-model MTP-versus-non-MTP output discrepancy.

## Recorded component results

Refinement run: [35561322174](https://github.com/yanun0323/Whallm/actions/runs/35561322174).
Initial trial, including the rejected multi-token fusion:
[35560929863](https://github.com/yanun0323/Whallm/actions/runs/35560929863).
[Machine-readable summary](validation/qwen-speed-20260921.json) records exact
source, output and artifact hashes. Artifacts contain all raw samples, environment,
additional-peak allocation measurements and test logs.

Each component uses seeded synthetic inputs, three warmup calls, 15 alternating
A/B-order rounds and three calls per sample. Timing includes Python graph building
and synchronous `mx.eval`; p95 is over these three-call averages, not an end-user
single-request percentile. Inputs and kernels are warm. QSA uses 24 query heads,
2 KV heads, dimension 256 and BF16. Resident experts use the model's actual
2560/640 dimensions, top-10 and MXFP4. No expert disk I/O is measured.

The QSA reference below is the **previous fused SDPA** path, not the slower manual
path. The benchmark also records the manual path, including cases where it wins.

| Component | Previous p50 ms | Refined p50 ms | Previous p95 ms | Refined p95 ms |
| --- | ---: | ---: | ---: | ---: |
| Hash, 1 new token + 2 history | 0.02160 | 0.01828 | 0.02509 | 0.01902 |
| Hash, 128 new tokens + 2 history | 0.47267 | 0.02682 | 0.53141 | 0.02962 |
| Hash, 8192 new tokens + 2 history | 30.32040 | 0.61201 | 40.21664 | 0.75127 |
| QSA, 1 query / 128 KV | 0.70461 | 0.51742 | 1.53380 | 0.60294 |
| QSA, 1 query / 2048 KV | 1.89072 | 1.01743 | 8.28092 | 9.23703 |
| QSA, 128 queries / 128 KV | 9.82278 | 0.85239 | 11.09732 | 1.33200 |
| QSA, 128 queries / 2048 KV | 100.33760 | 5.44599 | 105.15761 | 14.11394 |
| QSA, 1 query / 8192 KV (same sparse path) | 3.10746 | 2.92446 | 4.35786 | 3.75322 |
| QSA, 128 queries / 8192 KV (same sparse path) | 130.64785 | 126.37475 | 140.84327 | 132.40613 |
| Hyper-connection, 1 token | 1.28433 | 1.25211 | 1.75847 | 1.67137 |
| Hyper-connection, 3 tokens (old path retained) | 2.04156 | 1.92196 | 2.40602 | 2.19787 |
| Hyper-connection, 128 tokens (old path retained) | 3.92269 | 3.84679 | 5.46924 | 4.48251 |
| Resident singleton expert dispatch | 1.71900 | 1.63458 | 3.76451 | 3.27898 |

The 1-query/2048-KV case has a worse p95 despite a better p50. Runner variance is
visible and preserved, not filtered out. The unchanged long-context path and
multi-token hyper-connection path do not receive credit for timing noise.
At 128 queries/8192 KV the manual path measured 123.35879 ms p50, faster than the
refined fused path's 126.37475 ms. Fused SDPA is not universally faster.

For 128 queries/2048 KV, a separate untimed call measured additional peak MLX
allocated bytes above live inputs/weights dropping from 826770148 to 15999496
(about 788.47 to 15.26 MiB). This is component allocation, not whole-App physical
footprint, resident model size, or a hard memory bound.

## Correctness and reproduction

The refinement run passed 124 Qwen-suite test results (120 passing, 4 skipped
because existing local archive evidence was unavailable), and all 148 Swift test
results (145 passing, 3 existing installed-model discovery skips). It covered
EOS/overflow/hash parity; dense-prefix masks, threshold crossing and rollback;
MXFP4 output bytes and duplicate-route fallback; live compiled parameters;
native MTP state/next-logit parity and acceptance/rejection cursor alignment.
Final read-only CI additionally exercises packed QSA cache fork, restore and
replay across the threshold. It always runs the actual checked-out code.

```sh
# Existing pinned Python environment and Apple Silicon required.
git fetch origin
make test-qwen-flash
make test
make benchmark-qwen-speed QWEN_FLASH_OUTPUT=scratch/qwen-speed/components.json
```

A complete history containing `113be486422fa88deeeb96fbc29dc19ab7b27414` is needed
for the microbenchmark. It imports the original `qwen4_exp.py` from that revision
against the same pinned dependencies and unchanged shared helper contracts; it
is not a second full-model process. Reports refuse to overwrite old evidence.

For an installed-model pilot, reuse `pilot-qwen-flash` with variants `baseline`,
`sparse-sdpa`, `compiled` and the existing MTP variants. The current `baseline`
variant still contains this commit's unconditional exact hash/dispatch changes;
compare separate Git worktrees at the old and new commits to isolate those.
Never compare with different cache budgets, quantizations or prompt-cache states.

No complete checkpoint was loaded here. Full-model TTFT, Prefill/Decode rates,
quality and physical-device behavior remain unmeasured. Keep the sparse SDPA
experiment opt-in, and require real-prompt quality, cancellation, continuation,
and representative context-length measurements before treating these component
wins as a production speed profile.
