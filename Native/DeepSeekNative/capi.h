/* C interface of the Whallm native DeepSeek V4.1 engine, for the Python runtime (ctypes).
   Calls on one engine must come from one thread at a time. Functions that return
   an int status return 0 on success, 1 on an error (see wd_error) and 2 when the
   cancel callback stopped the call. After any nonzero status the engine must be
   reset (wd_reset) or restored (wd_restore) before it is used again. */
#ifndef WHALLM_DEEPSEEK_CAPI_H
#define WHALLM_DEEPSEEK_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define WD_EXPORT __attribute__((visibility("default")))
extern "C" {
#endif

typedef struct wd_engine wd_engine;
typedef struct wd_state wd_state;

typedef struct {
  int32_t slots;              /* expert cache size; 0 sizes it from the free memory */
  int32_t read_threads;       /* parallel SSD reads */
  int32_t chunk;              /* prompt rows whose attention runs together; 0: default */
  const int32_t* token_map;   /* compressed Engram id per vocabulary id */
  int32_t token_map_count;
  int32_t dspark;             /* load the DSpark draft layers */
} wd_options;

typedef struct {
  int64_t expert_hits, expert_misses, experts_seeded, experts_streamed;
  double read_seconds, gpu_seconds;
  uint64_t gpu_bytes;
  int32_t slots;
  int32_t resident;
} wd_counters;

/* Nonzero stops the running call with status 2. */
typedef int (*wd_cancel_fn)(void* context);

/* NULL on failure, with a message in error[0 .. error_size). */
WD_EXPORT wd_engine* wd_open(const char* model_dir, const wd_options* options, char* error, size_t error_size);
WD_EXPORT void wd_close(wd_engine* engine);
WD_EXPORT const char* wd_error(const wd_engine* engine);
WD_EXPORT void wd_set_cancel(wd_engine* engine, wd_cancel_fn check, void* context);

WD_EXPORT int32_t wd_vocab(const wd_engine* engine);
WD_EXPORT int32_t wd_max_context(const wd_engine* engine);
WD_EXPORT int32_t wd_position(const wd_engine* engine);
/* The tokens consumed since the last reset or restore; *count receives their number. */
WD_EXPORT const int32_t* wd_tokens(const wd_engine* engine, int32_t* count);
WD_EXPORT const char* wd_contract(const wd_engine* engine);
WD_EXPORT void wd_counters_get(const wd_engine* engine, wd_counters* out);

WD_EXPORT int wd_reset(wd_engine* engine);
/* logits: [logit_rows, vocab], valid until the next call that returns logits. */
WD_EXPORT int wd_eval(wd_engine* engine, const int32_t* tokens, int32_t count, int32_t logit_rows, const float** logits);

/* Speculative decoding (see Engine::draft in engine.h). */
WD_EXPORT int32_t wd_has_dspark(const wd_engine* engine);
WD_EXPORT int32_t wd_draft_block(const wd_engine* engine);
WD_EXPORT int wd_draft(wd_engine* engine, int32_t anchor);
/* logits: [vocab] for block position `index`; *confidence from 0 to 1. */
WD_EXPORT int wd_draft_step(wd_engine* engine, int32_t index, int32_t previous, const float** logits, float* confidence);
/* logits: [count, vocab]; count from 1 to 8. */
WD_EXPORT int wd_verify(wd_engine* engine, const int32_t* tokens, int32_t count, const float** logits);
WD_EXPORT int wd_commit(wd_engine* engine, int32_t keep);

/* Prompt cache. */
WD_EXPORT wd_state* wd_snapshot(wd_engine* engine);
WD_EXPORT int wd_restore(wd_engine* engine, const wd_state* state);
WD_EXPORT wd_state* wd_load(wd_engine* engine, const char* path);
WD_EXPORT int wd_save(wd_engine* engine, const wd_state* state, const char* path);
WD_EXPORT void wd_state_free(wd_state* state);
WD_EXPORT int32_t wd_state_token_count(const wd_state* state);
WD_EXPORT const int32_t* wd_state_tokens(const wd_state* state);
WD_EXPORT uint64_t wd_state_bytes(const wd_state* state);

#ifdef __cplusplus
}
#endif
#endif
