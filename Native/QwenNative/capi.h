/* C interface of the Whallm native Qwen engine, for the Python runtime (ctypes).
   Calls on one engine must come from one thread at a time. Functions that return
   an int status return 0 on success, 1 on an error (see wq_error) and 2 when the
   cancel callback stopped the call. After any nonzero status the engine must be
   reset (wq_reset) or restored (wq_restore) before it is used again. */
#ifndef WHALLM_QWEN_CAPI_H
#define WHALLM_QWEN_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#define WQ_EXPORT __attribute__((visibility("default")))
extern "C" {
#endif

typedef struct wq_engine wq_engine;
typedef struct wq_state wq_state;

typedef struct {
  int32_t slots;          /* expert cache size; 0 sizes it from the free memory */
  int32_t read_threads;   /* parallel SSD reads */
  int32_t mtp;            /* load the draft layer when installed */
  int32_t eight_bit;      /* 8-bit resident weights */
  const char* vision_catalog; /* QwenVision.json path, or NULL */
} wq_options;

typedef struct {
  int64_t expert_hits, expert_misses, experts_seeded;
  double read_seconds, gpu_seconds;
  uint64_t gpu_bytes;
  int32_t slots;
  int32_t resident;
  int32_t tensor_ops;
} wq_counters;

/* Nonzero stops the running call with status 2. */
typedef int (*wq_cancel_fn)(void* context);

/* NULL on failure, with a message in error[0 .. error_size). */
WQ_EXPORT wq_engine* wq_open(const char* model_dir, const wq_options* options, char* error, size_t error_size);
WQ_EXPORT void wq_close(wq_engine* engine);
WQ_EXPORT const char* wq_error(const wq_engine* engine);
WQ_EXPORT void wq_set_cancel(wq_engine* engine, wq_cancel_fn check, void* context);

WQ_EXPORT int32_t wq_vocab(const wq_engine* engine);
WQ_EXPORT int32_t wq_position(const wq_engine* engine);
WQ_EXPORT int32_t wq_has_mtp(const wq_engine* engine);
/* The tokens consumed since the last reset or restore; *count receives their number. */
WQ_EXPORT const int32_t* wq_tokens(const wq_engine* engine, int32_t* count);
WQ_EXPORT const char* wq_contract(const wq_engine* engine);
WQ_EXPORT void wq_counters_get(const wq_engine* engine, wq_counters* out);

WQ_EXPORT int wq_reset(wq_engine* engine);
/* logits: [logit_rows, vocab], valid until the next call that returns logits. */
WQ_EXPORT int wq_eval(wq_engine* engine, const int32_t* tokens, int32_t count, int32_t logit_rows, const float** logits);
WQ_EXPORT int wq_draft(wq_engine* engine, int32_t token, const float** logits);
WQ_EXPORT int wq_verify(wq_engine* engine, const int32_t* tokens, int32_t count, const float** logits);
WQ_EXPORT int wq_commit(wq_engine* engine, int32_t keep);
/* grids: image_count pairs (rows, columns) of 16-pixel patches; patches[i]: rows *
   columns * 1536 floats. Starts a new request. */
WQ_EXPORT int wq_eval_images(wq_engine* engine, const int32_t* tokens, int32_t count, int32_t image_count,
                   const int32_t* grids, const float* const* patches, int32_t logit_rows, const float** logits);

/* Prompt cache. */
WQ_EXPORT wq_state* wq_snapshot(wq_engine* engine);
WQ_EXPORT int wq_restore(wq_engine* engine, const wq_state* state);
WQ_EXPORT wq_state* wq_load(wq_engine* engine, const char* path);
WQ_EXPORT int wq_save(wq_engine* engine, const wq_state* state, const char* path);
WQ_EXPORT void wq_state_free(wq_state* state);
WQ_EXPORT int32_t wq_state_token_count(const wq_state* state);
WQ_EXPORT const int32_t* wq_state_tokens(const wq_state* state);
WQ_EXPORT uint64_t wq_state_bytes(const wq_state* state);

#ifdef __cplusplus
}
#endif
#endif
