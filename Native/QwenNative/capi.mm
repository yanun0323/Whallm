#include "capi.h"

#include <cstring>
#include <string>

#include "engine.h"

struct wq_engine {
  std::unique_ptr<wq::Engine> engine;
  std::string error;
};

struct wq_state {
  std::shared_ptr<const wq::State> state;
};

namespace {

// Runs `body`, turning exceptions into a status and the engine's error message.
template <typename Body>
int guarded(wq_engine* e, Body body) {
  try {
    @autoreleasepool {
      body();
    }
    return 0;
  } catch (const wq::Cancelled&) {
    e->error = "cancelled";
    return 2;
  } catch (const std::exception& error) {
    e->error = error.what();
    return 1;
  } catch (...) {
    e->error = "unknown error";
    return 1;
  }
}

}  // namespace

extern "C" {

wq_engine* wq_open(const char* model_dir, const wq_options* options, char* error, size_t error_size) {
  try {
    wq::Options o;
    if (options != nullptr) {
      o.slots = options->slots;
      if (options->read_threads > 0) o.read_threads = options->read_threads;
      o.mtp = options->mtp != 0;
      o.eight_bit = options->eight_bit != 0;
      if (options->vision_catalog != nullptr) o.vision_catalog = options->vision_catalog;
    }
    auto* e = new wq_engine;
    e->engine = wq::Engine::open(model_dir, o);
    return e;
  } catch (const std::exception& failure) {
    if (error != nullptr && error_size > 0) {
      strncpy(error, failure.what(), error_size - 1);
      error[error_size - 1] = '\0';
    }
    return nullptr;
  }
}

void wq_close(wq_engine* e) { delete e; }
const char* wq_error(const wq_engine* e) { return e->error.c_str(); }

void wq_set_cancel(wq_engine* e, wq_cancel_fn check, void* context) {
  if (check == nullptr) e->engine->set_cancel(nullptr);
  else e->engine->set_cancel([check, context] { return check(context) != 0; });
}

int32_t wq_vocab(const wq_engine* e) { return e->engine->vocab(); }
int32_t wq_position(const wq_engine* e) { return e->engine->position(); }
int32_t wq_has_mtp(const wq_engine* e) { return e->engine->has_mtp() ? 1 : 0; }
const char* wq_contract(const wq_engine* e) { return e->engine->contract().c_str(); }

const int32_t* wq_tokens(const wq_engine* e, int32_t* count) {
  const std::vector<int32_t>& tokens = e->engine->tokens();
  *count = (int32_t)tokens.size();
  return tokens.data();
}

void wq_counters_get(const wq_engine* e, wq_counters* out) {
  wq::Counters c = e->engine->counters();
  out->expert_hits = c.expert_hits;
  out->expert_misses = c.expert_misses;
  out->experts_seeded = c.experts_seeded;
  out->read_seconds = c.read_seconds;
  out->gpu_seconds = c.gpu_seconds;
  out->gpu_bytes = c.gpu_bytes;
  out->slots = c.slots;
  out->resident = c.resident;
  out->tensor_ops = c.tensor_ops ? 1 : 0;
}

int wq_reset(wq_engine* e) { return guarded(e, [&] { e->engine->reset(); }); }

int wq_eval(wq_engine* e, const int32_t* tokens, int32_t count, int32_t logit_rows, const float** logits) {
  return guarded(e, [&] { *logits = e->engine->eval(tokens, count, logit_rows); });
}

int wq_draft(wq_engine* e, int32_t token, const float** logits) {
  return guarded(e, [&] { *logits = e->engine->draft(token); });
}

int wq_verify(wq_engine* e, const int32_t* tokens, int32_t count, const float** logits) {
  return guarded(e, [&] { *logits = e->engine->verify(tokens, count); });
}

int wq_commit(wq_engine* e, int32_t keep) { return guarded(e, [&] { e->engine->commit(keep); }); }

int wq_eval_images(wq_engine* e, const int32_t* tokens, int32_t count, int32_t image_count,
                   const int32_t* grids, const float* const* patches, int32_t logit_rows, const float** logits) {
  return guarded(e, [&] {
    std::vector<wq::ImageInput> images((size_t)std::max(0, image_count));
    for (int32_t i = 0; i < image_count; ++i) {
      images[i].grid_h = grids[2 * i];
      images[i].grid_w = grids[2 * i + 1];
      size_t floats = (size_t)images[i].grid_h * images[i].grid_w * 1536;
      images[i].patches.assign(patches[i], patches[i] + floats);
    }
    *logits = e->engine->eval_images(tokens, count, images, logit_rows);
  });
}

wq_state* wq_snapshot(wq_engine* e) {
  wq_state* result = nullptr;
  if (guarded(e, [&] { result = new wq_state{e->engine->snapshot()}; }) != 0) return nullptr;
  return result;
}

int wq_restore(wq_engine* e, const wq_state* state) {
  return guarded(e, [&] { e->engine->restore(*state->state); });
}

wq_state* wq_load(wq_engine* e, const char* path) {
  wq_state* result = nullptr;
  if (guarded(e, [&] { result = new wq_state{e->engine->load(path)}; }) != 0) return nullptr;
  return result;
}

int wq_save(wq_engine* e, const wq_state* state, const char* path) {
  return guarded(e, [&] { e->engine->save(*state->state, path); });
}

void wq_state_free(wq_state* state) { delete state; }
int32_t wq_state_token_count(const wq_state* state) { return (int32_t)state->state->tokens.size(); }
const int32_t* wq_state_tokens(const wq_state* state) { return state->state->tokens.data(); }
uint64_t wq_state_bytes(const wq_state* state) {
  return state->state->data.size() + state->state->tokens.size() * sizeof(int32_t);
}

}  // extern "C"
