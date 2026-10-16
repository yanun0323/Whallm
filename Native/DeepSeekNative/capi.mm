#include "capi.h"

#include <cstring>
#include <string>

#include "engine.h"

struct wd_engine {
  std::unique_ptr<wd::Engine> engine;
  std::string error;
};

struct wd_state {
  std::shared_ptr<const wd::State> state;
};

namespace {

// Runs `body`, turning exceptions into a status and the engine's error message.
template <typename Body>
int guarded(wd_engine* e, Body body) {
  try {
    @autoreleasepool {
      body();
    }
    return 0;
  } catch (const wd::Cancelled&) {
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

wd_engine* wd_open(const char* model_dir, const wd_options* options, char* error, size_t error_size) {
  try {
    wd::Options o;
    if (options != nullptr) {
      o.slots = options->slots;
      if (options->read_threads > 0) o.read_threads = options->read_threads;
      if (options->chunk > 0) o.chunk = options->chunk;
      if (options->token_map != nullptr && options->token_map_count > 0)
        o.token_map.assign(options->token_map, options->token_map + options->token_map_count);
      o.dspark = options->dspark != 0;
    }
    auto* e = new wd_engine;
    @autoreleasepool {
      e->engine = wd::Engine::open(model_dir, o);
    }
    return e;
  } catch (const std::exception& failure) {
    if (error != nullptr && error_size > 0) {
      strncpy(error, failure.what(), error_size - 1);
      error[error_size - 1] = '\0';
    }
    return nullptr;
  }
}

void wd_close(wd_engine* e) { delete e; }
const char* wd_error(const wd_engine* e) { return e->error.c_str(); }

void wd_set_cancel(wd_engine* e, wd_cancel_fn check, void* context) {
  if (check == nullptr) e->engine->set_cancel(nullptr);
  else e->engine->set_cancel([check, context] { return check(context) != 0; });
}

int32_t wd_vocab(const wd_engine* e) { return e->engine->vocab(); }
int32_t wd_max_context(const wd_engine* e) { return e->engine->max_context(); }
int32_t wd_position(const wd_engine* e) { return e->engine->position(); }
const char* wd_contract(const wd_engine* e) { return e->engine->contract().c_str(); }

const int32_t* wd_tokens(const wd_engine* e, int32_t* count) {
  const std::vector<int32_t>& tokens = e->engine->tokens();
  *count = (int32_t)tokens.size();
  return tokens.data();
}

void wd_counters_get(const wd_engine* e, wd_counters* out) {
  wd::Counters c = e->engine->counters();
  out->expert_hits = c.expert_hits;
  out->expert_misses = c.expert_misses;
  out->experts_seeded = c.experts_seeded;
  out->experts_streamed = c.experts_streamed;
  out->read_seconds = c.read_seconds;
  out->gpu_seconds = c.gpu_seconds;
  out->gpu_bytes = c.gpu_bytes;
  out->slots = c.slots;
  out->resident = c.resident;
}

int wd_reset(wd_engine* e) { return guarded(e, [&] { e->engine->reset(); }); }

int wd_eval(wd_engine* e, const int32_t* tokens, int32_t count, int32_t logit_rows, const float** logits) {
  return guarded(e, [&] { *logits = e->engine->eval(tokens, count, logit_rows); });
}

int32_t wd_has_dspark(const wd_engine* e) { return e->engine->has_dspark() ? 1 : 0; }
int32_t wd_draft_block(const wd_engine* e) { return e->engine->draft_block(); }

int wd_draft(wd_engine* e, int32_t anchor) { return guarded(e, [&] { e->engine->draft(anchor); }); }

int wd_draft_step(wd_engine* e, int32_t index, int32_t previous, const float** logits, float* confidence) {
  return guarded(e, [&] { *logits = e->engine->draft_step(index, previous, confidence); });
}

int wd_verify(wd_engine* e, const int32_t* tokens, int32_t count, const float** logits) {
  return guarded(e, [&] { *logits = e->engine->verify(tokens, count); });
}

int wd_commit(wd_engine* e, int32_t keep) { return guarded(e, [&] { e->engine->commit(keep); }); }

wd_state* wd_snapshot(wd_engine* e) {
  wd_state* result = nullptr;
  if (guarded(e, [&] { result = new wd_state{e->engine->snapshot()}; }) != 0) return nullptr;
  return result;
}

int wd_restore(wd_engine* e, const wd_state* state) {
  return guarded(e, [&] { e->engine->restore(*state->state); });
}

wd_state* wd_load(wd_engine* e, const char* path) {
  wd_state* result = nullptr;
  if (guarded(e, [&] { result = new wd_state{e->engine->load(path)}; }) != 0) return nullptr;
  return result;
}

int wd_save(wd_engine* e, const wd_state* state, const char* path) {
  return guarded(e, [&] { e->engine->save(*state->state, path); });
}

void wd_state_free(wd_state* state) { delete state; }
int32_t wd_state_token_count(const wd_state* state) { return (int32_t)state->state->tokens.size(); }
const int32_t* wd_state_tokens(const wd_state* state) { return state->state->tokens.data(); }
uint64_t wd_state_bytes(const wd_state* state) {
  return state->state->data.size() + state->state->tokens.size() * sizeof(int32_t);
}

}  // extern "C"
