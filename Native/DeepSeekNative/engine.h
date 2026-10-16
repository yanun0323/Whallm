// Whallm native DeepSeek V4.1 engine: C++ and Metal, no MLX.
// Reads an installed model (manifest.json, config.json, common.bin, experts/, engram/).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wd {

// Everything the model holds after some tokens (prompt cache). `data` is
// engine-internal; only Engine::restore and Engine::save read it.
struct State {
  std::vector<int32_t> tokens;
  std::vector<uint8_t> data;
};

// Thrown when the cancel check of Engine::set_cancel returns true. The engine must
// be reset (or restored) before it is used again; the same holds for any other error.
struct Cancelled : std::runtime_error {
  Cancelled() : std::runtime_error("cancelled") {}
};

// Running totals since the engine was opened.
struct Counters {
  long expert_hits = 0, expert_misses = 0, experts_seeded = 0, experts_streamed = 0;
  double read_seconds = 0, gpu_seconds = 0;
  size_t gpu_bytes = 0;  // Metal memory the engine holds now
  int slots = 0;         // expert cache size in use
  int resident = 0;      // slots holding an expert
};

struct Options {
  int slots = 0;         // expert blobs kept in memory; 0: sized from the free memory
  int read_threads = 8;  // parallel SSD reads
  int chunk = 1024;      // prompt rows whose attention runs together inside one layer
  bool seed = true;      // after a long prompt, keep the experts it used most in the cache
  bool dspark = false;   // load the DSpark draft layers (the installed model must have them)
  // Compressed Engram token id of every vocabulary id. The tokenizer defines it
  // (see Scripts/deepseek_v41_reference.py); the engine checks its size and range.
  std::vector<int32_t> token_map;
  std::string dump_dir;  // debug: write the hidden stream after every layer of the first call here
};

class Engine {
 public:
  // Throws std::runtime_error with a readable message when the model cannot be used.
  static std::unique_ptr<Engine> open(const std::string& model_dir, const Options& options);
  virtual ~Engine() = default;

  // Forget all tokens.
  virtual void reset() = 0;
  // Tokens the model has consumed since the last reset.
  virtual int position() const = 0;
  // Consume tokens after the current position. Returns the logits of the last
  // `logit_rows` tokens (at most 8), row-major [logit_rows, vocab]; valid until
  // the next call that returns logits.
  virtual const float* eval(const int32_t* tokens, int count, int logit_rows) = 0;

  // Speculative decoding. A round is:
  //   draft(anchor)                draft layers over [anchor, 4 placeholder tokens]
  //   draft_step(i, previous, &c)  draft logits [vocab] for block position i, given the
  //                                token before it (anchor for i = 0); c is the draft's
  //                                confidence in that position (0 to 1)
  //   verify({anchor, d0, ...})    the model's own logits [count, vocab] after each token
  //   commit(keep)                 keep the first `keep` verified tokens
  // `anchor` is the token sampled from the last logits and not yet consumed. verify and
  // commit also work without DSpark. With the same tokens, verify gives the same numbers
  // as eval of one token at a time.
  virtual bool has_dspark() const = 0;
  virtual int draft_block() const = 0;
  virtual void draft(int32_t anchor) = 0;
  virtual const float* draft_step(int index, int32_t previous, float* confidence) = 0;
  // At most 8 tokens.
  virtual const float* verify(const int32_t* tokens, int count) = 0;
  // `keep` is from 1 to the verified count.
  virtual void commit(int keep) = 0;

  // Prompt cache. snapshot() copies the state after every consumed token;
  // restore() continues from it exactly as if those tokens were consumed again.
  virtual std::shared_ptr<const State> snapshot() = 0;
  virtual void restore(const State& state) = 0;
  // The tokens consumed since the last reset or restore.
  virtual const std::vector<int32_t>& tokens() const = 0;
  // Disk copies. load() refuses a file from another model, another engine format or
  // another DSpark setting.
  virtual void save(const State& state, const std::string& path) const = 0;
  virtual std::shared_ptr<const State> load(const std::string& path) const = 0;

  virtual int vocab() const = 0;
  virtual int max_context() const = 0;
  // Called between layers; returning true stops the call with Cancelled.
  virtual void set_cancel(std::function<bool()> check) = 0;
  virtual Counters counters() const = 0;
  // Identifies the model and the engine format (prompt cache files).
  virtual const std::string& contract() const = 0;
  // Counters for development: GPU time, submissions, expert cache hits and reads.
  virtual std::string stats() const = 0;
};

}  // namespace wd
