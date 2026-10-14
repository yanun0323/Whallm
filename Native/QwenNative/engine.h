// Whallm native Qwen engine: C++ and Metal, no MLX.
// Reads an installed model (manifest.json, common.bin, experts/, ngram.bin, mtp/).
#pragma once
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace wq {

// Everything the model holds after some tokens (prompt cache). `data` is
// engine-internal; only Engine::restore and Engine::save read it.
// One image after the Qwen2-VL image processor: patches [grid_h * grid_w, 1536]
// (3 channels x 2 frames x 16 x 16) in the processor's order, grid t = 1.
// `features` may hold encode_image() output instead; patches are then not needed.
struct ImageInput {
  int grid_h = 0, grid_w = 0;
  std::vector<float> patches;
  std::vector<float> features;
};

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
  long expert_hits = 0, expert_misses = 0, experts_seeded = 0;
  double read_seconds = 0, gpu_seconds = 0;
  size_t gpu_bytes = 0;  // Metal memory the engine holds now
  int slots = 0;         // expert cache size in use
  int resident = 0;      // slots holding an expert
  bool tensor_ops = false;
};

struct Options {
  int slots = 0;          // expert blobs kept in memory; 0: sized from the free memory
  int read_threads = 8;   // parallel SSD reads
  int chunk = 1024;       // prompt rows processed together inside one layer
  bool mtp = true;        // load the draft layer when the installed model has one
  bool seed = true;       // after a long prompt, keep the experts it used most in the cache
  // Store projections with at least 256 outputs as 8-bit (64-value groups), as the
  // MLX engine's "8-bit resident weights" does. Output tokens can change.
  bool eight_bit = false;
  std::string dump_dir;   // debug: write every layer input of single-token calls here
  // The vision catalog (QwenVision.json) that lists the tensors of vision/common.bin.
  // Needed only for image input.
  std::string vision_catalog;
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

  // Speculative decoding with the draft layer (MTP). One round is:
  //   draft(anchor), draft(d1), ...   each returns the draft logits for the next token
  //   verify({anchor, d1, ..., dn})   the model's own logits after each of them
  //   commit(keep)                    keep the first `keep` verified tokens
  // `anchor` is the token sampled from the last logits and not yet consumed.
  // Every draft() call consumes its token in the draft layer only. A round with
  // no accepted draft is verify({anchor}) + commit(1).
  virtual bool has_mtp() const = 0;
  virtual const float* draft(int32_t token) = 0;
  // At most 8 tokens. Returns [count, vocab] logits.
  virtual const float* verify(const int32_t* tokens, int count) = 0;
  // `keep` is from 1 to the verified count. The tokens after it are forgotten.
  virtual void commit(int keep) = 0;

  // Image input. encode_image() returns [grid_h * grid_w / 4, 2560] features.
  // eval_images() starts a new request (position 0) with a prompt in which every
  // image appears as its vision-start token, grid_h * grid_w / 4 image tokens
  // and its vision-end token, in the order of `images`. Image prompts use
  // T/H/W rotary positions and are not kept in the prompt cache.
  virtual std::vector<float> encode_image(const float* patches, int grid_h, int grid_w) = 0;
  virtual const float* eval_images(const int32_t* tokens, int count, const std::vector<ImageInput>& images,
                                   int logit_rows) = 0;

  // Prompt cache. snapshot() copies the state after every consumed token;
  // restore() continues from it exactly as if those tokens were consumed again.
  // Neither may run inside a draft round.
  virtual std::shared_ptr<const State> snapshot() = 0;
  virtual void restore(const State& state) = 0;
  // The tokens consumed since the last reset or restore.
  virtual const std::vector<int32_t>& tokens() const = 0;
  // Disk copies. load() refuses a file from another model, another engine
  // format, or other settings that change the numbers (8-bit, draft layer).
  virtual void save(const State& state, const std::string& path) const = 0;
  virtual std::shared_ptr<const State> load(const std::string& path) const = 0;

  virtual int vocab() const = 0;
  // Called between layers and draft steps; returning true stops the call with Cancelled.
  virtual void set_cancel(std::function<bool()> check) = 0;
  virtual Counters counters() const = 0;
  // Identifies the model and the settings that change the numbers (prompt cache files).
  virtual const std::string& contract() const = 0;
  // Counters for development: GPU time, submissions, expert cache hits and reads.
  virtual std::string stats() const = 0;
};

}  // namespace wq
