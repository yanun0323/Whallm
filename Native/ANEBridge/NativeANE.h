// Optional ANE projections shared by the native DeepSeek and Qwen engines.
// The existing WhallmANE bridge owns the private framework and IOSurfaces.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace whallm_ane {

uint16_t half_bits(float value);
float half_value(uint16_t bits);

class ProjectionSet {
 public:
  ProjectionSet(float ratio, int input_channels, int output_channels, int layer_count);
  ~ProjectionSet();
  ProjectionSet(const ProjectionSet&) = delete;
  ProjectionSet& operator=(const ProjectionSet&) = delete;

  int channels() const { return channels_; }
  int gpu_channels() const { return output_channels_ - channels_; }
  bool active() const { return active_; }
  bool requested() const { return channels_ != 0; }
  size_t reserved_bytes() const;
  bool add(int layer, const uint16_t* weights, size_t elements);
  bool finish();
  bool evaluate(int layer, const float* input);
  const uint16_t* output() const { return output_.data(); }
  void disable(const std::string& reason);
  const std::string& diagnostic() const { return diagnostic_; }
  std::string stats() const;

 private:
  using Create = void* (*)(const uint16_t*, size_t, int, int, int);
  using Evaluate = int (*)(void*, const uint16_t*, size_t, uint16_t*, size_t);
  using Destroy = void (*)(void*);
  using LastError = const char* (*)();
  std::string bridge_error(const char* operation) const;
  void close_projections();

  int input_channels_, output_channels_, layer_count_, channels_ = 0;
  bool active_ = false;
  void* library_ = nullptr;
  Create create_ = nullptr;
  Evaluate evaluate_ = nullptr;
  Destroy destroy_ = nullptr;
  LastError last_error_ = nullptr;
  std::vector<void*> projections_;
  std::vector<uint16_t> input_, output_;
  std::string diagnostic_;
  long evaluations_ = 0, fallbacks_ = 0;
  double ane_seconds_ = 0;
};

}  // namespace whallm_ane
