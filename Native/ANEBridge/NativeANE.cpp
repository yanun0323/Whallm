#include "NativeANE.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <stdexcept>
#include <chrono>

namespace whallm_ane {

uint16_t half_bits(float value) {
  __fp16 half = static_cast<__fp16>(value);
  uint16_t bits;
  memcpy(&bits, &half, sizeof(bits));
  return bits;
}

float half_value(uint16_t bits) {
  __fp16 half;
  memcpy(&half, &bits, sizeof(bits));
  return static_cast<float>(half);
}

ProjectionSet::ProjectionSet(float ratio, int input_channels, int output_channels, int layer_count)
    : input_channels_(input_channels), output_channels_(output_channels), layer_count_(layer_count),
      projections_(static_cast<size_t>(layer_count), nullptr) {
  if (!std::isfinite(ratio) || ratio < 0 || ratio > 1 || input_channels <= 0 ||
      output_channels <= 0 || output_channels % 256 || layer_count <= 0)
    throw std::invalid_argument("ANE share must be finite and between 0 and 1, with a valid projection shape");
  channels_ = std::min(output_channels, static_cast<int>(std::floor(ratio * output_channels / 256 + 0.5f)) * 256);
  if (!channels_) return;
  const char* configured = getenv("WHALLM_ANE_BRIDGE");
  const char* path = configured && *configured ? configured : "@loader_path/libWhallmANE.dylib";
  library_ = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (!library_ && !(configured && *configured))
    library_ = dlopen("@loader_path/../native/libWhallmANE.dylib", RTLD_NOW | RTLD_LOCAL);
  if (!library_) {
    disable(std::string("ANE bridge unavailable: ") + dlerror());
    return;
  }
  create_ = reinterpret_cast<Create>(dlsym(library_, "whallm_ane_projection_create"));
  evaluate_ = reinterpret_cast<Evaluate>(dlsym(library_, "whallm_ane_projection_evaluate"));
  destroy_ = reinterpret_cast<Destroy>(dlsym(library_, "whallm_ane_projection_destroy"));
  last_error_ = reinterpret_cast<LastError>(dlsym(library_, "whallm_ane_last_error"));
  if (!create_ || !evaluate_ || !destroy_ || !last_error_) {
    disable("ANE bridge is missing required symbols");
    return;
  }
  input_.resize(static_cast<size_t>(1024) * input_channels_);
  output_.resize(static_cast<size_t>(1024) * channels_);
}

ProjectionSet::~ProjectionSet() {
  close_projections();
  if (library_) dlclose(library_);
}

size_t ProjectionSet::reserved_bytes() const {
  if (!channels_ || !active_) return 0;
  // Compiled FP16 weights, both IOSurfaces per layer, and reusable host staging.
  return static_cast<size_t>(layer_count_) * (static_cast<size_t>(channels_) * input_channels_ * 2 +
      static_cast<size_t>(1024) * (input_channels_ + channels_) * 2) +
      static_cast<size_t>(1024) * (input_channels_ + channels_) * 2;
}

std::string ProjectionSet::bridge_error(const char* operation) const {
  const char* detail = last_error_ ? last_error_() : nullptr;
  return std::string(operation) + (detail && *detail ? std::string(": ") + detail : ": unknown bridge error");
}

void ProjectionSet::close_projections() {
  if (destroy_) for (void*& handle : projections_) {
    if (handle) destroy_(handle);
    handle = nullptr;
  }
  active_ = false;
}

void ProjectionSet::disable(const std::string& reason) {
  if (diagnostic_.empty()) fprintf(stderr, "Whallm ANE disabled; GPU fallback: %s\n", reason.c_str());
  diagnostic_ = reason;
  close_projections();
  ++fallbacks_;
}

bool ProjectionSet::add(int layer, const uint16_t* weights, size_t elements) {
  if (!channels_ || !library_ || !create_ || !diagnostic_.empty()) return false;
  if (layer < 0 || layer >= layer_count_ || projections_[layer] || !weights ||
      elements != static_cast<size_t>(channels_) * input_channels_) {
    disable("invalid ANE layer or weight shape");
    return false;
  }
  void* handle = create_(weights, elements, input_channels_, channels_, 1024);
  if (!handle) {
    disable(bridge_error("ANE projection compilation failed"));
    return false;
  }
  projections_[layer] = handle;
  return true;
}

bool ProjectionSet::finish() {
  if (!channels_ || !diagnostic_.empty()) return false;
  for (void* handle : projections_) if (!handle) {
    disable("ANE projection set is incomplete");
    return false;
  }
  active_ = true;
  fprintf(stderr, "Whallm ANE active: %d of %d projection channels across %d layers\n",
          channels_, output_channels_, layer_count_);
  return true;
}

bool ProjectionSet::evaluate(int layer, const float* input) {
  if (!active_ || layer < 0 || layer >= layer_count_ || !input) return false;
  for (size_t i = 0; i < input_.size(); ++i) {
    input_[i] = half_bits(input[i]);
    if ((input_[i] & 0x7c00u) == 0x7c00u) {
      disable("ANE input is not finite in FP16");
      return false;
    }
  }
  auto start = std::chrono::steady_clock::now();
  int status = evaluate_(projections_[layer], input_.data(), input_.size(), output_.data(), output_.size());
  ane_seconds_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (status != 0) {
    disable(bridge_error("ANE projection evaluation failed"));
    return false;
  }
  for (uint16_t bits : output_) if ((bits & 0x7c00u) == 0x7c00u) {
    disable("ANE projection produced a non-finite value");
    return false;
  }
  ++evaluations_;
  return true;
}

std::string ProjectionSet::stats() const {
  return "ane_active=" + std::to_string(active_) + " ane_channels=" + std::to_string(active_ ? channels_ : 0) +
      " ane_calls=" + std::to_string(evaluations_) + " ane_fallbacks=" + std::to_string(fallbacks_) +
      " ane_seconds=" + std::to_string(ane_seconds_) +
      (diagnostic_.empty() ? "" : " ane_error=" + diagnostic_);
}

}  // namespace whallm_ane
