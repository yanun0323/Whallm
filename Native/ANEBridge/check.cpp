#include "NativeANE.h"
#include "../QwenNative/ANEWeights.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  using whallm_ane::ProjectionSet;
  for (float bad : {-0.1f, 1.1f, NAN, INFINITY}) {
    bool rejected = false;
    try { ProjectionSet projection(bad, 256, 1024, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  ProjectionSet off(0, 256, 1024, 1);
  assert(!off.requested() && !off.active());
  std::vector<uint16_t> converted;
  std::vector<uint16_t> bf16(2 * 64, 0x3f80);
  bf16[64] = 0xbf80;
  wq_ane::extract_rows(bf16.data(), nullptr, nullptr, 64, 1, 1, converted);
  assert(whallm_ane::half_value(converted[0]) == -1);
  assert(whallm_ane::half_value(converted[1]) == 1);
  std::vector<uint8_t> q8(2 * 64, 10);
  float sb[4] = {1, 0, 0.25f, -1};
  wq_ane::extract_rows(nullptr, q8.data(), sb, 64, 1, 1, converted);
  assert(whallm_ane::half_value(converted[0]) == 1.5f);
  setenv("WHALLM_ANE_BRIDGE", "/no/such/libWhallmANE.dylib", 1);
  ProjectionSet missing(0.25f, 256, 1024, 1);
  assert(missing.requested() && !missing.active() && !missing.diagnostic().empty());

  void* mock_library = dlopen(argv[2], RTLD_NOW);
  assert(mock_library);
  auto live = reinterpret_cast<int (*)()>(dlsym(mock_library, "whallm_mock_ane_live"));
  assert(live);
  setenv("WHALLM_ANE_BRIDGE", argv[2], 1);
  std::vector<uint16_t> mock_weights(256 * 256, whallm_ane::half_bits(1));
  setenv("WHALLM_MOCK_ANE_CREATE_FAIL_AFTER", "1", 1);
  {
    ProjectionSet compilation_failure(0.25f, 256, 1024, 2);
    assert(compilation_failure.add(0, mock_weights.data(), mock_weights.size()));
    assert(!compilation_failure.add(1, mock_weights.data(), mock_weights.size()));
    assert(!compilation_failure.active() && live() == 0);
  }
  unsetenv("WHALLM_MOCK_ANE_CREATE_FAIL_AFTER");
  setenv("WHALLM_MOCK_ANE_EVAL_FAIL", "1", 1);
  {
    ProjectionSet evaluation_failure(0.25f, 256, 1024, 1);
    assert(evaluation_failure.add(0, mock_weights.data(), mock_weights.size()));
    assert(evaluation_failure.finish());
    std::vector<float> zeros(1024 * 256, 0);
    assert(!evaluation_failure.evaluate(0, zeros.data()));
    assert(!evaluation_failure.active() && live() == 0);
  }
  unsetenv("WHALLM_MOCK_ANE_EVAL_FAIL");
  {
    ProjectionSet nonfinite_input(0.25f, 256, 1024, 1);
    assert(nonfinite_input.add(0, mock_weights.data(), mock_weights.size()));
    assert(nonfinite_input.finish());
    std::vector<float> input(1024 * 256, 0);
    input[0] = INFINITY;
    assert(!nonfinite_input.evaluate(0, input.data()));
    assert(!nonfinite_input.active() && live() == 0);
  }
  setenv("WHALLM_MOCK_ANE_NONFINITE", "1", 1);
  {
    ProjectionSet nonfinite_output(0.25f, 256, 1024, 1);
    assert(nonfinite_output.add(0, mock_weights.data(), mock_weights.size()));
    assert(nonfinite_output.finish());
    std::vector<float> input(1024 * 256, 0);
    assert(!nonfinite_output.evaluate(0, input.data()));
    assert(!nonfinite_output.active() && live() == 0);
  }
  unsetenv("WHALLM_MOCK_ANE_NONFINITE");
  dlclose(mock_library);

  setenv("WHALLM_ANE_BRIDGE", argv[1], 1);
  std::vector<float> input(1024 * 256);
  for (int row = 0; row < 1024; ++row)
    for (int col = 0; col < 256; ++col) input[(size_t)row * 256 + col] = (float)(row % 16 + col % 7);
  for (float share : {0.25f, 0.5f, 1.0f}) {
    ProjectionSet projection(share, 256, 1024, 1);
    assert(projection.channels() == (int)(share * 1024));
    std::vector<uint16_t> weights((size_t)projection.channels() * 256, 0);
    for (int out = 0; out < projection.channels(); ++out)
      weights[(size_t)out * 256 + out % 256] = whallm_ane::half_bits(1);
    assert(projection.add(0, weights.data(), weights.size()));
    assert(projection.finish());
    assert(projection.evaluate(0, input.data()));
    const uint16_t* output = projection.output();
    for (int row = 0; row < 1024; ++row)
      for (int col = 0; col < projection.channels(); ++col)
        assert(whallm_ane::half_value(output[(size_t)row * projection.channels() + col]) ==
               input[(size_t)row * 256 + col % 256]);
    printf("share %.2f channels %d %s\n", share, projection.channels(), projection.stats().c_str());
  }
}
