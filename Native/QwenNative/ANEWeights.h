// QSA q_proj weight conversion for the optional ANE projection.
#pragma once

#include "../ANEBridge/NativeANE.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace wq_ane {

inline void extract_rows(const uint16_t* bf16, const uint8_t* q8, const float* scale_bias,
                         int input_channels, int first_row, int row_count, std::vector<uint16_t>& out) {
  if (input_channels <= 0 || first_row < 0 || row_count <= 0 ||
      (q8 && (!scale_bias || input_channels % 64)) || (!q8 && !bf16))
    throw std::invalid_argument("invalid QSA ANE weight layout");
  out.resize((size_t)row_count * input_channels);
  for (int row = 0; row < row_count; ++row) {
    int source_row = first_row + row;
    for (int in = 0; in < input_channels; ++in) {
      float value;
      if (q8) {
        size_t group = ((size_t)source_row * input_channels / 64 + in / 64) * 2;
        value = (float)q8[(size_t)source_row * input_channels + in] * scale_bias[group] + scale_bias[group + 1];
      } else {
        uint32_t bits = (uint32_t)bf16[(size_t)source_row * input_channels + in] << 16;
        memcpy(&value, &bits, sizeof(value));
      }
      if (!std::isfinite(value)) throw std::invalid_argument("non-finite QSA ANE weight");
      uint16_t half = whallm_ane::half_bits(value);
      if ((half & 0x7c00u) == 0x7c00u) throw std::invalid_argument("QSA ANE weight overflows FP16");
      out[(size_t)row * input_channels + in] = half;
    }
  }
}

}  // namespace wq_ane
