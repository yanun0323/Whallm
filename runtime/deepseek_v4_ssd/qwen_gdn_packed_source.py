"""Packed scalar-gate Dk=128 shader from mlx-lm PR #1559.
Pinned source: e1cc9c51442aef48ab206d23400720a1e2022448.
Source file SHA-256: ff4b382f593a8c5b400a51f141e6ac2743a9a878dc08c91b5fbc99b601a93a52

MIT License

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

PACKED_SOURCE = r"""
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;

        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);

        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;

        // q, k: [B, T, Hk, Dk]
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;

        // v, y: [B, T, Hv, Dv]
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;

        // state_in, state_out: [B, Hv, Dv, Dk]
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;

        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }

        // g, beta: [B, T, Hv]
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);

          // Partials mirror the generic kernel: each 4-element chain is one
          // original lane's sequential accumulation.
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          // Butterfly levels xor 1,2,4 stay inside this lane (commutative
          // pairwise tree); levels xor 8,16 become the row-group shuffles.
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);

          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);

          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }

          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """
