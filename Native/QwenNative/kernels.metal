// Metal kernels for the Whallm native Qwen engine.
// Activations are FP32. Resident weights stay in the checkpoint's BF16 bytes and
// routed experts stay in MXFP4; both are decoded where they are read.
#include <metal_stdlib>
using namespace metal;

static inline float bf16(ushort v) { return as_type<float>(uint(v) << 16); }
static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float silu(float x) { return x * sigm(x); }

// ---------------------------------------------------------------- linear

struct LinearP { uint T; uint I; uint O; uint x_stride; uint y_stride; };

// y[t, o] = sum_i x[t, i] * w[o, i]
// One simdgroup per (output, row): each of the 32 lanes takes every 32nd group of
// four weights. Grid (32, O, T).
kernel void linear_bf16(const device float* x [[buffer(0)]],
                        const device ushort* w [[buffer(1)]],
                        device float* y [[buffer(2)]],
                        constant LinearP& p [[buffer(3)]],
                        uint3 g [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const device packed_ushort4* wr = (const device packed_ushort4*)(w + ulong(g.y) * p.I);
  const device packed_float4* xq = (const device packed_float4*)(x + ulong(g.z) * p.x_stride);
  uint quads = p.I / 4;
  float4 sum = 0;
  for (uint i = lane; i < quads; i += 32)
    sum += as_type<float4>(uint4(ushort4(wr[i])) << 16) * float4(xq[i]);
  float total = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
  if (lane == 0) y[ulong(g.z) * p.y_stride + g.y] = total;
}

// The same for blocks of 4 rows: a block reads the weight row once for all of
// its rows. Rows past the block's end repeat its last row. Grid (32, O, blocks).
kernel void linear_bf16_4(const device float* x [[buffer(0)]],
                          const device ushort* w [[buffer(1)]],
                          device float* y [[buffer(2)]],
                          constant LinearP& p [[buffer(3)]],
                          uint3 g [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * 4;
  uint rows = min(4u, p.T - first);
  const device packed_ushort4* wr = (const device packed_ushort4*)(w + ulong(g.y) * p.I);
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  uint quads = p.I / 4;
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint i = lane; i < quads; i += 32) {
    float4 weight = as_type<float4>(uint4(ushort4(wr[i])) << 16);
    a0 += weight * float4(x0[i]); a1 += weight * float4(x1[i]);
    a2 += weight * float4(x2[i]); a3 += weight * float4(x3[i]);
  }
  float t0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w));
  float t1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float t2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w));
  float t3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  if (lane != 0) return;
  y[ulong(first) * p.y_stride + g.y] = t0;
  if (1u < rows) y[ulong(first + 1) * p.y_stride + g.y] = t1;
  if (2u < rows) y[ulong(first + 2) * p.y_stride + g.y] = t2;
  if (3u < rows) y[ulong(first + 3) * p.y_stride + g.y] = t3;
}

// ---------------------------------------------------------------- norms

struct NormP { uint T; uint groups; uint D; uint row_stride; uint base; uint w_period; uint has_w; float eps; float scale; };

// Group (t, n) holds D values at t * row_stride + base + n * D; in and out share the layout.
// One simdgroup per group; D is a multiple of 32.
kernel void rms_norm(const device float* x [[buffer(0)]],
                     const device float* w [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant NormP& p [[buffer(3)]],
                     uint3 g [[thread_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
  ulong at = ulong(g.z) * p.row_stride + p.base + g.y * p.D;
  const device float* xr = x + at;
  device float* yr = y + at;
  float ss = 0;
  for (uint i = lane; i < p.D; i += 32) ss += xr[i] * xr[i];
  float r = rsqrt(simd_sum(ss) / float(p.D) + p.eps) * p.scale;
  if (p.has_w != 0) {
    const device float* wr = w + (g.y % p.w_period) * p.D;
    for (uint i = lane; i < p.D; i += 32) yr[i] = xr[i] * r * wr[i];
  } else {
    for (uint i = lane; i < p.D; i += 32) yr[i] = xr[i] * r;
  }
}

// ---------------------------------------------------------------- hyper-connection

struct CountP { uint n; float scale; };

kernel void silu_scaled(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                        constant CountP& p [[buffer(2)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  y[g] = silu(x[g] * p.scale);
}

struct HcP { uint T; uint H; uint C; };

// mixed[t, i] = mean_s sigmoid(up[t, s, i]) * normed[t, s, i]
kernel void hc_mix(const device float* up [[buffer(0)]], const device float* normed [[buffer(1)]],
                   device float* mixed [[buffer(2)]], constant HcP& p [[buffer(3)]],
                   uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H || g.y >= p.T) return;
  ulong base = ulong(g.y) * p.H * p.C + g.x;
  float acc = 0;
  for (uint s = 0; s < p.C; ++s) acc += sigm(up[base + s * p.H]) * normed[base + s * p.H];
  mixed[ulong(g.y) * p.H + g.x] = acc / float(p.C);
}

// hidden[t, s, i] += result[t, i] * 2 * sigmoid(inject[t, s] / C)
kernel void hc_inject(device float* hidden [[buffer(0)]], const device float* result [[buffer(1)]],
                      const device float* inject [[buffer(2)]], constant HcP& p [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H * p.C || g.y >= p.T) return;
  uint s = g.x / p.H, i = g.x % p.H;
  float gate = 2.0f * sigm(inject[g.y * p.C + s] / float(p.C));
  hidden[ulong(g.y) * p.H * p.C + g.x] += result[ulong(g.y) * p.H + i] * gate;
}

// ---------------------------------------------------------------- MoE

kernel void mul_silu(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                     device float* y [[buffer(2)]], constant CountP& p [[buffer(3)]],
                     uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  y[g] = silu(a[g]) * b[g];
}

// x_row: input row; slot: cache slot of the expert; out: output row (also the input row
// of the down projection).
struct Pair { uint x_row; uint slot; uint out; };
struct ExpertP { uint pairs; uint I; uint O; uint x_stride; uint w_offset; uint s_offset; uint blob; uint x_is_pair; };

static inline float fp4(uint n) {
  const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float v = table[n & 7];
  return (n & 8) != 0 ? -v : v;
}

// y[pair, o] = sum_i x[x_row, i] * mxfp4(w[slot][o, i]); 32 values share one E8M0 scale.
// One simdgroup per (pair, output); each lane takes every 32nd word of 8 values.
kernel void expert_linear(const device float* x [[buffer(0)]],
                          const device uchar* blobs [[buffer(1)]],
                          const device Pair* pairs [[buffer(2)]],
                          device float* y [[buffer(3)]],
                          constant ExpertP& p [[buffer(4)]],
                          uint3 g [[thread_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
  Pair pair = pairs[g.z];
  const device uchar* blob = blobs + ulong(pair.slot) * p.blob;
  const device uint* wr = (const device uint*)(blob + p.w_offset) + ulong(g.y) * (p.I / 8);
  const device uchar* sr = blob + p.s_offset + ulong(g.y) * (p.I / 32);
  const device float* xr = x + ulong(p.x_is_pair != 0 ? pair.out : pair.x_row) * p.x_stride;
  float total = 0;
  for (uint word = lane; word < p.I / 8; word += 32) {
    uint bits = wr[word];
    const device float* xs = xr + word * 8;
    float acc = xs[0] * fp4(bits) + xs[1] * fp4(bits >> 4) + xs[2] * fp4(bits >> 8) + xs[3] * fp4(bits >> 12)
              + xs[4] * fp4(bits >> 16) + xs[5] * fp4(bits >> 20) + xs[6] * fp4(bits >> 24) + xs[7] * fp4(bits >> 28);
    total += acc * as_type<float>(uint(sr[word / 4]) << 23);
  }
  total = simd_sum(total);
  if (lane == 0) y[ulong(pair.out) * p.O + g.y] = total;
}

struct ActP { uint pairs; uint M; };

// act[pair, i] = silu(gate_up[pair, i]) * gate_up[pair, M + i]
kernel void expert_act(const device float* gate_up [[buffer(0)]], device float* act [[buffer(1)]],
                       constant ActP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.M || g.y >= p.pairs) return;
  ulong base = ulong(g.y) * 2 * p.M;
  act[ulong(g.y) * p.M + g.x] = silu(gate_up[base + g.x]) * gate_up[base + p.M + g.x];
}

struct CombineP { uint T; uint H; uint K; };

// out[t, i] = sum_k score[t, k] * routed[t * K + k, i] + sigmoid(shared_gate[t]) * shared[t, i]
kernel void moe_combine(const device float* routed [[buffer(0)]], const device float* score [[buffer(1)]],
                        const device float* shared [[buffer(2)]], const device float* shared_gate [[buffer(3)]],
                        device float* y [[buffer(4)]], constant CombineP& p [[buffer(5)]],
                        uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H || g.y >= p.T) return;
  float acc = 0;
  for (uint k = 0; k < p.K; ++k)
    acc += score[g.y * p.K + k] * routed[(ulong(g.y) * p.K + k) * p.H + g.x];
  y[ulong(g.y) * p.H + g.x] = acc + sigm(shared_gate[g.y]) * shared[ulong(g.y) * p.H + g.x];
}

// ---------------------------------------------------------------- depthwise causal convolution

struct ConvP { uint T; uint C; uint K; uint dilation; uint state_rows; uint apply_silu; };

static inline float conv_in(const device float* state, const device float* x, uint row, uint c, uint S, uint C) {
  return row < S ? state[ulong(row) * C + c] : x[ulong(row - S) * C + c];
}

// y[t, c] = sum_j w[c, j] * in[t + j * dilation, c], in = concat(state, x)
kernel void conv_depthwise(const device float* state [[buffer(0)]], const device float* x [[buffer(1)]],
                           const device ushort* w [[buffer(2)]], device float* y [[buffer(3)]],
                           constant ConvP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.C || g.y >= p.T) return;
  float acc = 0;
  for (uint j = 0; j < p.K; ++j)
    acc += bf16(w[g.x * p.K + j]) * conv_in(state, x, g.y + j * p.dilation, g.x, p.state_rows, p.C);
  y[ulong(g.y) * p.C + g.x] = p.apply_silu != 0 ? silu(acc) : acc;
}

// next = the last state_rows rows of concat(state, x). The old state is left
// untouched so a rejected verification can start again from it.
kernel void conv_state_next(const device float* state [[buffer(0)]], const device float* x [[buffer(1)]],
                            device float* next [[buffer(2)]], constant ConvP& p [[buffer(3)]],
                            uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.C || g.y >= p.state_rows) return;
  next[ulong(g.y) * p.C + g.x] = conv_in(state, x, p.T + g.y, g.x, p.state_rows, p.C);
}

// ---------------------------------------------------------------- Gated DeltaNet

struct GdnP { uint T; uint Hk; uint Hv; uint Dk; uint Dv; uint row; uint key_dim; };

// g = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b)
kernel void gdn_gates(const device float* a [[buffer(0)]], const device float* b [[buffer(1)]],
                      const device float* a_log [[buffer(2)]], const device float* dt_bias [[buffer(3)]],
                      device float* decay [[buffer(4)]], device float* beta [[buffer(5)]],
                      constant GdnP& p [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.Hv || g.y >= p.T) return;
  uint i = g.y * p.Hv + g.x;
  float v = a[i] + dt_bias[g.x];
  float softplus = v > 20.0f ? v : log(1.0f + exp(v));
  decay[i] = exp(-exp(a_log[g.x]) * softplus);
  beta[i] = sigm(b[i]);
}

// The delta-rule recurrence, one simdgroup per (value head, value row). Each of
// the 32 lanes holds Dk / 32 state values. qkv rows are [q | k | v].
kernel void gdn_step(const device float* qkv [[buffer(0)]],
                     const device float* decay [[buffer(1)]],
                     const device float* beta [[buffer(2)]],
                     const device float* state_in [[buffer(3)]],
                     device float* state_out [[buffer(4)]],
                     device float* y [[buffer(5)]],
                     constant GdnP& p [[buffer(6)]],
                     uint3 grid [[thread_position_in_grid]],
                     uint3 local [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
  const uint per = 4;  // Dk / 32
  uint hv = grid.z, dv = grid.y, dk = local.x;
  uint hk = hv / (p.Hv / p.Hk);
  ulong so = (ulong(hv) * p.Dv + dv) * p.Dk + dk * per;
  float state[per];
  for (uint i = 0; i < per; ++i) state[i] = state_in[so + i];
  for (uint t = 0; t < p.T; ++t) {
    const device float* row = qkv + ulong(t) * p.row;
    const device float* q = row + hk * p.Dk + dk * per;
    const device float* k = row + p.key_dim + hk * p.Dk + dk * per;
    float v = row[2 * p.key_dim + hv * p.Dv + dv];
    float gt = decay[t * p.Hv + hv];
    float mem = 0;
    for (uint i = 0; i < per; ++i) { state[i] *= gt; mem += state[i] * k[i]; }
    mem = simd_sum(mem);
    float delta = (v - mem) * beta[t * p.Hv + hv];
    float out = 0;
    for (uint i = 0; i < per; ++i) { state[i] += k[i] * delta; out += state[i] * q[i]; }
    out = simd_sum(out);
    if (lane == 0) y[(ulong(t) * p.Hv + hv) * p.Dv + dv] = out;
  }
  for (uint i = 0; i < per; ++i) state_out[so + i] = state[i];
}

struct GatedNormP { uint N; uint D; float eps; };

// y = rms_norm(x, w) * sigmoid(z), one head per thread
kernel void gated_rms_norm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                           const device float* z [[buffer(2)]], device float* y [[buffer(3)]],
                           constant GatedNormP& p [[buffer(4)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.N) return;
  ulong base = ulong(g) * p.D;
  float ss = 0;
  for (uint i = 0; i < p.D; ++i) ss += x[base + i] * x[base + i];
  float r = rsqrt(ss / float(p.D) + p.eps);
  for (uint i = 0; i < p.D; ++i) y[base + i] = x[base + i] * r * w[i] * sigm(z[base + i]);
}

// ---------------------------------------------------------------- QSA attention

struct RopeP { uint T; uint groups; uint D; uint in_stride; uint in_offset; uint group_stride; uint rotary; uint pos0; float eps; float ln_theta; uint has_norm;
               uint map_len; int delta; };

// Position of sequence index i on rotary axis a (0: time, 1: height, 2: width). An image
// prompt maps its indices through `map`; later indices continue at i + delta. Text has map_len 0.
static inline float axis_position(const device int4* map, uint map_len, int delta, uint i, uint a) {
  return i < map_len ? float(map[i][a]) : float(int(i) + delta);
}

// Frequency pair i rotates by the position on axis i % 3 (interleaved T/H/W sections).
static inline void norm_rope(const device float* x, const device float* w, thread float* out,
                             uint D, uint rotary, uint index, float eps, float ln_theta, bool normalize,
                             const device int4* map, uint map_len, int delta) {
  float r = 1.0f;
  if (normalize) {
    float ss = 0;
    for (uint i = 0; i < D; ++i) ss += x[i] * x[i];
    r = rsqrt(ss / float(D) + eps);
  }
  uint half_dims = rotary / 2;
  for (uint i = 0; i < half_dims; ++i) {
    float a = normalize ? x[i] * r * w[i] : x[i];
    float b = normalize ? x[i + half_dims] * r * w[i + half_dims] : x[i + half_dims];
    float position = map_len > 0 ? axis_position(map, map_len, delta, index, i % 3) : float(index);
    float angle = position * exp(-ln_theta * float(2 * i) / float(rotary));
    float c = cos(angle), s = sin(angle);
    out[i] = a * c - b * s;
    out[i + half_dims] = b * c + a * s;
  }
  for (uint i = rotary; i < D; ++i) out[i] = normalize ? x[i] * r * w[i] : x[i];
}

// Group (t, n) reads D values at t * in_stride + in_offset + n * group_stride.
// group_stride is D for packed heads and 2 * D for the query's [query | gate] pairs.

kernel void norm_rope_f32(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                          device float* y [[buffer(2)]], const device int4* map [[buffer(3)]],
                          constant RopeP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.groups || g.y >= p.T) return;
  float out[256];
  norm_rope(x + ulong(g.y) * p.in_stride + p.in_offset + g.x * p.group_stride, w, out,
            p.D, p.rotary, p.pos0 + g.y, p.eps, p.ln_theta, p.has_norm != 0, map, p.map_len, p.delta);
  device float* yr = y + (ulong(g.y) * p.groups + g.x) * p.D;
  for (uint i = 0; i < p.D; ++i) yr[i] = out[i];
}

kernel void norm_rope_f16(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                          device half* y [[buffer(2)]], const device int4* map [[buffer(3)]],
                          constant RopeP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.groups || g.y >= p.T) return;
  float out[256];
  norm_rope(x + ulong(g.y) * p.in_stride + p.in_offset + g.x * p.group_stride, w, out,
            p.D, p.rotary, p.pos0 + g.y, p.eps, p.ln_theta, p.has_norm != 0, map, p.map_len, p.delta);
  device half* yr = y + (ulong(g.y) * p.groups + g.x) * p.D;
  for (uint i = 0; i < p.D; ++i) yr[i] = half(out[i]);
}

struct CopyP { uint T; uint D; uint in_stride; uint in_offset; };

kernel void copy_f16(const device float* x [[buffer(0)]], device half* y [[buffer(1)]],
                     constant CopyP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.D || g.y >= p.T) return;
  y[ulong(g.y) * p.D + g.x] = half(x[ulong(g.y) * p.in_stride + p.in_offset + g.x]);
}

struct PoolP { uint blocks; uint first; uint ratio; uint D; uint rotary; float eps; float ln_theta; uint map_len; int delta; };

// pooled[b] = rope(k_layernorm(mean of the block's raw index keys), position b * ratio)
kernel void qsa_pool(const device half* raw [[buffer(0)]], const device float* w [[buffer(1)]],
                     device float* pooled [[buffer(2)]], const device int4* map [[buffer(3)]],
                     constant PoolP& p [[buffer(4)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.blocks) return;
  uint block = p.first + g;
  float mean[128];
  float ss = 0;
  for (uint i = 0; i < p.D; ++i) {
    float acc = 0;
    for (uint j = 0; j < p.ratio; ++j) acc += float(raw[(ulong(block) * p.ratio + j) * p.D + i]);
    mean[i] = acc / float(p.ratio);
    ss += mean[i] * mean[i];
  }
  float r = rsqrt(ss / float(p.D) + p.eps);
  device float* out = pooled + ulong(block) * p.D;
  uint half_dims = p.rotary / 2;
  uint index = block * p.ratio;
  for (uint i = 0; i < half_dims; ++i) {
    float a = mean[i] * r * w[i], b = mean[i + half_dims] * r * w[i + half_dims];
    float position = p.map_len > 0 ? axis_position(map, p.map_len, p.delta, index, i % 3) : float(index);
    float angle = position * exp(-p.ln_theta * float(2 * i) / float(p.rotary));
    float c = cos(angle), s = sin(angle);
    out[i] = a * c - b * s;
    out[i + half_dims] = b * c + a * s;
  }
  for (uint i = p.rotary; i < p.D; ++i) out[i] = mean[i] * r * w[i];
}

struct BlockScoreP { uint T; uint blocks; uint heads; uint D; uint pos0; uint ratio; };

// score[t, b] = sum_h relu(index_query[t, h] . pooled[b]); blocks not yet complete get -FLT_MAX.
kernel void qsa_block_scores(const device float* index_query [[buffer(0)]], const device float* pooled [[buffer(1)]],
                             device float* score [[buffer(2)]], constant BlockScoreP& p [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.blocks || g.y >= p.T) return;
  float total = -FLT_MAX;
  if (g.x * p.ratio + p.ratio - 1 <= p.pos0 + g.y) {
    total = 0;
    const device float* k = pooled + ulong(g.x) * p.D;
    for (uint h = 0; h < p.heads; ++h) {
      const device float* q = index_query + (ulong(g.y) * p.heads + h) * p.D;
      float acc = 0;
      for (uint i = 0; i < p.D; ++i) acc += q[i] * k[i];
      total += max(acc, 0.0f);
    }
  }
  score[ulong(g.y) * p.blocks + g.x] = total;
}

struct AttentionP { uint T; uint K; uint heads; uint kv_heads; uint D; uint pos0; uint use_selection; float scale; };

static inline int key_index(const device int* selection, constant AttentionP& p, uint t, uint j) {
  int index = p.use_selection != 0 ? selection[ulong(t) * p.K + j] : int(j);
  return (index >= 0 && uint(index) <= p.pos0 + t) ? index : -1;
}

// score[t, h, j] = query[t, h] . key[j, kv(h)] * scale, for all query heads of one KV head at once.
// 12 query heads share each KV head (checked when the model is opened).
kernel void qsa_scores(const device float* query [[buffer(0)]], const device half* keys [[buffer(1)]],
                       const device int* selection [[buffer(2)]], device float* score [[buffer(3)]],
                       constant AttentionP& p [[buffer(4)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.K || g.y >= p.kv_heads || g.z >= p.T) return;
  const uint per = 12;
  int index = key_index(selection, p, g.z, g.x);
  float acc[per];
  for (uint h = 0; h < per; ++h) acc[h] = 0;
  if (index >= 0) {
    const device packed_float4* q = (const device packed_float4*)(query + (ulong(g.z) * p.heads + g.y * per) * p.D);
    const device packed_half4* k = (const device packed_half4*)(keys + (ulong(index) * p.kv_heads + g.y) * p.D);
    uint quads = p.D / 4;
    for (uint i = 0; i < quads; ++i) {
      float4 key = float4(half4(k[i]));
      for (uint h = 0; h < per; ++h) acc[h] += dot(float4(q[h * quads + i]), key);
    }
  }
  device float* out = score + (ulong(g.z) * p.heads + g.y * per) * p.K + g.x;
  for (uint h = 0; h < per; ++h) out[ulong(h) * p.K] = index >= 0 ? acc[h] * p.scale : -FLT_MAX;
}

// In place: one simdgroup per (query head, token), lanes stride over the keys.
kernel void qsa_softmax(device float* score [[buffer(0)]], constant AttentionP& p [[buffer(1)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  device float* row = score + (ulong(g.z) * p.heads + g.y) * p.K;
  float top = -FLT_MAX;
  for (uint j = lane; j < p.K; j += 32) top = max(top, row[j]);
  top = simd_max(top);
  float total = 0;
  for (uint j = lane; j < p.K; j += 32) {
    float e = row[j] == -FLT_MAX ? 0.0f : exp(row[j] - top);
    row[j] = e;
    total += e;
  }
  float inverse = 1.0f / simd_sum(total);
  for (uint j = lane; j < p.K; j += 32) row[j] *= inverse;
}

// out[t, h, d] = sigmoid(gate[t, h, d]) * sum_j weight[t, h, j] * value[key_j, kv(h), d]
// One simdgroup per (4 value dims, query head, token): lanes stride over the keys.
// The gate is the second half of each head's q_proj output.
kernel void qsa_output(const device float* weight [[buffer(0)]], const device half* values [[buffer(1)]],
                       const device int* selection [[buffer(2)]], const device float* q_proj [[buffer(3)]],
                       device float* y [[buffer(4)]], constant AttentionP& p [[buffer(5)]],
                       uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint t = g.z / p.heads, head = g.z % p.heads, kv = head / (p.heads / p.kv_heads);
  const device float* row = weight + (ulong(t) * p.heads + head) * p.K;
  float4 acc = 0;
  for (uint j = lane; j < p.K; j += 32) {
    // Keys outside the selection have weight 0; read key 0 for them instead of branching.
    uint index = uint(max(key_index(selection, p, t, j), 0));
    acc += row[j] * float4(half4(((const device packed_half4*)(values + (ulong(index) * p.kv_heads + kv) * p.D))[g.y]));
  }
  float4 total = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
  if (lane != 0) return;
  float4 gate = ((const device packed_float4*)(q_proj + ulong(t) * p.heads * 2 * p.D + head * 2 * p.D + p.D))[g.y];
  ((device packed_float4*)(y + (ulong(t) * p.heads + head) * p.D))[g.y] = total / (1.0f + exp(-gate));
}

// ---------------------------------------------------------------- PLE

struct PleP { uint T; uint H; uint C; };

// gate[t, s] = sigmoid(signed_sqrt(key[t, s] . query[t, s] / sqrt(H)))
kernel void ple_gate(const device float* key [[buffer(0)]], const device float* query [[buffer(1)]],
                     device float* gate [[buffer(2)]], constant PleP& p [[buffer(3)]],
                     uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.C || g.y >= p.T) return;
  ulong base = (ulong(g.y) * p.C + g.x) * p.H;
  float acc = 0;
  for (uint i = 0; i < p.H; ++i) acc += key[base + i] * query[base + i];
  acc /= sqrt(float(p.H));
  float root = sqrt(max(abs(acc), 1e-6f));
  gate[g.y * p.C + g.x] = sigm(acc > 0 ? root : (acc < 0 ? -root : 0.0f));
}

// value[t, s, i] = gate[t, s] * projected[t, i]
kernel void ple_value(const device float* gate [[buffer(0)]], const device float* projected [[buffer(1)]],
                      device float* value [[buffer(2)]], constant PleP& p [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H * p.C || g.y >= p.T) return;
  value[ulong(g.y) * p.H * p.C + g.x] = gate[g.y * p.C + g.x / p.H] * projected[ulong(g.y) * p.H + g.x % p.H];
}

// hidden += value + conv
kernel void add2(device float* hidden [[buffer(0)]], const device float* a [[buffer(1)]],
                 const device float* b [[buffer(2)]], constant CountP& p [[buffer(3)]],
                 uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  hidden[g] += a[g] + b[g];
}

// One MXFP4 value times its group's E8M0 scale, built from the bits alone:
// value = (-1)^s * (e == 0 ? m / 2 : (1 + m / 2) * 2^(e - 1)) * 2^(scale - 127).
static inline float fp4_scaled(uint nibble, uint scale) {
  uint m = nibble & 1, e = (nibble >> 1) & 3;
  if (scale == 0) return 0.0f;  // an all-zero group
  uint bits = e != 0 ? (((e - 1 + scale) << 23) | (m << 22)) : (m != 0 ? ((scale - 1) << 23) : 0u);
  return as_type<float>(bits | ((nibble & 8) << 28));
}

// ---------------------------------------------------------------- batched matrix products (prompt reading)
// Rows are first packed 8 at a time as [block][i][8], so one thread reads its
// inputs as a single stream and keeps an 8-row by 4-output tile in registers.

struct PackP { uint T; uint I; uint x_stride; uint blocks; };

kernel void pack_rows8(const device float* x [[buffer(0)]], device float* packed [[buffer(1)]],
                       constant PackP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.I || g.y >= p.blocks) return;
  device float* out = packed + (ulong(g.y) * p.I + g.x) * 8;
  for (uint r = 0; r < 8; ++r) {
    uint t = g.y * 8 + r;
    out[r] = t < p.T ? x[ulong(t) * p.x_stride + g.x] : 0.0f;
  }
}

// Block b packs the rows listed in rows[b * 8 ...]; 0xFFFFFFFF marks an empty place.
kernel void pack_gather8(const device float* x [[buffer(0)]], const device uint* rows [[buffer(1)]],
                         device float* packed [[buffer(2)]], constant PackP& p [[buffer(3)]],
                         uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.I || g.y >= p.blocks) return;
  device float* out = packed + (ulong(g.y) * p.I + g.x) * 8;
  for (uint r = 0; r < 8; ++r) {
    uint t = rows[g.y * 8 + r];
    out[r] = t != 0xFFFFFFFFu ? x[ulong(t) * p.x_stride + g.x] : 0.0f;
  }
}

struct GemmP { uint T; uint I; uint O; uint y_stride; };

// y[t, o] = sum_i x[t, i] * w[o, i] for the 8 rows of block g.y and outputs 4 * g.x ... + 3.
kernel void gemm_bf16(const device float* packed [[buffer(0)]], const device ushort* w [[buffer(1)]],
                      device float* y [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                      uint2 g [[thread_position_in_grid]]) {
  uint o = g.x * 4;
  if (o >= p.O || g.y * 8 >= p.T) return;
  const device packed_float4* xb = (const device packed_float4*)(packed + ulong(g.y) * p.I * 8);
  const device packed_ushort4* w0 = (const device packed_ushort4*)(w + ulong(o) * p.I);
  const device packed_ushort4* w1 = (const device packed_ushort4*)(w + ulong(min(o + 1, p.O - 1)) * p.I);
  const device packed_ushort4* w2 = (const device packed_ushort4*)(w + ulong(min(o + 2, p.O - 1)) * p.I);
  const device packed_ushort4* w3 = (const device packed_ushort4*)(w + ulong(min(o + 3, p.O - 1)) * p.I);
  float4 a0 = 0, b0 = 0, a1 = 0, b1 = 0, a2 = 0, b2 = 0, a3 = 0, b3 = 0;
  for (uint q = 0; q < p.I / 4; ++q) {
    float4 c0 = as_type<float4>(uint4(ushort4(w0[q])) << 16);
    float4 c1 = as_type<float4>(uint4(ushort4(w1[q])) << 16);
    float4 c2 = as_type<float4>(uint4(ushort4(w2[q])) << 16);
    float4 c3 = as_type<float4>(uint4(ushort4(w3[q])) << 16);
    for (uint k = 0; k < 4; ++k) {
      float4 lo = xb[(q * 4 + k) * 2], hi = xb[(q * 4 + k) * 2 + 1];
      a0 += c0[k] * lo; b0 += c0[k] * hi;
      a1 += c1[k] * lo; b1 += c1[k] * hi;
      a2 += c2[k] * lo; b2 += c2[k] * hi;
      a3 += c3[k] * lo; b3 += c3[k] * hi;
    }
  }
  for (uint r = 0; r < 8; ++r) {
    uint t = g.y * 8 + r;
    if (t >= p.T) break;
    device float* out = y + ulong(t) * p.y_stride + o;
    out[0] = r < 4 ? a0[r] : b0[r - 4];
    if (o + 1 < p.O) out[1] = r < 4 ? a1[r] : b1[r - 4];
    if (o + 2 < p.O) out[2] = r < 4 ? a2[r] : b2[r - 4];
    if (o + 3 < p.O) out[3] = r < 4 ? a3[r] : b3[r - 4];
  }
}

struct Tile { uint slot; uint out[8]; };
struct ExpertGemmP { uint tiles; uint I; uint O; uint w_offset; uint s_offset; uint blob; };

// The 8 packed rows of tile g.y through one expert's MXFP4 matrix, outputs 4 * g.x ... + 3.
// Row r of the tile is written to y[tile.out[r]].
kernel void expert_gemm(const device float* packed [[buffer(0)]], const device uchar* blobs [[buffer(1)]],
                        const device Tile* tiles [[buffer(2)]], device float* y [[buffer(3)]],
                        constant ExpertGemmP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  uint o = g.x * 4;
  if (o >= p.O || g.y >= p.tiles) return;
  const device packed_float4* xb = (const device packed_float4*)(packed + ulong(g.y) * p.I * 8);
  const device uchar* blob = blobs + ulong(tiles[g.y].slot) * p.blob;
  uint words = p.I / 8, groups = p.I / 32;
  const device uint* w0 = (const device uint*)(blob + p.w_offset) + ulong(o) * words;
  const device uchar* s0 = blob + p.s_offset + ulong(o) * groups;
  float4 a0 = 0, b0 = 0, a1 = 0, b1 = 0, a2 = 0, b2 = 0, a3 = 0, b3 = 0;
  for (uint word = 0; word < words; ++word) {
    uint group = word / 4;
    uint n0 = w0[word], n1 = w0[words + word], n2 = w0[2 * words + word], n3 = w0[3 * words + word];
    uint e0 = s0[group], e1 = s0[groups + group], e2 = s0[2 * groups + group], e3 = s0[3 * groups + group];
    for (uint k = 0; k < 8; ++k) {
      float4 lo = xb[(word * 8 + k) * 2], hi = xb[(word * 8 + k) * 2 + 1];
      float c0 = fp4_scaled(n0 >> (4 * k), e0), c1 = fp4_scaled(n1 >> (4 * k), e1);
      float c2 = fp4_scaled(n2 >> (4 * k), e2), c3 = fp4_scaled(n3 >> (4 * k), e3);
      a0 += c0 * lo; b0 += c0 * hi;
      a1 += c1 * lo; b1 += c1 * hi;
      a2 += c2 * lo; b2 += c2 * hi;
      a3 += c3 * lo; b3 += c3 * hi;
    }
  }
  for (uint r = 0; r < 8; ++r) {
    uint row = tiles[g.y].out[r];
    if (row == 0xFFFFFFFFu) continue;
    device float* out = y + ulong(row) * p.O + o;
    out[0] = r < 4 ? a0[r] : b0[r - 4];
    out[1] = r < 4 ? a1[r] : b1[r - 4];
    out[2] = r < 4 ? a2[r] : b2[r - 4];
    out[3] = r < 4 ? a3[r] : b3[r - 4];
  }
}

// ---------------------------------------------------------------- simdgroup matrix products
// The fastest batched path: one simdgroup holds a 16-row by 32-output tile as
// eight 8x8 matrices. Weights are placed straight into the matrix registers, so
// they need no FP32 copy. That relies on where each lane's two elements sit in
// an 8x8 matrix; sg_probe reports it and the engine checks it before using
// these kernels.

kernel void sg_probe(device float* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 m;
  m.thread_elements()[0] = float(lane * 2);
  m.thread_elements()[1] = float(lane * 2 + 1);
  simdgroup_store(m, out, 8);
}

// Row and first column of this lane's two adjacent elements (the checked layout:
// four 4x4 quadrants, each row-major, lanes in order).
static inline uint2 sg_place(uint lane) {
  uint quadrant = lane / 8, local = (lane % 8) * 2;
  return uint2((quadrant / 2) * 4 + local / 4, (quadrant % 2) * 4 + local % 4);
}

// y[t, o] = sum_i x[t, i] * w[o, i]. Accumulates the transposed tile:
// C[o, t] += W[o, i] * X^T[i, t]. O is a multiple of 32, T is padded to 16.
kernel void gemm_bf16_sg(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                         device float* y [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                         uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  const device float* x0 = x + ulong(t) * p.I;
  const device float* x1 = x0 + ulong(8) * p.I;
  const device packed_ushort2* w0 = (const device packed_ushort2*)(w + ulong(o + place.x) * p.I + place.y);
  const device packed_ushort2* w1 = (const device packed_ushort2*)(w + ulong(o + 8 + place.x) * p.I + place.y);
  const device packed_ushort2* w2 = (const device packed_ushort2*)(w + ulong(o + 16 + place.x) * p.I + place.y);
  const device packed_ushort2* w3 = (const device packed_ushort2*)(w + ulong(o + 24 + place.x) * p.I + place.y);
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f),
                     c11 = simdgroup_float8x8(0.0f), c20 = simdgroup_float8x8(0.0f), c21 = simdgroup_float8x8(0.0f),
                     c30 = simdgroup_float8x8(0.0f), c31 = simdgroup_float8x8(0.0f);
  for (uint i = 0; i < p.I; i += 8) {
    simdgroup_load(a0, x0 + i, p.I, 0, true);
    simdgroup_load(a1, x1 + i, p.I, 0, true);
    uint q = i / 2;
    float2 v0 = as_type<float2>(uint2(ushort2(w0[q])) << 16), v1 = as_type<float2>(uint2(ushort2(w1[q])) << 16);
    float2 v2 = as_type<float2>(uint2(ushort2(w2[q])) << 16), v3 = as_type<float2>(uint2(ushort2(w3[q])) << 16);
    b0.thread_elements()[0] = v0.x; b0.thread_elements()[1] = v0.y;
    b1.thread_elements()[0] = v1.x; b1.thread_elements()[1] = v1.y;
    b2.thread_elements()[0] = v2.x; b2.thread_elements()[1] = v2.y;
    b3.thread_elements()[0] = v3.x; b3.thread_elements()[1] = v3.y;
    simdgroup_multiply_accumulate(c00, b0, a0, c00); simdgroup_multiply_accumulate(c01, b0, a1, c01);
    simdgroup_multiply_accumulate(c10, b1, a0, c10); simdgroup_multiply_accumulate(c11, b1, a1, c11);
    simdgroup_multiply_accumulate(c20, b2, a0, c20); simdgroup_multiply_accumulate(c21, b2, a1, c21);
    simdgroup_multiply_accumulate(c30, b3, a0, c30); simdgroup_multiply_accumulate(c31, b3, a1, c31);
  }
  device float* y0 = y + ulong(t) * p.y_stride + o;
  device float* y1 = y0 + ulong(8) * p.y_stride;
  simdgroup_store(c00, y0, p.y_stride, 0, true); simdgroup_store(c01, y1, p.y_stride, 0, true);
  simdgroup_store(c10, y0 + 8, p.y_stride, 0, true); simdgroup_store(c11, y1 + 8, p.y_stride, 0, true);
  simdgroup_store(c20, y0 + 16, p.y_stride, 0, true); simdgroup_store(c21, y1 + 16, p.y_stride, 0, true);
  simdgroup_store(c30, y0 + 24, p.y_stride, 0, true); simdgroup_store(c31, y1 + 24, p.y_stride, 0, true);
}

struct GatherP { uint tiles; uint I; uint x_stride; };

// Copies each tile's 16 rows next to each other; 0xFFFFFFFF marks an empty row.
kernel void gather_rows16(const device float* x [[buffer(0)]], const device uint* rows [[buffer(1)]],
                          device float* out [[buffer(2)]], constant GatherP& p [[buffer(3)]],
                          uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.I || g.y >= p.tiles * 16) return;
  uint row = rows[g.y];
  out[ulong(g.y) * p.I + g.x] = row != 0xFFFFFFFFu ? x[ulong(row) * p.x_stride + g.x] : 0.0f;
}

struct ExpertSgP { uint tiles; uint I; uint O; uint w_offset; uint s_offset; uint blob; };

constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// Rows tile * 16 ... + 15 of x through the tile's expert: y[row, o] = sum_i x[row, i] * mxfp4(w[o, i]).
// A lane loads one uint4 per weight row: 32 values sharing one scale, used over four 8-wide steps.
kernel void expert_gemm_sg(const device float* x [[buffer(0)]], const device uchar* blobs [[buffer(1)]],
                           const device uint* slots [[buffer(2)]], device float* y [[buffer(3)]],
                           constant ExpertSgP& p [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  uint groups = p.I / 32, shift = place.y * 4;
  const device uchar* blob = blobs + ulong(slots[g.z]) * p.blob;
  const device uint4* w0 = (const device uint4*)(blob + p.w_offset) + ulong(o + place.x) * groups;
  const device uint4* w1 = w0 + ulong(8) * groups;
  const device uint4* w2 = w0 + ulong(16) * groups;
  const device uint4* w3 = w0 + ulong(24) * groups;
  const device uchar* s0 = blob + p.s_offset + ulong(o + place.x) * groups;
  const device float* x0 = x + ulong(t) * p.I;
  const device float* x1 = x0 + ulong(8) * p.I;
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f),
                     c11 = simdgroup_float8x8(0.0f), c20 = simdgroup_float8x8(0.0f), c21 = simdgroup_float8x8(0.0f),
                     c30 = simdgroup_float8x8(0.0f), c31 = simdgroup_float8x8(0.0f);
  for (uint group = 0; group < groups; ++group) {
    uint4 q0 = w0[group] >> shift, q1 = w1[group] >> shift, q2 = w2[group] >> shift, q3 = w3[group] >> shift;
    // A zero E8M0 byte (2^-127) is below FP32's normal range; as_type gives 0 there, as wanted.
    float f0 = as_type<float>(uint(s0[group]) << 23), f1 = as_type<float>(uint(s0[8 * groups + group]) << 23);
    float f2 = as_type<float>(uint(s0[16 * groups + group]) << 23), f3 = as_type<float>(uint(s0[24 * groups + group]) << 23);
    for (uint k = 0; k < 4; ++k) {
      uint i = group * 32 + k * 8;
      simdgroup_load(a0, x0 + i, p.I, 0, true);
      simdgroup_load(a1, x1 + i, p.I, 0, true);
      uint n0 = q0[k], n1 = q1[k], n2 = q2[k], n3 = q3[k];
      b0.thread_elements()[0] = FP4_VALUES[n0 & 15] * f0; b0.thread_elements()[1] = FP4_VALUES[(n0 >> 4) & 15] * f0;
      b1.thread_elements()[0] = FP4_VALUES[n1 & 15] * f1; b1.thread_elements()[1] = FP4_VALUES[(n1 >> 4) & 15] * f1;
      b2.thread_elements()[0] = FP4_VALUES[n2 & 15] * f2; b2.thread_elements()[1] = FP4_VALUES[(n2 >> 4) & 15] * f2;
      b3.thread_elements()[0] = FP4_VALUES[n3 & 15] * f3; b3.thread_elements()[1] = FP4_VALUES[(n3 >> 4) & 15] * f3;
      simdgroup_multiply_accumulate(c00, b0, a0, c00); simdgroup_multiply_accumulate(c01, b0, a1, c01);
      simdgroup_multiply_accumulate(c10, b1, a0, c10); simdgroup_multiply_accumulate(c11, b1, a1, c11);
      simdgroup_multiply_accumulate(c20, b2, a0, c20); simdgroup_multiply_accumulate(c21, b2, a1, c21);
      simdgroup_multiply_accumulate(c30, b3, a0, c30); simdgroup_multiply_accumulate(c31, b3, a1, c31);
    }
  }
  device float* y0 = y + ulong(t) * p.O + o;
  device float* y1 = y0 + ulong(8) * p.O;
  simdgroup_store(c00, y0, p.O, 0, true); simdgroup_store(c01, y1, p.O, 0, true);
  simdgroup_store(c10, y0 + 8, p.O, 0, true); simdgroup_store(c11, y1 + 8, p.O, 0, true);
  simdgroup_store(c20, y0 + 16, p.O, 0, true); simdgroup_store(c21, y1 + 16, p.O, 0, true);
  simdgroup_store(c30, y0 + 24, p.O, 0, true); simdgroup_store(c31, y1 + 24, p.O, 0, true);
}

struct CombineRowsP { uint T; uint H; uint K; };

// moe_combine for tiled experts: pair (t, k) lives in routed row rows[t * K + k].
kernel void moe_combine_rows(const device float* routed [[buffer(0)]], const device float* score [[buffer(1)]],
                             const device float* shared [[buffer(2)]], const device float* shared_gate [[buffer(3)]],
                             const device uint* rows [[buffer(4)]], device float* y [[buffer(5)]],
                             constant CombineRowsP& p [[buffer(6)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H || g.y >= p.T) return;
  float acc = 0;
  for (uint k = 0; k < p.K; ++k)
    acc += score[g.y * p.K + k] * routed[ulong(rows[g.y * p.K + k]) * p.H + g.x];
  y[ulong(g.y) * p.H + g.x] = acc + sigm(shared_gate[g.y]) * shared[ulong(g.y) * p.H + g.x];
}

// qsa_output for whole tiles: one simdgroup per (32 value dims, KV head, token).
// C[head, d] += weight[head, j] * value[key_j, d]. The KV head's 12 query heads
// are covered by two 8-row tiles that overlap by four heads. K is a multiple of 8.
// Writes the ungated sum; qsa_gate applies the gate.
kernel void qsa_output_sg(const device float* weight [[buffer(0)]], const device half* values [[buffer(1)]],
                          const device int* selection [[buffer(2)]], device float* y [[buffer(3)]],
                          constant AttentionP& p [[buffer(4)]],
                          uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint t = g.z / p.kv_heads, kv = g.z % p.kv_heads, first = kv * (p.heads / p.kv_heads), d = g.y * 32;
  uint2 place = sg_place(lane);
  const device float* w0 = weight + (ulong(t) * p.heads + first) * p.K;
  const device float* w1 = w0 + ulong(4) * p.K;
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c02 = simdgroup_float8x8(0.0f),
                     c03 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f), c11 = simdgroup_float8x8(0.0f),
                     c12 = simdgroup_float8x8(0.0f), c13 = simdgroup_float8x8(0.0f);
  for (uint j = 0; j < p.K; j += 8) {
    simdgroup_load(a0, w0 + j, p.K);
    simdgroup_load(a1, w1 + j, p.K);
    // Keys outside the selection have weight 0; read key 0 for them.
    uint index = uint(max(key_index(selection, p, t, j + place.x), 0));
    const device packed_half2* value = (const device packed_half2*)(values + (ulong(index) * p.kv_heads + kv) * p.D + d + place.y);
    float2 v0 = float2(half2(value[0])), v1 = float2(half2(value[4])), v2 = float2(half2(value[8])), v3 = float2(half2(value[12]));
    b0.thread_elements()[0] = v0.x; b0.thread_elements()[1] = v0.y;
    b1.thread_elements()[0] = v1.x; b1.thread_elements()[1] = v1.y;
    b2.thread_elements()[0] = v2.x; b2.thread_elements()[1] = v2.y;
    b3.thread_elements()[0] = v3.x; b3.thread_elements()[1] = v3.y;
    simdgroup_multiply_accumulate(c00, a0, b0, c00); simdgroup_multiply_accumulate(c01, a0, b1, c01);
    simdgroup_multiply_accumulate(c02, a0, b2, c02); simdgroup_multiply_accumulate(c03, a0, b3, c03);
    simdgroup_multiply_accumulate(c10, a1, b0, c10); simdgroup_multiply_accumulate(c11, a1, b1, c11);
    simdgroup_multiply_accumulate(c12, a1, b2, c12); simdgroup_multiply_accumulate(c13, a1, b3, c13);
  }
  device float* y0 = y + (ulong(t) * p.heads + first) * p.D + d;
  device float* y1 = y0 + ulong(4) * p.D;
  simdgroup_store(c00, y0, p.D); simdgroup_store(c01, y0 + 8, p.D); simdgroup_store(c02, y0 + 16, p.D); simdgroup_store(c03, y0 + 24, p.D);
  simdgroup_store(c10, y1, p.D); simdgroup_store(c11, y1 + 8, p.D); simdgroup_store(c12, y1 + 16, p.D); simdgroup_store(c13, y1 + 24, p.D);
}

// y[t, h, d] *= sigmoid(gate[t, h, d]); the gate is the second half of each head's q_proj output.
kernel void qsa_gate(device float* y [[buffer(0)]], const device float* q_proj [[buffer(1)]],
                     constant AttentionP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.heads * p.D || g.y >= p.T) return;
  uint head = g.x / p.D, d = g.x % p.D;
  y[ulong(g.y) * p.heads * p.D + g.x] *= sigm(q_proj[ulong(g.y) * p.heads * 2 * p.D + head * 2 * p.D + p.D + d]);
}

// hidden[t, s, i] += x[t, i]
kernel void add_broadcast(device float* hidden [[buffer(0)]], const device float* x [[buffer(1)]],
                          constant HcP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H * p.C || g.y >= p.T) return;
  hidden[ulong(g.y) * p.H * p.C + g.x] += x[ulong(g.y) * p.H + g.x % p.H];
}

// ---------------------------------------------------------------- decode and verification products (1-8 rows)

struct SplitP { uint T; uint I; uint O; uint x_stride; uint span; uint splits; };

// Narrow products: partial[s, t, o] = sum over the s-th span of i of x[t, i] * w[o, i],
// for blocks of up to 4 rows. Grid (32, O, blocks * splits).
kernel void linear_bf16_split(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                              device float* partial [[buffer(2)]], constant SplitP& p [[buffer(3)]],
                              uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint split = g.z % p.splits, first = (g.z / p.splits) * 4;
  uint rows = min(4u, p.T - first);
  uint begin = split * p.span / 4, end = min(p.I, (split + 1) * p.span) / 4;
  const device packed_ushort4* wr = (const device packed_ushort4*)(w + ulong(g.y) * p.I);
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint i = begin + lane; i < end; i += 32) {
    float4 weight = as_type<float4>(uint4(ushort4(wr[i])) << 16);
    a0 += weight * float4(x0[i]); a1 += weight * float4(x1[i]);
    a2 += weight * float4(x2[i]); a3 += weight * float4(x3[i]);
  }
  float t0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w));
  float t1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float t2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w));
  float t3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  if (lane != 0) return;
  device float* out = partial + (ulong(split) * p.T + first) * p.O + g.y;
  out[0] = t0;
  if (1u < rows) out[p.O] = t1;
  if (2u < rows) out[2 * p.O] = t2;
  if (3u < rows) out[3 * p.O] = t3;
}

// y[t, o] = sum_s partial[s, t, o]
kernel void split_sum(const device float* partial [[buffer(0)]], device float* y [[buffer(1)]],
                      constant SplitP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.O || g.y >= p.T) return;
  float total = 0;
  for (uint s = 0; s < p.splits; ++s) total += partial[(ulong(s) * p.T + g.y) * p.O + g.x];
  y[ulong(g.y) * p.O + g.x] = total;
}


// ---------------------------------------------------------------- 8-bit resident weights
// w[o, i] = q[o, i] * scale + bias, one (scale, bias) pair per 64 inputs.

static inline float4 q8x4(uchar4 q, float2 sb) { return float4(q) * sb.x + sb.y; }

kernel void linear_q8(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                      const device float2* sb [[buffer(2)]], device float* y [[buffer(3)]],
                      constant LinearP& p [[buffer(4)]],
                      uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const device packed_uchar4* wr = (const device packed_uchar4*)(w + ulong(g.y) * p.I);
  const device float2* sr = sb + ulong(g.y) * (p.I / 64);
  const device packed_float4* xq = (const device packed_float4*)(x + ulong(g.z) * p.x_stride);
  float4 sum = 0;
  for (uint i = lane; i < p.I / 4; i += 32) sum += q8x4(uchar4(wr[i]), sr[i / 16]) * float4(xq[i]);
  float total = simd_sum((sum.x + sum.y) + (sum.z + sum.w));
  if (lane == 0) y[ulong(g.z) * p.y_stride + g.y] = total;
}

kernel void linear_q8_4(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                        const device float2* sb [[buffer(2)]], device float* y [[buffer(3)]],
                        constant LinearP& p [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * 4;
  uint rows = min(4u, p.T - first);
  const device packed_uchar4* wr = (const device packed_uchar4*)(w + ulong(g.y) * p.I);
  const device float2* sr = sb + ulong(g.y) * (p.I / 64);
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint i = lane; i < p.I / 4; i += 32) {
    float4 weight = q8x4(uchar4(wr[i]), sr[i / 16]);
    a0 += weight * float4(x0[i]); a1 += weight * float4(x1[i]);
    a2 += weight * float4(x2[i]); a3 += weight * float4(x3[i]);
  }
  float t0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w));
  float t1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float t2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w));
  float t3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  if (lane != 0) return;
  y[ulong(first) * p.y_stride + g.y] = t0;
  if (1u < rows) y[ulong(first + 1) * p.y_stride + g.y] = t1;
  if (2u < rows) y[ulong(first + 2) * p.y_stride + g.y] = t2;
  if (3u < rows) y[ulong(first + 3) * p.y_stride + g.y] = t3;
}

// span is a multiple of 64 here.
kernel void linear_q8_split(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                            const device float2* sb [[buffer(2)]], device float* partial [[buffer(3)]],
                            constant SplitP& p [[buffer(4)]],
                            uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint split = g.z % p.splits, first = (g.z / p.splits) * 4;
  uint rows = min(4u, p.T - first);
  uint begin = split * p.span / 4, end = min(p.I, (split + 1) * p.span) / 4;
  const device packed_uchar4* wr = (const device packed_uchar4*)(w + ulong(g.y) * p.I);
  const device float2* sr = sb + ulong(g.y) * (p.I / 64);
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint i = begin + lane; i < end; i += 32) {
    float4 weight = q8x4(uchar4(wr[i]), sr[i / 16]);
    a0 += weight * float4(x0[i]); a1 += weight * float4(x1[i]);
    a2 += weight * float4(x2[i]); a3 += weight * float4(x3[i]);
  }
  float t0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w));
  float t1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float t2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w));
  float t3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  if (lane != 0) return;
  device float* out = partial + (ulong(split) * p.T + first) * p.O + g.y;
  out[0] = t0;
  if (1u < rows) out[p.O] = t1;
  if (2u < rows) out[2 * p.O] = t2;
  if (3u < rows) out[3 * p.O] = t3;
}

// gemm_bf16_sg with 8-bit weights.
kernel void gemm_q8_sg(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                       const device float2* sb [[buffer(2)]], device float* y [[buffer(3)]],
                       constant GemmP& p [[buffer(4)]],
                       uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  uint groups = p.I / 64;
  const device float* x0 = x + ulong(t) * p.I;
  const device float* x1 = x0 + ulong(8) * p.I;
  const device packed_uchar2* w0 = (const device packed_uchar2*)(w + ulong(o + place.x) * p.I + place.y);
  const device packed_uchar2* w1 = (const device packed_uchar2*)(w + ulong(o + 8 + place.x) * p.I + place.y);
  const device packed_uchar2* w2 = (const device packed_uchar2*)(w + ulong(o + 16 + place.x) * p.I + place.y);
  const device packed_uchar2* w3 = (const device packed_uchar2*)(w + ulong(o + 24 + place.x) * p.I + place.y);
  const device float2* s0 = sb + ulong(o + place.x) * groups;
  const device float2* s1 = s0 + ulong(8) * groups;
  const device float2* s2 = s0 + ulong(16) * groups;
  const device float2* s3 = s0 + ulong(24) * groups;
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f),
                     c11 = simdgroup_float8x8(0.0f), c20 = simdgroup_float8x8(0.0f), c21 = simdgroup_float8x8(0.0f),
                     c30 = simdgroup_float8x8(0.0f), c31 = simdgroup_float8x8(0.0f);
  for (uint i = 0; i < p.I; i += 8) {
    simdgroup_load(a0, x0 + i, p.I, 0, true);
    simdgroup_load(a1, x1 + i, p.I, 0, true);
    uint q = i / 2, group = i / 64;
    float2 e0 = s0[group], e1 = s1[group], e2 = s2[group], e3 = s3[group];
    float2 v0 = float2(uchar2(w0[q])) * e0.x + e0.y, v1 = float2(uchar2(w1[q])) * e1.x + e1.y;
    float2 v2 = float2(uchar2(w2[q])) * e2.x + e2.y, v3 = float2(uchar2(w3[q])) * e3.x + e3.y;
    b0.thread_elements()[0] = v0.x; b0.thread_elements()[1] = v0.y;
    b1.thread_elements()[0] = v1.x; b1.thread_elements()[1] = v1.y;
    b2.thread_elements()[0] = v2.x; b2.thread_elements()[1] = v2.y;
    b3.thread_elements()[0] = v3.x; b3.thread_elements()[1] = v3.y;
    simdgroup_multiply_accumulate(c00, b0, a0, c00); simdgroup_multiply_accumulate(c01, b0, a1, c01);
    simdgroup_multiply_accumulate(c10, b1, a0, c10); simdgroup_multiply_accumulate(c11, b1, a1, c11);
    simdgroup_multiply_accumulate(c20, b2, a0, c20); simdgroup_multiply_accumulate(c21, b2, a1, c21);
    simdgroup_multiply_accumulate(c30, b3, a0, c30); simdgroup_multiply_accumulate(c31, b3, a1, c31);
  }
  device float* y0 = y + ulong(t) * p.y_stride + o;
  device float* y1 = y0 + ulong(8) * p.y_stride;
  simdgroup_store(c00, y0, p.y_stride, 0, true); simdgroup_store(c01, y1, p.y_stride, 0, true);
  simdgroup_store(c10, y0 + 8, p.y_stride, 0, true); simdgroup_store(c11, y1 + 8, p.y_stride, 0, true);
  simdgroup_store(c20, y0 + 16, p.y_stride, 0, true); simdgroup_store(c21, y1 + 16, p.y_stride, 0, true);
  simdgroup_store(c30, y0 + 24, p.y_stride, 0, true); simdgroup_store(c31, y1 + 24, p.y_stride, 0, true);
}

struct DequantP { uint O; uint I; };

// BF16 copy of an 8-bit matrix (round to nearest even), for long products.
kernel void dequant_q8(const device uchar* w [[buffer(0)]], const device float2* sb [[buffer(1)]],
                       device ushort* out [[buffer(2)]], constant DequantP& p [[buffer(3)]],
                       uint2 g [[thread_position_in_grid]]) {
  uint i = g.x * 4;
  if (i >= p.I || g.y >= p.O) return;
  float2 s = sb[ulong(g.y) * (p.I / 64) + i / 64];
  float4 v = float4(uchar4(*(const device packed_uchar4*)(w + ulong(g.y) * p.I + i))) * s.x + s.y;
  uint4 bits = as_type<uint4>(v);
  bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16;
  *(device packed_ushort4*)(out + ulong(g.y) * p.I + i) = ushort4(bits);
}

// ---------------------------------------------------------------- image encoder (ViT)

struct VisionNormP { uint N; uint D; float eps; };

// y = (x - mean) / sqrt(var + eps) * w + b, one simdgroup per row.
kernel void layer_norm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                       const device float* b [[buffer(2)]], device float* y [[buffer(3)]],
                       constant VisionNormP& p [[buffer(4)]],
                       uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (g.y >= p.N) return;
  const device float* xr = x + ulong(g.y) * p.D;
  device float* yr = y + ulong(g.y) * p.D;
  float sum = 0;
  for (uint i = lane; i < p.D; i += 32) sum += xr[i];
  float mean = simd_sum(sum) / float(p.D);
  float squares = 0;
  for (uint i = lane; i < p.D; i += 32) { float d = xr[i] - mean; squares += d * d; }
  float r = rsqrt(simd_sum(squares) / float(p.D) + p.eps);
  for (uint i = lane; i < p.D; i += 32) yr[i] = (xr[i] - mean) * r * w[i] + b[i];
}

struct BiasP { uint N; uint O; uint act; };

static inline float erf_approx(float x) {
  // Abramowitz and Stegun 7.1.26, absolute error below 1.5e-7.
  float t = 1.0f / (1.0f + 0.3275911f * abs(x));
  float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t * exp(-x * x);
  return x < 0 ? -y : y;
}

// y[n, o] = act(y[n, o] + b[o]); act 0: none, 1: tanh GELU, 2: exact GELU.
kernel void bias_act(device float* y [[buffer(0)]], const device float* b [[buffer(1)]],
                     constant BiasP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.O || g.y >= p.N) return;
  ulong at = ulong(g.y) * p.O + g.x;
  float v = y[at] + b[g.x];
  if (p.act == 1) v = 0.5f * v * (1.0f + precise::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
  else if (p.act == 2) v = 0.5f * v * (1.0f + erf_approx(v * 0.7071067812f));
  y[at] = v;
}

kernel void add_in_place(device float* y [[buffer(0)]], const device float* x [[buffer(1)]],
                         constant CountP& p [[buffer(2)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  y[g] += x[g];
}

struct VisionAttnP { uint N; uint Q0; uint Qn; uint heads; uint D; float scale; };

// Rotate q and k of every head in place: x[i] and x[i + D/2] by angle[n, i].
// qkv rows are [q | k | v], each heads * D wide.
kernel void vision_rope(device float* qkv [[buffer(0)]], const device float* angles [[buffer(1)]],
                        constant VisionAttnP& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  uint half_dims = p.D / 2;
  if (g.x >= half_dims || g.y >= p.heads * 2 || g.z >= p.N) return;
  device float* x = qkv + ulong(g.z) * 3 * p.heads * p.D + g.y * p.D;
  float angle = angles[ulong(g.z) * half_dims + g.x];
  float c = cos(angle), s = sin(angle), a = x[g.x], b = x[g.x + half_dims];
  x[g.x] = a * c - b * s;
  x[g.x + half_dims] = b * c + a * s;
}

// score[q, h, k] for queries Q0 ... Q0 + Qn - 1 against all N keys.
kernel void vision_scores(const device float* qkv [[buffer(0)]], device float* score [[buffer(1)]],
                          constant VisionAttnP& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= p.N || g.y >= p.heads || g.z >= p.Qn) return;
  ulong row = 3 * p.heads * p.D;
  const device float* q = qkv + ulong(p.Q0 + g.z) * row + g.y * p.D;
  const device float* k = qkv + ulong(g.x) * row + p.heads * p.D + g.y * p.D;
  float acc = 0;
  for (uint i = 0; i < p.D; ++i) acc += q[i] * k[i];
  score[(ulong(g.z) * p.heads + g.y) * p.N + g.x] = acc * p.scale;
}

kernel void vision_softmax(device float* score [[buffer(0)]], constant VisionAttnP& p [[buffer(1)]],
                           uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  if (g.y >= p.heads || g.z >= p.Qn) return;
  device float* row = score + (ulong(g.z) * p.heads + g.y) * p.N;
  float top = -FLT_MAX;
  for (uint j = lane; j < p.N; j += 32) top = max(top, row[j]);
  top = simd_max(top);
  float total = 0;
  for (uint j = lane; j < p.N; j += 32) { float e = exp(row[j] - top); row[j] = e; total += e; }
  float inverse = 1.0f / simd_sum(total);
  for (uint j = lane; j < p.N; j += 32) row[j] *= inverse;
}

// out[q, h, d] = sum_k weight[q, h, k] * v[k, h, d]; one simdgroup per (d, h, q), lanes over keys.
kernel void vision_output(const device float* score [[buffer(0)]], const device float* qkv [[buffer(1)]],
                          device float* out [[buffer(2)]], constant VisionAttnP& p [[buffer(3)]],
                          uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint d = g.y % p.D, h = g.y / p.D;
  if (h >= p.heads || g.z >= p.Qn) return;
  const device float* weight = score + (ulong(g.z) * p.heads + h) * p.N;
  ulong row = 3 * p.heads * p.D;
  const device float* v = qkv + 2 * p.heads * p.D + h * p.D + d;
  float acc = 0;
  for (uint k = lane; k < p.N; k += 32) acc += weight[k] * v[ulong(k) * row];
  acc = simd_sum(acc);
  if (lane == 0) out[(ulong(p.Q0 + g.z) * p.heads + h) * p.D + d] = acc;
}

// ---------------------------------------------------------------- staged GEMM (long prompts)
// y[t, o] = sum_i x[t, i] * w[o, i]. Follows the structure of MLX's Steel GEMM: a
// threadgroup of 2 x 2 simdgroups computes a 64 x 64 tile; each K step of 16 stages
// x (FP32) and w (BF16 widened to FP32) in threadgroup memory; each lane loads its two
// fragment elements and keeps its accumulators as float2. Simdgroup arrays of more than
// 8 accumulators ran 4-7 times slower on M2 Max (register spills), float2 arrays did not.
// Requires O % 64 == 0 and I % 16 == 0. Rows past T are read clamped and not written.
constant constexpr short TG_BM = 64, TG_BN = 64, TG_BK = 16, TG_LD = TG_BK + 4;

kernel void gemm_bf16_tg(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                         device float* y [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float As[TG_BM * TG_LD];
  threadgroup float Bs[TG_BN * TG_LD];
  uint row0 = tid.y * TG_BM, col0 = tid.x * TG_BN;
  ushort t = sg * 32 + lane;
  // 128 threads: each copies 8 consecutive K values of one A row and one B row.
  short load_row = t / 2, load_col = (t % 2) * 8;
  const device float* ap = x + ulong(min(row0 + load_row, p.T - 1)) * p.I + load_col;
  const device ushort* bp = w + ulong(col0 + load_row) * p.I + load_col;
  short qid = lane / 4;
  short fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2;
  short tm = 8 * (sg / 2), tn = 8 * (sg % 2);
  float2 c[4][4];
  for (short i = 0; i < 4; ++i)
    for (short j = 0; j < 4; ++j) c[i][j] = 0;
  for (uint k = 0; k < p.I; k += TG_BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 a0 = *(const device packed_float4*)ap, a1 = *(const device packed_float4*)(ap + 4);
    float4 b0 = as_type<float4>(uint4(ushort4(*(const device packed_ushort4*)bp)) << 16);
    float4 b1 = as_type<float4>(uint4(ushort4(*(const device packed_ushort4*)(bp + 4))) << 16);
    threadgroup float* da = As + load_row * TG_LD + load_col;
    threadgroup float* db = Bs + load_row * TG_LD + load_col;
    da[0] = a0.x; da[1] = a0.y; da[2] = a0.z; da[3] = a0.w; da[4] = a1.x; da[5] = a1.y; da[6] = a1.z; da[7] = a1.w;
    db[0] = b0.x; db[1] = b0.y; db[2] = b0.z; db[3] = b0.w; db[4] = b1.x; db[5] = b1.y; db[6] = b1.z; db[7] = b1.w;
    ap += TG_BK;
    bp += TG_BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (short kk = 0; kk < TG_BK; kk += 8) {
      float2 af[4], bf[4];
      #pragma unroll
      for (short i = 0; i < 4; ++i) {
        const threadgroup float* s = As + (tm + i * 16 + fm) * TG_LD + kk + fn;
        af[i] = float2(s[0], s[1]);
      }
      #pragma unroll
      for (short j = 0; j < 4; ++j) {
        const threadgroup float* s = Bs + (tn + j * 16 + fn) * TG_LD + kk + fm;
        bf[j] = float2(s[0], s[TG_LD]);
      }
      #pragma unroll
      for (short i = 0; i < 4; ++i) {
        simdgroup_float8x8 am;
        am.thread_elements()[0] = af[i].x;
        am.thread_elements()[1] = af[i].y;
        #pragma unroll
        for (short j = 0; j < 4; ++j) {
          simdgroup_float8x8 bm, cm;
          bm.thread_elements()[0] = bf[j].x;
          bm.thread_elements()[1] = bf[j].y;
          cm.thread_elements()[0] = c[i][j].x;
          cm.thread_elements()[1] = c[i][j].y;
          simdgroup_multiply_accumulate(cm, am, bm, cm);
          c[i][j] = float2(cm.thread_elements()[0], cm.thread_elements()[1]);
        }
      }
    }
  }
  #pragma unroll
  for (short i = 0; i < 4; ++i) {
    uint row = row0 + tm + i * 16 + fm;
    if (row >= p.T) continue;
    #pragma unroll
    for (short j = 0; j < 4; ++j)
      *(device float2*)(y + ulong(row) * p.y_stride + col0 + tn + j * 16 + fn) = c[i][j];
  }
}

// expert_linear for 4 outputs per simdgroup: each lane loads its 8 inputs once and uses
// them for all four weight rows (the one-row kernel reread the input for every row).
// Grid (32, O / 4, pairs). 1.7-2.1 times faster on M2 Max.
kernel void expert_linear4(const device float* x [[buffer(0)]],
                           const device uchar* blobs [[buffer(1)]],
                           const device Pair* pairs [[buffer(2)]],
                           device float* y [[buffer(3)]],
                           constant ExpertP& p [[buffer(4)]],
                           uint3 g [[thread_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
  Pair pair = pairs[g.z];
  uint o0 = g.y * 4, words = p.I / 8, groups = p.I / 32;
  const device uchar* blob = blobs + ulong(pair.slot) * p.blob;
  const device uint* wr = (const device uint*)(blob + p.w_offset) + ulong(o0) * words;
  const device uchar* sr = blob + p.s_offset + ulong(o0) * groups;
  const device float4* xr = (const device float4*)(x + ulong(p.x_is_pair != 0 ? pair.out : pair.x_row) * p.x_stride);
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  for (uint word = lane; word < words; word += 32) {
    float4 lo = xr[word * 2], hi = xr[word * 2 + 1];
    uint group = word / 4;
    #define WQ_ROW(r, acc) { \
      uint bits = wr[ulong(r) * words + word]; \
      float scale = as_type<float>(uint(sr[ulong(r) * groups + group]) << 23); \
      float4 wl = float4(FP4_VALUES[bits & 15], FP4_VALUES[(bits >> 4) & 15], FP4_VALUES[(bits >> 8) & 15], FP4_VALUES[(bits >> 12) & 15]); \
      float4 wh = float4(FP4_VALUES[(bits >> 16) & 15], FP4_VALUES[(bits >> 20) & 15], FP4_VALUES[(bits >> 24) & 15], FP4_VALUES[bits >> 28]); \
      acc += (dot(wl, lo) + dot(wh, hi)) * scale; }
    WQ_ROW(0, a0) WQ_ROW(1, a1) WQ_ROW(2, a2) WQ_ROW(3, a3)
    #undef WQ_ROW
  }
  float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
  if (lane != 0) return;
  device float* out = y + ulong(pair.out) * p.O + o0;
  out[0] = t0; out[1] = t1; out[2] = t2; out[3] = t3;
}

// linear_q8_2o: one row, two outputs per simdgroup. Same arithmetic order as the one-output
// kernel. Grid (32, O / 2, T).
kernel void linear_q8_2o(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                   const device float2* sb [[buffer(2)]],
                   device float* y [[buffer(3)]], constant LinearP& p [[buffer(4)]],
                   uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 2;
  const device packed_uchar4* w0 = (const device packed_uchar4*)(w + ulong(o) * p.I);
  const device packed_uchar4* w1 = (const device packed_uchar4*)(w + ulong(o + 1) * p.I);
  const device float2* s0 = sb + ulong(o) * (p.I / 64);
  const device float2* s1 = s0 + p.I / 64;
  const device packed_float4* xq = (const device packed_float4*)(x + ulong(g.z) * p.x_stride);
  float4 a = 0, b = 0;
  for (uint i = lane; i < p.I / 4; i += 32) {
    float4 v = float4(xq[i]);
    a += q8x4(uchar4(w0[i]), s0[i / 16]) * v;
    b += q8x4(uchar4(w1[i]), s1[i / 16]) * v;
  }
  float ta = simd_sum((a.x + a.y) + (a.z + a.w)), tb = simd_sum((b.x + b.y) + (b.z + b.w));
  if (lane != 0) return;
  y[ulong(g.z) * p.y_stride + o] = ta;
  y[ulong(g.z) * p.y_stride + o + 1] = tb;
}

// linear_q8_4_2o: two outputs per simdgroup for blocks of up to 4 rows; the input loads serve both
// weight rows. Same arithmetic order as the one-output kernel. Grid (32, O / 2, blocks).
kernel void linear_q8_4_2o(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                   const device float2* sb [[buffer(2)]],
                   device float* y [[buffer(3)]], constant LinearP& p [[buffer(4)]],
                   uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * 4, rows = min(4u, p.T - first), o = g.y * 2;
  const device packed_uchar4* w0 = (const device packed_uchar4*)(w + ulong(o) * p.I);
  const device packed_uchar4* w1 = (const device packed_uchar4*)(w + ulong(o + 1) * p.I);
  const device float2* s0 = sb + ulong(o) * (p.I / 64);
  const device float2* s1 = s0 + p.I / 64;
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
  for (uint i = lane; i < p.I / 4; i += 32) {
    float4 v0 = float4(x0[i]), v1 = float4(x1[i]), v2 = float4(x2[i]), v3 = float4(x3[i]);
    float4 wa = q8x4(uchar4(w0[i]), s0[i / 16]), wb = q8x4(uchar4(w1[i]), s1[i / 16]);
    a0 += wa * v0; a1 += wa * v1; a2 += wa * v2; a3 += wa * v3;
    b0 += wb * v0; b1 += wb * v1; b2 += wb * v2; b3 += wb * v3;
  }
  float ta0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w)), ta1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float ta2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w)), ta3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  float tb0 = simd_sum((b0.x + b0.y) + (b0.z + b0.w)), tb1 = simd_sum((b1.x + b1.y) + (b1.z + b1.w));
  float tb2 = simd_sum((b2.x + b2.y) + (b2.z + b2.w)), tb3 = simd_sum((b3.x + b3.y) + (b3.z + b3.w));
  if (lane != 0) return;
  device float* out = y + ulong(first) * p.y_stride + o;
  out[0] = ta0; out[1] = tb0;
  if (1u < rows) { out[p.y_stride] = ta1; out[p.y_stride + 1] = tb1; }
  if (2u < rows) { out[2 * p.y_stride] = ta2; out[2 * p.y_stride + 1] = tb2; }
  if (3u < rows) { out[3 * p.y_stride] = ta3; out[3 * p.y_stride + 1] = tb3; }
}

// linear_bf16_2o: one row, two outputs per simdgroup. Same arithmetic order as the one-output
// kernel. Grid (32, O / 2, T).
kernel void linear_bf16_2o(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                   device float* y [[buffer(2)]], constant LinearP& p [[buffer(3)]],
                   uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 2;
  const device packed_ushort4* w0 = (const device packed_ushort4*)(w + ulong(o) * p.I);
  const device packed_ushort4* w1 = (const device packed_ushort4*)(w + ulong(o + 1) * p.I);
  const device packed_float4* xq = (const device packed_float4*)(x + ulong(g.z) * p.x_stride);
  float4 a = 0, b = 0;
  for (uint i = lane; i < p.I / 4; i += 32) {
    float4 v = float4(xq[i]);
    a += as_type<float4>(uint4(ushort4(w0[i])) << 16) * v;
    b += as_type<float4>(uint4(ushort4(w1[i])) << 16) * v;
  }
  float ta = simd_sum((a.x + a.y) + (a.z + a.w)), tb = simd_sum((b.x + b.y) + (b.z + b.w));
  if (lane != 0) return;
  y[ulong(g.z) * p.y_stride + o] = ta;
  y[ulong(g.z) * p.y_stride + o + 1] = tb;
}

// linear_bf16_4_2o: two outputs per simdgroup for blocks of up to 4 rows; the input loads serve both
// weight rows. Same arithmetic order as the one-output kernel. Grid (32, O / 2, blocks).
kernel void linear_bf16_4_2o(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                   device float* y [[buffer(2)]], constant LinearP& p [[buffer(3)]],
                   uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * 4, rows = min(4u, p.T - first), o = g.y * 2;
  const device packed_ushort4* w0 = (const device packed_ushort4*)(w + ulong(o) * p.I);
  const device packed_ushort4* w1 = (const device packed_ushort4*)(w + ulong(o + 1) * p.I);
  const device packed_float4* x0 = (const device packed_float4*)(x + ulong(first) * p.x_stride);
  const device packed_float4* x1 = (const device packed_float4*)(x + ulong(first + min(1u, rows - 1)) * p.x_stride);
  const device packed_float4* x2 = (const device packed_float4*)(x + ulong(first + min(2u, rows - 1)) * p.x_stride);
  const device packed_float4* x3 = (const device packed_float4*)(x + ulong(first + min(3u, rows - 1)) * p.x_stride);
  float4 a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0;
  for (uint i = lane; i < p.I / 4; i += 32) {
    float4 v0 = float4(x0[i]), v1 = float4(x1[i]), v2 = float4(x2[i]), v3 = float4(x3[i]);
    float4 wa = as_type<float4>(uint4(ushort4(w0[i])) << 16), wb = as_type<float4>(uint4(ushort4(w1[i])) << 16);
    a0 += wa * v0; a1 += wa * v1; a2 += wa * v2; a3 += wa * v3;
    b0 += wb * v0; b1 += wb * v1; b2 += wb * v2; b3 += wb * v3;
  }
  float ta0 = simd_sum((a0.x + a0.y) + (a0.z + a0.w)), ta1 = simd_sum((a1.x + a1.y) + (a1.z + a1.w));
  float ta2 = simd_sum((a2.x + a2.y) + (a2.z + a2.w)), ta3 = simd_sum((a3.x + a3.y) + (a3.z + a3.w));
  float tb0 = simd_sum((b0.x + b0.y) + (b0.z + b0.w)), tb1 = simd_sum((b1.x + b1.y) + (b1.z + b1.w));
  float tb2 = simd_sum((b2.x + b2.y) + (b2.z + b2.w)), tb3 = simd_sum((b3.x + b3.y) + (b3.z + b3.w));
  if (lane != 0) return;
  device float* out = y + ulong(first) * p.y_stride + o;
  out[0] = ta0; out[1] = tb0;
  if (1u < rows) { out[p.y_stride] = ta1; out[p.y_stride + 1] = tb1; }
  if (2u < rows) { out[2 * p.y_stride] = ta2; out[2 * p.y_stride + 1] = tb2; }
  if (3u < rows) { out[3 * p.y_stride] = ta3; out[3 * p.y_stride + 1] = tb3; }
}

// expert_act for the rows of a list of pairs: act[out] = silu(gate) * up.
kernel void expert_act_pairs(const device float* gate_up [[buffer(0)]], device float* act [[buffer(1)]],
                             const device Pair* pairs [[buffer(2)]], constant ActP& p [[buffer(3)]],
                             uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.M || g.y >= p.pairs) return;
  ulong row = pairs[g.y].out;
  act[row * p.M + g.x] = silu(gate_up[row * 2 * p.M + g.x]) * gate_up[row * 2 * p.M + p.M + g.x];
}
