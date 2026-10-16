// Metal kernels for the Whallm native DeepSeek V4.1 engine.
// Activations are FP32. Resident matrices stay in the checkpoint's FP8 E4M3 bytes
// (one E8M0 scale per 32 x 32 block) or BF16, and routed experts stay in MXFP4;
// all are decoded where they are read.
#include <metal_stdlib>
using namespace metal;

static inline float bf16(ushort v) { return as_type<float>(uint(v) << 16); }
// Round to nearest even. Cached activations are exact in BF16 (see the fake-quant kernels).
static inline ushort to_bf16(float v) {
  uint bits = as_type<uint>(v);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static inline float sigm(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float silu(float x) { return x * sigm(x); }

// FP8 E4M3 bytes as FP16 bit patterns: the value divided by 256, exactly. The 256
// is folded into the block scale below (MLX converts FP8 the same way).
static inline float4 fp8x4(uint bytes) {
  ushort4 b = ushort4(as_type<uchar4>(bytes));
  return float4(as_type<half4>(ushort4(((b & 0x80) << 8) | ((b & 0x7F) << 7))));
}
static inline float fp8f(uchar byte) {
  ushort b = byte;
  return float(as_type<half>(ushort(((b & 0x80) << 8) | ((b & 0x7F) << 7))));
}
// An E8M0 scale (2^(s - 127)) times 256.
static inline float e8m0_256(uchar s) { return as_type<float>(uint(s + 8) << 23); }

// Four MXFP4 (E2M1) nibbles, the low 16 bits of `bits`, as their values divided by 2^14:
// sign, exponent and mantissa bits land in an FP16 pattern, exactly (zero, subnormal 0.5
// and normal values alike). Multiply the block scale by 2^14 (fp4_scale) to compensate;
// both products are exact, so the results equal those of a table lookup.
static inline float4 fp4x4(uint bits) {
  ushort4 n = ushort4(ushort(bits), ushort(bits >> 4), ushort(bits >> 8), ushort(bits >> 12)) & ushort4(15);
  return float4(as_type<half4>(ushort4(((n & 8) << 12) | ((n & 7) << 9))));
}
static inline float fp4_scale(uchar s) { return as_type<float>(uint(s) << 23) * 16384.0f; }

// Decode with two values per 32-bit operation: a half2 holds the values of two fields 16
// bits apart, at the same scale as fp4x4 and fp8x4 (exact).
// FP4: nibbles of `t` at bits 0-3 and 16-19.
static inline float2 fp4pair(uint t) {
  return float2(as_type<half2>(((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9)));
}
// FP8 E4M3: bytes of `t` at bits 0-7 and 16-23.
static inline float2 fp8pair(uint t) {
  return float2(as_type<half2>(((t & 0x00800080u) << 8) | ((t & 0x007F007Fu) << 7)));
}

// The 8 MXFP4 values of one word, in order: w[0..7].
struct Fp4x8 { float2 a, b, c, d; };  // a = (w0, w4), b = (w2, w6), c = (w1, w5), d = (w3, w7)
static inline Fp4x8 fp4x8(uint v) {
  uint e = v & 0x0F0F0F0Fu, o = (v >> 4) & 0x0F0F0F0Fu;
  return {fp4pair(e), fp4pair(e >> 8), fp4pair(o), fp4pair(o >> 8)};
}
// sum_i w_i x_i with x = (lo, hi) in a fixed order.
static inline float dot8(Fp4x8 w, float4 lo, float4 hi) {
  return ((w.a.x * lo.x + w.c.x * lo.y) + (w.b.x * lo.z + w.d.x * lo.w)) +
         ((w.a.y * hi.x + w.c.y * hi.y) + (w.b.y * hi.z + w.d.y * hi.w));
}

// The 8 FP8 values of two words, in order.
struct Fp8x8 { float2 a, b, c, d; };  // a = (w0, w2), b = (w1, w3), c = (w4, w6), d = (w5, w7)
static inline Fp8x8 fp8x8(uint2 v) { return {fp8pair(v.x), fp8pair(v.x >> 8), fp8pair(v.y), fp8pair(v.y >> 8)}; }
static inline float dot8(Fp8x8 w, float4 lo, float4 hi) {
  return ((w.a.x * lo.x + w.b.x * lo.y) + (w.a.y * lo.z + w.b.y * lo.w)) +
         ((w.c.x * hi.x + w.d.x * hi.y) + (w.c.y * hi.z + w.d.y * hi.w));
}

// ---------------------------------------------------------------- FP8 products
// y[t, o] = sum_i x[t, i] * w[o, i]. w is FP8 E4M3 [O, I] with scales s[o / 32, i / 32].

struct LinearP { uint T; uint I; uint O; uint x_stride; uint y_stride; };

// Products for 1 to 8 rows, two outputs per simdgroup: a block of R rows reads each weight
// once. Rows past the block's end repeat its last row and are not written. Every row's
// sum has the same terms in the same order whatever R and the row count, so a token's
// result does not depend on how many tokens run together.
// The FP8 kernels take 8 inputs per lane step, the BF16 kernels 4.

template <uint R>
static inline void fp8_rows(const device float* x, const device uchar* w, const device uchar* s, uint x_stride, uint I,
                            uint first, uint rows, uint o, uint begin, uint end, uint lane, thread float* a, thread float* b) {
  const device uint2* w0 = (const device uint2*)(w + ulong(o) * I);
  const device uint2* w1 = (const device uint2*)(w + ulong(o + 1) * I);
  const device uchar* sr = s + ulong(o / 32) * (I / 32);
  const device float4* xr[R];
  for (uint r = 0; r < R; ++r) xr[r] = (const device float4*)(x + ulong(first + min(r, rows - 1)) * x_stride);
  for (uint r = 0; r < R; ++r) { a[r] = 0; b[r] = 0; }
  for (uint i = begin + lane; i < end; i += 32) {
    float scale = e8m0_256(sr[i / 4]);
    Fp8x8 wa = fp8x8(w0[i]), wb = fp8x8(w1[i]);
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      float4 l = xr[r][2 * i], h = xr[r][2 * i + 1];
      a[r] += dot8(wa, l, h) * scale;
      b[r] += dot8(wb, l, h) * scale;
    }
  }
  for (uint r = 0; r < R; ++r) { a[r] = simd_sum(a[r]); b[r] = simd_sum(b[r]); }
}

// Grid (32, O / 2, ceil(T / R)). O is even, I a multiple of 32.
template <uint R>
kernel void linear_fp8(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                       const device uchar* s [[buffer(2)]], device float* y [[buffer(3)]],
                       constant LinearP& p [[buffer(4)]],
                       uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * R, rows = min(R, p.T - first), o = g.y * 2;
  float a[R], b[R];
  fp8_rows<R>(x, w, s, p.x_stride, p.I, first, rows, o, 0, p.I / 8, lane, a, b);
  if (lane != 0) return;
  for (uint r = 0; r < rows; ++r) {
    device float* out = y + ulong(first + r) * p.y_stride + o;
    out[0] = a[r]; out[1] = b[r];
  }
}
template [[host_name("linear_fp8_1")]] kernel void linear_fp8<1>(const device float*, const device uchar*, const device uchar*,
    device float*, constant LinearP&, uint3, uint);
template [[host_name("linear_fp8")]] kernel void linear_fp8<4>(const device float*, const device uchar*, const device uchar*,
    device float*, constant LinearP&, uint3, uint);
template [[host_name("linear_fp8_8")]] kernel void linear_fp8<8>(const device float*, const device uchar*, const device uchar*,
    device float*, constant LinearP&, uint3, uint);

// The split products in one dispatch: simdgroup s of a threadgroup computes span s and
// simdgroup 0 adds the spans in order (the same sums as linear_*_split + split_sum).
// Grid: threadgroups (O / 2, ceil(T / R), groups), 32 * splits threads. Group g reads
// x + g * x_group, w + g * w_group, s + g * s_group and writes y + g * y_group.
struct TgSplitP { uint T; uint I; uint O; uint x_stride; uint y_stride; uint span; uint splits;
                  uint x_group; uint w_group; uint s_group; uint y_group; };
constant constexpr uint MAX_SPLITS = 16;

template <uint R>
kernel void linear_fp8_tgsplit(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                               const device uchar* s [[buffer(2)]], device float* y [[buffer(3)]],
                               constant TgSplitP& p [[buffer(4)]],
                               uint3 tg [[threadgroup_position_in_grid]], ushort split [[simdgroup_index_in_threadgroup]],
                               ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[MAX_SPLITS][2 * R];
  uint o = tg.x * 2, first = tg.y * R, rows = min(R, p.T - first);
  x += ulong(tg.z) * p.x_group; w += ulong(tg.z) * p.w_group; s += ulong(tg.z) * p.s_group; y += ulong(tg.z) * p.y_group;
  float a[R], b[R];
  fp8_rows<R>(x, w, s, p.x_stride, p.I, first, rows, o, split * p.span / 8, min(p.I, (split + 1) * p.span) / 8, lane, a, b);
  if (lane == 0)
    for (uint r = 0; r < rows; ++r) { part[split][2 * r] = a[r]; part[split][2 * r + 1] = b[r]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split != 0 || lane >= 2 * rows) return;
  float total = 0;
  for (uint k = 0; k < p.splits; ++k) total += part[k][lane];
  y[ulong(first + lane / 2) * p.y_stride + o + lane % 2] = total;
}
template [[host_name("linear_fp8_tgsplit_1")]] kernel void linear_fp8_tgsplit<1>(const device float*, const device uchar*,
    const device uchar*, device float*, constant TgSplitP&, uint3, ushort, ushort);
template [[host_name("linear_fp8_tgsplit")]] kernel void linear_fp8_tgsplit<4>(const device float*, const device uchar*,
    const device uchar*, device float*, constant TgSplitP&, uint3, ushort, ushort);
template [[host_name("linear_fp8_tgsplit_8")]] kernel void linear_fp8_tgsplit<8>(const device float*, const device uchar*,
    const device uchar*, device float*, constant TgSplitP&, uint3, ushort, ushort);

// ---------------------------------------------------------------- BF16 products (router, compressor, indexer, head)

template <uint R>
static inline void bf16_rows(const device float* x, const device ushort* w, uint x_stride, uint I, uint first, uint rows,
                             uint o, uint begin, uint end, uint lane, thread float* ta, thread float* tb) {
  const device packed_ushort4* w0 = (const device packed_ushort4*)(w + ulong(o) * I);
  const device packed_ushort4* w1 = (const device packed_ushort4*)(w + ulong(o + 1) * I);
  const device float4* xr[R];
  for (uint r = 0; r < R; ++r) xr[r] = (const device float4*)(x + ulong(first + min(r, rows - 1)) * x_stride);
  float4 a[R], b[R];
  for (uint r = 0; r < R; ++r) { a[r] = 0; b[r] = 0; }
  for (uint i = begin + lane; i < end; i += 32) {
    float4 wa = as_type<float4>(uint4(ushort4(w0[i])) << 16), wb = as_type<float4>(uint4(ushort4(w1[i])) << 16);
    for (uint r = 0; r < R; ++r) {
      if (r >= rows) break;
      float4 v = xr[r][i];
      a[r] += wa * v;
      b[r] += wb * v;
    }
  }
  for (uint r = 0; r < R; ++r) {
    ta[r] = simd_sum((a[r].x + a[r].y) + (a[r].z + a[r].w));
    tb[r] = simd_sum((b[r].x + b[r].y) + (b[r].z + b[r].w));
  }
}

// Grid (32, O / 2, ceil(T / R)). O is even.
template <uint R>
kernel void linear_bf16(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant LinearP& p [[buffer(3)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint first = g.z * R, rows = min(R, p.T - first), o = g.y * 2;
  float a[R], b[R];
  bf16_rows<R>(x, w, p.x_stride, p.I, first, rows, o, 0, p.I / 4, lane, a, b);
  if (lane != 0) return;
  for (uint r = 0; r < rows; ++r) {
    device float* out = y + ulong(first + r) * p.y_stride + o;
    out[0] = a[r]; out[1] = b[r];
  }
}
template [[host_name("linear_bf16_1")]] kernel void linear_bf16<1>(const device float*, const device ushort*, device float*,
    constant LinearP&, uint3, uint);
template [[host_name("linear_bf16")]] kernel void linear_bf16<4>(const device float*, const device ushort*, device float*,
    constant LinearP&, uint3, uint);
template [[host_name("linear_bf16_8")]] kernel void linear_bf16<8>(const device float*, const device ushort*, device float*,
    constant LinearP&, uint3, uint);

// linear_fp8_tgsplit for BF16 weights (x_group, w_group in elements; s_group unused).
template <uint R>
kernel void linear_bf16_tgsplit(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                                device float* y [[buffer(2)]], constant TgSplitP& p [[buffer(3)]],
                                uint3 tg [[threadgroup_position_in_grid]], ushort split [[simdgroup_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[MAX_SPLITS][2 * R];
  uint o = tg.x * 2, first = tg.y * R, rows = min(R, p.T - first);
  x += ulong(tg.z) * p.x_group; w += ulong(tg.z) * p.w_group; y += ulong(tg.z) * p.y_group;
  float a[R], b[R];
  bf16_rows<R>(x, w, p.x_stride, p.I, first, rows, o, split * p.span / 4, min(p.I, (split + 1) * p.span) / 4, lane, a, b);
  if (lane == 0)
    for (uint r = 0; r < rows; ++r) { part[split][2 * r] = a[r]; part[split][2 * r + 1] = b[r]; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (split != 0 || lane >= 2 * rows) return;
  float total = 0;
  for (uint k = 0; k < p.splits; ++k) total += part[k][lane];
  y[ulong(first + lane / 2) * p.y_stride + o + lane % 2] = total;
}
template [[host_name("linear_bf16_tgsplit_1")]] kernel void linear_bf16_tgsplit<1>(const device float*, const device ushort*,
    device float*, constant TgSplitP&, uint3, ushort, ushort);
template [[host_name("linear_bf16_tgsplit")]] kernel void linear_bf16_tgsplit<4>(const device float*, const device ushort*,
    device float*, constant TgSplitP&, uint3, ushort, ushort);
template [[host_name("linear_bf16_tgsplit_8")]] kernel void linear_bf16_tgsplit<8>(const device float*, const device ushort*,
    device float*, constant TgSplitP&, uint3, ushort, ushort);

// ---------------------------------------------------------------- simdgroup matrix products (9 to 63 rows)
// One simdgroup holds a 16-row by 32-output tile as eight 8x8 matrices; weights are
// placed straight into the matrix registers. That relies on where each lane's two
// elements sit in an 8x8 matrix; sg_probe reports it and the engine checks it.

kernel void sg_probe(device float* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 m;
  m.thread_elements()[0] = float(lane * 2);
  m.thread_elements()[1] = float(lane * 2 + 1);
  simdgroup_store(m, out, 8);
}

static inline uint2 sg_place(uint lane) {
  uint quadrant = lane / 8, local = (lane % 8) * 2;
  return uint2((quadrant / 2) * 4 + local / 4, (quadrant % 2) * 4 + local % 4);
}

struct GemmP { uint T; uint I; uint O; uint x_stride; uint y_stride; };

#define SG_ACCUMULATE \
  simdgroup_multiply_accumulate(c00, b0, a0, c00); simdgroup_multiply_accumulate(c01, b0, a1, c01); \
  simdgroup_multiply_accumulate(c10, b1, a0, c10); simdgroup_multiply_accumulate(c11, b1, a1, c11); \
  simdgroup_multiply_accumulate(c20, b2, a0, c20); simdgroup_multiply_accumulate(c21, b2, a1, c21); \
  simdgroup_multiply_accumulate(c30, b3, a0, c30); simdgroup_multiply_accumulate(c31, b3, a1, c31);

#define SG_STORE(y, stride) { \
  device float* y0 = y; device float* y1 = y0 + ulong(8) * stride; \
  simdgroup_store(c00, y0, stride, 0, true); simdgroup_store(c01, y1, stride, 0, true); \
  simdgroup_store(c10, y0 + 8, stride, 0, true); simdgroup_store(c11, y1 + 8, stride, 0, true); \
  simdgroup_store(c20, y0 + 16, stride, 0, true); simdgroup_store(c21, y1 + 16, stride, 0, true); \
  simdgroup_store(c30, y0 + 24, stride, 0, true); simdgroup_store(c31, y1 + 24, stride, 0, true); }

#define SG_ZERO \
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3; \
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f), \
                     c11 = simdgroup_float8x8(0.0f), c20 = simdgroup_float8x8(0.0f), c21 = simdgroup_float8x8(0.0f), \
                     c30 = simdgroup_float8x8(0.0f), c31 = simdgroup_float8x8(0.0f);

// Accumulates the transposed tile C[o, t] += W[o, i] * X^T[i, t]. O is a multiple of 32;
// x and y hold rows up to the next multiple of 16. Grid (32, O / 32, ceil(T / 16)).
kernel void gemm_fp8_sg(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                        const device uchar* s [[buffer(2)]], device float* y [[buffer(3)]],
                        constant GemmP& p [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  const device float* x0 = x + ulong(t) * p.x_stride;
  const device float* x1 = x0 + ulong(8) * p.x_stride;
  const device packed_uchar2* w0 = (const device packed_uchar2*)(w + ulong(o + place.x) * p.I + place.y);
  const device packed_uchar2* w1 = (const device packed_uchar2*)(w + ulong(o + 8 + place.x) * p.I + place.y);
  const device packed_uchar2* w2 = (const device packed_uchar2*)(w + ulong(o + 16 + place.x) * p.I + place.y);
  const device packed_uchar2* w3 = (const device packed_uchar2*)(w + ulong(o + 24 + place.x) * p.I + place.y);
  const device uchar* sr = s + ulong(o / 32) * (p.I / 32);  // the 32 outputs share one row of blocks
  SG_ZERO
  for (uint i = 0; i < p.I; i += 8) {
    simdgroup_load(a0, x0 + i, p.x_stride, 0, true);
    simdgroup_load(a1, x1 + i, p.x_stride, 0, true);
    float scale = e8m0_256(sr[i / 32]);
    uint q = i / 2;
    uchar2 v0 = uchar2(w0[q]), v1 = uchar2(w1[q]), v2 = uchar2(w2[q]), v3 = uchar2(w3[q]);
    b0.thread_elements()[0] = fp8f(v0.x) * scale; b0.thread_elements()[1] = fp8f(v0.y) * scale;
    b1.thread_elements()[0] = fp8f(v1.x) * scale; b1.thread_elements()[1] = fp8f(v1.y) * scale;
    b2.thread_elements()[0] = fp8f(v2.x) * scale; b2.thread_elements()[1] = fp8f(v2.y) * scale;
    b3.thread_elements()[0] = fp8f(v3.x) * scale; b3.thread_elements()[1] = fp8f(v3.y) * scale;
    SG_ACCUMULATE
  }
  SG_STORE(y + ulong(t) * p.y_stride + o, p.y_stride)
}

kernel void gemm_bf16_sg(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                         device float* y [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                         uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  const device float* x0 = x + ulong(t) * p.x_stride;
  const device float* x1 = x0 + ulong(8) * p.x_stride;
  const device packed_ushort2* w0 = (const device packed_ushort2*)(w + ulong(o + place.x) * p.I + place.y);
  const device packed_ushort2* w1 = (const device packed_ushort2*)(w + ulong(o + 8 + place.x) * p.I + place.y);
  const device packed_ushort2* w2 = (const device packed_ushort2*)(w + ulong(o + 16 + place.x) * p.I + place.y);
  const device packed_ushort2* w3 = (const device packed_ushort2*)(w + ulong(o + 24 + place.x) * p.I + place.y);
  SG_ZERO
  for (uint i = 0; i < p.I; i += 8) {
    simdgroup_load(a0, x0 + i, p.x_stride, 0, true);
    simdgroup_load(a1, x1 + i, p.x_stride, 0, true);
    uint q = i / 2;
    float2 v0 = as_type<float2>(uint2(ushort2(w0[q])) << 16), v1 = as_type<float2>(uint2(ushort2(w1[q])) << 16);
    float2 v2 = as_type<float2>(uint2(ushort2(w2[q])) << 16), v3 = as_type<float2>(uint2(ushort2(w3[q])) << 16);
    b0.thread_elements()[0] = v0.x; b0.thread_elements()[1] = v0.y;
    b1.thread_elements()[0] = v1.x; b1.thread_elements()[1] = v1.y;
    b2.thread_elements()[0] = v2.x; b2.thread_elements()[1] = v2.y;
    b3.thread_elements()[0] = v3.x; b3.thread_elements()[1] = v3.y;
    SG_ACCUMULATE
  }
  SG_STORE(y + ulong(t) * p.y_stride + o, p.y_stride)
}

// ---------------------------------------------------------------- staged products (64 rows and more)
// A threadgroup of 2 x 2 simdgroups computes a 64 x 64 tile; each K step of 16 stages
// x and the decoded weights in threadgroup memory (MLX Steel GEMM layout, as in the
// native Qwen engine). Requires O % 64 == 0 and I % 16 == 0. Rows past T are read
// clamped and not written.
constant constexpr short TG_BM = 64, TG_BN = 64, TG_BK = 16, TG_LD = TG_BK + 4;

#define TG_BODY(LOAD_B) \
  threadgroup float As[TG_BM * TG_LD]; \
  threadgroup float Bs[TG_BN * TG_LD]; \
  uint row0 = tid.y * TG_BM, col0 = tid.x * TG_BN; \
  ushort t = sg * 32 + lane; \
  short load_row = t / 2, load_col = (t % 2) * 8; \
  const device float* ap = x + ulong(min(row0 + load_row, p.T - 1)) * p.x_stride + load_col; \
  short qid = lane / 4; \
  short fm = (qid & 4) + ((lane / 2) % 4), fn = (qid & 2) * 2 + (lane % 2) * 2; \
  short tm = 8 * (sg / 2), tn = 8 * (sg % 2); \
  float2 c[4][4]; \
  for (short i = 0; i < 4; ++i) \
    for (short j = 0; j < 4; ++j) c[i][j] = 0; \
  for (uint k = 0; k < p.I; k += TG_BK) { \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    float4 a0 = *(const device packed_float4*)ap, a1 = *(const device packed_float4*)(ap + 4); \
    float4 b0, b1; \
    LOAD_B \
    threadgroup float* da = As + load_row * TG_LD + load_col; \
    threadgroup float* db = Bs + load_row * TG_LD + load_col; \
    da[0] = a0.x; da[1] = a0.y; da[2] = a0.z; da[3] = a0.w; da[4] = a1.x; da[5] = a1.y; da[6] = a1.z; da[7] = a1.w; \
    db[0] = b0.x; db[1] = b0.y; db[2] = b0.z; db[3] = b0.w; db[4] = b1.x; db[5] = b1.y; db[6] = b1.z; db[7] = b1.w; \
    ap += TG_BK; \
    threadgroup_barrier(mem_flags::mem_threadgroup); \
    _Pragma("unroll") \
    for (short kk = 0; kk < TG_BK; kk += 8) { \
      float2 af[4], bf[4]; \
      for (short i = 0; i < 4; ++i) { \
        const threadgroup float* src = As + (tm + i * 16 + fm) * TG_LD + kk + fn; \
        af[i] = float2(src[0], src[1]); \
      } \
      for (short j = 0; j < 4; ++j) { \
        const threadgroup float* src = Bs + (tn + j * 16 + fn) * TG_LD + kk + fm; \
        bf[j] = float2(src[0], src[TG_LD]); \
      } \
      for (short i = 0; i < 4; ++i) { \
        simdgroup_float8x8 am; \
        am.thread_elements()[0] = af[i].x; \
        am.thread_elements()[1] = af[i].y; \
        for (short j = 0; j < 4; ++j) { \
          simdgroup_float8x8 bm, cm; \
          bm.thread_elements()[0] = bf[j].x; \
          bm.thread_elements()[1] = bf[j].y; \
          cm.thread_elements()[0] = c[i][j].x; \
          cm.thread_elements()[1] = c[i][j].y; \
          simdgroup_multiply_accumulate(cm, am, bm, cm); \
          c[i][j] = float2(cm.thread_elements()[0], cm.thread_elements()[1]); \
        } \
      } \
    } \
  } \
  for (short i = 0; i < 4; ++i) { \
    uint row = row0 + tm + i * 16 + fm; \
    if (row >= p.T) continue; \
    for (short j = 0; j < 4; ++j) \
      *(device float2*)(y + ulong(row) * p.y_stride + col0 + tn + j * 16 + fn) = c[i][j]; \
  }

kernel void gemm_fp8_tg(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                        const device uchar* s [[buffer(2)]], device float* y [[buffer(3)]],
                        constant GemmP& p [[buffer(4)]],
                        uint3 tid [[threadgroup_position_in_grid]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const device uchar* bp = w + ulong(tid.x * TG_BN + (sg * 32 + lane) / 2) * p.I + ((sg * 32 + lane) % 2) * 8;
  const device uchar* sp = s + ulong((tid.x * TG_BN + (sg * 32 + lane) / 2) / 32) * (p.I / 32);
  TG_BODY({
    uint2 bits = *(const device uint2*)bp;
    float scale = e8m0_256(sp[k / 32]);
    b0 = fp8x4(bits.x) * scale; b1 = fp8x4(bits.y) * scale;
    bp += TG_BK;
  })
}

kernel void gemm_bf16_tg(const device float* x [[buffer(0)]], const device ushort* w [[buffer(1)]],
                         device float* y [[buffer(2)]], constant GemmP& p [[buffer(3)]],
                         uint3 tid [[threadgroup_position_in_grid]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  const device ushort* bp = w + ulong(tid.x * TG_BN + (sg * 32 + lane) / 2) * p.I + ((sg * 32 + lane) % 2) * 8;
  TG_BODY({
    b0 = as_type<float4>(uint4(ushort4(*(const device packed_ushort4*)bp)) << 16);
    b1 = as_type<float4>(uint4(ushort4(*(const device packed_ushort4*)(bp + 4))) << 16);
    bp += TG_BK;
  })
}

// ---------------------------------------------------------------- norms

struct NormP { uint T; uint groups; uint D; uint row_stride; uint base; float eps; uint has_w; };

// Group (t, n) holds D values at t * row_stride + base + n * D; in and out share the layout.
// One simdgroup per group. Grid (32, groups, T).
kernel void rms_norm(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                     device float* y [[buffer(2)]], constant NormP& p [[buffer(3)]],
                     uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  ulong at = ulong(g.z) * p.row_stride + p.base + g.y * p.D;
  const device float* xr = x + at;
  device float* yr = y + at;
  float ss = 0;
  for (uint i = lane; i < p.D; i += 32) ss += xr[i] * xr[i];
  float r = rsqrt(simd_sum(ss) / float(p.D) + p.eps);
  if (p.has_w != 0) {
    for (uint i = lane; i < p.D; i += 32) yr[i] = xr[i] * r * w[i];
  } else {
    for (uint i = lane; i < p.D; i += 32) yr[i] = xr[i] * r;
  }
}

// rms_norm for long rows: 256 threads per group, partial sums added in a fixed order.
// Grid: threadgroups (groups, T), 256 threads.
kernel void rms_norm_tg(const device float* x [[buffer(0)]], const device float* w [[buffer(1)]],
                        device float* y [[buffer(2)]], constant NormP& p [[buffer(3)]],
                        uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  ulong at = ulong(tg.y) * p.row_stride + p.base + tg.x * p.D;
  const device float* xr = x + at;
  device float* yr = y + at;
  float ss = 0;
  for (uint i = tid; i < p.D; i += 256) ss += xr[i] * xr[i];
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0;  // every thread adds the parts in the same order
  for (uint k = 0; k < 8; ++k) total += part[k];
  float r = rsqrt(total / float(p.D) + p.eps);
  if (p.has_w != 0) {
    for (uint i = tid; i < p.D; i += 256) yr[i] = xr[i] * r * w[i];
  } else {
    for (uint i = tid; i < p.D; i += 256) yr[i] = xr[i] * r;
  }
}

// ---------------------------------------------------------------- hyper-connections
// The stream of a token is 4 copies of the hidden state, [4, H]. A block's mixing
// weights are 24 numbers per token: pre[4] | post[4] | comb[4][4] (row j, column k).

struct MixP { uint T; uint W; float norm_eps; float hc_eps; uint iters; };

// mixes = (h . fn^T) * rsqrt(mean(h^2) + eps), split into pre, post and a
// Sinkhorn-normalized comb, in two steps. hc_mix_partial: threadgroup (s, t) of 256
// threads takes 1,024 of the token's W values (one float4 each) and writes the 24 dot
// products and the sum of squares, partial[t][s][25]. Grid: threadgroups (W / 1024, T).
constant constexpr uint MIX_SPLIT = 1024;

kernel void hc_mix_partial(const device float* h [[buffer(0)]], const device float* fn [[buffer(1)]],
                           device float* partial [[buffer(2)]], constant MixP& p [[buffer(3)]],
                           uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8][25];
  uint t = tg.y, i = tg.x * (MIX_SPLIT / 4) + tid;
  float4 v = ((const device float4*)(h + ulong(t) * p.W))[i];
  for (uint m = 0; m < 24; ++m) {
    float d = simd_sum(dot(v, ((const device float4*)(fn + ulong(m) * p.W))[i]));
    if (lane == 0) part[sg][m] = d;
  }
  float ss = simd_sum(dot(v, v));
  if (lane == 0) part[sg][24] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid >= 25) return;
  float total = 0;
  for (uint k = 0; k < 8; ++k) total += part[k][tid];
  partial[(ulong(t) * (p.W / MIX_SPLIT) + tg.x) * 25 + tid] = total;
}

// Sum of v from lanes first, first + step, first + 2 * step, first + 3 * step, added left to right.
static inline float lane_sum4(float v, uint first, uint step) {
  return ((simd_shuffle(v, first) + simd_shuffle(v, first + step)) + simd_shuffle(v, first + 2 * step)) +
         simd_shuffle(v, first + 3 * step);
}

// hc_mix_finish: one thread per token adds the parts in order and splits the mixes.
// One simdgroup per token: lane k < 25 adds part k over the splits in order, lane 0 splits
// the mixes. Grid: threadgroups (T), 32 threads.
kernel void hc_mix_finish(const device float* partial [[buffer(0)]], const device float* scale [[buffer(1)]],
                          const device float* base [[buffer(2)]], device float* mix [[buffer(3)]],
                          constant MixP& p [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                          ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[25];
  uint t = tg.x, parts = p.W / MIX_SPLIT;
  if (lane < 25) {
    float total = 0;
    for (uint s = 0; s < parts; ++s) total += partial[(ulong(t) * parts + s) * 25 + lane];
    sums[lane] = total;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Lane (j, k) = (lane / 4 % 4, lane % 4) holds comb[j][k]; sums gather the four values
  // with shuffles and add them left to right, as the one-thread loop does.
  float r = rsqrt(sums[24] / float(p.W) + p.norm_eps);
  device float* out = mix + ulong(t) * 24;
  if (lane < 4) out[lane] = sigm(sums[lane] * r * scale[0] + base[lane]) + p.hc_eps;
  else if (lane < 8) out[lane] = 2.0f * sigm(sums[lane] * r * scale[1] + base[lane]);
  uint j = (lane >> 2) & 3, k = lane & 3, row = lane & ~3u, column = (lane & 16) + k;
  float v = sums[8 + j * 4 + k] * r * scale[2] + base[8 + j * 4 + k];
  float top = max(max(simd_shuffle(v, row), simd_shuffle(v, row + 1)), max(simd_shuffle(v, row + 2), simd_shuffle(v, row + 3)));
  float e = exp(v - top);
  float c = e / lane_sum4(e, row, 1) + p.hc_eps;
  for (uint iter = 0; iter < p.iters; ++iter) {
    if (iter > 0) c /= lane_sum4(c, row, 1) + p.hc_eps;
    c /= lane_sum4(c, column, 4) + p.hc_eps;
  }
  if (lane < 16) out[8 + lane] = c;
}

struct HcP { uint T; uint H; uint mix_stride; };

struct MeanP { uint T; uint H; uint out_stride; };

// out[t, d] = mean_k h[t, k, d], rows out_stride apart. Grid (H / 4, T).
kernel void hc_mean(const device float* h [[buffer(0)]], device float* out [[buffer(1)]],
                    constant MeanP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x * 4 >= p.H || g.y >= p.T) return;
  const device float4* hr = (const device float4*)(h + ulong(g.y) * 4 * p.H);
  uint q = p.H / 4;
  ((device float4*)(out + ulong(g.y) * p.out_stride))[g.x] = (hr[g.x] + hr[q + g.x] + hr[2 * q + g.x] + hr[3 * q + g.x]) * 0.25f;
}

// x[t, d] = sum_k pre[t, k] * h[t, k, d]. Grid (H / 4, T).
kernel void hc_pre(const device float* h [[buffer(0)]], const device float* pre [[buffer(1)]],
                   device float* x [[buffer(2)]], constant HcP& p [[buffer(3)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x * 4 >= p.H || g.y >= p.T) return;
  const device float* m = pre + ulong(g.y) * p.mix_stride;
  const device float4* hr = (const device float4*)(h + ulong(g.y) * 4 * p.H);
  uint q = p.H / 4;
  float4 v = m[0] * hr[g.x] + m[1] * hr[q + g.x] + m[2] * hr[2 * q + g.x] + m[3] * hr[3 * q + g.x];
  ((device float4*)(x + ulong(g.y) * p.H))[g.x] = v;
}

// h[t, k, d] = post[t, k] * y[t, d] + sum_j comb[t, j, k] * h[t, j, d], in place. Grid (H / 4, T).
kernel void hc_post(device float* h [[buffer(0)]], const device float* y [[buffer(1)]],
                    const device float* mix [[buffer(2)]], constant HcP& p [[buffer(3)]],
                    uint2 g [[thread_position_in_grid]]) {
  if (g.x * 4 >= p.H || g.y >= p.T) return;
  const device float* m = mix + ulong(g.y) * 24;
  device float4* hr = (device float4*)(h + ulong(g.y) * 4 * p.H);
  uint q = p.H / 4;
  float4 r0 = hr[g.x], r1 = hr[q + g.x], r2 = hr[2 * q + g.x], r3 = hr[3 * q + g.x];
  float4 v = ((const device float4*)(y + ulong(g.y) * p.H))[g.x];
  for (uint k = 0; k < 4; ++k)
    hr[k * q + g.x] = m[4 + k] * v + (m[8 + k] * r0 + m[12 + k] * r1 + m[16 + k] * r2 + m[20 + k] * r3);
}

// ---------------------------------------------------------------- rotary embedding
// Rotates adjacent pairs (x[2i], x[2i + 1]) of the last 64 channels of each head by
// position * freq[i]. Group (t, n) starts at t * row_stride + base + n * D.

struct RopeP { uint T; uint groups; uint D; uint row_stride; uint base; uint pos0; uint pos_step; uint inverse; };

kernel void rope_tail(device float* x [[buffer(0)]], const device float* freq [[buffer(1)]],
                      constant RopeP& p [[buffer(2)]], uint3 g [[thread_position_in_grid]]) {
  if (g.x >= 32 || g.y >= p.groups || g.z >= p.T) return;
  device float* v = x + ulong(g.z) * p.row_stride + p.base + g.y * p.D + (p.D - 64) + 2 * g.x;
  float angle = float((p.pos0 + g.z) * p.pos_step) * freq[g.x];
  float c = precise::cos(angle), s = precise::sin(angle);
  if (p.inverse != 0) s = -s;
  float re = v[0], im = v[1];
  v[0] = re * c - im * s;
  v[1] = re * s + im * c;
}

// ---------------------------------------------------------------- activation fake quantization
// V4.1 rounds some activations through FP8 or FP4, as its training did. Every result
// is exactly representable in BF16, so the rounded rows are stored as BF16.

// PyTorch's float -> FP8 E4M3 (round to nearest even, saturating), as MLX uses it.
static inline uchar to_e4m3(float f) {
  uint fp8_max = 543u << 21, denorm_mask = 141u << 23;
  uint bits = as_type<uint>(f);
  uint sign = bits & 0x80000000u;
  bits ^= sign;
  uchar out;
  if (bits >= fp8_max) {
    out = 0x7E;
  } else if (bits < (121u << 23)) {
    bits = as_type<uint>(as_type<float>(bits) + as_type<float>(denorm_mask));
    out = uchar(bits - denorm_mask);
  } else {
    uint odd = (bits >> 20) & 1;
    bits += ((uint)(7 - 127) << 23) + 0x7FFFF;
    bits += odd;
    out = uchar(bits >> 20);
  }
  return out | uchar(sign >> 24);
}
static inline float from_e4m3(uchar b) { return fp8f(b) * 256.0f; }

// 2^ceil(log2(v)) from the bits of v.
static inline float pow2_ceil(float v) {
  uint bits = as_type<uint>(v);
  int e = int((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0 ? 1 : 0);
  return as_type<float>(uint(e + 127) << 23);
}

// Nearest E2M1 value of |v| <= 6, ties to the even code, sign kept.
static inline float e2m1(float v) {
  const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  float m = abs(v);
  uint i = uint(m > 0.25f) + uint(m >= 0.75f) + uint(m > 1.25f) + uint(m >= 1.75f) + uint(m > 2.5f) + uint(m >= 3.5f) + uint(m > 5.0f);
  return v > 0 ? table[i] : (v < 0 ? -table[i] : 0.0f);
}

// mode 0: FP8 E4M3, blocks of 32, power-of-two scales (window KV)
// mode 1: FP4 E2M1, blocks of 32, power-of-two scales (indexer queries and keys)
// mode 2: FP4 E2M1, blocks of 16, E4M3 scales (compressed KV)
struct QuantP { uint T; uint D; uint in_stride; uint out_stride; uint mode; uint write_f32; };

// Block b of row t: x[t * in_stride + b * block ...]; rounded values go to out (BF16,
// row stride out_stride) and, with write_f32, back into x. Grid (D / block, T).
kernel void fake_quant(device float* x [[buffer(0)]], device ushort* out [[buffer(1)]],
                       constant QuantP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  uint block = p.mode == 2 ? 16 : 32;
  if (g.x * block >= p.D || g.y >= p.T) return;
  device float* v = x + ulong(g.y) * p.in_stride + g.x * block;
  device ushort* o = out + ulong(g.y) * p.out_stride + g.x * block;
  float amax = 0;
  for (uint i = 0; i < block; ++i) amax = max(amax, abs(v[i]));
  float scale;
  if (p.mode == 0) scale = pow2_ceil(max(amax, 1e-4f) * (1.0f / 448.0f));
  else if (p.mode == 1) scale = pow2_ceil(max(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
  else scale = from_e4m3(to_e4m3(max(amax, 6.0f * 0x1p-9f) / 6.0f));
  for (uint i = 0; i < block; ++i) {
    float q;
    if (p.mode == 0) q = from_e4m3(to_e4m3(clamp(v[i] / scale, -448.0f, 448.0f))) * scale;
    else q = e2m1(clamp(v[i] / scale, -6.0f, 6.0f)) * scale;
    o[i] = to_bf16(q);
    if (p.write_f32 != 0) v[i] = q;
  }
}

// ---------------------------------------------------------------- KV compression (ratio 2)
// Group g pools rows 2g and 2g + 1: out = sum_j kv[j] * softmax_j(score[j]) per channel.
// Row p comes from the chunk (p >= pos0) or the carried row (p == pos0 - 1).
struct PoolP { uint groups; uint g0; uint pos0; uint D; };

kernel void compress_pool(const device float* kv [[buffer(0)]], const device float* score [[buffer(1)]],
                          const device float* carry_kv [[buffer(2)]], const device float* carry_score [[buffer(3)]],
                          device float* out [[buffer(4)]], constant PoolP& p [[buffer(5)]],
                          uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.D || g.y >= p.groups) return;
  uint position = (p.g0 + g.y) * 2;
  float k0, s0;
  if (position < p.pos0) { k0 = carry_kv[g.x]; s0 = carry_score[g.x]; }
  else { k0 = kv[ulong(position - p.pos0) * p.D + g.x]; s0 = score[ulong(position - p.pos0) * p.D + g.x]; }
  float k1 = kv[ulong(position + 1 - p.pos0) * p.D + g.x], s1 = score[ulong(position + 1 - p.pos0) * p.D + g.x];
  float top = max(s0, s1);
  float e0 = exp(s0 - top), e1 = exp(s1 - top), sum = e0 + e1;
  out[ulong(g.y) * p.D + g.x] = k0 * (e0 / sum) + k1 * (e1 / sum);
}

struct CopyRowsP { uint rows; uint D; uint in_stride; uint out_stride; };

kernel void copy_rows(const device float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant CopyRowsP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.D || g.y >= p.rows) return;
  y[ulong(g.y) * p.out_stride + g.x] = x[ulong(g.y) * p.in_stride + g.x];
}

kernel void copy_rows_u16(const device ushort* x [[buffer(0)]], device ushort* y [[buffer(1)]],
                          constant CopyRowsP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.D || g.y >= p.rows) return;
  y[ulong(g.y) * p.out_stride + g.x] = x[ulong(g.y) * p.in_stride + g.x];
}

// ---------------------------------------------------------------- sparse indexer

struct IndexScoreP { uint T; uint keys; uint heads; uint D; uint pos0; uint ratio; float scale; };

// score[t, j] = sum_h relu(q[t, h] . k[j]) * weight[t, h] for keys visible to query t
// (j < (pos0 + t + 1) / ratio), -inf otherwise. Grid (keys, T).
kernel void index_scores(const device float* q [[buffer(0)]], const device ushort* keys [[buffer(1)]],
                         const device float* weight [[buffer(2)]], device float* score [[buffer(3)]],
                         constant IndexScoreP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.keys || g.y >= p.T) return;
  float total = -INFINITY;
  if (g.x < (p.pos0 + g.y + 1) / p.ratio) {
    const device packed_ushort4* k = (const device packed_ushort4*)(keys + ulong(g.x) * p.D);
    float4 key[32];
    for (uint i = 0; i < p.D / 4; ++i) key[i] = as_type<float4>(uint4(ushort4(k[i])) << 16);
    total = 0;
    for (uint h = 0; h < p.heads; ++h) {
      const device float4* qh = (const device float4*)(q + (ulong(g.y) * p.heads + h) * p.D);
      float acc = 0;
      for (uint i = 0; i < p.D / 4; ++i) acc += dot(qh[i], key[i]);
      total += max(acc, 0.0f) * (weight[g.y * p.heads + h] * p.scale);
    }
  }
  score[ulong(g.y) * p.keys + g.x] = total;
}

static inline float2 bf16x2(const device ushort* key, uint at) {
  if (key == nullptr) return float2(0);
  return as_type<float2>(uint2(ushort2(*(const device packed_ushort2*)(key + at))) << 16);
}

// index_scores on the simdgroup matrix units: S^T[key, head] = K . Q^T per query, 16 keys
// by 32 heads per simdgroup (four simdgroups per threadgroup), then relu, the head weights
// and the head sum in index_scores' order. Queries and keys are FP4 values times power-of-
// two scales, so the dot products are exact in FP32 either way (checked on full prompts).
// Requires heads == 32, D % 8 == 0. Grid: threadgroups (ceil(keys / 64), T), 128 threads.
kernel void index_scores_mma(const device float* q [[buffer(0)]], const device ushort* keys [[buffer(1)]],
                             const device float* weight [[buffer(2)]], device float* score [[buffer(3)]],
                             constant IndexScoreP& p [[buffer(4)]],
                             uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float S[4][16][33];
  uint t = tg.y, j0 = (tg.x * 4 + sg) * 16;
  if (j0 >= p.keys) return;
  uint2 place = sg_place(lane);
  const device ushort* k0 = keys + ulong(min(j0 + place.x, p.keys - 1)) * p.D;
  const device ushort* k1 = keys + ulong(min(j0 + 8 + place.x, p.keys - 1)) * p.D;
  const device float* qt = q + ulong(t) * p.heads * p.D;
  simdgroup_float8x8 a0, a1, b0, b1, b2, b3;
  simdgroup_float8x8 c00 = simdgroup_float8x8(0.0f), c01 = simdgroup_float8x8(0.0f), c02 = simdgroup_float8x8(0.0f),
                     c03 = simdgroup_float8x8(0.0f), c10 = simdgroup_float8x8(0.0f), c11 = simdgroup_float8x8(0.0f),
                     c12 = simdgroup_float8x8(0.0f), c13 = simdgroup_float8x8(0.0f);
  for (uint d = 0; d < p.D; d += 8) {
    float2 v0 = bf16x2(k0, d + place.y), v1 = bf16x2(k1, d + place.y);
    a0.thread_elements()[0] = v0.x; a0.thread_elements()[1] = v0.y;
    a1.thread_elements()[0] = v1.x; a1.thread_elements()[1] = v1.y;
    simdgroup_load(b0, qt + d, p.D, 0, true);
    simdgroup_load(b1, qt + 8 * p.D + d, p.D, 0, true);
    simdgroup_load(b2, qt + 16 * p.D + d, p.D, 0, true);
    simdgroup_load(b3, qt + 24 * p.D + d, p.D, 0, true);
    simdgroup_multiply_accumulate(c00, a0, b0, c00); simdgroup_multiply_accumulate(c01, a0, b1, c01);
    simdgroup_multiply_accumulate(c02, a0, b2, c02); simdgroup_multiply_accumulate(c03, a0, b3, c03);
    simdgroup_multiply_accumulate(c10, a1, b0, c10); simdgroup_multiply_accumulate(c11, a1, b1, c11);
    simdgroup_multiply_accumulate(c12, a1, b2, c12); simdgroup_multiply_accumulate(c13, a1, b3, c13);
  }
  threadgroup float* out = &S[sg][0][0];
  simdgroup_store(c00, out, 33); simdgroup_store(c01, out + 8, 33); simdgroup_store(c02, out + 16, 33); simdgroup_store(c03, out + 24, 33);
  simdgroup_store(c10, out + 8 * 33, 33); simdgroup_store(c11, out + 8 * 33 + 8, 33);
  simdgroup_store(c12, out + 8 * 33 + 16, 33); simdgroup_store(c13, out + 8 * 33 + 24, 33);
  simdgroup_barrier(mem_flags::mem_threadgroup);
  uint j = j0 + lane;
  if (lane >= 16 || j >= p.keys) return;
  float total = -INFINITY;
  if (j < (p.pos0 + t + 1) / p.ratio) {
    total = 0;
    const device float* w = weight + t * p.heads;
    for (uint h = 0; h < 32; ++h) total += max(S[sg][lane][h], 0.0f) * (w[h] * p.scale);
  }
  score[ulong(t) * p.keys + j] = total;
}

// ---------------------------------------------------------------- attention
// Query t (position pos0 + t) reads `window` sliding-window keys, keys j = 0 .. window - 1
// at positions pos0 + t - window + 1 + j (window row = position - first), then `comp`
// selected compressed keys (selection row t; -1 masked); keys from window + comp up to K
// (a multiple of 32) are padding. Key and value are the same 512-wide vector. The
// learned sink only adds to the softmax denominator.

struct AttnP { uint T; uint pos0; uint first; uint window; uint comp; uint K; uint heads; uint D; uint sel_stride; float scale; };

static inline const device ushort* attn_key(const device ushort* window, const device ushort* comp,
                                            const device int* selection, constant AttnP& p, uint t, uint j) {
  if (j < p.window) {
    int position = int(p.pos0 + t) - int(p.window - 1) + int(j);
    if (position < 0) return nullptr;
    return window + ulong(uint(position) - p.first) * p.D;
  }
  if (j >= p.window + p.comp) return nullptr;
  int index = selection[ulong(t) * p.sel_stride + (j - p.window)];
  return index >= 0 ? comp + ulong(index) * p.D : nullptr;
}


// score[t, h, j] = scale * q[t, h] . kv[key j], -inf for masked keys. One simdgroup per
// (32 keys, 8 heads, token) computes the transposed tile S^T[key, head] = KV . Q^T.
// Grid (32, K / 32, heads / 8 * T).
kernel void attn_scores(const device float* q [[buffer(0)]], const device ushort* window [[buffer(1)]],
                        const device ushort* comp [[buffer(2)]], const device int* selection [[buffer(3)]],
                        device float* score [[buffer(4)]], constant AttnP& p [[buffer(5)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint t = g.z / (p.heads / 8), h0 = (g.z % (p.heads / 8)) * 8, j0 = g.y * 32;
  uint2 place = sg_place(lane);
  const device ushort* k0 = attn_key(window, comp, selection, p, t, j0 + place.x);
  const device ushort* k1 = attn_key(window, comp, selection, p, t, j0 + 8 + place.x);
  const device ushort* k2 = attn_key(window, comp, selection, p, t, j0 + 16 + place.x);
  const device ushort* k3 = attn_key(window, comp, selection, p, t, j0 + 24 + place.x);
  const device float* qh = q + (ulong(t) * p.heads + h0) * p.D;
  simdgroup_float8x8 a0, a1, a2, a3, b;
  simdgroup_float8x8 c0 = simdgroup_float8x8(0.0f), c1 = simdgroup_float8x8(0.0f), c2 = simdgroup_float8x8(0.0f),
                     c3 = simdgroup_float8x8(0.0f);
  for (uint d = 0; d < p.D; d += 8) {
    simdgroup_load(b, qh + d, p.D, 0, true);
    float2 v0 = bf16x2(k0, d + place.y), v1 = bf16x2(k1, d + place.y), v2 = bf16x2(k2, d + place.y), v3 = bf16x2(k3, d + place.y);
    a0.thread_elements()[0] = v0.x; a0.thread_elements()[1] = v0.y;
    a1.thread_elements()[0] = v1.x; a1.thread_elements()[1] = v1.y;
    a2.thread_elements()[0] = v2.x; a2.thread_elements()[1] = v2.y;
    a3.thread_elements()[0] = v3.x; a3.thread_elements()[1] = v3.y;
    simdgroup_multiply_accumulate(c0, a0, b, c0); simdgroup_multiply_accumulate(c1, a1, b, c1);
    simdgroup_multiply_accumulate(c2, a2, b, c2); simdgroup_multiply_accumulate(c3, a3, b, c3);
  }
  device float* out = score + (ulong(t) * p.heads + h0) * p.K + j0;
  simdgroup_store(c0, out, p.K, 0, true); simdgroup_store(c1, out + 8, p.K, 0, true);
  simdgroup_store(c2, out + 16, p.K, 0, true); simdgroup_store(c3, out + 24, p.K, 0, true);
  simdgroup_barrier(mem_flags::mem_device);
  bool masked = attn_key(window, comp, selection, p, t, j0 + lane) == nullptr;
  for (uint h = 0; h < 8; ++h) {
    device float* s = out + ulong(h) * p.K + lane;
    *s = masked ? -INFINITY : *s * p.scale;
  }
}

// In place: weights = exp(score - max) / (sum + exp(sink - max)), max including the sink.
// One simdgroup per (head, token). Grid (32, heads, T).
kernel void attn_softmax(device float* score [[buffer(0)]], const device float* sink [[buffer(1)]],
                         constant AttnP& p [[buffer(2)]],
                         uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  device float* row = score + (ulong(g.z) * p.heads + g.y) * p.K;
  float s = sink[g.y];
  float top = -INFINITY;
  for (uint j = lane; j < p.K; j += 32) top = max(top, row[j]);
  top = max(simd_max(top), s);
  float total = 0;
  for (uint j = lane; j < p.K; j += 32) {
    float e = row[j] == -INFINITY ? 0.0f : exp(row[j] - top);
    row[j] = e;
    total += e;
  }
  float inverse = 1.0f / (simd_sum(total) + exp(s - top));
  for (uint j = lane; j < p.K; j += 32) row[j] *= inverse;
}

// out[t, h, d] = sum_j weight[t, h, j] * kv[key j, d]. One simdgroup per (32 dims, 8 heads,
// token): C[head, dim] += W[head, key] . KV[key, dim]. Grid (32, D / 32, heads / 8 * T).
kernel void attn_output(const device float* weight [[buffer(0)]], const device ushort* window [[buffer(1)]],
                        const device ushort* comp [[buffer(2)]], const device int* selection [[buffer(3)]],
                        device float* out [[buffer(4)]], constant AttnP& p [[buffer(5)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint t = g.z / (p.heads / 8), h0 = (g.z % (p.heads / 8)) * 8, d = g.y * 32;
  uint2 place = sg_place(lane);
  const device float* w = weight + (ulong(t) * p.heads + h0) * p.K;
  simdgroup_float8x8 a, b0, b1, b2, b3;
  simdgroup_float8x8 c0 = simdgroup_float8x8(0.0f), c1 = simdgroup_float8x8(0.0f), c2 = simdgroup_float8x8(0.0f),
                     c3 = simdgroup_float8x8(0.0f);
  for (uint j = 0; j < p.K; j += 8) {
    simdgroup_load(a, w + j, p.K);
    const device ushort* key = attn_key(window, comp, selection, p, t, j + place.x);
    float2 v0 = bf16x2(key, d + place.y), v1 = bf16x2(key, d + 8 + place.y);
    float2 v2 = bf16x2(key, d + 16 + place.y), v3 = bf16x2(key, d + 24 + place.y);
    b0.thread_elements()[0] = v0.x; b0.thread_elements()[1] = v0.y;
    b1.thread_elements()[0] = v1.x; b1.thread_elements()[1] = v1.y;
    b2.thread_elements()[0] = v2.x; b2.thread_elements()[1] = v2.y;
    b3.thread_elements()[0] = v3.x; b3.thread_elements()[1] = v3.y;
    simdgroup_multiply_accumulate(c0, a, b0, c0); simdgroup_multiply_accumulate(c1, a, b1, c1);
    simdgroup_multiply_accumulate(c2, a, b2, c2); simdgroup_multiply_accumulate(c3, a, b3, c3);
  }
  device float* o = out + (ulong(t) * p.heads + h0) * p.D + d;
  simdgroup_store(c0, o, p.D); simdgroup_store(c1, o + 8, p.D); simdgroup_store(c2, o + 16, p.D); simdgroup_store(c3, o + 24, p.D);
}

// ---------------------------------------------------------------- MoE

static inline float clamped_swiglu(float gate, float up, float limit) {
  return silu(min(gate, limit)) * clamp(up, -limit, limit);
}

struct ActP { uint n; uint M; float limit; };

// act = silu(min(gate, limit)) * clamp(up, -limit, limit), in place over gate. Grid (n).
kernel void swiglu(device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                   constant ActP& p [[buffer(2)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  gate[g] = clamped_swiglu(gate[g], up[g], p.limit);
}

// One (token, expert) pair: input row, expert blob slot, output row.
struct Pair { uint x_row; uint slot; uint out; };
struct ExpertP { uint pairs; uint I; uint O; uint x_stride; uint w_offset; uint s_offset; uint blob; uint x_is_pair; };

struct PairActP { uint pairs; uint M; float limit; uint stride; };

// swiglu over the output rows of a list of pairs (rows `stride` values apart). Grid (M, pairs).
kernel void swiglu_pairs(device float* gate [[buffer(0)]], const device float* up [[buffer(1)]],
                         const device Pair* pairs [[buffer(2)]], constant PairActP& p [[buffer(3)]],
                         uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.M || g.y >= p.pairs) return;
  ulong at = ulong(pairs[g.y].out) * p.stride + g.x;
  gate[at] = clamped_swiglu(gate[at], up[at], p.limit);
}

// The tokens that share one expert: their input rows and output rows.
constant constexpr uint GROUP_ROWS = 8;
struct ExpertGroup { uint slot; uint count; uint in[GROUP_ROWS]; uint out[GROUP_ROWS]; };
struct GroupP { uint groups; uint I; uint O; uint x_stride; uint w_offset; uint s_offset; uint blob;
                uint O2; uint w_offset2; uint s_offset2; uint y_stride; };

// y[out_r, o] = sum_i x[in_r, i] * mxfp4(w[slot][o, i]) for the rows of a group, four outputs
// per simdgroup; the group reads its expert's weights once. Each row's sum has the same
// terms in the same order whatever the group's size. Grid (32, O / 4, groups).
// Templated on R, the most rows of any group in the dispatch (one-token decode keeps few
// registers), and K, the outputs per simdgroup. A lane takes 16 values (8 bytes) of a row
// at a time, so short rows still use every lane; each weight word is decoded once for all
// the group's rows, and every row's sum has the same order whatever R. Output rows from
// p.O on come from the second matrix (w_offset2, s_offset2): gate and up projections in
// one dispatch. y has p.y_stride values per row. Grid (32, (O + O2) / K, groups).
template <uint R, uint K>
kernel void expert_group(const device float* x [[buffer(0)]], const device uchar* blobs [[buffer(1)]],
                         const device ExpertGroup* groups [[buffer(2)]], device float* y [[buffer(3)]],
                         constant GroupP& p [[buffer(4)]],
                         uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const device ExpertGroup& group = groups[g.z];
  uint count = min(group.count, R), o0 = g.y * K, blocks = p.I / 32;
  bool second = o0 >= p.O;
  uint row0 = second ? o0 - p.O : o0;
  const device uchar* blob = blobs + ulong(group.slot) * p.blob;
  const device uint2* wr = (const device uint2*)(blob + (second ? p.w_offset2 : p.w_offset)) + ulong(row0) * blocks * 2;
  const device uchar* sr = blob + (second ? p.s_offset2 : p.s_offset) + ulong(row0) * blocks;
  const device float4* xr[R];
  for (uint r = 0; r < R; ++r) xr[r] = (const device float4*)(x + ulong(group.in[min(r, count - 1)]) * p.x_stride);
  float acc[R][K];
  for (uint r = 0; r < R; ++r)
    for (uint k = 0; k < K; ++k) acc[r][k] = 0;
  for (uint h = lane; h < blocks * 2; h += 32) {
    if (R == 1) {
      // One row: its 16 inputs are loaded once for all K outputs.
      float4 x0 = xr[0][h * 4], x1 = xr[0][h * 4 + 1], x2 = xr[0][h * 4 + 2], x3 = xr[0][h * 4 + 3];
      for (uint k = 0; k < K; ++k) {
        uint2 q = wr[ulong(k) * blocks * 2 + h];
        acc[0][k] += (dot8(fp4x8(q.x), x0, x1) + dot8(fp4x8(q.y), x2, x3)) * fp4_scale(sr[ulong(k) * blocks + h / 2]);
      }
      continue;
    }
    for (uint k = 0; k < K; ++k) {
      uint2 q = wr[ulong(k) * blocks * 2 + h];
      float scale = fp4_scale(sr[ulong(k) * blocks + h / 2]);
      Fp4x8 lo = fp4x8(q.x), hi = fp4x8(q.y);
      for (uint r = 0; r < R; ++r) {
        if (r >= count) break;
        const device float4* xh = xr[r] + h * 4;
        acc[r][k] += (dot8(lo, xh[0], xh[1]) + dot8(hi, xh[2], xh[3])) * scale;
      }
    }
  }
  for (uint r = 0; r < R; ++r) {
    if (r >= count) break;
    float t[K];
    for (uint k = 0; k < K; ++k) t[k] = simd_sum(acc[r][k]);
    if (lane == 0) {
      device float* out = y + ulong(group.out[r]) * p.y_stride + o0;
      for (uint k = 0; k < K; ++k) out[k] = t[k];
    }
  }
}
#define WD_EXPERT_GROUP(R, K) template [[host_name("expert_group" #R "_" #K)]] kernel void expert_group<R, K>( \
    const device float*, const device uchar*, const device ExpertGroup*, device float*, constant GroupP&, uint3, uint);
WD_EXPERT_GROUP(1, 4) WD_EXPERT_GROUP(2, 4) WD_EXPERT_GROUP(4, 4) WD_EXPERT_GROUP(8, 4)
WD_EXPERT_GROUP(1, 8) WD_EXPERT_GROUP(2, 8) WD_EXPERT_GROUP(4, 8) WD_EXPERT_GROUP(8, 8)
#undef WD_EXPERT_GROUP

struct CombineP { uint T; uint H; uint K; };

// y[t, i] = shared[t, i] + sum_k weight[t, k] * routed[t * K + k, i], in order of k. Grid (H, T).
kernel void moe_combine(const device float* routed [[buffer(0)]], const device float* weight [[buffer(1)]],
                        const device float* shared [[buffer(2)]], device float* y [[buffer(3)]],
                        constant CombineP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.H || g.y >= p.T) return;
  float acc = shared[ulong(g.y) * p.H + g.x];
  for (uint k = 0; k < p.K; ++k) acc += weight[g.y * p.K + k] * routed[(ulong(g.y) * p.K + k) * p.H + g.x];
  y[ulong(g.y) * p.H + g.x] = acc;
}

// ---- expert tiles (prompt reading): 16 rows of one expert per tile

struct GatherP { uint rows; uint I; uint x_stride; };

// out[r] = x[rows[r]]; 0xFFFFFFFF marks an empty row. Grid (I / 4, rows).
kernel void gather_rows(const device float* x [[buffer(0)]], const device uint* rows [[buffer(1)]],
                        device float* out [[buffer(2)]], constant GatherP& p [[buffer(3)]],
                        uint2 g [[thread_position_in_grid]]) {
  if (g.x * 4 >= p.I || g.y >= p.rows) return;
  uint row = rows[g.y];
  float4 v = row != 0xFFFFFFFFu ? ((const device float4*)(x + ulong(row) * p.x_stride))[g.x] : float4(0);
  ((device float4*)(out + ulong(g.y) * p.I))[g.x] = v;
}

// A tile's expert lives in the cache (slot < 2^31) or in the staging buffer (high bit set).
constant float FP4_VALUES[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                                 -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct ExpertSgP { uint tiles; uint I; uint O; uint w_offset; uint s_offset; uint blob; };

kernel void expert_gemm_sg(const device float* x [[buffer(0)]], const device uchar* cache [[buffer(1)]],
                           const device uchar* staging [[buffer(2)]], const device uint* slots [[buffer(3)]],
                           device float* y [[buffer(4)]], constant ExpertSgP& p [[buffer(5)]],
                           uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  uint o = g.y * 32, t = g.z * 16;
  uint2 place = sg_place(lane);
  uint groups = p.I / 32, shift = place.y * 4;
  uint slot = slots[g.z];
  const device uchar* blob = (slot & 0x80000000u) != 0 ? staging + ulong(slot & 0x7FFFFFFFu) * p.blob
                                                        : cache + ulong(slot) * p.blob;
  const device uint4* w0 = (const device uint4*)(blob + p.w_offset) + ulong(o + place.x) * groups;
  const device uint4* w1 = w0 + ulong(8) * groups;
  const device uint4* w2 = w0 + ulong(16) * groups;
  const device uint4* w3 = w0 + ulong(24) * groups;
  const device uchar* s0 = blob + p.s_offset + ulong(o + place.x) * groups;
  const device float* x0 = x + ulong(t) * p.I;
  const device float* x1 = x0 + ulong(8) * p.I;
  SG_ZERO
  for (uint group = 0; group < groups; ++group) {
    uint4 q0 = w0[group] >> shift, q1 = w1[group] >> shift, q2 = w2[group] >> shift, q3 = w3[group] >> shift;
    // A zero E8M0 byte (2^-127) is below FP32's normal range; as_type gives 0 there, as wanted.
    // The table lookup measured faster here than the bit decode (2026-10-16, M2 Max).
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
      SG_ACCUMULATE
    }
  }
  SG_STORE(y + ulong(t) * p.O + o, p.O)
}

// For each token of a wave, add its pairs' outputs in expert order:
// y[token] += weight * routed[row] for each (row, weight) of the token, in list order.
// list[token_index] = (token, first entry, entry count); entries are (row, weight bits).
struct WaveP { uint tokens; uint H; };

kernel void wave_combine(const device float* routed [[buffer(0)]], const device uint* list [[buffer(1)]],
                         const device uint* entries [[buffer(2)]], device float* y [[buffer(3)]],
                         constant WaveP& p [[buffer(4)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x * 4 >= p.H || g.y >= p.tokens) return;
  uint token = list[g.y * 3], first = list[g.y * 3 + 1], count = list[g.y * 3 + 2];
  device float4* out = (device float4*)(y + ulong(token) * p.H) + g.x;
  float4 acc = *out;
  for (uint e = 0; e < count; ++e) {
    uint row = entries[(first + e) * 2];
    float w = as_type<float>(entries[(first + e) * 2 + 1]);
    acc += w * ((const device float4*)(routed + ulong(row) * p.H))[g.x];
  }
  *out = acc;
}

// ---------------------------------------------------------------- Engram

struct EngramP { uint T; uint H; float eps; };

// gate[t, k] = sigmoid(signed_sqrt(dot)), dot = sum(h * weight * key) * rsqrt(mean(h^2) + eps)
// * rsqrt(mean(key^2) + eps) / sqrt(H), with key = kv[t, k * H ...]. Grid (32, 4, T).
kernel void engram_gate(const device float* h [[buffer(0)]], const device float* kv [[buffer(1)]],
                        const device float* weight [[buffer(2)]], device float* gate [[buffer(3)]],
                        constant EngramP& p [[buffer(4)]],
                        uint3 g [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const device float* hr = h + (ulong(g.z) * 4 + g.y) * p.H;
  const device float* kr = kv + ulong(g.z) * 5 * p.H + g.y * p.H;
  const device float* wr = weight + ulong(g.y) * p.H;
  float hh = 0, kk = 0, hk = 0;
  for (uint i = lane; i < p.H; i += 32) {
    float a = hr[i], b = kr[i];
    hh += a * a; kk += b * b; hk += a * wr[i] * b;
  }
  hh = simd_sum(hh); kk = simd_sum(kk); hk = simd_sum(hk);
  if (lane != 0) return;
  float rstd = rsqrt(hh / float(p.H) + p.eps) * rsqrt(kk / float(p.H) + p.eps);
  float d = hk * rstd * rsqrt(float(p.H));
  float m = sqrt(max(abs(d), 1e-6f));
  gate[g.z * 4 + g.y] = sigm(d < 0 ? -m : m);
}

// h[t, k, i] += gate[t, k] * value[t, i], value = kv[t, 4 * H ...]. Grid (4 * H, T).
kernel void engram_add(device float* h [[buffer(0)]], const device float* kv [[buffer(1)]],
                       const device float* gate [[buffer(2)]], constant EngramP& p [[buffer(3)]],
                       uint2 g [[thread_position_in_grid]]) {
  if (g.x >= 4 * p.H || g.y >= p.T) return;
  h[ulong(g.y) * 4 * p.H + g.x] += gate[g.y * 4 + g.x / p.H] * kv[ulong(g.y) * 5 * p.H + 4 * p.H + g.x % p.H];
}

kernel void add_in_place(device float* y [[buffer(0)]], const device float* x [[buffer(1)]],
                         constant ActP& p [[buffer(2)]], uint g [[thread_position_in_grid]]) {
  if (g >= p.n) return;
  y[g] += x[g];
}
