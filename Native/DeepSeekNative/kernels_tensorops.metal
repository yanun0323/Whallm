// TensorOps kernels (Metal 4, Metal Performance Primitives) for the Neural Accelerators
// in M5-class GPUs. Compiled separately with Metal language 4.0; the engine uses them
// only when that compile succeeds and the GPU has the accelerators (or
// WHALLM_DEEPSEEK_TENSOROPS=1). The accelerators take BF16 inputs: weights decode to BF16
// exactly (an FP8 or FP4 value times a power-of-two scale), activations round to BF16,
// and products accumulate in FP32. Outputs can therefore differ slightly from the
// simdgroup kernels, which keep FP32 activations.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using ext2 = dextents<int32_t, 2>;

// FP8 E4M3 byte as an FP16 bit pattern: the value divided by 256, exactly.
static inline float tops_fp8(uchar byte) {
  ushort b = byte;
  return float(as_type<half>(ushort(((b & 0x80) << 8) | ((b & 0x7F) << 7))));
}
// An E8M0 scale (2^(s - 127)) times 256.
static inline float tops_e8m0_256(uchar s) { return as_type<float>(uint(s + 8) << 23); }

struct TopsGemmP { uint T; uint I; uint O; uint x_stride; uint y_stride; };

// y[t, o] = sum_i x[t, i] * w[o, i] with w FP8 E4M3 [O, I] and scales s[o / 32, i / 32].
// One threadgroup of 4 simdgroups per 64 x 64 output tile; each K step of 32 stages
// 64 rows of x and 64 weight rows as BF16. Rows past T are read clamped; TensorOps
// leaves output rows past T unwritten. Requires O % 64 == 0, I % 32 == 0.
// Grid: threadgroups (O / 64, ceil(T / 64)), 128 threads.
kernel void tops_fp8(const device float* x [[buffer(0)]], const device uchar* w [[buffer(1)]],
                     const device uchar* s [[buffer(2)]], device float* y [[buffer(3)]],
                     constant TopsGemmP& p [[buffer(4)]], uint2 tid [[threadgroup_position_in_grid]],
                     ushort t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat As[64 * 32];
  threadgroup bfloat Bs[64 * 32];
  tensor<threadgroup bfloat, ext2, tensor_inline> tA(As, ext2(32, 64));
  tensor<threadgroup bfloat, ext2, tensor_inline> tB(Bs, ext2(32, 64));
  constexpr auto desc = matmul2d_descriptor(64, 64, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
  uint row0 = tid.y * 64, col0 = tid.x * 64, blocks = p.I / 32;
  ushort r = t / 2, half_k = (t % 2) * 16;  // each thread stages 16 values of one row of A and of B
  const device float* arow = x + ulong(min(row0 + r, p.T - 1)) * p.x_stride + half_k;
  const device uchar* brow = w + ulong(col0 + r) * p.I + half_k;
  const device uchar* srow = s + ulong((col0 + r) / 32) * blocks;
  for (uint b = 0; b < blocks; ++b) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup bfloat* da = As + r * 32 + half_k;
    for (ushort q = 0; q < 4; ++q) {
      float4 v = *(const device packed_float4*)(arow + b * 32 + q * 4);
      da[q * 4] = bfloat(v.x); da[q * 4 + 1] = bfloat(v.y); da[q * 4 + 2] = bfloat(v.z); da[q * 4 + 3] = bfloat(v.w);
    }
    float scale = tops_e8m0_256(srow[b]);
    threadgroup bfloat* db = Bs + r * 32 + half_k;
    const device uchar* wb = brow + b * 32;
    for (ushort k = 0; k < 16; ++k) db[k] = bfloat(tops_fp8(wb[k]) * scale);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(tA, tB, cT);
  }
  array<int32_t, 2> strides = {1, int32_t(p.y_stride)};
  tensor<device float, ext2, tensor_inline> C(y, ext2(p.O, p.T), strides);
  auto mC = C.slice(col0, row0);
  cT.store(mC);
}

constant float TOPS_FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct TopsExpertP { uint tiles; uint I; uint O; uint w_offset; uint s_offset; uint blob; };

// expert_gemm_sg on the accelerators: rows tile * 16 ... + 15 of x through the tile's
// expert, 64 outputs per threadgroup. A tile's expert is in the cache (slot < 2^31) or in
// the staging buffer (high bit set). Each 32-wide K step (one scale) widens the weights
// and rows to BF16 in threadgroup memory. Grid: threadgroups (O / 64, tiles), 128 threads.
kernel void tops_expert(const device float* x [[buffer(0)]], const device uchar* cache [[buffer(1)]],
                        const device uchar* staging [[buffer(2)]], const device uint* slots [[buffer(3)]],
                        device float* y [[buffer(4)]], constant TopsExpertP& p [[buffer(5)]],
                        uint2 tid [[threadgroup_position_in_grid]], ushort t [[thread_index_in_threadgroup]]) {
  threadgroup bfloat As[16 * 32];
  threadgroup bfloat Bs[64 * 32];
  tensor<threadgroup bfloat, ext2, tensor_inline> tA(As, ext2(32, 16));
  tensor<threadgroup bfloat, ext2, tensor_inline> tB(Bs, ext2(32, 64));
  constexpr auto desc = matmul2d_descriptor(16, 64, 32, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto cT = op.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
  for (uint16_t i = 0; i < cT.get_capacity(); ++i)
    if (cT.is_valid_element(i)) cT[i] = 0;
  uint row0 = tid.y * 16, col0 = tid.x * 64, words = p.I / 8, groups = p.I / 32;
  uint slot = slots[tid.y];
  const device uchar* blob = (slot & 0x80000000u) != 0 ? staging + ulong(slot & 0x7FFFFFFFu) * p.blob
                                                        : cache + ulong(slot) * p.blob;
  ushort br = t / 2, bw = (t % 2) * 2;  // output row, first of two words (16 values)
  const device uint* wrow = (const device uint*)(blob + p.w_offset) + ulong(col0 + br) * words + bw;
  const device uchar* srow = blob + p.s_offset + ulong(col0 + br) * groups;
  ushort ar = t / 8, ac = (t % 8) * 4;  // 16 rows x 32 values, four per thread
  const device float* arow = x + ulong(row0 + ar) * p.I + ac;
  for (uint g = 0; g < groups; ++g) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = as_type<float>(uint(srow[g]) << 23);
    uint w0 = wrow[g * 4], w1 = wrow[g * 4 + 1];
    threadgroup bfloat* db = Bs + br * 32 + bw * 8;
    for (ushort k = 0; k < 8; ++k) {
      db[k] = bfloat(TOPS_FP4[(w0 >> (4 * k)) & 15] * scale);
      db[8 + k] = bfloat(TOPS_FP4[(w1 >> (4 * k)) & 15] * scale);
    }
    float4 a = *(const device packed_float4*)(arow + g * 32);
    threadgroup bfloat* da = As + ar * 32 + ac;
    da[0] = bfloat(a.x); da[1] = bfloat(a.y); da[2] = bfloat(a.z); da[3] = bfloat(a.w);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    op.run(tA, tB, cT);
  }
  tensor<device float, ext2, tensor_inline> C(y, ext2(p.O, p.tiles * 16));
  auto mC = C.slice(col0, row0);
  cT.store(mC);
}
