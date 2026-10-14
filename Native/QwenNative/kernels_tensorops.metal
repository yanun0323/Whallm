// TensorOps kernels (Metal 4, Metal Performance Primitives) for the Neural Accelerators
// in M5-class GPUs. Compiled separately with Metal language 4.0; the engine uses them
// only when that compile succeeds and the GPU has the accelerators (or
// WHALLM_QWEN_TENSOROPS=1). Inputs are BF16, the format the accelerators take,
// as in MLX's M5 kernels; results accumulate in FP32.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
using ext2 = dextents<int32_t, 2>;

struct TopsCopyP { uint T; uint I; uint x_stride; };

// BF16 copy of [T, I] FP32 rows (round to nearest even).
kernel void to_bf16_rows(const device float* x [[buffer(0)]], device bfloat* y [[buffer(1)]],
                         constant TopsCopyP& p [[buffer(2)]], uint2 g [[thread_position_in_grid]]) {
  if (g.x >= p.I || g.y >= p.T) return;
  y[ulong(g.y) * p.I + g.x] = bfloat(x[ulong(g.y) * p.x_stride + g.x]);
}

struct TopsGemmP { uint T; uint I; uint O; uint y_stride; };

// y[t, o] = sum_i x[t, i] * w[o, i]; one threadgroup of 4 simdgroups per 64 x 64 output tile.
// TensorOps checks the matrix edges.
kernel void tops_gemm_bf16(const device bfloat* x [[buffer(0)]], const device bfloat* w [[buffer(1)]],
                           device float* y [[buffer(2)]], constant TopsGemmP& p [[buffer(3)]],
                           uint2 tid [[threadgroup_position_in_grid]]) {
  tensor<device bfloat, ext2, tensor_inline> A((device bfloat*)x, ext2(p.I, p.T));
  tensor<device bfloat, ext2, tensor_inline> B((device bfloat*)w, ext2(p.I, p.O));
  array<int32_t, 2> strides = {1, int32_t(p.y_stride)};
  tensor<device float, ext2, tensor_inline> C(y, ext2(p.O, p.T), strides);
  constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent), false, true, false);
  matmul2d<desc, execution_simdgroups<4>> op;
  auto mA = A.slice(0, tid.y * 64);
  auto mB = B.slice(0, tid.x * 64);
  auto mC = C.slice(tid.x * 64, tid.y * 64);
  op.run(mA, mB, mC);
}

constant float TOPS_FP4[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                               -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

struct TopsExpertP { uint tiles; uint I; uint O; uint w_offset; uint s_offset; uint blob; };

// Rows tile * 16 ... + 15 of x (gathered rows of one expert) through that expert's
// MXFP4 matrix, 64 outputs per threadgroup. Each 32-wide K step (one scale group)
// widens the weights and rows to BF16 in threadgroup memory for TensorOps.
kernel void tops_expert(const device float* x [[buffer(0)]], const device uchar* blobs [[buffer(1)]],
                        const device uint* slots [[buffer(2)]], device float* y [[buffer(3)]],
                        constant TopsExpertP& p [[buffer(4)]], uint2 tid [[threadgroup_position_in_grid]],
                        ushort t [[thread_index_in_threadgroup]]) {
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
  const device uchar* blob = blobs + ulong(slots[tid.y]) * p.blob;
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
