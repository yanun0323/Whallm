#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <mach/mach.h>
#include <memory>
#include <set>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <unordered_map>

#include "engine.h"

extern const char* const kWhallmDeepSeekKernels;        // kernels.metal, embedded at build time
extern const char* const kWhallmDeepSeekTensorKernels;  // kernels_tensorops.metal

namespace wd {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

// Model dimensions. open() checks them against config.json and the manifest.
constexpr int H = 5120, HC = 4, W = H * HC, LAYERS = 40, VOCAB = 129280;
constexpr int HEADS = 64, HEAD_DIM = 512, ROPE = 64, Q_RANK = 1280, O_RANK = 1024, O_GROUPS = 8, WINDOW = 128;
constexpr int O_IN = HEADS * HEAD_DIM / O_GROUPS;  // 4096 inputs per output group
constexpr int EXPERTS = 384, TOP_K = 6, MOE_M = 2304;
constexpr int IDX_HEADS = 32, IDX_DIM = 128, IDX_TOPK = 512, CAND_BLOCKS = 2048, CAND_BLOCK = 8;
constexpr int MIX = 24, SINKHORN = 20, MIX_SPLIT = 1024;  // hc_mix_partial: W values per threadgroup
constexpr float HC_EPS = 1e-6f, NORM_EPS = 1e-20f, SWIGLU_LIMIT = 10.0f, ROUTE_SCALE = 1.5f;
constexpr int ENGRAM_NGRAM = 4, ENGRAM_HEADS = 8, ENGRAM_COLS = (ENGRAM_NGRAM - 1) * ENGRAM_HEADS, ENGRAM_DIM = 256;
constexpr int ENGRAM_BLOCK = 32, ENGRAM_IN = ENGRAM_COLS * ENGRAM_DIM;
constexpr int64_t ENGRAM_VOCAB = 16000000;
constexpr int ENGRAM_COMPRESSED = 99092, ENGRAM_PAD_TOKEN = 2;
constexpr int ENGRAM_LAYER_IDS[2] = {1, 14};
// numpy.random.default_rng(10007 * layer).integers(0, bound, 4, int64) * 2 + 1 with
// bound = (2^63 - 1) // 99092 // 2, for layers 1 and 14 (deepseek_v41/engram.py).
constexpr int64_t ENGRAM_MULTIPLIERS[2][ENGRAM_NGRAM] = {
    {76632096046245, 4839876093313, 35959672319349, 73987337458391},
    {67716810739261, 51510806800915, 30921347202721, 82619226485591}};
// Expert blob: w1, w2 and w3 as MXFP4 rows with one E8M0 scale per 32 inputs.
constexpr size_t BLOB = 18800640;
constexpr uint32_t W1 = 0, W1_SCALE = 5898240, W2 = 6266880, W2_SCALE = 12165120, W3 = 12533760, W3_SCALE = 18432000;
constexpr int MAX_CONTEXT = 1048576;
// DSpark: three draft layers with 128 experts (3 per token) draft a block of 5 tokens
// from the hidden states of target layers 37, 38 and 39.
constexpr int DSPARK_LAYERS = 3, DSPARK_EXPERTS = 128, DSPARK_TOP_K = 3, DSPARK_BLOCK = 5, MARKOV_RANK = 256;
constexpr int DSPARK_NOISE = 128799;
constexpr int DSPARK_TARGETS[DSPARK_LAYERS] = {37, 38, 39};
// Tokens of one verification (the newest token and up to 7 drafts).
constexpr int VERIFY_ROWS = 8;
// Fewer rows than this use the simdgroup-per-output kernels (decode); from
// STAGED_GEMM_ROWS rows the threadgroup-staged GEMM.
constexpr int GEMM_ROWS = 9, STAGED_GEMM_ROWS = 64;
// From this many rows FP8 products and expert waves go through TensorOps when the GPU has
// Neural Accelerators (as in the native Qwen engine; unmeasured here, no M5).
constexpr int TENSOR_OPS_ROWS = 256;
// From this many tokens a call streams each layer's routed experts from SSD in waves
// of WAVE experts, reading every needed expert once for all of the call's tokens.
constexpr int PASS_ROWS = 16, WAVE = 32, STAGING = 4;
// From this many tokens a call reads each layer's first waves ahead (see prefetch).
constexpr int PREFETCH_ROWS = 512;
// Longest pass over the 40 layers. A pass holds 80 KB per token of hidden stream;
// longer calls run several passes.
constexpr int MAX_PASS = 16384;
// Fewest missing experts in one layer for which decode starts the cached ones first.
constexpr size_t OVERLAP_MIN_READS = 2;
constexpr int MIN_AUTOMATIC_SLOTS = 256, MAX_AUTOMATIC_SLOTS = 4096;
constexpr size_t READ_PIECE = 8u << 20;

const char kStateMagic[8] = {'W', 'D', 'S', 'T', 'A', 'T', 'E', '1'};
const MTLSize kSimdRows = MTLSizeMake(32, 32, 1);

struct Buf {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  Buf at(size_t bytes) const { return {buffer, offset + bytes}; }
  void* contents() const { return (char*)buffer.contents + offset; }
};

// ------------------------------------------------------------------ kernel parameter blocks (match kernels.metal)
struct LinearP { uint32_t T, I, O, x_stride, y_stride; };
struct TgSplitP { uint32_t T, I, O, x_stride, y_stride, span, splits, x_group, w_group, s_group, y_group; };
struct GemmP { uint32_t T, I, O, x_stride, y_stride; };
struct NormP { uint32_t T, groups, D, row_stride, base; float eps; uint32_t has_w; };
struct MixP { uint32_t T, W; float norm_eps, hc_eps; uint32_t iters; };
struct HcP { uint32_t T, H, mix_stride; };
struct RopeP { uint32_t T, groups, D, row_stride, base, pos0, pos_step, inverse; };
struct QuantP { uint32_t T, D, in_stride, out_stride, mode, write_f32; };
struct PoolP { uint32_t groups, g0, pos0, D; };
struct CopyRowsP { uint32_t rows, D, in_stride, out_stride; };
struct IndexScoreP { uint32_t T, keys, heads, D, pos0, ratio; float scale; };
struct AttnP { uint32_t T, pos0, first, window, comp, K, heads, D, sel_stride; float scale; };
struct ActP { uint32_t n, M; float limit; };
struct Pair { uint32_t x_row, slot, out; };
constexpr uint32_t GROUP_ROWS = 8;
struct ExpertGroup { uint32_t slot, count, in[GROUP_ROWS], out[GROUP_ROWS]; };
struct GroupP { uint32_t groups, I, O, x_stride, w_offset, s_offset, blob, O2, w_offset2, s_offset2, y_stride; };
struct PairActP { uint32_t pairs, M; float limit; uint32_t stride; };
struct CombineP { uint32_t T, H, K; };
struct GatherP { uint32_t rows, I, x_stride; };
struct ExpertSgP { uint32_t tiles, I, O, w_offset, s_offset, blob; };
struct WaveP { uint32_t tokens, H; };
struct TopsGemmP { uint32_t T, I, O, x_stride, y_stride; };
struct TopsExpertP { uint32_t tiles, I, O, w_offset, s_offset, blob; };
struct EngramP { uint32_t T, H; float eps; };
struct MeanP { uint32_t T, H, out_stride; };

// ------------------------------------------------------------------ Metal
class Gpu {
 public:
  id<MTLDevice> device;

  Gpu() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) fail("no Metal device");
    queue_ = [device newCommandQueue];
    // Every large allocation joins a residency set attached to the queue, so a commit
    // does not check all referenced memory (see the native Qwen engine). Allocations
    // made after loading go to a second set, so committing them leaves the large one alone.
    NSError* set_error = nil;
    residency_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&set_error];
    later_residency_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&set_error];
    if (residency_ == nil || later_residency_ == nil) fail("Metal residency sets are not available");
    [queue_ addResidencySet:residency_];
    [queue_ addResidencySet:later_residency_];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    // Without a fixed version Metal picks one from the SDK the host process was linked
    // against: the App's Python (SDK 14.5) got an older one than wd-cli (SDK 27).
    options.languageVersion = MTLLanguageVersion3_1;
    NSError* error = nil;
    library_ = [device newLibraryWithSource:[NSString stringWithUTF8String:kWhallmDeepSeekKernels] options:options error:&error];
    if (library_ == nil) fail(std::string("Metal kernels did not compile: ") + error.localizedDescription.UTF8String);
    pipelines_ = [NSMutableDictionary new];
    if (!probe()) fail("this GPU lays out simdgroup matrices differently than the engine expects");
    tensor_ops = open_tensor_ops();
  }

  // True when long products run through TensorOps (the M5 Neural Accelerators).
  bool tensor_ops = false;

  Buf alloc(size_t bytes) {
    id<MTLBuffer> buffer = [device newBufferWithLength:std::max<size_t>(bytes, 16) options:MTLResourceStorageModeShared];
    if (buffer == nil) fail("Metal could not allocate " + std::to_string(bytes) + " bytes");
    allocated_bytes += buffer.length;
    if (buffer.length >= kResidentBytes) {
      [settled_ ? later_residency_ : residency_ addAllocation:buffer];
      residency_dirty_ = true;
    }
    return {buffer, 0};
  }

  void release(Buf& buffer) {
    if (buffer.buffer != nil) allocated_bytes -= buffer.buffer.length;
    if (buffer.buffer != nil && buffer.buffer.length >= kResidentBytes) {
      if ([residency_ containsAllocation:buffer.buffer]) {
        [residency_ removeAllocation:buffer.buffer];
        loaded_dirty_ = true;
      } else {
        [later_residency_ removeAllocation:buffer.buffer];
      }
      residency_dirty_ = true;
    }
    buffer = {};
  }

  // Keep `buffer` at least `bytes` long; a replacement does not keep the contents.
  void ensure(Buf& buffer, size_t bytes) {
    if (buffer.buffer != nil && buffer.buffer.length >= bytes) return;
    flush();
    release(buffer);
    buffer = alloc(bytes);
  }

  void settle() {
    begin();
    flush();
    settled_ = true;
  }

  size_t allocated_bytes = 0;
  static constexpr size_t kResidentBytes = 16u << 20;

  template <typename P>
  void run(const char* name, std::initializer_list<Buf> buffers, const P& params, size_t gx, size_t gy = 1, size_t gz = 1,
           MTLSize group = MTLSizeMake(0, 0, 0)) {
    if (gx == 0 || gy == 0 || gz == 0) return;
    id<MTLComputePipelineState> pipeline = this->pipeline(name);
    begin();
    [encoder_ setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (const Buf& buffer : buffers) [encoder_ setBuffer:buffer.buffer offset:buffer.offset atIndex:index++];
    [encoder_ setBytes:&params length:sizeof(P) atIndex:index];
    if (group.width == 0) {
      NSUInteger width = pipeline.threadExecutionWidth;
      NSUInteger height = gy > 1 || gz > 1 ? std::max<NSUInteger>(1, pipeline.maxTotalThreadsPerThreadgroup / width) : 1;
      if (gz > 1) height = std::min<NSUInteger>(height, 8);
      group = MTLSizeMake(width, height, 1);
    }
    NSUInteger limit = pipeline.maxTotalThreadsPerThreadgroup;
    while (group.width * group.height * group.depth > limit && (group.height > 1 || group.depth > 1)) {
      if (group.depth > 1) group.depth = std::max<NSUInteger>(1, group.depth / 2);
      else group.height = std::max<NSUInteger>(1, group.height / 2);
    }
    if (group.width * group.height * group.depth > limit) fail(std::string("Metal kernel ") + name + " cannot run 32 threads together");
    [encoder_ dispatchThreads:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:group];
    after(name);
  }

  template <typename P>
  void run_groups(const char* name, std::initializer_list<Buf> buffers, const P& params, MTLSize groups, MTLSize group) {
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) return;
    id<MTLComputePipelineState> pipeline = this->pipeline(name);
    if (group.width * group.height * group.depth > pipeline.maxTotalThreadsPerThreadgroup)
      fail(std::string("Metal kernel ") + name + " cannot run its threadgroup size");
    begin();
    [encoder_ setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (const Buf& buffer : buffers) [encoder_ setBuffer:buffer.buffer offset:buffer.offset atIndex:index++];
    [encoder_ setBytes:&params length:sizeof(P) atIndex:index];
    [encoder_ dispatchThreadgroups:groups threadsPerThreadgroup:group];
    after(name);
  }

  // WHALLM_DEEPSEEK_KERNEL_PROFILE=1: time each kernel alone (slow; for development).
  std::string profile() const {
    std::vector<std::pair<double, std::string>> rows;
    for (const auto& item : kernel_seconds_) rows.emplace_back(item.second, item.first);
    std::sort(rows.rbegin(), rows.rend());
    std::string text;
    char line[160];
    for (const auto& row : rows) {
      snprintf(line, sizeof(line), "\n  %-20s %8.3fs %8ld calls", row.second.c_str(), row.first, kernel_calls_.at(row.second));
      text += line;
    }
    return text;
  }

  double gpu_seconds = 0, wait_seconds = 0;
  long flushes = 0, dispatches = 0;

  void begin() {
    if (encoder_ != nil) return;
    if (residency_dirty_) {
      if (!settled_ || loaded_dirty_) {
        [residency_ commit];
        [residency_ requestResidency];
        loaded_dirty_ = false;
      }
      [later_residency_ commit];
      [later_residency_ requestResidency];
      residency_dirty_ = false;
    }
    command_ = [queue_ commandBuffer];
    encoder_ = [command_ computeCommandEncoder];
  }

  // Start everything encoded so far without waiting for it. Returns its command
  // buffer (nil when nothing was encoded); the next flush() waits for all of them.
  id<MTLCommandBuffer> commit() {
    if (encoder_ == nil) return pending_.empty() ? nil : pending_.back();
    [encoder_ endEncoding];
    [command_ commit];
    pending_.push_back(command_);
    id<MTLCommandBuffer> committed = command_;
    encoder_ = nil;
    command_ = nil;
    return committed;
  }

  // Wait for one committed command buffer.
  void wait(id<MTLCommandBuffer> committed) {
    if (committed == nil) return;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    [committed waitUntilCompleted];
    wait_seconds += CFAbsoluteTimeGetCurrent() - started;
    if (committed.error != nil) fail(std::string("Metal command failed: ") + committed.error.localizedDescription.UTF8String);
  }

  // Run everything encoded so far and wait for it.
  void flush() {
    if (encoder_ == nil && pending_.empty()) return;
    if (encoder_ == nil) begin();
    [encoder_ endEncoding];
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    [command_ commit];
    [command_ waitUntilCompleted];
    wait_seconds += CFAbsoluteTimeGetCurrent() - started;
    gpu_seconds += command_.GPUEndTime - command_.GPUStartTime;
    ++flushes;
    NSError* error = command_.error;
    for (id<MTLCommandBuffer> earlier : pending_) {
      gpu_seconds += earlier.GPUEndTime - earlier.GPUStartTime;
      if (error == nil) error = earlier.error;
    }
    pending_.clear();
    encoder_ = nil;
    command_ = nil;
    if (error != nil) fail(std::string("Metal command failed: ") + error.localizedDescription.UTF8String);
  }

 private:
  void after(const char* name) {
    ++dispatches;
    if (!profile_) return;
    double before = gpu_seconds;
    flush();
    kernel_seconds_[name] += gpu_seconds - before;
    ++kernel_calls_[name];
  }

  // The GPU architecture generation, e.g. 14 for "applegpu_g14s" (M2 Max); 0 when unknown.
  int architecture_generation() const {
    const char* name = device.architecture.name.UTF8String;
    const char* at = name != nullptr ? strstr(name, "_g") : nullptr;
    return at != nullptr ? atoi(at + 2) : 0;
  }

  // TensorOps need Metal 4 (macOS 26). They are used on generation 17 (M5) and later,
  // where they reach the Neural Accelerators; WHALLM_DEEPSEEK_TENSOROPS=1 or 0 overrides.
  bool open_tensor_ops() {
    const char* force = getenv("WHALLM_DEEPSEEK_TENSOROPS");
    bool wanted = force != nullptr ? strcmp(force, "0") != 0 : architecture_generation() >= 17;
    if (!wanted) return false;
    if (@available(macOS 26.0, *)) {
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.languageVersion = MTLLanguageVersion4_0;
      NSError* error = nil;
      tensor_library_ = [device newLibraryWithSource:[NSString stringWithUTF8String:kWhallmDeepSeekTensorKernels]
                                             options:options error:&error];
      if (tensor_library_ == nil && force != nullptr)
        fprintf(stderr, "TensorOps kernels did not compile: %s\n", error.localizedDescription.UTF8String);
      return tensor_library_ != nil;
    }
    return false;
  }

  bool probe() {
    Buf out = alloc(64 * 4);
    struct { uint32_t unused; } none{0};
    run("sg_probe", {out}, none, 32, 1, 1, MTLSizeMake(32, 1, 1));
    flush();
    const float* m = (const float*)out.buffer.contents;
    bool ok = true;
    for (int lane = 0; lane < 32; ++lane) {
      int quadrant = lane / 8, local = (lane % 8) * 2;
      int row = (quadrant / 2) * 4 + local / 4, column = (quadrant % 2) * 4 + local % 4;
      if (m[row * 8 + column] != lane * 2 || m[row * 8 + column + 1] != lane * 2 + 1) ok = false;
    }
    release(out);
    return ok;
  }

  id<MTLComputePipelineState> pipeline(const char* name) {
    NSString* key = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> found = pipelines_[key];
    if (found != nil) return found;
    id<MTLFunction> function = [library_ newFunctionWithName:key];
    if (function == nil && tensor_library_ != nil) function = [tensor_library_ newFunctionWithName:key];
    if (function == nil) fail(std::string("missing Metal kernel ") + name);
    NSError* error = nil;
    found = [device newComputePipelineStateWithFunction:function error:&error];
    if (found == nil) fail(std::string("Metal kernel ") + name + ": " + error.localizedDescription.UTF8String);
    pipelines_[key] = found;
    return found;
  }

  bool profile_ = getenv("WHALLM_DEEPSEEK_KERNEL_PROFILE") != nullptr;
  std::unordered_map<std::string, double> kernel_seconds_;
  std::unordered_map<std::string, long> kernel_calls_;
  id<MTLCommandQueue> queue_;
  id<MTLResidencySet> residency_ = nil, later_residency_ = nil;
  bool residency_dirty_ = false, loaded_dirty_ = false, settled_ = false;
  id<MTLLibrary> library_, tensor_library_ = nil;
  NSMutableDictionary<NSString*, id<MTLComputePipelineState>>* pipelines_;
  id<MTLCommandBuffer> command_ = nil;
  std::vector<id<MTLCommandBuffer>> pending_;
  id<MTLComputeCommandEncoder> encoder_ = nil;
};

// ------------------------------------------------------------------ files
struct File {
  int fd = -1;
  size_t size = 0;
  File() = default;
  // `uncached`: reads bypass the page cache (the streamed expert files are far larger than memory).
  explicit File(const std::string& path, bool uncached = false) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat info;
    fstat(fd, &info);
    size = (size_t)info.st_size;
    if (uncached) fcntl(fd, F_NOCACHE, 1);
  }
  File(File&& other) noexcept : fd(other.fd), size(other.size) { other.fd = -1; }
  File& operator=(File&& other) noexcept { std::swap(fd, other.fd); std::swap(size, other.size); return *this; }
  ~File() { if (fd >= 0) ::close(fd); }
  void read(void* destination, size_t length, size_t offset) const {
    char* out = (char*)destination;
    while (length > 0) {
      ssize_t got = ::pread(fd, out, length, (off_t)offset);
      if (got <= 0) fail("short read from model file");
      out += got; offset += (size_t)got; length -= (size_t)got;
    }
  }
};

// One contiguous read.
struct ReadJob { const File* file; size_t offset, length; char* destination; };

// Run reads in pieces of READ_PIECE on the global concurrent queue; returns false on a failed read.
bool run_reads(const std::vector<ReadJob>& jobs) {
  std::vector<ReadJob> pieces;
  for (const ReadJob& job : jobs)
    for (size_t at = 0; at < job.length; at += READ_PIECE)
      pieces.push_back({job.file, job.offset + at, std::min(READ_PIECE, job.length - at), job.destination + at});
  __block bool failed = false;
  const ReadJob* items = pieces.data();
  dispatch_apply(pieces.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
    try { items[i].file->read(items[i].destination, items[i].length, items[i].offset); }
    catch (...) { failed = true; }
  });
  return !failed;
}

struct Tensor { size_t offset = 0, length = 0; std::string dtype; std::vector<int> shape; };

struct Fp8 { Buf w, s; int O = 0, I = 0; };   // FP8 E4M3 [O, I], E8M0 scales [ceil(O / 32), I / 32]
struct Bf16 { Buf w; int O = 0, I = 0; };

float bf16_value(uint16_t bits) {
  uint32_t wide = (uint32_t)bits << 16;
  float value;
  memcpy(&value, &wide, 4);
  return value;
}

// common.bin held whole in one Metal buffer; tensors are views of it.
class Weights {
 public:
  Weights(Gpu& gpu, const std::string& path, NSArray* tensors, int threads) : gpu_(gpu) {
    File file(path, true);
    data_ = gpu.alloc(file.size);
    if (data_.buffer.length < file.size) fail("Metal buffer is smaller than " + path);
    std::vector<ReadJob> jobs{{&file, 0, file.size, (char*)data_.buffer.contents}};
    if (!run_reads(jobs)) fail("could not read " + path);
    for (NSDictionary* item in tensors) {
      Tensor tensor;
      tensor.offset = [item[@"offset"] unsignedLongLongValue];
      tensor.length = [item[@"length"] unsignedLongLongValue];
      tensor.dtype = [item[@"dtype"] UTF8String];
      for (NSNumber* dimension in item[@"shape"]) tensor.shape.push_back(dimension.intValue);
      if (tensor.offset + tensor.length > file.size || tensor.offset % 16 != 0) fail("tensor outside or misaligned in " + path);
      tensors_[[item[@"name"] UTF8String]] = tensor;
    }
  }
  ~Weights() {
    gpu_.release(data_);
    gpu_.release(vectors_);
  }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  const Tensor& tensor(const std::string& name, const char* dtype, std::initializer_list<int> shape) const {
    auto found = tensors_.find(name);
    if (found == tensors_.end()) fail("model has no tensor " + name);
    if (found->second.dtype != dtype || found->second.shape != std::vector<int>(shape)) fail("unexpected shape or type for " + name);
    return found->second;
  }
  Fp8 fp8(const std::string& name, int O, int I) const {
    const Tensor& w = tensor(name + ".weight", "F8_E4M3", {O, I});
    const Tensor& s = tensor(name + ".scale", "F8_E8M0", {(O + 31) / 32, I / 32});
    if (O % 32 != 0 || I % 32 != 0) fail("unsupported FP8 matrix shape for " + name);
    return {data_.at(w.offset), data_.at(s.offset), O, I};
  }
  Bf16 bf16(const std::string& name, int O, int I) const {
    return {data_.at(tensor(name, "BF16", {O, I}).offset), O, I};
  }
  Buf f32(const std::string& name, std::initializer_list<int> shape) const { return data_.at(tensor(name, "F32", shape).offset); }
  const void* bytes(const Tensor& t) const { return (const char*)data_.buffer.contents + t.offset; }
  // A BF16 vector widened to FP32; returns its float index in vectors().
  size_t vector(const std::string& name, int elements) {
    const uint16_t* source = (const uint16_t*)bytes(tensor(name, "BF16", {elements}));
    size_t start = floats_.size();
    for (int i = 0; i < elements; ++i) floats_.push_back(bf16_value(source[i]));
    return start;
  }
  size_t push(const std::vector<float>& values) {
    size_t start = floats_.size();
    floats_.insert(floats_.end(), values.begin(), values.end());
    return start;
  }
  void seal() {
    vectors_ = gpu_.alloc(floats_.size() * 4);
    memcpy(vectors_.buffer.contents, floats_.data(), floats_.size() * 4);
  }
  Buf vec(size_t index) const { return vectors_.at(index * 4); }

 private:
  Gpu& gpu_;
  Buf data_, vectors_;
  std::vector<float> floats_;
  std::unordered_map<std::string, Tensor> tensors_;
};

// ------------------------------------------------------------------ expert cache
// Keeps expert blobs in one Metal buffer; the least used (oldest among equals) leave first.
class ExpertArena {
 public:
  ExpertArena(Gpu& gpu, int slots) : slots_(slots) {
    if (slots < PASS_ROWS * TOP_K) fail("the expert cache needs at least " + std::to_string(PASS_ROWS * TOP_K) + " slots");
    buffer_ = gpu.alloc((size_t)slots * BLOB);
    owner_.assign(slots, -1);
    stamp_.assign(slots, 0);
    uses_.assign(slots, 0);
  }
  // Returns the file's index. The experts of a `kept` file leave the cache only when no
  // other expert can (DSpark's: every draft uses them, while a verification brings in
  // hundreds of experts that would otherwise push them out).
  int add_file(const std::string& path, int experts = EXPERTS, bool kept = false) {
    cached_.emplace_back(path);
    uncached_.emplace_back(path, true);
    kept_.push_back(kept);
    if (cached_.back().size != (size_t)experts * BLOB) fail("unexpected expert file size: " + path);
    // WHALLM_DEEPSEEK_PAGE_CACHE_PROBE=1 (development): map the file to ask the OS which
    // expert bytes are in its page cache before each read.
    void* map = nullptr;
    if (getenv("WHALLM_DEEPSEEK_PAGE_CACHE_PROBE") != nullptr) {
      map = mmap(nullptr, cached_.back().size, PROT_READ, MAP_SHARED, cached_.back().fd, 0);
      if (map == MAP_FAILED) map = nullptr;
    }
    maps_.push_back(map);
    return (int)cached_.size() - 1;
  }
  Buf buffer() const { return buffer_; }
  char* slot_memory(int slot) const { return (char*)buffer_.buffer.contents + (size_t)slot * BLOB; }
  const File& file(int index, bool uncached) const { return uncached ? uncached_[index] : cached_[index]; }

  // The slot holding this expert, or -1.
  int find(int file, int expert) const {
    auto found = where_.find((long)file * EXPERTS + expert);
    return found == where_.end() ? -1 : found->second;
  }

  // Start a step: slots handed out from now on stay reserved until the next begin().
  void begin() { ++clock_; }

  // The slot of this expert; a missing one gets a slot and `missing` is set (fill it with load()).
  int acquire(int file, int expert, bool& missing, long uses = 1) {
    long key = (long)file * EXPERTS + expert;
    touch();
    auto found = where_.find(key);
    if (found != where_.end()) {
      stamp_[found->second] = clock_;
      uses_[found->second] += uses;
      ++hits;
      missing = false;
      return found->second;
    }
    int slot = victim();
    if (owner_[slot] >= 0) where_.erase(owner_[slot]);
    owner_[slot] = key;
    where_[key] = slot;
    stamp_[slot] = clock_;
    uses_[slot] = uses;
    missing = true;
    return slot;
  }

  // Read (expert, slot) pairs of one file into their slots.
  void load(int file, const std::vector<std::pair<int, int>>& missing) {
    if (missing.empty()) return;
    if (maps_[file] != nullptr) {
      size_t page = (size_t)getpagesize(), pages = (BLOB + page - 1) / page;
      std::vector<char> resident(pages + 1);
      for (const auto& item : missing) {
        size_t at = (size_t)item.first * BLOB, first = at / page * page;
        mincore((char*)maps_[file] + first, at + BLOB - first, resident.data());
        for (size_t i = 0; i < (at + BLOB - first + page - 1) / page; ++i) page_cache_bytes += (resident[i] & 1) ? page : 0;
      }
    }
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    std::vector<ReadJob> jobs;
    for (const auto& item : missing)
      jobs.push_back({&cached_[file], (size_t)item.first * BLOB, BLOB, slot_memory(item.second)});
    bool ok = run_reads(jobs);
    read_seconds += CFAbsoluteTimeGetCurrent() - started;
    misses += (long)missing.size();
    if (!ok) fail("could not read an expert from SSD");
  }

  ~ExpertArena() {
    for (size_t i = 0; i < maps_.size(); ++i)
      if (maps_[i] != nullptr) munmap(maps_[i], cached_[i].size);
  }

  int slots() const { return slots_; }
  int resident() const { return used_; }
  int files() const { return (int)cached_.size(); }

  long hits = 0, misses = 0, seeded = 0, streamed = 0;
  double read_seconds = 0, page_cache_bytes = 0;

 private:
  int victim() {
    if (used_ < slots_) return used_++;
    int best = -1;
    bool best_kept = true;
    for (int slot = 0; slot < slots_; ++slot) {
      if (stamp_[slot] == clock_) continue;
      bool kept = kept_[owner_[slot] / EXPERTS];
      if (best < 0 || (best_kept && !kept) ||
          (kept == best_kept && (uses_[slot] < uses_[best] || (uses_[slot] == uses_[best] && stamp_[slot] < stamp_[best])))) {
        best = slot;
        best_kept = kept;
      }
    }
    if (best < 0) fail("the expert cache is too small for one step");
    return best;
  }
  // Use counts are halved now and then, so experts that stopped being used can leave.
  void touch() {
    if (++touches_ < std::max<long>((long)slots_ * 8, 64)) return;
    touches_ = 0;
    for (long& uses : uses_) uses = std::max<long>(1, uses / 2);
  }

  int slots_, used_ = 0;
  long clock_ = 0, touches_ = 0;
  Buf buffer_;
  std::vector<File> cached_, uncached_;
  std::vector<bool> kept_;
  std::vector<void*> maps_;
  std::vector<long> owner_, stamp_, uses_;
  std::unordered_map<long, int> where_;
};

// ------------------------------------------------------------------ Engram tables
bool is_prime(int64_t n) {
  if (n < 2) return false;
  if (n % 2 == 0) return n == 2;
  for (int64_t i = 3; i * i <= n; i += 2)
    if (n % i == 0) return false;
  return true;
}

struct EngramTable {
  File weight, scale;
  int64_t rows = 0;
  int64_t primes[ENGRAM_COLS], offsets[ENGRAM_COLS];
  const int64_t* multipliers = nullptr;
};

// ------------------------------------------------------------------ per-layer weights and state
struct LayerW {
  Weights* w = nullptr;                 // the file holding this layer's vectors
  int file = 0;                         // expert file in the arena
  int experts = EXPERTS, top_k = TOP_K;
  int ratio = 0;
  bool kv_source = false, index_source = false, owns_keys = false, candidate_source = false, uses_candidates = false;
  int kv_from = -1;    // layer whose compressed cache this layer reads
  int keys_from = -1;  // index sources: layer whose index keys they score
  int engram = -1;     // Engram table, or -1
  Buf attn_fn, attn_scale, attn_base, ffn_fn, ffn_scale, ffn_base, sink;
  size_t attn_norm = 0, ffn_norm = 0, q_norm = 0, kv_norm = 0, comp_norm = 0, idx_k_norm = 0, engram_weight = 0;
  Fp8 wq_a, wq_b, wkv, wo_a, wo_b, sh_w1, sh_w2, sh_w3, idx_wq_b, engram_wkv;
  Bf16 gate, comp_wkv, comp_wgate, idx_weights, idx_wk;
  std::vector<float> gate_bias;
};

struct LayerState {
  Buf ring;         // the last min(position, WINDOW) window KV rows (BF16), oldest first
  Buf comp, keys;   // kv sources: compressed KV (BF16 [capacity, 512]); key owners: index keys (BF16 [capacity, 128])
  Buf carry;        // ratio-2 sources: the unpaired row's kv and score (FP32 [2, 512])
  int capacity = 0; // rows of comp and keys
};

// ------------------------------------------------------------------ engine
class EngineImpl : public Engine {
 public:
  EngineImpl(const std::string& root, const Options& options) : options_(options) {
    if (options_.chunk < 1) options_.chunk = 1024;
    if (options_.read_threads < 1) options_.read_threads = 8;
    NSDictionary* manifest = json(root + "/manifest.json");
    NSDictionary* config = json(root + "/config.json")[@"text_config"];
    check(config, manifest);
    if (options_.token_map.size() != (size_t)VOCAB) fail("the Engram token map must have one entry per vocabulary id");
    for (int32_t id : options_.token_map)
      if (id < 0 || id >= ENGRAM_COMPRESSED) fail("the Engram token map holds an id outside the compressed vocabulary");
    contract_ = std::string("wd-state-1|") + [manifest[@"modelID"] UTF8String] + "@" + [manifest[@"revision"] UTF8String] +
                "|dspark=" + (options_.dspark ? "1" : "0");

    common_.reset(new Weights(gpu_, root + "/common.bin", manifest[@"commonTensors"], options_.read_threads));
    NSArray* ratios = config[@"compress_ratios"];
    std::set<int> kv_sources, index_sources;
    for (NSNumber* layer in config[@"kv_source_layer_ids"]) kv_sources.insert(layer.intValue);
    for (NSNumber* layer in config[@"index_source_layer_ids"]) index_sources.insert(layer.intValue);
    int candidate = [config[@"candidate_source_layer_id"] intValue];
    int last_kv = -1, last_keys = -1;
    for (int l = 0; l < LAYERS; ++l) {
      LayerW lw;
      std::string p = "layers." + std::to_string(l) + ".";
      lw.file = l;
      lw.ratio = [ratios[l] intValue];
      lw.kv_source = kv_sources.count(l) != 0;
      lw.index_source = index_sources.count(l) != 0;
      lw.owns_keys = lw.kv_source && lw.index_source;
      lw.candidate_source = l == candidate;
      lw.uses_candidates = candidate >= 0 && candidate < l;
      if (lw.kv_source) last_kv = l;
      if (lw.owns_keys) last_keys = l;
      if (lw.ratio > 0) {
        if (last_kv < 0 || [ratios[last_kv] intValue] != lw.ratio) fail("unsupported model: compressed cache sharing");
        if (!lw.index_source && !index_source_before(index_sources, l)) fail("unsupported model: index sharing");
        lw.kv_from = last_kv;
        if (lw.index_source) {
          if (last_keys < 0 || [ratios[last_keys] intValue] != lw.ratio) fail("unsupported model: index key sharing");
          lw.keys_from = last_keys;
        }
      } else if (lw.kv_source || lw.index_source) {
        fail("unsupported model: an uncompressed layer owns a compressed cache");
      }
      load_block(*common_, p, lw);
      std::string a = p + "attn.";
      if (lw.kv_source) {
        lw.comp_norm = common_->vector(a + "compressor.norm.weight", HEAD_DIM);
        lw.comp_wkv = common_->bf16(a + "compressor.wkv.weight", HEAD_DIM, H);
        if (lw.ratio == 2) lw.comp_wgate = common_->bf16(a + "compressor.wgate.weight", HEAD_DIM, H);
        else if (lw.ratio != 1) fail("unsupported model: compress ratio");
      }
      if (lw.index_source) {
        lw.idx_wq_b = common_->fp8(a + "indexer.wq_b", IDX_HEADS * IDX_DIM, Q_RANK);
        lw.idx_weights = common_->bf16(a + "indexer.weights_proj.weight", IDX_HEADS, H);
        if (lw.owns_keys) {
          lw.idx_wk = common_->bf16(a + "indexer.wk.weight", IDX_DIM, HEAD_DIM);
          lw.idx_k_norm = common_->vector(a + "indexer.k_norm.weight", IDX_DIM);
        }
      }
      for (int e = 0; e < 2; ++e) {
        if (l != ENGRAM_LAYER_IDS[e]) continue;
        lw.engram = e;
        lw.engram_wkv = common_->fp8(p + "engram.wkv", H * (HC + 1), ENGRAM_IN);
        const uint16_t* q = (const uint16_t*)common_->bytes(common_->tensor(p + "engram.q_weight", "BF16", {HC, H}));
        const uint16_t* k = (const uint16_t*)common_->bytes(common_->tensor(p + "engram.k_weight", "BF16", {HC, H}));
        std::vector<float> product(HC * H);
        for (int i = 0; i < HC * H; ++i) product[i] = bf16_value(q[i]) * bf16_value(k[i]);
        lw.engram_weight = common_->push(product);
      }
      layers_.push_back(lw);
    }
    embed_ = (const uint16_t*)common_->bytes(common_->tensor("embed.weight", "BF16", {VOCAB, H}));
    head_ = common_->bf16("head.weight", VOCAB, H);
    final_norm_ = common_->vector("norm.weight", H);
    // Rotary frequencies: plain (theta 10,000) for the window-only layers, YaRN on
    // theta 160,000 for the compressing layers; as deepseek_v41/layers.py computes them.
    rope_ = common_->push(frequencies([config[@"rope_theta"] doubleValue], 0, 0, 0, 0));
    NSDictionary* yarn = config[@"rope_scaling"];
    yarn_ = common_->push(frequencies([config[@"compress_rope_theta"] doubleValue], [yarn[@"original_max_position_embeddings"] intValue],
                                      [yarn[@"factor"] doubleValue], [yarn[@"beta_fast"] doubleValue], [yarn[@"beta_slow"] doubleValue]));
    common_->seal();
    open_engram(root, config, manifest);
    if (options_.dspark) open_dspark(root, manifest);

    slots_ = options_.slots > 0 ? options_.slots : automatic_slots(gpu_.allocated_bytes);
    arena_.reset(new ExpertArena(gpu_, slots_));
    for (int l = 0; l < LAYERS; ++l) {
      char name[64];
      snprintf(name, sizeof(name), "/experts/layer_%02d.bin", l);
      arena_->add_file(root + name);
    }
    // DSpark's experts share the cache and stay in it (see ExpertArena::add_file).
    for (int stage = 0; stage < (int)stages_.size(); ++stage) {
      char name[64];
      snprintf(name, sizeof(name), "/dspark/experts/layer_%02d.bin", stage);
      stages_[stage].file = arena_->add_file(root + name, DSPARK_EXPERTS, true);
    }
    for (int l = 0; l < LAYERS; ++l) {
      LayerState s;
      s.ring = gpu_.alloc((size_t)WINDOW * HEAD_DIM * 2);
      if (layers_[l].kv_source && layers_[l].ratio == 2) s.carry = gpu_.alloc((size_t)2 * HEAD_DIM * 4);
      state_.push_back(s);
    }
    logits_ = gpu_.alloc((size_t)8 * VOCAB * 4);
    route_ = gpu_.alloc((size_t)PASS_ROWS * TOP_K * 4);
    pairs_ = gpu_.alloc((size_t)PASS_ROWS * TOP_K * sizeof(Pair));
    groups_ = gpu_.alloc((size_t)PASS_ROWS * TOP_K * sizeof(ExpertGroup));
    down_groups_ = gpu_.alloc((size_t)PASS_ROWS * TOP_K * sizeof(ExpertGroup));
    for (int l = 0; l < LAYERS; ++l) {
      VerifyLog log;
      log.window = gpu_.alloc((size_t)(WINDOW + VERIFY_ROWS) * HEAD_DIM * 2);
      if (state_[l].carry.buffer != nil) {
        log.ckv = gpu_.alloc((size_t)VERIFY_ROWS * HEAD_DIM * 4);
        log.cscore = gpu_.alloc((size_t)VERIFY_ROWS * HEAD_DIM * 4);
      }
      vlog_.push_back(log);
    }
    reset();
    gpu_.settle();
  }

  ~EngineImpl() override {
    wait_engram_reads();
    if (reads_ != nil) dispatch_group_wait(reads_, DISPATCH_TIME_FOREVER);
  }

  void reset() override {
    gpu_.flush();
    for (LayerState& s : state_)
      if (s.carry.buffer != nil) memset(s.carry.contents(), 0, s.carry.buffer.length);
    position_ = 0;
    tokens_.clear();
    context_rows_ = 0;
    drafted_ = verifying_ = false;
  }

  int position() const override { return position_; }
  bool has_dspark() const override { return dspark_ != nullptr; }
  int draft_block() const override { return DSPARK_BLOCK; }
  int vocab() const override { return VOCAB; }
  int max_context() const override { return MAX_CONTEXT; }
  void set_cancel(std::function<bool()> check) override { cancel_ = std::move(check); }
  const std::string& contract() const override { return contract_; }
  const std::vector<int32_t>& tokens() const override { return tokens_; }

  Counters counters() const override {
    Counters c;
    c.expert_hits = arena_->hits;
    c.expert_misses = arena_->misses;
    c.experts_seeded = arena_->seeded;
    c.experts_streamed = arena_->streamed;
    c.read_seconds = arena_->read_seconds;
    c.gpu_seconds = gpu_.gpu_seconds;
    c.gpu_bytes = gpu_.allocated_bytes;
    c.slots = slots_;
    c.resident = arena_->resident();
    return c;
  }

  std::string stats() const override {
    char text[400];
    snprintf(text, sizeof(text),
             "slots %d %swait %.3fs gpu %.3fs flushes %ld dispatches %ld expert hits %ld misses %ld streamed %ld seeded %ld "
             "read %.3fs engram %.3fs route %.3fs select %.3fs page-cached %.2fGB",
             slots_, gpu_.tensor_ops ? "tensorops " : "", gpu_.wait_seconds, gpu_.gpu_seconds, gpu_.flushes, gpu_.dispatches, arena_->hits, arena_->misses,
             arena_->streamed, arena_->seeded, arena_->read_seconds, engram_seconds_, route_seconds_, select_seconds_,
             arena_->page_cache_bytes / 1e9);
    return text + gpu_.profile();
  }

  const float* eval(const int32_t* tokens, int count, int logit_rows) override {
    if (verifying_) fail("commit the open verification first");
    if (count < 1 || logit_rows < 0 || logit_rows > std::min(count, 8)) fail("invalid eval request");
    for (int i = 0; i < count; ++i)
      if (tokens[i] < 0 || tokens[i] >= VOCAB) fail("token outside the vocabulary");
    if ((long)position_ + count > MAX_CONTEXT) fail("the context exceeds " + std::to_string(MAX_CONTEXT) + " tokens");
    int passes = (count + MAX_PASS - 1) / MAX_PASS, size = (count + passes - 1) / passes;
    drafted_ = false;
    for (int at = 0; at < count; at += size) {
      int n = std::min(size, count - at);
      pass(tokens + at, n, at + n == count ? logit_rows : 0);
    }
    return (const float*)logits_.buffer.contents;
  }

  // ------------------------------------------------------------ DSpark
  void draft(int32_t anchor) override {
    if (dspark_ == nullptr) fail("this model was opened without DSpark");
    if (verifying_) fail("commit the open verification first");
    if (position_ < 1) fail("DSpark drafts after a consumed prompt");
    if (anchor < 0 || anchor >= VOCAB) fail("token outside the vocabulary");
    gpu_.flush();
    settle_reads();
    const int T = DSPARK_BLOCK;
    reserve(T, T);
    float* hidden = (float*)s_.all.contents();
    float* pre = (float*)s_.pre_mix.contents();
    for (int t = 0; t < T; ++t) {
      embed(t == 0 ? anchor : DSPARK_NOISE, hidden + (size_t)t * W);
      float* m = pre + (size_t)t * MIX;
      m[0] = 1; m[1] = m[2] = m[3] = 0;
    }
    for (const LayerW& lw : stages_) {
      check_cancel();
      draft_attention(lw, T);
      feed_forward(lw, T);
      std::swap(s_.pre_mix, s_.ffn_mix);
    }
    hc_pre(s_.all, s_.pre_mix, draft_hidden_, T);
    norm(draft_hidden_, dspark_->vec(draft_norm_), s_.x, T, 1, H, H);
    linear(s_.x, head_, draft_logits_, T);
    gpu_.flush();
    drafted_ = true;
  }

  const float* draft_step(int index, int32_t previous, float* confidence) override {
    if (!drafted_) fail("draft a block first");
    if (index < 0 || index >= DSPARK_BLOCK) fail("draft position outside the block");
    if (previous < 0 || previous >= VOCAB) fail("token outside the vocabulary");
    float* embedding = (float*)markov_embedding_.contents();
    const uint16_t* row = markov_embed_ + (size_t)previous * MARKOV_RANK;
    for (int i = 0; i < MARKOV_RANK; ++i) embedding[i] = bf16_value(row[i]);
    linear(markov_embedding_, markov_head_, draft_row_, 1);
    gpu_.run("add_in_place", {draft_row_, draft_logits_.at((size_t)index * VOCAB * 4)}, ActP{(uint32_t)VOCAB, 0, 0}, VOCAB);
    gpu_.flush();
    if (confidence != nullptr) {
      // sigmoid(w . [head hidden, Markov embedding])
      const float* hidden = (const float*)draft_hidden_.contents() + (size_t)index * H;
      float total = 0;
      for (int i = 0; i < H; ++i) total += bf16_value(confidence_[i]) * hidden[i];
      for (int i = 0; i < MARKOV_RANK; ++i) total += bf16_value(confidence_[H + i]) * embedding[i];
      *confidence = 1.0f / (1.0f + std::exp(-total));
    }
    return (const float*)draft_row_.contents();
  }

  const float* verify(const int32_t* tokens, int count) override {
    if (verifying_) fail("commit the open verification first");
    if (count < 1 || count > VERIFY_ROWS) fail("verify takes 1 to 8 tokens");
    for (int i = 0; i < count; ++i)
      if (tokens[i] < 0 || tokens[i] >= VOCAB) fail("token outside the vocabulary");
    if ((long)position_ + count > MAX_CONTEXT) fail("the context exceeds " + std::to_string(MAX_CONTEXT) + " tokens");
    drafted_ = false;
    verify_position_ = position_;
    verify_count_ = count;
    verifying_ = true;
    try {
      pass(tokens, count, count);
    } catch (...) {
      verifying_ = false;
      throw;
    }
    return (const float*)logits_.buffer.contents;
  }

  // Keep the first `keep` verified tokens. The window rows and the compressors' unpaired
  // rows go back to what they were after those tokens; compressed rows past them are
  // rewritten before anything reads them.
  void commit(int keep) override {
    if (!verifying_ || keep < 1 || keep > verify_count_) fail("commit does not match the verification");
    verifying_ = false;
    gpu_.flush();
    int p0 = verify_position_;
    if (keep < verify_count_) {
      for (int l = 0; l < LAYERS; ++l) {
        const VerifyLog& log = vlog_[l];
        LayerState& st = state_[l];
        int rows = log.kept + keep, n = std::min(rows, WINDOW);
        copy_u16(log.window.at((size_t)(rows - n) * HEAD_DIM * 2), st.ring, n, HEAD_DIM);
        if (st.carry.buffer != nil && (p0 + keep) % 2 != 0) {
          CopyRowsP one{1, HEAD_DIM, HEAD_DIM, HEAD_DIM};
          gpu_.run("copy_rows", {log.ckv.at((size_t)(keep - 1) * HEAD_DIM * 4), st.carry}, one, HEAD_DIM, 1);
          gpu_.run("copy_rows", {log.cscore.at((size_t)(keep - 1) * HEAD_DIM * 4), st.carry.at(HEAD_DIM * 4)}, one, HEAD_DIM, 1);
        }
      }
      position_ = p0 + keep;
      tokens_.resize(position_);
    }
    if (dspark_ != nullptr) seed_context(keep, p0);
    gpu_.flush();
  }

  std::shared_ptr<const State> snapshot() override {
    if (verifying_) fail("commit the open verification before taking a snapshot");
    gpu_.flush();
    auto state = std::make_shared<State>();
    state->tokens = tokens_;
    std::vector<uint8_t>& out = state->data;
    auto put = [&](const void* data, size_t bytes) { out.insert(out.end(), (const uint8_t*)data, (const uint8_t*)data + bytes); };
    int64_t position = position_;
    put(&position, sizeof(position));
    for (int l = 0; l < LAYERS; ++l) {
      const LayerW& lw = layers_[l];
      const LayerState& s = state_[l];
      put(s.ring.contents(), (size_t)std::min(position_, WINDOW) * HEAD_DIM * 2);
      if (!lw.kv_source) continue;
      size_t rows = (size_t)position_ / lw.ratio;
      if (rows > 0) put(s.comp.contents(), rows * HEAD_DIM * 2);
      if (lw.owns_keys && rows > 0) put(s.keys.contents(), rows * IDX_DIM * 2);
      if (s.carry.buffer != nil) put(s.carry.contents(), (size_t)2 * HEAD_DIM * 4);
    }
    if (dspark_ != nullptr) {
      int64_t rows = context_rows_;
      put(&rows, sizeof(rows));
      for (const Buf& context : context_) put(context.contents(), (size_t)context_rows_ * HEAD_DIM * 2);
    }
    return state;
  }

  void restore(const State& state) override {
    if (verifying_) fail("commit the open verification before restoring a snapshot");
    drafted_ = false;
    gpu_.flush();
    const uint8_t* at = state.data.data();
    const uint8_t* end = at + state.data.size();
    auto take = [&](void* data, size_t bytes) {
      if ((size_t)(end - at) < bytes) fail("the prompt cache state is incomplete");
      memcpy(data, at, bytes);
      at += bytes;
    };
    int64_t position;
    take(&position, sizeof(position));
    if (position != (int64_t)state.tokens.size() || position > MAX_CONTEXT) fail("the prompt cache state does not match its tokens");
    position_ = 0;  // stays invalid until every layer is restored
    tokens_.clear();
    for (int l = 0; l < LAYERS; ++l) {
      const LayerW& lw = layers_[l];
      LayerState& s = state_[l];
      take(s.ring.contents(), (size_t)std::min<int64_t>(position, WINDOW) * HEAD_DIM * 2);
      if (!lw.kv_source) continue;
      size_t rows = (size_t)position / lw.ratio;
      grow(l, (int)rows);
      if (rows > 0) take(s.comp.contents(), rows * HEAD_DIM * 2);
      if (lw.owns_keys && rows > 0) take(s.keys.contents(), rows * IDX_DIM * 2);
      if (s.carry.buffer != nil) take(s.carry.contents(), (size_t)2 * HEAD_DIM * 4);
    }
    context_rows_ = 0;
    if (dspark_ != nullptr) {
      int64_t rows;
      take(&rows, sizeof(rows));
      if (rows < 0 || rows > std::min<int64_t>(position, WINDOW)) fail("the prompt cache state is invalid");
      for (Buf& context : context_) take(context.contents(), (size_t)rows * HEAD_DIM * 2);
      context_rows_ = (int)rows;
    }
    if (at != end) fail("the prompt cache state has unexpected data");
    position_ = (int)position;
    tokens_ = state.tokens;
  }

  void save(const State& state, const std::string& path) const override {
    std::string temporary = path + ".partial";
    FILE* file = fopen(temporary.c_str(), "wb");
    if (file == nullptr) fail("cannot write " + temporary);
    uint64_t sizes[3] = {contract_.size(), state.tokens.size(), state.data.size()};
    bool written = fwrite(kStateMagic, 1, 8, file) == 8 && fwrite(sizes, 8, 3, file) == 3 &&
                   fwrite(contract_.data(), 1, contract_.size(), file) == contract_.size() &&
                   fwrite(state.tokens.data(), 4, state.tokens.size(), file) == state.tokens.size() &&
                   fwrite(state.data.data(), 1, state.data.size(), file) == state.data.size();
    written = fclose(file) == 0 && written;
    if (!written || rename(temporary.c_str(), path.c_str()) != 0) {
      unlink(temporary.c_str());
      fail("cannot write " + path);
    }
  }

  std::shared_ptr<const State> load(const std::string& path) const override {
    File file(path);
    char magic[8];
    uint64_t sizes[3];
    if (file.size < 32) fail(path + " is not a prompt cache file");
    file.read(magic, 8, 0);
    file.read(sizes, 24, 8);
    if (memcmp(magic, kStateMagic, 8) != 0 || sizes[0] > 4096 || sizes[1] > (uint64_t)MAX_CONTEXT ||
        32 + sizes[0] + sizes[1] * 4 + sizes[2] != file.size)
      fail(path + " is not a prompt cache file");
    std::string contract(sizes[0], '\0');
    file.read(contract.data(), sizes[0], 32);
    if (contract != contract_) fail(path + " was written for another model or engine format");
    auto state = std::make_shared<State>();
    state->tokens.resize(sizes[1]);
    state->data.resize(sizes[2]);
    file.read(state->tokens.data(), sizes[1] * 4, 32 + sizes[0]);
    file.read(state->data.data(), sizes[2], 32 + sizes[0] + sizes[1] * 4);
    return state;
  }

 private:
  // -------------------------------------------------------------- loading helpers
  // What every block has: hyper-connection mixing, attention without compression, and the
  // MoE with its shared expert. lw.experts must be set.
  static void load_block(Weights& w, const std::string& p, LayerW& lw) {
    lw.w = &w;
    lw.attn_fn = w.f32(p + "hc_attn_fn", {MIX, W});
    lw.attn_scale = w.f32(p + "hc_attn_scale", {3});
    lw.attn_base = w.f32(p + "hc_attn_base", {MIX});
    lw.ffn_fn = w.f32(p + "hc_ffn_fn", {MIX, W});
    lw.ffn_scale = w.f32(p + "hc_ffn_scale", {3});
    lw.ffn_base = w.f32(p + "hc_ffn_base", {MIX});
    lw.attn_norm = w.vector(p + "attn_norm.weight", H);
    lw.ffn_norm = w.vector(p + "ffn_norm.weight", H);
    std::string a = p + "attn.";
    lw.sink = w.f32(a + "attn_sink", {HEADS});
    lw.q_norm = w.vector(a + "q_norm.weight", Q_RANK);
    lw.kv_norm = w.vector(a + "kv_norm.weight", HEAD_DIM);
    lw.wq_a = w.fp8(a + "wq_a", Q_RANK, H);
    lw.wq_b = w.fp8(a + "wq_b", HEADS * HEAD_DIM, Q_RANK);
    lw.wkv = w.fp8(a + "wkv", HEAD_DIM, H);
    lw.wo_a = w.fp8(a + "wo_a", O_GROUPS * O_RANK, O_IN);
    lw.wo_b = w.fp8(a + "wo_b", H, O_GROUPS * O_RANK);
    std::string f = p + "ffn.";
    lw.gate = w.bf16(f + "gate.weight", lw.experts, H);
    const float* bias = (const float*)w.bytes(w.tensor(f + "gate.bias", "F32", {lw.experts}));
    lw.gate_bias.assign(bias, bias + lw.experts);
    lw.sh_w1 = w.fp8(f + "shared_experts.w1", MOE_M, H);
    lw.sh_w2 = w.fp8(f + "shared_experts.w2", H, MOE_M);
    lw.sh_w3 = w.fp8(f + "shared_experts.w3", MOE_M, H);
  }

  static NSDictionary* json(const std::string& path) {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (data == nil) fail("cannot read " + path);
    NSDictionary* parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![parsed isKindOfClass:[NSDictionary class]]) fail("invalid JSON in " + path);
    return parsed;
  }

  static bool index_source_before(const std::set<int>& sources, int layer) {
    for (int s : sources)
      if (s < layer) return true;
    return false;
  }

  static void check(NSDictionary* config, NSDictionary* manifest) {
    if (![config isKindOfClass:[NSDictionary class]]) fail("unsupported model: config.json has no text_config");
    auto same = [&](NSString* key, double expected) {
      if (config[key] == nil || (float)[config[key] doubleValue] != (float)expected) fail(std::string("unsupported model: ") + key.UTF8String);
    };
    same(@"hidden_size", H); same(@"num_hidden_layers", LAYERS); same(@"vocab_size", VOCAB);
    same(@"num_attention_heads", HEADS); same(@"head_dim", HEAD_DIM); same(@"qk_rope_head_dim", ROPE);
    same(@"q_lora_rank", Q_RANK); same(@"o_lora_rank", O_RANK); same(@"o_groups", O_GROUPS); same(@"sliding_window", WINDOW);
    same(@"n_routed_experts", EXPERTS); same(@"num_experts_per_tok", TOP_K); same(@"moe_intermediate_size", MOE_M);
    same(@"n_shared_experts", 1); same(@"routed_scaling_factor", ROUTE_SCALE); same(@"swiglu_limit", SWIGLU_LIMIT);
    same(@"index_n_heads", IDX_HEADS); same(@"index_head_dim", IDX_DIM); same(@"index_topk", IDX_TOPK);
    same(@"candidate_topk_blocks", CAND_BLOCKS); same(@"candidate_block_size", CAND_BLOCK);
    same(@"hc_mult", HC); same(@"hc_sinkhorn_iters", SINKHORN); same(@"hc_eps", HC_EPS); same(@"rms_norm_eps", NORM_EPS);
    same(@"engram_max_ngram_size", ENGRAM_NGRAM); same(@"engram_n_heads", ENGRAM_HEADS); same(@"engram_head_dim", ENGRAM_DIM);
    same(@"engram_vocab_size", (double)ENGRAM_VOCAB); same(@"engram_compressed_vocab_size", ENGRAM_COMPRESSED);
    same(@"engram_pad_token_id", ENGRAM_PAD_TOKEN);
    if (![config[@"scoring_func"] isEqualToString:@"sqrtsoftplus"] || ![config[@"norm_topk_prob"] boolValue])
      fail("unsupported model: expert scoring");
    if ([config[@"compress_ratios"] count] < (NSUInteger)LAYERS) fail("unsupported model: compress_ratios");
    NSArray* engram = config[@"engram_layer_ids"];
    if (engram.count != 2 || [engram[0] intValue] != ENGRAM_LAYER_IDS[0] || [engram[1] intValue] != ENGRAM_LAYER_IDS[1])
      fail("unsupported model: engram_layer_ids");
    if (![manifest[@"modelKind"] isEqualToString:@"deepseek-v4.1"] || [manifest[@"expertBlobSize"] unsignedLongLongValue] != BLOB ||
        [manifest[@"expertCount"] intValue] != EXPERTS || [manifest[@"layerCount"] intValue] != LAYERS)
      fail("unsupported model: expert blob layout");
    const uint32_t offsets[6] = {W1, W1_SCALE, W2, W2_SCALE, W3, W3_SCALE};
    const char* names[6] = {"w1.weight", "w1.scale", "w2.weight", "w2.scale", "w3.weight", "w3.scale"};
    NSArray* regions = manifest[@"expertRegions"];
    if (regions.count != 6) fail("unsupported model: expert regions");
    for (int i = 0; i < 6; ++i)
      if (![regions[i][@"name"] isEqualToString:[NSString stringWithUTF8String:names[i]]] || [regions[i][@"offset"] unsignedIntValue] != offsets[i])
        fail("unsupported model: expert regions");
  }

  // deepseek_v41/layers.py precompute_freqs_cis, in FP32 as MLX evaluates it.
  static std::vector<float> frequencies(double base, int original, double factor, double beta_fast, double beta_slow) {
    std::vector<float> out(ROPE / 2);
    for (int i = 0; i < ROPE / 2; ++i) out[i] = 1.0f / std::pow((float)base, (float)(2 * i) / (float)ROPE);
    if (original > 0) {
      auto corrected = [&](double rotations) { return ROPE * std::log(original / (rotations * 2 * M_PI)) / (2 * std::log(base)); };
      double low = std::max(std::floor(corrected(beta_fast)), 0.0), high = std::min(std::ceil(corrected(beta_slow)), (double)ROPE - 1);
      for (int i = 0; i < ROPE / 2; ++i) {
        float ramp = std::min(1.0f, std::max(0.0f, ((float)i - (float)low) / (float)std::max(high - low, 1e-3)));
        float smooth = 1.0f - ramp;
        out[i] = out[i] / (float)factor * (1.0f - smooth) + out[i] * smooth;
      }
    }
    return out;
  }

  void open_engram(const std::string& root, NSDictionary* config, NSDictionary* manifest) {
    NSArray* sizes = config[@"engram_num_embeddings"];
    NSArray* tables = manifest[@"engram"][@"tables"];
    if (sizes.count != 2 || tables.count != 2) fail("unsupported model: Engram tables");
    std::set<int64_t> seen;
    for (int e = 0; e < 2; ++e) {
      EngramTable& table = engram_[e];
      NSDictionary* item = tables[e];
      if ([item[@"layer"] intValue] != ENGRAM_LAYER_IDS[e] || [item[@"dimension"] intValue] != ENGRAM_DIM ||
          [item[@"blockSize"] intValue] != ENGRAM_BLOCK)
        fail("unsupported model: Engram table layout");
      table.rows = [item[@"rows"] longLongValue];
      table.weight = File(root + "/" + [item[@"weightFile"] UTF8String]);
      table.scale = File(root + "/" + [item[@"scaleFile"] UTF8String]);
      if (table.weight.size != (size_t)table.rows * ENGRAM_DIM || table.scale.size != (size_t)table.rows * (ENGRAM_DIM / ENGRAM_BLOCK))
        fail("Engram table sizes do not match the manifest");
      // Prime bucket sizes drawn in order, never reused (deepseek_v41/engram.py build_layout).
      int64_t total = 0;
      for (int n = 0; n < ENGRAM_NGRAM - 1; ++n) {
        int64_t current = ENGRAM_VOCAB - 1;
        for (int head = 0; head < ENGRAM_HEADS; ++head) {
          do ++current; while (!is_prime(current) || seen.count(current) != 0);
          seen.insert(current);
          table.offsets[n * ENGRAM_HEADS + head] = total;
          table.primes[n * ENGRAM_HEADS + head] = current;
          total += current;
        }
      }
      if (total != [sizes[e] longLongValue] || total != table.rows) fail("Engram bucket layout does not match the model");
      table.multipliers = ENGRAM_MULTIPLIERS[e];
    }
  }

  // Expert cache size when Options::slots is 0: the engine's footprint stays under 60% of
  // the RAM, and at least 7 GB or 15% of the RAM (whichever is more) of the memory macOS
  // can give without swapping stays free (as the native Qwen engine sizes it).
  static int automatic_slots(size_t resident) {
    uint64_t ram = 0;
    size_t size = sizeof(ram);
    sysctlbyname("hw.memsize", &ram, &size, nullptr, 0);
    vm_statistics64_data_t vm;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    uint64_t available = 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&vm, &count) == KERN_SUCCESS)
      available = ((uint64_t)vm.free_count + vm.inactive_count + vm.purgeable_count + vm.speculative_count) * vm_page_size;
    const double gib = 1024.0 * 1024.0 * 1024.0;
    // Staging for streamed experts, the pass buffers and the attention scratch.
    double later = (double)(STAGING * WAVE) * BLOB + 4.0 * gib;
    double keep = std::max(7.0 * gib, 0.15 * (double)ram);
    double budget = std::min(0.6 * (double)ram - (double)resident - later, (double)available - keep - later);
    int slots = (int)std::max(0.0, budget / (double)BLOB);
    return std::max(MIN_AUTOMATIC_SLOTS, std::min(MAX_AUTOMATIC_SLOTS, slots));
  }

  void check_cancel() {
    if (cancel_ && cancel_()) throw Cancelled();
  }

  // -------------------------------------------------------------- building blocks
  // The instance of a few-row product kernel for T rows: 1, up to 4 or up to 8 rows per
  // block. Each row's sum has the same order in all three (see kernels.metal).
  static const char* rows_kernel(const char* base, int T) {
    static std::unordered_map<std::string, std::string> names;
    std::string key = std::string(base) + (T == 1 ? "_1" : T > 4 ? "_8" : "");
    auto found = names.find(key);
    if (found == names.end()) found = names.emplace(key, key).first;
    return found->second.c_str();
  }
  void linear(Buf x, const Fp8& m, Buf y, int T, int x_stride = 0, int y_stride = 0) {
    uint32_t I = m.I, O = m.O, xs = x_stride ? x_stride : I, ys = y_stride ? y_stride : O;
    if (T < GEMM_ROWS) {
      if (O < 2048 && I >= 1024) {
        int splits = std::min<int>(I / 512, (2048 + O - 1) / O);
        if (splits > 1) {
          uint32_t span = (I / splits + 31) / 32 * 32;
          splits = (I + span - 1) / span;
          TgSplitP sp{(uint32_t)T, I, O, xs, ys, span, (uint32_t)splits, 0, 0, 0, 0};
          gpu_.run_groups(rows_kernel("linear_fp8_tgsplit", T), {x, m.w, m.s, y}, sp,
                          MTLSizeMake(O / 2, 1, 1), MTLSizeMake(32 * splits, 1, 1));
          return;
        }
      }
      gpu_.run(rows_kernel("linear_fp8", T), {x, m.w, m.s, y}, LinearP{(uint32_t)T, I, O, xs, ys}, 32, O / 2, 1, kSimdRows);
      return;
    }
    GemmP gp{(uint32_t)T, I, O, xs, ys};
    if (gpu_.tensor_ops && T >= TENSOR_OPS_ROWS && O % 64 == 0 && I % 32 == 0) {
      gpu_.run_groups("tops_fp8", {x, m.w, m.s, y}, TopsGemmP{(uint32_t)T, I, O, xs, ys},
                      MTLSizeMake(O / 64, (T + 63) / 64, 1), MTLSizeMake(128, 1, 1));
      return;
    }
    if (T >= STAGED_GEMM_ROWS && O % 64 == 0 && I % 16 == 0) {
      gpu_.run_groups("gemm_fp8_tg", {x, m.w, m.s, y}, gp, MTLSizeMake(O / 64, (T + 63) / 64, 1), MTLSizeMake(128, 1, 1));
      return;
    }
    gpu_.run("gemm_fp8_sg", {x, m.w, m.s, y}, gp, 32, O / 32, (T + 15) / 16, MTLSizeMake(32, 4, 1));
  }

  void linear(Buf x, const Bf16& m, Buf y, int T, int x_stride = 0, int y_stride = 0) {
    uint32_t I = m.I, O = m.O, xs = x_stride ? x_stride : I, ys = y_stride ? y_stride : O;
    if (T < GEMM_ROWS) {
      if (O < 2048 && I >= 1024) {
        int splits = std::min<int>(I / 512, (2048 + O - 1) / O);
        if (splits > 1) {
          uint32_t span = (I / splits + 3) / 4 * 4;
          splits = (I + span - 1) / span;
          TgSplitP sp{(uint32_t)T, I, O, xs, ys, span, (uint32_t)splits, 0, 0, 0, 0};
          gpu_.run_groups(rows_kernel("linear_bf16_tgsplit", T), {x, m.w, y}, sp,
                          MTLSizeMake(O / 2, 1, 1), MTLSizeMake(32 * splits, 1, 1));
          return;
        }
      }
      gpu_.run(rows_kernel("linear_bf16", T), {x, m.w, y}, LinearP{(uint32_t)T, I, O, xs, ys}, 32, O / 2, 1, kSimdRows);
      return;
    }
    GemmP gp{(uint32_t)T, I, O, xs, ys};
    if (T >= STAGED_GEMM_ROWS && O % 64 == 0 && I % 16 == 0) {
      gpu_.run_groups("gemm_bf16_tg", {x, m.w, y}, gp, MTLSizeMake(O / 64, (T + 63) / 64, 1), MTLSizeMake(128, 1, 1));
      return;
    }
    gpu_.run("gemm_bf16_sg", {x, m.w, y}, gp, 32, O / 32, (T + 15) / 16, MTLSizeMake(32, 4, 1));
  }

  void norm(Buf x, Buf weight, Buf y, int T, int groups, int D, int row_stride, int base = 0) {
    NormP p{(uint32_t)T, (uint32_t)groups, (uint32_t)D, (uint32_t)row_stride, (uint32_t)base, NORM_EPS, 1};
    if (D >= 1024) {
      gpu_.run_groups("rms_norm_tg", {x, weight, y}, p, MTLSizeMake(groups, T, 1), MTLSizeMake(256, 1, 1));
      return;
    }
    gpu_.run("rms_norm", {x, weight, y}, p, 32, groups, T, kSimdRows);
  }

  void rope(Buf x, int T, int groups, int D, int row_stride, int pos0, int step, bool inverse, bool yarn) {
    RopeP p{(uint32_t)T, (uint32_t)groups, (uint32_t)D, (uint32_t)row_stride, 0, (uint32_t)pos0, (uint32_t)step, inverse ? 1u : 0u};
    gpu_.run("rope_tail", {x, common_->vec(yarn ? yarn_ : rope_)}, p, 32, groups, T);
  }

  // mode 0: FP8 window KV; 1: FP4 indexer rows; 2: FP4 compressed KV (see kernels.metal).
  void fake_quant(Buf x, Buf out, int T, int D, int in_stride, int out_stride, int mode, bool write_f32) {
    int block = mode == 2 ? 16 : 32;
    QuantP p{(uint32_t)T, (uint32_t)D, (uint32_t)in_stride, (uint32_t)out_stride, (uint32_t)mode, write_f32 ? 1u : 0u};
    gpu_.run("fake_quant", {x, out}, p, D / block, T);
  }

  void mixes(Buf h, Buf fn, Buf scale, Buf base, Buf mix, int T) {
    MixP p{(uint32_t)T, W, NORM_EPS, HC_EPS, SINKHORN};
    gpu_.run_groups("hc_mix_partial", {h, fn, s_.mix_part}, p, MTLSizeMake(W / MIX_SPLIT, T, 1), MTLSizeMake(256, 1, 1));
    gpu_.run_groups("hc_mix_finish", {s_.mix_part, scale, base, mix}, p, MTLSizeMake(T, 1, 1), MTLSizeMake(32, 1, 1));
  }

  void hc_pre(Buf h, Buf pre, Buf x, int T) { gpu_.run("hc_pre", {h, pre, x}, HcP{(uint32_t)T, H, MIX}, H / 4, T); }
  void hc_post(Buf h, Buf y, Buf mix, int T) { gpu_.run("hc_post", {h, y, mix}, HcP{(uint32_t)T, H, MIX}, H / 4, T); }

  void copy_u16(Buf from, Buf to, int rows, int D) {
    gpu_.run("copy_rows_u16", {from, to}, CopyRowsP{(uint32_t)rows, (uint32_t)D, (uint32_t)D, (uint32_t)D}, D, rows);
  }

  // Room for `rows` compressed rows in a kv source's caches.
  void grow(int layer, int rows) {
    LayerState& s = state_[layer];
    if (rows <= s.capacity) return;
    gpu_.flush();
    int capacity = std::max(4096, s.capacity * 2);
    while (capacity < rows) capacity *= 2;
    auto resize = [&](Buf& buffer, size_t row_bytes) {
      Buf next = gpu_.alloc((size_t)capacity * row_bytes);
      if (buffer.buffer != nil) memcpy(next.contents(), buffer.contents(), (size_t)s.capacity * row_bytes);
      gpu_.release(buffer);
      buffer = next;
    };
    resize(s.comp, (size_t)HEAD_DIM * 2);
    if (layers_[layer].owns_keys) resize(s.keys, (size_t)IDX_DIM * 2);
    s.capacity = capacity;
  }

  // -------------------------------------------------------------- scratch
  struct Scratch {
    int rows = 0, chunk = 0;
    // per pass: [rows, ...]
    Buf all, x, result, pre_mix, attn_mix, ffn_mix, selection, gate_logits, shared_gate, shared_up, mix_part;
    // per chunk: [chunk, ...]
    Buf qr, q, kv, o, oa, res, ckv, cscore, latent, ik, iq, iq_bf16, iw, scores, window, erows, ekv, egate;
    // decode experts: [PASS_ROWS * TOP_K, ...]
    Buf rgate, routed;
    Buf iscore; size_t iscore_bytes = 0;
  } s_;

  static size_t pad16(int rows) { return ((size_t)rows + 15) / 16 * 16; }

  void reserve(int count, int chunk) {
    if (count > s_.rows) {
      gpu_.flush();
      for (Buf* b : {&s_.all, &s_.x, &s_.result, &s_.pre_mix, &s_.attn_mix, &s_.ffn_mix, &s_.selection, &s_.gate_logits,
                     &s_.shared_gate, &s_.shared_up, &s_.mix_part})
        gpu_.release(*b);
      size_t n = pad16(count);
      auto f = [&](size_t per_row) { return gpu_.alloc(n * per_row * 4); };
      s_.all = f(W); s_.x = f(H); s_.result = f(H);
      s_.pre_mix = f(MIX); s_.attn_mix = f(MIX); s_.ffn_mix = f(MIX);
      s_.selection = f(IDX_TOPK); s_.gate_logits = f(EXPERTS); s_.shared_gate = f(MOE_M); s_.shared_up = f(MOE_M);
      s_.mix_part = f((size_t)(W / MIX_SPLIT) * 25);
      s_.rows = count;
    }
    if (chunk > s_.chunk) {
      gpu_.flush();
      for (Buf* b : {&s_.qr, &s_.q, &s_.kv, &s_.o, &s_.oa, &s_.res, &s_.ckv, &s_.cscore, &s_.latent, &s_.ik, &s_.iq,
                     &s_.iq_bf16, &s_.iw, &s_.scores, &s_.window, &s_.erows, &s_.ekv, &s_.egate})
        gpu_.release(*b);
      size_t n = pad16(chunk);
      auto f = [&](size_t per_row) { return gpu_.alloc(n * per_row * 4); };
      s_.qr = f(Q_RANK); s_.q = f(HEADS * HEAD_DIM); s_.kv = f(HEAD_DIM); s_.o = f(HEADS * HEAD_DIM);
      s_.oa = f(O_GROUPS * O_RANK); s_.res = f(H); s_.ckv = f(HEAD_DIM); s_.cscore = f(HEAD_DIM); s_.latent = f(HEAD_DIM);
      s_.ik = f(IDX_DIM); s_.iq = f(IDX_HEADS * IDX_DIM); s_.iq_bf16 = f(IDX_HEADS * IDX_DIM / 2); s_.iw = f(IDX_HEADS);
      s_.scores = f((size_t)HEADS * (WINDOW + IDX_TOPK));
      s_.window = gpu_.alloc(((size_t)WINDOW + n) * HEAD_DIM * 2);
      s_.erows = f(ENGRAM_IN); s_.ekv = f((size_t)H * (HC + 1)); s_.egate = f(HC);
      s_.chunk = chunk;
    }
    if (s_.rgate.buffer == nil) {
      size_t n = (size_t)PASS_ROWS * TOP_K;
      s_.rgate = gpu_.alloc(n * 2 * MOE_M * 4);  // [gate | up] per pair
      s_.routed = gpu_.alloc(n * H * 4);
    }
  }

  // -------------------------------------------------------------- one pass over the 40 layers
  void pass(const int32_t* tokens, int count, int logit_rows) {
    gpu_.flush();
    settle_reads();  // reads left by a failed call
    // A verification runs as one chunk, so its window log covers all its rows.
    int chunk = verifying_ ? count : std::min(count, options_.chunk);
    reserve(count, chunk);
    pos0_ = position_;
    for (int l = 0; l < LAYERS; ++l)
      if (layers_[l].kv_source) grow(l, (pos0_ + count) / layers_[l].ratio);
    // Compressed Engram ids of positions pos0 - 3 ... pos0 + count - 1 (pad before position 0).
    cids_.assign((size_t)count + 3, options_.token_map[ENGRAM_PAD_TOKEN]);
    for (int i = 0; i < count + 3; ++i) {
      int position = pos0_ - 3 + i;
      if (position < 0) continue;
      int32_t token = position < pos0_ ? tokens_[position] : tokens[position - pos0_];
      cids_[i] = options_.token_map[token];
    }
    candidates_.assign(count, {});
    start_engram_reads(count);
    float* hidden = (float*)s_.all.contents();
    float* pre = (float*)s_.pre_mix.contents();
    for (int t = 0; t < count; ++t) {
      embed(tokens[t], hidden + (size_t)t * W);
      float* m = pre + (size_t)t * MIX;
      m[0] = 1; m[1] = m[2] = m[3] = 0;
    }
    if (count >= PREFETCH_ROWS) prefetch(0);
    for (int l = 0; l < LAYERS; ++l) {
      check_cancel();
      const LayerW& lw = layers_[l];
      if (lw.engram >= 0) engram(l, count);
      // DSpark reads the stream entering target layers 37 to 39, averaged over the copies,
      // for the rows its window can still see (all rows of a verification).
      for (int c = 0; c < DSPARK_LAYERS && dspark_ != nullptr; ++c) {
        if (l != DSPARK_TARGETS[c]) continue;
        int rows = verifying_ ? count : std::min(count, WINDOW);
        gpu_.run("hc_mean", {s_.all.at((size_t)(count - rows) * W * 4), capture_.at((size_t)c * H * 4)},
                 MeanP{(uint32_t)rows, H, (uint32_t)(DSPARK_LAYERS * H)}, H / 4, rows);
      }
      for (int start = 0; start < count; start += chunk) attention(l, start, std::min(chunk, count - start));
      feed_forward(lw, count);
      std::swap(s_.pre_mix, s_.ffn_mix);
      if (!options_.dump_dir.empty() && pos0_ == 0) dump(l, count);
    }
    if (logit_rows > 0) {
      int first = count - logit_rows;
      hc_pre(s_.all.at((size_t)first * W * 4), s_.pre_mix.at((size_t)first * MIX * 4), s_.x, logit_rows);
      norm(s_.x, common_->vec(final_norm_), s_.x, logit_rows, 1, H, H);
      linear(s_.x, head_, logits_, logit_rows);
    }
    if (dspark_ != nullptr && !verifying_) {
      int rows = std::min(count, WINDOW);
      seed_context(rows, pos0_ + count - rows);
    }
    gpu_.flush();
    position_ += count;
    tokens_.insert(tokens_.end(), tokens, tokens + count);
  }

  void embed(int32_t token, float* out) const {
    const uint16_t* row = embed_ + (size_t)token * H;
    for (int i = 0; i < H; ++i) out[i] = bf16_value(row[i]);
    for (int copy = 1; copy < HC; ++copy) memcpy(out + (size_t)copy * H, out, (size_t)H * 4);
  }

  void dump(int layer, int count) {
    gpu_.flush();
    char name[32];
    snprintf(name, sizeof(name), "/layer_%02d.f32", layer);
    FILE* file = fopen((options_.dump_dir + name).c_str(), "wb");
    if (file == nullptr) fail("cannot write to the dump directory");
    fwrite(s_.all.contents(), 4, (size_t)count * W, file);
    fclose(file);
  }

  // Bytes of one token's rows in one table: 24 rows of 256 FP8 values and 8 scales.
  static constexpr size_t ENGRAM_TOKEN_BYTES = (size_t)ENGRAM_COLS * (ENGRAM_DIM + ENGRAM_DIM / ENGRAM_BLOCK);

  // Start reading both tables' rows for the pass's tokens in the background. The rows
  // depend only on the token ids (cids_), so the reads run while the layers before each
  // Engram layer compute.
  void start_engram_reads(int count) {
    wait_engram_reads();
    engram_failed_ = false;
    auto cids = std::make_shared<std::vector<int32_t>>(cids_);
    for (int e = 0; e < 2; ++e) {
      if (engram_raw_[e].size() < (size_t)count * ENGRAM_TOKEN_BYTES) engram_raw_[e].resize((size_t)count * ENGRAM_TOKEN_BYTES);
      uint8_t* raw = engram_raw_[e].data();
      const EngramTable* tb = &engram_[e];
      std::atomic<bool>* failed = &engram_failed_;
      if (engram_group_[e] == nil) engram_group_[e] = dispatch_group_create();
      dispatch_group_async(engram_group_[e], dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        const int32_t* c = cids->data() + 3;  // c[t - s]: the token s places before t
        dispatch_apply((size_t)count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
          int64_t product[ENGRAM_NGRAM];
          for (int s = 0; s < ENGRAM_NGRAM; ++s) product[s] = (int64_t)c[(int)t - s] * tb->multipliers[s];
          int64_t rolling = product[0];
          uint8_t* out = raw + t * ENGRAM_TOKEN_BYTES;
          for (int n = 1; n < ENGRAM_NGRAM; ++n) {
            rolling ^= product[n];
            for (int head = 0; head < ENGRAM_HEADS; ++head) {
              int col = (n - 1) * ENGRAM_HEADS + head;
              int64_t row = rolling % tb->primes[col] + tb->offsets[col];
              uint8_t* dst = out + (size_t)col * (ENGRAM_DIM + ENGRAM_DIM / ENGRAM_BLOCK);
              try {
                tb->weight.read(dst, ENGRAM_DIM, (size_t)row * ENGRAM_DIM);
                tb->scale.read(dst + ENGRAM_DIM, ENGRAM_DIM / ENGRAM_BLOCK, (size_t)row * (ENGRAM_DIM / ENGRAM_BLOCK));
              } catch (...) {
                *failed = true;
                return;
              }
            }
          }
        });
      });
    }
  }

  // No Engram read outstanding afterwards (pass start, errors, exit).
  void wait_engram_reads() {
    for (dispatch_group_t group : engram_group_)
      if (group != nil) dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
  }

  // -------------------------------------------------------------- Engram: adds the gated N-gram rows to the stream
  void engram(int layer, int count) {
    const LayerW& lw = layers_[layer];
    int chunk = s_.chunk;
    CFAbsoluteTime waited = CFAbsoluteTimeGetCurrent();
    dispatch_group_wait(engram_group_[lw.engram], DISPATCH_TIME_FOREVER);
    engram_seconds_ += CFAbsoluteTimeGetCurrent() - waited;
    if (engram_failed_) fail("could not read Engram rows");
    float table_fp8[256];
    for (int b = 0; b < 256; ++b) {
      int e = (b >> 3) & 15, m = b & 7;
      float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
      table_fp8[b] = (b & 0x80) ? -v : v;
    }
    const uint8_t* raw = engram_raw_[lw.engram].data();
    for (int start = 0; start < count; start += chunk) {
      int T = std::min(chunk, count - start);
      gpu_.flush();  // the previous chunk's product may still read the rows
      CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
      float* rows = (float*)s_.erows.contents();
      const uint8_t* src = raw + (size_t)start * ENGRAM_TOKEN_BYTES;
      const float* lut = table_fp8;
      dispatch_apply((size_t)T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        for (int col = 0; col < ENGRAM_COLS; ++col) {
          const uint8_t* weight = src + t * ENGRAM_TOKEN_BYTES + (size_t)col * (ENGRAM_DIM + ENGRAM_DIM / ENGRAM_BLOCK);
          const uint8_t* scale = weight + ENGRAM_DIM;
          float* out = rows + t * ENGRAM_IN + (size_t)col * ENGRAM_DIM;
          for (int i = 0; i < ENGRAM_DIM; ++i) out[i] = lut[weight[i]] * std::ldexp(1.0f, (int)scale[i / ENGRAM_BLOCK] - 127);
        }
      });
      engram_seconds_ += CFAbsoluteTimeGetCurrent() - started;
      Buf h = s_.all.at((size_t)start * W * 4);
      linear(s_.erows, lw.engram_wkv, s_.ekv, T);
      EngramP p{(uint32_t)T, H, NORM_EPS};
      gpu_.run("engram_gate", {h, s_.ekv, common_->vec(lw.engram_weight), s_.egate}, p, 32, HC, T, MTLSizeMake(32, HC, 1));
      gpu_.run("engram_add", {h, s_.ekv, s_.egate}, p, W, T);
    }
  }

  // -------------------------------------------------------------- attention over one chunk: stream rows += attention
  void attention(int layer, int start, int T) {
    const LayerW& lw = layers_[layer];
    LayerState& st = state_[layer];
    bool yarn = lw.ratio > 0;
    int p0 = pos0_ + start;
    Buf h = s_.all.at((size_t)start * W * 4), mix = s_.attn_mix.at((size_t)start * MIX * 4);
    Buf x = s_.x.at((size_t)start * H * 4);
    mixes(h, lw.attn_fn, lw.attn_scale, lw.attn_base, mix, T);
    hc_pre(h, s_.pre_mix.at((size_t)start * MIX * 4), x, T);
    norm(x, lw.w->vec(lw.attn_norm), x, T, 1, H, H);
    linear(x, lw.wq_a, s_.qr, T);
    norm(s_.qr, lw.w->vec(lw.q_norm), s_.qr, T, 1, Q_RANK, Q_RANK);
    linear(s_.qr, lw.wq_b, s_.q, T);
    rope(s_.q, T, HEADS, HEAD_DIM, HEADS * HEAD_DIM, p0, 1, false, yarn);
    linear(x, lw.wkv, s_.kv, T);
    norm(s_.kv, lw.w->vec(lw.kv_norm), s_.kv, T, 1, HEAD_DIM, HEAD_DIM);
    rope(s_.kv, T, 1, HEAD_DIM, HEAD_DIM, p0, 1, false, yarn);
    // The window: the kept rows before this chunk, then the chunk's rows rounded through FP8.
    int kept = std::min(p0, WINDOW);
    if (kept > 0) copy_u16(st.ring, s_.window, kept, HEAD_DIM);
    fake_quant(s_.kv, s_.window.at((size_t)kept * HEAD_DIM * 2), T, HEAD_DIM, HEAD_DIM, HEAD_DIM, 0, false);
    if (verifying_) {
      copy_u16(s_.window, vlog_[layer].window, kept + T, HEAD_DIM);
      vlog_[layer].kept = kept;
    }
    int comp = 0;
    if (lw.ratio > 0) {
      if (lw.kv_source) compress(layer, x, T, p0);
      if (lw.index_source) select(layer, x, start, T, p0);
      comp = std::min(IDX_TOPK, (p0 + T) / lw.ratio);
    }
    Buf comp_kv = comp > 0 ? state_[lw.kv_from].comp : s_.window;
    uint32_t keys = (WINDOW + comp + 31) / 32 * 32;
    AttnP ap{(uint32_t)T, (uint32_t)p0, (uint32_t)(p0 - kept), WINDOW, (uint32_t)comp, keys, HEADS, HEAD_DIM,
             IDX_TOPK, 1.0f / std::sqrt((float)HEAD_DIM)};
    Buf selection = s_.selection.at((size_t)start * IDX_TOPK * 4);
    MTLSize tiles = MTLSizeMake(32, 4, 1);
    gpu_.run("attn_scores", {s_.q, s_.window, comp_kv, selection, s_.scores}, ap, 32, keys / 32, (size_t)HEADS / 8 * T, tiles);
    gpu_.run("attn_softmax", {s_.scores, lw.sink}, ap, 32, HEADS, T, kSimdRows);
    gpu_.run("attn_output", {s_.scores, s_.window, comp_kv, selection, s_.o}, ap, 32, HEAD_DIM / 32, (size_t)HEADS / 8 * T, tiles);
    rope(s_.o, T, HEADS, HEAD_DIM, HEADS * HEAD_DIM, p0, 1, true, yarn);
    output_projection(lw, T);
    hc_post(h, s_.res, mix, T);
    int rows = kept + T, keep = std::min(rows, WINDOW);
    copy_u16(s_.window.at((size_t)(rows - keep) * HEAD_DIM * 2), st.ring, keep, HEAD_DIM);
  }

  // Grouped output projection (s_.o -> s_.res): group g maps heads 8g ... 8g + 7 (4,096
  // values) to 1,024, then wo_b. Few rows run the eight groups in one dispatch, with the
  // split sums linear() would use.
  void output_projection(const LayerW& lw, int T) {
    if (T < GEMM_ROWS) {
      uint32_t splits = std::min<uint32_t>(O_IN / 512, (2048 + O_RANK - 1) / O_RANK);
      uint32_t span = (O_IN / splits + 31) / 32 * 32;
      splits = (O_IN + span - 1) / span;
      TgSplitP sp{(uint32_t)T, O_IN, O_RANK, HEADS * HEAD_DIM, O_GROUPS * O_RANK, span, splits,
                  O_IN, (uint32_t)O_RANK * O_IN, (O_RANK / 32) * (O_IN / 32), O_RANK};
      gpu_.run_groups(rows_kernel("linear_fp8_tgsplit", T), {s_.o, lw.wo_a.w, lw.wo_a.s, s_.oa}, sp,
                      MTLSizeMake(O_RANK / 2, 1, O_GROUPS), MTLSizeMake(32 * splits, 1, 1));
    } else {
      for (int g = 0; g < O_GROUPS; ++g) {
        Fp8 part{lw.wo_a.w.at((size_t)g * O_RANK * O_IN), lw.wo_a.s.at((size_t)g * (O_RANK / 32) * (O_IN / 32)), O_RANK, O_IN};
        linear(s_.o.at((size_t)g * O_IN * 4), part, s_.oa.at((size_t)g * O_RANK * 4), T, HEADS * HEAD_DIM, O_GROUPS * O_RANK);
      }
    }
    linear(s_.oa, lw.wo_b, s_.res, T);
  }

  // A kv source's compressor: the latents of the groups this chunk completes, then
  // their index keys (key owners) and compressed KV rows.
  void compress(int layer, Buf x, int T, int p0) {
    const LayerW& lw = layers_[layer];
    LayerState& st = state_[layer];
    int ratio = lw.ratio, g0, groups;
    if (ratio == 2) {
      linear(x, lw.comp_wkv, s_.ckv, T);
      linear(x, lw.comp_wgate, s_.cscore, T);
      if (verifying_) {
        CopyRowsP rows{(uint32_t)T, HEAD_DIM, HEAD_DIM, HEAD_DIM};
        gpu_.run("copy_rows", {s_.ckv, vlog_[layer].ckv}, rows, HEAD_DIM, T);
        gpu_.run("copy_rows", {s_.cscore, vlog_[layer].cscore}, rows, HEAD_DIM, T);
      }
      g0 = p0 / 2;
      groups = (p0 + T) / 2 - g0;
      if (groups > 0)
        gpu_.run("compress_pool", {s_.ckv, s_.cscore, st.carry, st.carry.at(HEAD_DIM * 4), s_.latent},
                 PoolP{(uint32_t)groups, (uint32_t)g0, (uint32_t)p0, HEAD_DIM}, HEAD_DIM, groups);
      if ((p0 + T) % 2 != 0) {
        CopyRowsP one{1, HEAD_DIM, HEAD_DIM, HEAD_DIM};
        gpu_.run("copy_rows", {s_.ckv.at((size_t)(T - 1) * HEAD_DIM * 4), st.carry}, one, HEAD_DIM, 1);
        gpu_.run("copy_rows", {s_.cscore.at((size_t)(T - 1) * HEAD_DIM * 4), st.carry.at(HEAD_DIM * 4)}, one, HEAD_DIM, 1);
      }
    } else {
      linear(x, lw.comp_wkv, s_.latent, T);
      g0 = p0;
      groups = T;
    }
    if (groups == 0) return;
    norm(s_.latent, lw.w->vec(lw.comp_norm), s_.latent, groups, 1, HEAD_DIM, HEAD_DIM);
    if (lw.owns_keys) {
      linear(s_.latent, lw.idx_wk, s_.ik, groups);
      norm(s_.ik, lw.w->vec(lw.idx_k_norm), s_.ik, groups, 1, IDX_DIM, IDX_DIM);
      rope(s_.ik, groups, 1, IDX_DIM, IDX_DIM, g0, ratio, false, true);
      fake_quant(s_.ik, st.keys.at((size_t)g0 * IDX_DIM * 2), groups, IDX_DIM, IDX_DIM, IDX_DIM, 1, false);
    }
    rope(s_.latent, groups, 1, HEAD_DIM, HEAD_DIM, g0, ratio, false, true);
    fake_quant(s_.latent, st.comp.at((size_t)g0 * HEAD_DIM * 2), groups, HEAD_DIM, HEAD_DIM, HEAD_DIM, 2, false);
  }

  // An index source picks, for each query, up to 512 compressed positions (sorted, -1
  // padded) into selection rows start ... start + T - 1. The layers up to the next
  // index source read the same rows. The GPU work that read these rows for the previous
  // source finished at that layer's expert routing, so the CPU may write them here.
  void select(int layer, Buf x, int start, int T, int p0) {
    const LayerW& lw = layers_[layer];
    int ratio = lw.ratio, keys = (p0 + T) / ratio;
    int32_t* selection = (int32_t*)s_.selection.contents() + (size_t)start * IDX_TOPK;
    if (keys <= IDX_TOPK) {
      // Every visible position fits the budget, whatever the scores (and the candidate
      // blocks, at most 2,048 blocks of 8, keep them all).
      for (int t = 0; t < T; ++t) {
        int visible = (p0 + t + 1) / ratio;
        int32_t* out = selection + (size_t)t * IDX_TOPK;
        for (int i = 0; i < IDX_TOPK; ++i) out[i] = i < visible ? i : -1;
      }
      return;
    }
    linear(s_.qr, lw.idx_wq_b, s_.iq, T);
    rope(s_.iq, T, IDX_HEADS, IDX_DIM, IDX_HEADS * IDX_DIM, p0, 1, false, true);
    fake_quant(s_.iq, s_.iq_bf16, T, IDX_HEADS * IDX_DIM, IDX_HEADS * IDX_DIM, IDX_HEADS * IDX_DIM, 1, true);
    linear(x, lw.idx_weights, s_.iw, T);
    size_t bytes = (size_t)T * keys * 4;
    if (bytes > s_.iscore_bytes) {
      gpu_.ensure(s_.iscore, bytes + bytes / 2);
      s_.iscore_bytes = bytes + bytes / 2;
    }
    IndexScoreP p{(uint32_t)T, (uint32_t)keys, IDX_HEADS, IDX_DIM, (uint32_t)p0, (uint32_t)ratio,
                  1.0f / std::sqrt((float)IDX_DIM) / std::sqrt((float)IDX_HEADS)};
    if (getenv("WHALLM_DEEPSEEK_SCALAR_INDEX") != nullptr)  // development: the scalar kernel, for comparison
      gpu_.run("index_scores", {s_.iq, state_[lw.keys_from].keys, s_.iw, s_.iscore}, p, keys, T);
    else
      gpu_.run_groups("index_scores_mma", {s_.iq, state_[lw.keys_from].keys, s_.iw, s_.iscore}, p,
                      MTLSizeMake((keys + 63) / 64, T, 1), MTLSizeMake(128, 1, 1));
    gpu_.flush();
    CFAbsoluteTime selecting = CFAbsoluteTimeGetCurrent();
    const float* scores = (const float*)s_.iscore.contents();
    std::vector<std::vector<int>>* candidates = &candidates_;
    bool source = lw.candidate_source, use = lw.uses_candidates;
    int first = start;
    dispatch_apply((size_t)T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
      int visible = (p0 + (int)t + 1) / ratio;
      const float* row = scores + t * (size_t)keys;
      int32_t* out = selection + t * IDX_TOPK;
      std::vector<int>& kept_blocks = (*candidates)[first + t];
      if (source) {
        // Two-level selection: the 2,048 blocks of 8 with the best maximum, the block of
        // the newest position always included.
        kept_blocks.clear();
        int blocks = (visible + CAND_BLOCK - 1) / CAND_BLOCK;
        if (blocks > CAND_BLOCKS) {
          std::vector<std::pair<float, int>> best(blocks);
          for (int b = 0; b < blocks; ++b) {
            float top = -INFINITY;
            for (int i = b * CAND_BLOCK; i < std::min(visible, (b + 1) * CAND_BLOCK); ++i) top = std::max(top, row[i]);
            best[b] = {b == blocks - 1 ? INFINITY : top, b};
          }
          std::nth_element(best.begin(), best.begin() + CAND_BLOCKS, best.end(),
                           [](const std::pair<float, int>& a, const std::pair<float, int>& b) { return a.first > b.first; });
          for (int i = 0; i < CAND_BLOCKS; ++i)
            if (best[i].first > -INFINITY) kept_blocks.push_back(best[i].second);
          std::sort(kept_blocks.begin(), kept_blocks.end());
        }
      }
      std::vector<int> pool;
      if (use && !kept_blocks.empty()) {
        for (int b : kept_blocks)
          for (int i = b * CAND_BLOCK; i < std::min(visible, (b + 1) * CAND_BLOCK); ++i) pool.push_back(i);
      } else {
        pool.resize(visible);
        for (int i = 0; i < visible; ++i) pool[i] = i;
      }
      if ((int)pool.size() > IDX_TOPK) {
        std::nth_element(pool.begin(), pool.begin() + IDX_TOPK, pool.end(), [&](int a, int b) { return row[a] > row[b]; });
        pool.resize(IDX_TOPK);
        std::sort(pool.begin(), pool.end());
      }
      for (int i = 0; i < IDX_TOPK; ++i) out[i] = i < (int)pool.size() ? pool[i] : -1;
    });
    select_seconds_ += CFAbsoluteTimeGetCurrent() - selecting;
  }

  // -------------------------------------------------------------- DSpark
  void open_dspark(const std::string& root, NSDictionary* manifest) {
    NSDictionary* d = manifest[@"dspark"];
    if (![d isKindOfClass:[NSDictionary class]]) fail("this installed model has no DSpark weights");
    NSArray* targets = d[@"targetLayerIDs"];
    bool same = [d[@"layerCount"] intValue] == DSPARK_LAYERS && [d[@"blockSize"] intValue] == DSPARK_BLOCK &&
                [d[@"noiseTokenID"] intValue] == DSPARK_NOISE && [d[@"markovRank"] intValue] == MARKOV_RANK &&
                targets.count == (NSUInteger)DSPARK_LAYERS;
    for (int c = 0; same && c < DSPARK_LAYERS; ++c) same = [targets[c] intValue] == DSPARK_TARGETS[c];
    if (!same) fail("unsupported DSpark layout");
    dspark_.reset(new Weights(gpu_, root + "/dspark/common.bin", d[@"commonTensors"], options_.read_threads));
    for (int stage = 0; stage < DSPARK_LAYERS; ++stage) {
      LayerW lw;
      lw.experts = DSPARK_EXPERTS;
      lw.top_k = DSPARK_TOP_K;
      load_block(*dspark_, "mtp." + std::to_string(stage) + ".", lw);
      stages_.push_back(lw);
    }
    main_proj_ = dspark_->fp8("mtp.0.main_proj", H, DSPARK_LAYERS * H);
    main_norm_ = dspark_->vector("mtp.0.main_norm.weight", H);
    std::string last = "mtp." + std::to_string(DSPARK_LAYERS - 1) + ".";
    draft_norm_ = dspark_->vector(last + "norm.weight", H);
    markov_embed_ = (const uint16_t*)dspark_->bytes(dspark_->tensor(last + "markov_head.embed.weight", "BF16", {VOCAB, MARKOV_RANK}));
    markov_head_ = dspark_->bf16(last + "markov_head.head.weight", VOCAB, MARKOV_RANK);
    confidence_ = (const uint16_t*)dspark_->bytes(dspark_->tensor(last + "confidence_head.proj.weight", "BF16", {1, H + MARKOV_RANK}));
    dspark_->seal();
    size_t rows = pad16(WINDOW);
    capture_ = gpu_.alloc(rows * DSPARK_LAYERS * H * 4);
    main_x_ = gpu_.alloc(rows * H * 4);
    context_kv_ = gpu_.alloc(rows * HEAD_DIM * 4);
    context_window_ = gpu_.alloc((size_t)2 * WINDOW * HEAD_DIM * 2);
    for (Buf& context : context_) context = gpu_.alloc((size_t)WINDOW * HEAD_DIM * 2);
    draft_keys_ = gpu_.alloc((size_t)(WINDOW + VERIFY_ROWS) * 4);
    int32_t* keys = (int32_t*)draft_keys_.contents();
    for (int i = 0; i < WINDOW + VERIFY_ROWS; ++i) keys[i] = i;
    draft_hidden_ = gpu_.alloc((size_t)16 * H * 4);
    draft_logits_ = gpu_.alloc((size_t)DSPARK_BLOCK * VOCAB * 4);
    draft_row_ = gpu_.alloc((size_t)VOCAB * 4);
    markov_embedding_ = gpu_.alloc((size_t)16 * MARKOV_RANK * 4);
  }

  // Each draft layer's context: the last WINDOW consumed tokens' keys (= values), from
  // the target layers' stream. `rows` captured rows hold positions first ... first + rows - 1.
  void seed_context(int rows, int first) {
    if (rows <= 0) return;
    linear(capture_, main_proj_, main_x_, rows);
    norm(main_x_, dspark_->vec(main_norm_), main_x_, rows, 1, H, H);
    int kept = context_rows_, total = kept + rows, keep = std::min(total, WINDOW);
    for (int stage = 0; stage < DSPARK_LAYERS; ++stage) {
      const LayerW& lw = stages_[stage];
      linear(main_x_, lw.wkv, context_kv_, rows);
      norm(context_kv_, lw.w->vec(lw.kv_norm), context_kv_, rows, 1, HEAD_DIM, HEAD_DIM);
      rope(context_kv_, rows, 1, HEAD_DIM, HEAD_DIM, first, 1, false, false);
      if (kept > 0) copy_u16(context_[stage], context_window_, kept, HEAD_DIM);
      fake_quant(context_kv_, context_window_.at((size_t)kept * HEAD_DIM * 2), rows, HEAD_DIM, HEAD_DIM, HEAD_DIM, 0, false);
      copy_u16(context_window_.at((size_t)(total - keep) * HEAD_DIM * 2), context_[stage], keep, HEAD_DIM);
    }
    context_rows_ = keep;
  }

  // A draft layer's attention: the T block rows (positions position_ ...) see the
  // context and every block row, the later ones included.
  void draft_attention(const LayerW& lw, int T) {
    int p0 = position_, stage = (int)(&lw - stages_.data());
    Buf h = s_.all, mix = s_.attn_mix, x = s_.x;
    mixes(h, lw.attn_fn, lw.attn_scale, lw.attn_base, mix, T);
    hc_pre(h, s_.pre_mix, x, T);
    norm(x, lw.w->vec(lw.attn_norm), x, T, 1, H, H);
    linear(x, lw.wq_a, s_.qr, T);
    norm(s_.qr, lw.w->vec(lw.q_norm), s_.qr, T, 1, Q_RANK, Q_RANK);
    linear(s_.qr, lw.wq_b, s_.q, T);
    rope(s_.q, T, HEADS, HEAD_DIM, HEADS * HEAD_DIM, p0, 1, false, false);
    linear(x, lw.wkv, s_.kv, T);
    norm(s_.kv, lw.w->vec(lw.kv_norm), s_.kv, T, 1, HEAD_DIM, HEAD_DIM);
    rope(s_.kv, T, 1, HEAD_DIM, HEAD_DIM, p0, 1, false, false);
    int kept = context_rows_;
    if (kept > 0) copy_u16(context_[stage], s_.window, kept, HEAD_DIM);
    fake_quant(s_.kv, s_.window.at((size_t)kept * HEAD_DIM * 2), T, HEAD_DIM, HEAD_DIM, HEAD_DIM, 0, false);
    // No window keys; the "compressed" keys are context + block, the same list for every row.
    uint32_t comp = kept + T, keys = (comp + 31) / 32 * 32;
    AttnP ap{(uint32_t)T, (uint32_t)p0, 0, 0, comp, keys, HEADS, HEAD_DIM, 0, 1.0f / std::sqrt((float)HEAD_DIM)};
    MTLSize tiles = MTLSizeMake(32, 4, 1);
    gpu_.run("attn_scores", {s_.q, s_.window, s_.window, draft_keys_, s_.scores}, ap, 32, keys / 32, (size_t)HEADS / 8 * T, tiles);
    gpu_.run("attn_softmax", {s_.scores, lw.sink}, ap, 32, HEADS, T, kSimdRows);
    gpu_.run("attn_output", {s_.scores, s_.window, s_.window, draft_keys_, s_.o}, ap, 32, HEAD_DIM / 32, (size_t)HEADS / 8 * T, tiles);
    rope(s_.o, T, HEADS, HEAD_DIM, HEADS * HEAD_DIM, p0, 1, true, false);
    output_projection(lw, T);
    hc_post(h, s_.res, mix, T);
  }

  // -------------------------------------------------------------- MoE over all rows of the pass
  void feed_forward(const LayerW& lw, int count) {
    mixes(s_.all, lw.ffn_fn, lw.ffn_scale, lw.ffn_base, s_.ffn_mix, count);
    hc_pre(s_.all, s_.attn_mix, s_.x, count);
    norm(s_.x, lw.w->vec(lw.ffn_norm), s_.x, count, 1, H, H);
    linear(s_.x, lw.gate, s_.gate_logits, count);
    // Shared expert into result.
    linear(s_.x, lw.sh_w1, s_.shared_gate, count);
    linear(s_.x, lw.sh_w3, s_.shared_up, count);
    gpu_.run("swiglu", {s_.shared_gate, s_.shared_up}, ActP{(uint32_t)(count * MOE_M), MOE_M, SWIGLU_LIMIT}, (size_t)count * MOE_M);
    linear(s_.shared_gate, lw.sh_w2, s_.result, count);
    gpu_.flush();  // the router's choice decides which experts to read
    CFAbsoluteTime routed = CFAbsoluteTimeGetCurrent();
    route(count, lw);
    route_seconds_ += CFAbsoluteTimeGetCurrent() - routed;
    if (count < PASS_ROWS) experts_decode(lw, count);
    else experts_pass(lw.file, count);
    hc_post(s_.all, s_.result, s_.ffn_mix, count);
  }

  // sqrt(softplus) scores; the bias picks the experts but does not weight them.
  void route(int count, const LayerW& lw) {
    const float* logits = (const float*)s_.gate_logits.contents();
    const int E = lw.experts, K = lw.top_k;
    chosen_.resize((size_t)count * K);
    weights_.resize((size_t)count * K);
    int* chosen = chosen_.data();
    float* weights = weights_.data();
    const float* b = lw.gate_bias.data();
    dispatch_apply((size_t)count, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
      const float* row = logits + t * E;
      float score[EXPERTS], biased[EXPERTS];
      int order[EXPERTS];
      for (int e = 0; e < E; ++e) {
        float v = row[e];
        score[e] = std::sqrt(std::max(v, 0.0f) + std::log1p(std::exp(-std::fabs(v))));
        biased[e] = score[e] + b[e];
        order[e] = e;
      }
      std::partial_sort(order, order + K, order + E, [&](int x, int y) { return biased[x] > biased[y] || (biased[x] == biased[y] && x < y); });
      float sum = 0;
      for (int k = 0; k < K; ++k) sum += score[order[k]];
      for (int k = 0; k < K; ++k) {
        chosen[t * K + k] = order[k];
        weights[t * K + k] = score[order[k]] / (sum + 1e-20f) * ROUTE_SCALE;
      }
    });
  }

  // Few rows: experts come from the cache; the cached ones run while the missing ones are read.
  void experts_decode(const LayerW& lw, int count) {
    const int layer = lw.file, K = lw.top_k;
    uint32_t n = (uint32_t)(count * K);
    memcpy(route_.contents(), weights_.data(), n * 4);
    arena_->begin();
    std::vector<int> unique(chosen_.begin(), chosen_.begin() + n);
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::vector<int> slot(unique.size());
    std::vector<char> late(unique.size(), 0);
    std::vector<std::pair<int, int>> missing;
    for (size_t u = 0; u < unique.size(); ++u) {
      bool miss = false;
      slot[u] = arena_->acquire(layer, unique[u], miss);
      if (miss) {
        late[u] = 1;
        missing.emplace_back(unique[u], slot[u]);
      }
    }
    // Pairs (token, expert) grouped by expert, cached experts first: their groups run while
    // the missing experts are read. A group reads its expert's weights once for its tokens.
    std::vector<uint32_t> order;  // unique indices, cached first
    for (int pass = 0; pass < 2; ++pass)
      for (size_t u = 0; u < unique.size(); ++u)
        if (late[u] == pass) order.push_back((uint32_t)u);
    Pair* pairs = (Pair*)pairs_.contents();
    ExpertGroup* groups = (ExpertGroup*)groups_.contents();
    uint32_t pair_count = 0, group_count = 0, ready_pairs = 0, ready_groups = 0;
    for (uint32_t u : order) {
      ExpertGroup* group = nullptr;
      for (uint32_t i = 0; i < n; ++i) {
        if (chosen_[i] != unique[u]) continue;
        if (group == nullptr || group->count == GROUP_ROWS) {
          group = &groups[group_count++];
          group->slot = (uint32_t)slot[u];
          group->count = 0;
        }
        group->in[group->count] = i / K;
        group->out[group->count] = i;
        ++group->count;
        pairs[pair_count++] = Pair{i / K, (uint32_t)slot[u], i};
      }
      if (!late[u]) ready_groups = group_count, ready_pairs = pair_count;  // the cached experts come first
    }
    // The down projection reads each pair's activation row: its input row is its output row.
    ExpertGroup* down = (ExpertGroup*)down_groups_.contents();
    for (uint32_t g = 0; g < group_count; ++g) {
      down[g] = groups[g];
      for (uint32_t r = 0; r < groups[g].count; ++r) down[g].in[r] = groups[g].out[r];
    }
    Buf blobs = arena_->buffer();
    auto run_groups = [&](uint32_t first_group, uint32_t groups_n, uint32_t first_pair, uint32_t pairs_n) {
      if (groups_n == 0) return;
      Buf list = groups_.at(first_group * sizeof(ExpertGroup));
      uint32_t most = 0;
      for (uint32_t g = first_group; g < first_group + groups_n; ++g) most = std::max(most, groups[g].count);
      // Outputs per simdgroup measured 2026-10-16 on M2 Max: 8 for gate+up (I = 5,120), 4 for down (I = 2,304).
      unsigned rows = most <= 1 ? 1 : most <= 2 ? 2 : most <= 4 ? 4 : 8;
      char kernel[32], down_kernel[32];
      snprintf(kernel, sizeof(kernel), "expert_group%u_8", rows);
      snprintf(down_kernel, sizeof(down_kernel), "expert_group%u_4", rows);
      // Gate (w1) and up (w3) rows in one dispatch: row i of rgate is [gate | up].
      gpu_.run(kernel, {s_.x, blobs, list, s_.rgate}, GroupP{groups_n, H, MOE_M, H, W1, W1_SCALE, (uint32_t)BLOB,
               MOE_M, W3, W3_SCALE, 2 * MOE_M}, 32, 2 * MOE_M / 8, groups_n, kSimdRows);
      gpu_.run("swiglu_pairs", {s_.rgate, s_.rgate.at(MOE_M * 4), pairs_.at(first_pair * sizeof(Pair))},
               PairActP{pairs_n, MOE_M, SWIGLU_LIMIT, 2 * MOE_M}, MOE_M, pairs_n);
      gpu_.run(down_kernel, {s_.rgate, blobs, down_groups_.at(first_group * sizeof(ExpertGroup)), s_.routed},
               GroupP{groups_n, MOE_M, H, 2 * MOE_M, W2, W2_SCALE, (uint32_t)BLOB, H, W2, W2_SCALE, H}, 32, H / 4, groups_n, kSimdRows);
    };
    run_groups(0, ready_groups, 0, ready_pairs);
    if (!missing.empty()) {
      if (ready_groups > 0 && missing.size() >= OVERLAP_MIN_READS) gpu_.commit();
      arena_->load(layer, missing);
      run_groups(ready_groups, group_count - ready_groups, ready_pairs, pair_count - ready_pairs);
    }
    gpu_.run("moe_combine", {s_.routed, route_, s_.result, s_.result}, CombineP{(uint32_t)count, H, (uint32_t)K}, H, count);
  }

  // Many rows: every needed expert of the layer is used once for all rows, in waves of
  // up to 32 in expert order. Cached experts are used in place; the rest are read into a
  // staging buffer of STAGING waves while the GPU works on earlier waves. The experts the
  // call uses most (slots / 40 per layer) are read into the cache instead, for the
  // tokens that follow. Each token adds its experts' outputs in expert order, so the
  // sums do not depend on what was cached. A long call starts reading the next layer's
  // first waves before that layer's attention runs (see prefetch).
  void experts_pass(int layer, int count) {
    std::vector<std::vector<uint32_t>> rows(EXPERTS);  // pair indices t * TOP_K + k, in token order
    for (int i = 0; i < count * TOP_K; ++i) rows[chosen_[i]].push_back((uint32_t)i);
    std::vector<int> needed;
    for (int e = 0; e < EXPERTS; ++e)
      if (!rows[e].empty()) needed.push_back(e);
    // Waves read ahead for this layer (prefetch); their experts are all below `ahead_end`.
    std::vector<WaveRead> ahead;
    if (ahead_layer_ == layer) ahead.swap(ahead_);
    ahead_layer_ = -1;
    int ahead_end = ahead.empty() ? 0 : ahead.back().experts.back() + 1;
    std::vector<long> staged_at(EXPERTS, -1);  // staging slot of a read-ahead expert
    for (const WaveRead& read : ahead)
      for (size_t i = 0; i < read.experts.size(); ++i) staged_at[read.experts[i]] = (long)(read.slot * WAVE + i);
    // Where each expert's blob is: a cache slot, or (high bit) a staging slot. Cached
    // experts are claimed first, so seeding cannot evict them.
    arena_->begin();
    std::vector<uint32_t> where(EXPERTS, 0xFFFFFFFFu);
    std::vector<char> read_into_cache(EXPERTS, 0);
    for (int e : needed)
      if (arena_->find(layer, e) >= 0) {
        bool miss = false;
        where[e] = (uint32_t)arena_->acquire(layer, e, miss, (long)rows[e].size());
      }
    if (options_.seed) {
      std::vector<int> order;
      for (int e : needed)
        if (where[e] == 0xFFFFFFFFu && e >= ahead_end) order.push_back(e);
      std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return rows[a].size() > rows[b].size(); });
      int keep = std::min<int>((int)order.size(), slots_ / LAYERS);
      for (int i = 0; i < keep; ++i) {
        bool miss = false;
        int e = order[i];
        where[e] = (uint32_t)arena_->acquire(layer, e, miss, (long)rows[e].size());
        read_into_cache[e] = 1;
        ++arena_->seeded;
      }
    }
    // Waves: one per read-ahead wave (its range of expert indices), then groups of WAVE.
    struct Wave { std::vector<int> experts; int slot = 0; dispatch_group_t read = nil;
                  uint32_t tiles = 0, tile_first = 0, tokens = 0, list_first = 0, entry_first = 0; };
    std::vector<Wave> waves;
    size_t next = 0;
    for (const WaveRead& read : ahead) {
      Wave wave;
      while (next < needed.size() && needed[next] <= read.experts.back()) wave.experts.push_back(needed[next++]);
      wave.slot = read.slot;
      wave.read = read.done;
      waves.push_back(wave);
    }
    for (; next < needed.size(); next += WAVE) {
      Wave wave;
      wave.experts.assign(needed.begin() + next, needed.begin() + std::min(needed.size(), next + WAVE));
      wave.slot = (int)(wave_seq_++ % STAGING);
      waves.push_back(wave);
    }
    auto staging_slot = [&](const Wave& wave, size_t i) -> uint32_t {
      int e = wave.experts[i];
      return staged_at[e] >= 0 ? (uint32_t)staged_at[e] : (uint32_t)(wave.slot * WAVE + i);
    };
    // Tiles of 16 rows per expert; the combine lists of each wave.
    std::vector<uint32_t> tile_slots, tile_rows, list, entries;
    std::vector<uint32_t> pair_row((size_t)count * TOP_K);
    uint32_t max_rows = 0;
    for (Wave& wave : waves) {
      wave.tile_first = (uint32_t)tile_slots.size();
      for (size_t i = 0; i < wave.experts.size(); ++i) {
        int e = wave.experts[i];
        uint32_t code = where[e] != 0xFFFFFFFFu ? where[e] : (0x80000000u | staging_slot(wave, i));
        const std::vector<uint32_t>& r = rows[e];
        for (size_t at = 0; at < r.size(); at += 16) {
          uint32_t tile = (uint32_t)tile_slots.size() - wave.tile_first;
          tile_slots.push_back(code);
          for (size_t j = 0; j < 16; ++j) {
            if (at + j < r.size()) {
              tile_rows.push_back(r[at + j] / TOP_K);
              pair_row[r[at + j]] = tile * 16 + (uint32_t)j;
            } else {
              tile_rows.push_back(0xFFFFFFFFu);
            }
          }
        }
      }
      wave.tiles = (uint32_t)tile_slots.size() - wave.tile_first;
      max_rows = std::max(max_rows, wave.tiles * 16);
      // Tokens of this wave in token order; each token's pairs in expert order.
      std::vector<std::pair<uint32_t, uint32_t>> items;  // (token, pair) sorted by token then expert
      for (int e : wave.experts)
        for (uint32_t pair : rows[e]) items.emplace_back(pair / TOP_K, pair);
      std::stable_sort(items.begin(), items.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      wave.list_first = (uint32_t)list.size() / 3;
      wave.entry_first = (uint32_t)entries.size() / 2;
      for (size_t i = 0; i < items.size();) {
        size_t j = i;
        uint32_t first = (uint32_t)entries.size() / 2 - wave.entry_first;
        while (j < items.size() && items[j].first == items[i].first) {
          float weight = weights_[items[j].second];
          uint32_t bits;
          memcpy(&bits, &weight, 4);
          entries.push_back(pair_row[items[j].second]);
          entries.push_back(bits);
          ++j;
        }
        list.push_back(items[i].first);
        list.push_back(first);
        list.push_back((uint32_t)(j - i));
        ++wave.tokens;
        i = j;
      }
    }
    auto upload = [&](Buf& buffer, const std::vector<uint32_t>& values) {
      gpu_.ensure(buffer, values.size() * 4 + 16);
      memcpy(buffer.contents(), values.data(), values.size() * 4);
    };
    upload(tile_slots_, tile_slots);
    upload(tile_rows_, tile_rows);
    upload(wave_list_, list);
    upload(wave_entries_, entries);
    gpu_.ensure(gx_, (size_t)max_rows * H * 4);
    gpu_.ensure(gg_, (size_t)max_rows * MOE_M * 4);
    gpu_.ensure(gu_, (size_t)max_rows * MOE_M * 4);
    gpu_.ensure(go_, (size_t)max_rows * H * 4);

    // A wave's reads: staged experts into its staging slot, seeded ones into the cache.
    const File& file = arena_->file(layer, true);
    auto issue_wave = [&](Wave& wave) {
      if (wave.read != nil) return;
      std::vector<ReadJob> jobs;
      for (size_t i = 0; i < wave.experts.size(); ++i) {
        int e = wave.experts[i];
        if (where[e] == 0xFFFFFFFFu) {
          add_read(jobs, file, e, staging_memory(staging_slot(wave, i)));
          ++arena_->streamed;
        } else if (read_into_cache[e]) {
          add_read(jobs, file, e, arena_->slot_memory((int)where[e]));
          ++arena_->misses;
        }
      }
      wave.read = issue_reads(std::move(jobs), wave.slot);
    };
    // A wave may be read once the wave that used its staging slot before has been committed.
    for (size_t w = 0; w < waves.size() && w < (size_t)STAGING; ++w) issue_wave(waves[w]);
    for (size_t w = 0; w < waves.size(); ++w) {
      check_cancel_reads();
      wait_reads(waves[w].read);
      const Wave& wave = waves[w];
      uint32_t R = wave.tiles * 16;
      Buf slots = tile_slots_.at((size_t)wave.tile_first * 4);
      gpu_.run("gather_rows", {s_.x, tile_rows_.at((size_t)wave.tile_first * 16 * 4), gx_}, GatherP{R, H, H}, H / 4, R);
      Buf cache = arena_->buffer();
      auto expert = [&](Buf in, Buf out, uint32_t I, uint32_t O, uint32_t w, uint32_t s) {
        if (gpu_.tensor_ops && (int)R >= TENSOR_OPS_ROWS)
          gpu_.run_groups("tops_expert", {in, cache, staging_, slots, out}, TopsExpertP{wave.tiles, I, O, w, s, (uint32_t)BLOB},
                          MTLSizeMake(O / 64, wave.tiles, 1), MTLSizeMake(128, 1, 1));
        else
          gpu_.run("expert_gemm_sg", {in, cache, staging_, slots, out}, ExpertSgP{wave.tiles, I, O, w, s, (uint32_t)BLOB},
                   32, O / 32, wave.tiles, MTLSizeMake(32, 4, 1));
      };
      expert(gx_, gg_, H, MOE_M, W1, W1_SCALE);
      expert(gx_, gu_, H, MOE_M, W3, W3_SCALE);
      gpu_.run("swiglu", {gg_, gu_}, ActP{R * MOE_M, MOE_M, SWIGLU_LIMIT}, (size_t)R * MOE_M);
      expert(gg_, go_, MOE_M, H, W2, W2_SCALE);
      gpu_.run("wave_combine", {go_, wave_list_.at((size_t)wave.list_first * 3 * 4), wave_entries_.at((size_t)wave.entry_first * 2 * 4), s_.result},
               WaveP{wave.tokens, H}, H / 4, wave.tokens);
      slot_user_[wave.slot] = gpu_.commit();
      if (w + STAGING < waves.size()) issue_wave(waves[w + STAGING]);
    }
    if (count >= PREFETCH_ROWS && layer + 1 < LAYERS) prefetch(layer + 1);
  }

  // Start reading the first STAGING waves of uncached experts of `layer`, in expert order.
  // A call this long uses nearly every expert, so few of these reads are wasted, and the
  // SSD keeps working while the layer's attention runs.
  void prefetch(int layer) {
    ahead_.clear();
    const File& file = arena_->file(layer, true);
    int e = 0;
    for (int w = 0; w < STAGING && e < EXPERTS; ++w) {
      WaveRead read;
      read.slot = (int)(wave_seq_++ % STAGING);
      std::vector<ReadJob> jobs;
      for (; e < EXPERTS && (int)read.experts.size() < WAVE; ++e) {
        if (arena_->find(layer, e) >= 0) continue;
        add_read(jobs, file, e, staging_memory((uint32_t)(read.slot * WAVE + read.experts.size())));
        read.experts.push_back(e);
        ++arena_->streamed;
      }
      if (read.experts.empty()) break;
      read.done = issue_reads(std::move(jobs), read.slot);
      ahead_.push_back(read);
    }
    ahead_layer_ = layer;
  }

  char* staging_memory(uint32_t slot) {
    if (staging_.buffer == nil) staging_ = gpu_.alloc((size_t)STAGING * WAVE * BLOB);
    return (char*)staging_.contents() + (size_t)slot * BLOB;
  }

  // Appends one expert read, merged with the previous one when both file and memory are adjacent.
  static void add_read(std::vector<ReadJob>& jobs, const File& file, int expert, char* destination) {
    if (!jobs.empty() && jobs.back().offset + jobs.back().length == (size_t)expert * BLOB &&
        jobs.back().destination + jobs.back().length == destination)
      jobs.back().length += BLOB;
    else
      jobs.push_back({&file, (size_t)expert * BLOB, BLOB, destination});
  }

  // Reads on the serial reader queue, after the GPU work that last used staging slot
  // `slot` (if any) has completed. Returns a group to wait for these reads.
  dispatch_group_t issue_reads(std::vector<ReadJob> list, int slot) {
    if (reads_ == nil) {
      reads_ = dispatch_group_create();
      reader_ = dispatch_queue_create("whallm.deepseek.reader", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0));
    }
    dispatch_group_t done = dispatch_group_create();
    dispatch_group_t all = reads_;
    id<MTLCommandBuffer> after = slot >= 0 ? slot_user_[slot] : nil;
    auto jobs = std::make_shared<std::vector<ReadJob>>(std::move(list));
    bool* flag = &reads_failed_;  // a member: reads can outlive the call that started them
    dispatch_group_enter(done);
    dispatch_group_enter(all);
    dispatch_async(reader_, ^{
      if (after != nil) [after waitUntilCompleted];
      if (!run_reads(*jobs)) *flag = true;
      dispatch_group_leave(done);
      dispatch_group_leave(all);
    });
    return done;
  }

  void wait_reads(dispatch_group_t group) {
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    arena_->read_seconds += CFAbsoluteTimeGetCurrent() - started;
    if (reads_failed_) {
      reads_failed_ = false;
      fail("could not read experts from SSD");
    }
  }

  // Every read in flight finished and the staging slots unused (pass start, errors).
  void settle_reads() {
    if (reads_ != nil) dispatch_group_wait(reads_, DISPATCH_TIME_FOREVER);
    reads_failed_ = false;
    for (auto& user : slot_user_) user = nil;
    ahead_layer_ = -1;
    ahead_.clear();
  }

  // Cancellation between waves; reads in flight are awaited first.
  void check_cancel_reads() {
    if (cancel_ && cancel_()) {
      if (reads_ != nil) dispatch_group_wait(reads_, DISPATCH_TIME_FOREVER);
      throw Cancelled();
    }
  }

  Options options_;
  Gpu gpu_;
  std::unique_ptr<Weights> common_;
  std::vector<LayerW> layers_;
  std::vector<LayerState> state_;
  EngramTable engram_[2];
  const uint16_t* embed_ = nullptr;
  Bf16 head_;
  size_t final_norm_ = 0, rope_ = 0, yarn_ = 0;
  std::unique_ptr<ExpertArena> arena_;
  int slots_ = 0;
  Buf logits_, route_, pairs_, groups_, down_groups_;
  Buf staging_, tile_slots_, tile_rows_, wave_list_, wave_entries_, gx_, gg_, gu_, go_;
  // Streaming experts: reads run on reader_ (all of them in reads_). A staging slot holds
  // one wave; slot_user_ is the GPU work that last read it. Waves are numbered across
  // layers (wave_seq_) and take staging slots in turn.
  struct WaveRead { std::vector<int> experts; int slot = 0; dispatch_group_t done = nil; };
  dispatch_group_t reads_ = nil;
  dispatch_queue_t reader_ = nil;
  bool reads_failed_ = false;
  id<MTLCommandBuffer> slot_user_[STAGING] = {};
  long wave_seq_ = 0;
  int ahead_layer_ = -1;  // the layer whose first waves ahead_ reads
  std::vector<WaveRead> ahead_;
  std::vector<int> chosen_;
  std::vector<float> weights_;
  std::vector<int32_t> cids_;
  std::vector<std::vector<int>> candidates_;
  int position_ = 0, pos0_ = 0;
  double engram_seconds_ = 0, route_seconds_ = 0, select_seconds_ = 0;
  // Engram rows of the current pass, read in the background (start_engram_reads).
  std::vector<uint8_t> engram_raw_[2];
  dispatch_group_t engram_group_[2] = {nil, nil};
  std::atomic<bool> engram_failed_{false};
  std::function<bool()> cancel_;
  std::vector<int32_t> tokens_;
  std::string contract_;
  // Verification (verify / commit): what commit() needs to keep only some rows.
  struct VerifyLog { Buf window, ckv, cscore; int kept = 0; };
  std::vector<VerifyLog> vlog_;
  bool verifying_ = false;
  int verify_position_ = 0, verify_count_ = 0;
  // DSpark
  std::unique_ptr<Weights> dspark_;
  std::vector<LayerW> stages_;
  Fp8 main_proj_;
  Bf16 markov_head_;
  size_t main_norm_ = 0, draft_norm_ = 0;
  const uint16_t* markov_embed_ = nullptr;
  const uint16_t* confidence_ = nullptr;
  Buf capture_, main_x_, context_kv_, context_window_, context_[DSPARK_LAYERS], draft_keys_;
  Buf draft_hidden_, draft_logits_, draft_row_, markov_embedding_;
  int context_rows_ = 0;
  bool drafted_ = false;
};

}  // namespace

std::unique_ptr<Engine> Engine::open(const std::string& model_dir, const Options& options) {
  @autoreleasepool {
    return std::unique_ptr<Engine>(new EngineImpl(model_dir, options));
  }
}

}  // namespace wd
