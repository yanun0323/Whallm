#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <initializer_list>
#include <stdexcept>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/sysctl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

#include "engine.h"

extern const char* const kWhallmQwenKernels;        // kernels.metal, embedded at build time
extern const char* const kWhallmQwenTensorKernels;  // kernels_tensorops.metal

namespace wq {
namespace {

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

// Model dimensions. open() checks them against config.json.
constexpr int H = 2560, C = 4, W = H * C, LAYERS = 48, VOCAB = 248320;
constexpr int HC_RANK = 320;
constexpr int HEADS = 24, KV_HEADS = 2, HEAD_DIM = 256, ROTARY = 64;
constexpr int INDEX_HEADS = 4, INDEX_DIM = 128, INDEX_BUDGET = 2048, RATIO = 4;
// Selected keys per query: the budget plus the unfinished tail, padded to a multiple of 8.
constexpr int SELECTED = (INDEX_BUDGET + RATIO + 7) / 8 * 8;
constexpr int GDN_HK = 16, GDN_HV = 48, GDN_D = 128, GDN_KEY = GDN_HK * GDN_D, GDN_VALUE = GDN_HV * GDN_D;
constexpr int EXPERTS = 512, TOP_K = 10, MOE_M = 640;
constexpr int NGRAM_HEADS = 16, NGRAM_ROW = 160, PLE_STATE = 9, CONV_STATE = 3;
constexpr size_t BLOB = 2611200, GATE_UP_SCALE = 1638400, DOWN_WEIGHT = 1740800, DOWN_SCALE = 2560000;
constexpr float EPS = 1e-6f;
constexpr int EOS = 248044;
constexpr int VISION_START = 248053, IMAGE_TOKEN = 248056, VISION_END = 248054;
// Image encoder (ViT) dimensions.
constexpr int V_HIDDEN = 1152, V_HEADS = 16, V_HEAD_DIM = 72, V_DEPTH = 27, V_MLP = 4304, V_MLP_PADDED = 4320;
constexpr int V_PATCH = 3 * 2 * 16 * 16, V_SIDE = 48, V_MERGED = V_HIDDEN * 4;
// Calls of more tokens than this read whole expert files, one layer at a time,
// instead of only the chosen experts (512, as measured for the MLX engine).
// On M2 Max (2026-10-14): 97 and 256 tokens took 1.7 and 2.7 s this way against 10.0
// and 9.0 s with whole files; 512 tokens took 9.5 s either way; 1,024 tokens 10.7 s
// with whole files against 13.0 s.
constexpr int WHOLE_LAYER_ROWS = 512;
// From this many tokens the routed experts run as 16-row tiles per expert.
constexpr int TILED_EXPERT_ROWS = 32;
// Fewest missing experts in one layer for which decode starts the cached ones first.
// M2 Max, 4,096-token Code prompt, MTP: 8,192 slots 19.8 tok/s (19.1 never, 18.1-18.4
// always); 2,089 slots 16.3 (15.4 never).
constexpr size_t OVERLAP_MIN_READS = 4;
// Bounds of the automatic expert cache: 1,024 slots are 2.6 GB, 8,192 are 21 GB. On M2 Max
// 64 GB (2026-10-14), 8,192 slots decoded at 16.7 tok/s against 14.1 with 2,089 and kept
// prompt reading at 296 tok/s; 11,438 slots decoded no faster and read prompts at 271.
constexpr int MIN_AUTOMATIC_SLOTS = 1024, MAX_AUTOMATIC_SLOTS = 8192;
// From this many rows a matrix product runs as 16-row tiles. Fewer rows (decode
// and verification) use simdgroup-per-output kernels, which cost less per weight.
constexpr int GEMM_ROWS = 9;
// From this many rows dense products use the threadgroup-staged kernel (3.2-3.7
// TMAC/s on M2 Max against 2.2 for the per-simdgroup tile kernel).
constexpr int STAGED_GEMM_ROWS = 64;
// From this many rows dense products go through TensorOps when the GPU has Neural
// Accelerators. Tungsten measured them ahead of simdgroup kernels from about 400 rows
// on M5 Max; unmeasured here (no M5).
constexpr int TENSOR_OPS_ROWS = 256;

// Prompt cache file header: magic, then the sizes of the contract, tokens and state.
const char kStateMagic[8] = {'W', 'Q', 'S', 'T', 'A', 'T', 'E', '1'};

// Threadgroups of whole 32-lane simdgroups, for kernels that reduce with simd_sum.
const MTLSize kSimdRows = MTLSizeMake(32, 32, 1);

struct Buf {
  id<MTLBuffer> buffer = nil;
  size_t offset = 0;
  Buf at(size_t bytes) const { return {buffer, offset + bytes}; }
};

// ------------------------------------------------------------------ kernel parameter blocks (match kernels.metal)
struct LinearP { uint32_t T, I, O, x_stride, y_stride; };
struct NormP { uint32_t T, groups, D, row_stride, base, w_period, has_w; float eps, scale; };
struct CountP { uint32_t n; float scale; };
struct HcP { uint32_t T, H, C; };
struct Pair { uint32_t x_row, slot, out; };
struct ExpertP { uint32_t pairs, I, O, x_stride, w_offset, s_offset, blob, x_is_pair; };
struct ActP { uint32_t pairs, M; };
struct CombineP { uint32_t T, H, K; };
struct ConvP { uint32_t T, C, K, dilation, state_rows, apply_silu; };
struct GdnP { uint32_t T, Hk, Hv, Dk, Dv, row, key_dim; };
struct GatedNormP { uint32_t N, D; float eps; };
struct VisionNormP { uint32_t N, D; float eps; };
struct BiasP { uint32_t N, O, act; };
struct VisionAttnP { uint32_t N, Q0, Qn, heads, D; float scale; };
struct RopeP { uint32_t T, groups, D, in_stride, in_offset, group_stride, rotary, pos0; float eps, ln_theta; uint32_t has_norm, map_len; int32_t delta; };
struct CopyP { uint32_t T, D, in_stride, in_offset; };
struct PoolP { uint32_t blocks, first, ratio, D, rotary; float eps, ln_theta; uint32_t map_len; int32_t delta; };
struct BlockScoreP { uint32_t T, blocks, heads, D, pos0, ratio; };
struct AttentionP { uint32_t T, K, heads, kv_heads, D, pos0, use_selection; float scale; };
struct PleP { uint32_t T, H, C; };
struct PackP { uint32_t T, I, x_stride, blocks; };
struct GemmP { uint32_t T, I, O, y_stride; };
struct Tile { uint32_t slot, out[8]; };
struct ExpertGemmP { uint32_t tiles, I, O, w_offset, s_offset, blob; };
struct GatherP { uint32_t tiles, I, x_stride; };
struct CombineRowsP { uint32_t T, H, K; };
struct DequantP { uint32_t O, I; };
struct TopsCopyP { uint32_t T, I, x_stride; };
struct TopsGemmP { uint32_t T, I, O, y_stride; };
struct TopsExpertP { uint32_t tiles, I, O, w_offset, s_offset, blob; };
struct SplitP { uint32_t T, I, O, x_stride, span, splits; };


// ------------------------------------------------------------------ Metal
class Gpu {
 public:
  id<MTLDevice> device;

  Gpu() {
    device = MTLCreateSystemDefaultDevice();
    if (device == nil) fail("no Metal device");
    queue_ = [device newCommandQueue];
    // Every allocation joins one residency set attached to the queue. Without it each
    // commit pays for checking all referenced memory: on M2 Max a commit referencing a
    // 5 GB buffer took 0.68 ms and a 20 GB one 2.3 ms, against 0.14 ms with the set.
    // Two sets: what exists when the model is loaded (weights, expert cache) is committed
    // once; later allocations (prompt scratch, attention state) go to a second set, so
    // committing them does not re-process the large one.
    NSError* set_error = nil;
    residency_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&set_error];
    later_residency_ = [device newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&set_error];
    if (residency_ == nil || later_residency_ == nil) fail("Metal residency sets are not available");
    [queue_ addResidencySet:residency_];
    [queue_ addResidencySet:later_residency_];
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    NSError* error = nil;
    library_ = [device newLibraryWithSource:[NSString stringWithUTF8String:kWhallmQwenKernels] options:options error:&error];
    if (library_ == nil) fail(std::string("Metal kernels did not compile: ") + error.localizedDescription.UTF8String);
    pipelines_ = [NSMutableDictionary new];
    simdgroup_matrices = probe();
    tensor_ops = open_tensor_ops();
  }

  // True when long products run through TensorOps (the M5 Neural Accelerators).
  bool tensor_ops = false;

  // True when an 8x8 simdgroup matrix keeps its elements where the tiled kernels assume.
  bool simdgroup_matrices = false;

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

  // Drop a buffer from alloc(). The residency set holds the large ones, so they must
  // leave it before their last reference goes.
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

  // Replace a buffer with a new allocation of `bytes`.
  Buf replace(Buf& buffer, size_t bytes) {
    release(buffer);
    buffer = alloc(bytes);
    return buffer;
  }

  // Make every allocation so far resident now, so the cost falls here and not in a request.
  void settle() {
    begin();
    flush();
    settled_ = true;
  }

  size_t allocated_bytes = 0;

  // Allocations from this size on join the residency set; smaller ones cost little.
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
    // A kernel that needs many registers allows fewer threads per group, and
    // Metal skips a dispatch that asks for more without an error. Rows of 32
    // threads stay whole, since the simdgroup kernels rely on them.
    NSUInteger limit = pipeline.maxTotalThreadsPerThreadgroup;
    while (group.width * group.height * group.depth > limit && (group.height > 1 || group.depth > 1)) {
      if (group.depth > 1) group.depth = std::max<NSUInteger>(1, group.depth / 2);
      else group.height = std::max<NSUInteger>(1, group.height / 2);
    }
    if (group.width * group.height * group.depth > limit) fail(std::string("Metal kernel ") + name + " cannot run 32 threads together");
    [encoder_ dispatchThreads:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:group];
    ++dispatches;
    if (profile_) {  // one submission per kernel, so its GPU time can be read
      double before = gpu_seconds;
      flush();
      kernel_seconds_[name] += gpu_seconds - before;
      ++kernel_calls_[name];
    }
  }

  // WHALLM_QWEN_KERNEL_PROFILE=1: time each kernel alone (slow; for development).
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

  // Dispatch whole threadgroups (kernels that index by threadgroup position).
  template <typename P>
  void run_groups(const char* name, std::initializer_list<Buf> buffers, const P& params, MTLSize groups, MTLSize group) {
    id<MTLComputePipelineState> pipeline = this->pipeline(name);
    if (group.width * group.height * group.depth > pipeline.maxTotalThreadsPerThreadgroup)
      fail(std::string("Metal kernel ") + name + " cannot run its threadgroup size");
    begin();
    [encoder_ setComputePipelineState:pipeline];
    NSUInteger index = 0;
    for (const Buf& buffer : buffers) [encoder_ setBuffer:buffer.buffer offset:buffer.offset atIndex:index++];
    [encoder_ setBytes:&params length:sizeof(P) atIndex:index];
    [encoder_ dispatchThreadgroups:groups threadsPerThreadgroup:group];
    ++dispatches;
    if (profile_) {
      double before = gpu_seconds;
      flush();
      kernel_seconds_[name] += gpu_seconds - before;
      ++kernel_calls_[name];
    }
  }

  // WHALLM_QWEN_NO_STAGED=1: use the per-simdgroup tile GEMM for long prompts (comparison).
  bool no_staged = getenv("WHALLM_QWEN_NO_STAGED") != nullptr;

  // Open a command buffer if none is open; new allocations become resident first.
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

  // Start everything encoded so far without waiting for it. The next flush() waits.
  void commit() {
    if (encoder_ == nil || profile_) return;
    [encoder_ endEncoding];
    [command_ commit];
    pending_.push_back(command_);
    encoder_ = nil;
    command_ = nil;
  }

  // Run everything encoded so far and wait for it.
  void flush() {
    if (encoder_ == nil && pending_.empty()) return;
    if (encoder_ == nil) begin();  // an empty buffer to wait behind the committed ones
    [encoder_ endEncoding];
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    [command_ commit];
    [command_ waitUntilCompleted];
    wait_seconds += CFAbsoluteTimeGetCurrent() - started;
    gpu_seconds += command_.GPUEndTime - command_.GPUStartTime;
    ++flushes;
    NSError* error = command_.error;
    for (id<MTLCommandBuffer> earlier : pending_) {  // the queue runs them in order: all done
      gpu_seconds += earlier.GPUEndTime - earlier.GPUStartTime;
      if (error == nil) error = earlier.error;
    }
    pending_.clear();
    encoder_ = nil;
    command_ = nil;
    if (error != nil) fail(std::string("Metal command failed: ") + error.localizedDescription.UTF8String);
  }

 private:
  // The GPU architecture generation, e.g. 14 for "applegpu_g14s" (M2 Max); 0 when unknown.
  int architecture_generation() const {
    const char* name = device.architecture.name.UTF8String;
    const char* at = name != nullptr ? strstr(name, "_g") : nullptr;
    return at != nullptr ? atoi(at + 2) : 0;
  }

  // TensorOps need Metal 4 (macOS 26). They are used on generation 17 (M5) and later,
  // where they reach the Neural Accelerators; WHALLM_QWEN_TENSOROPS=1 or 0 overrides.
  bool open_tensor_ops() {
    const char* force = getenv("WHALLM_QWEN_TENSOROPS");
    bool wanted = force != nullptr ? strcmp(force, "0") != 0 : architecture_generation() >= 17;
    if (!wanted || !simdgroup_matrices) return false;
    if (@available(macOS 26.0, *)) {
      MTLCompileOptions* options = [MTLCompileOptions new];
      options.languageVersion = MTLLanguageVersion4_0;
      NSError* error = nil;
      tensor_library_ = [device newLibraryWithSource:[NSString stringWithUTF8String:kWhallmQwenTensorKernels]
                                             options:options error:&error];
      if (tensor_library_ == nil && force != nullptr)
        fprintf(stderr, "TensorOps kernels did not compile: %s\n", error.localizedDescription.UTF8String);
      return tensor_library_ != nil;
    }
    return false;
  }

  bool probe() {
    if (getenv("WHALLM_QWEN_NO_SIMDGROUP") != nullptr) return false;
    Buf out = alloc(64 * 4);
    struct { uint32_t unused; } none{0};
    run("sg_probe", {out}, none, 32, 1, 1, MTLSizeMake(32, 1, 1));
    flush();
    const float* m = (const float*)out.buffer.contents;
    for (int lane = 0; lane < 32; ++lane) {
      int quadrant = lane / 8, local = (lane % 8) * 2;
      int row = (quadrant / 2) * 4 + local / 4, column = (quadrant % 2) * 4 + local % 4;
      if (m[row * 8 + column] != lane * 2 || m[row * 8 + column + 1] != lane * 2 + 1) return false;
    }
    return true;
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

  bool profile_ = getenv("WHALLM_QWEN_KERNEL_PROFILE") != nullptr;
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
  explicit File(const std::string& path) {
    fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail("cannot open " + path);
    struct stat info;
    fstat(fd, &info);
    size = (size_t)info.st_size;
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
  // Read a range with several threads.
  void read_parallel(void* destination, size_t length, size_t offset, int threads) const {
    const size_t piece = 32u << 20;
    size_t pieces = (length + piece - 1) / piece;
    const File* self = this;
    __block bool failed = false;
    dispatch_queue_t queue = dispatch_queue_create("whallm.read", dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0));
    dispatch_semaphore_t limit = dispatch_semaphore_create(threads);
    dispatch_group_t group = dispatch_group_create();
    for (size_t index = 0; index < pieces; ++index) {
      dispatch_semaphore_wait(limit, DISPATCH_TIME_FOREVER);
      dispatch_group_async(group, queue, ^{
        size_t start = index * piece;
        try { self->read((char*)destination + start, std::min(piece, length - start), offset + start); }
        catch (...) { failed = true; }
        dispatch_semaphore_signal(limit);
      });
    }
    dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    if (failed) fail("short read from model file");
  }
};

struct Tensor { size_t offset = 0, length = 0; std::string dtype; std::vector<int> shape; };

// A weight matrix [O, I] for linear(): BF16, or 8-bit with a (scale, bias) pair per
// 64 inputs (w = q * scale + bias).
struct Mat { Buf w, sb; bool q8 = false; int O = 0, I = 0; };
constexpr int Q8_GROUP = 64;
// 8-bit weights apply to projections with at least this many outputs, as in the MLX engine.
constexpr int Q8_MIN_ROWS = 256;

// One tensor file. BF16 mode holds the whole file in one Metal buffer. 8-bit mode
// stores each eligible matrix as 8-bit and copies only the other matrices; the
// file stays mapped for vectors and the embedding rows, which are read on the CPU.
class Weights {
 public:
  Weights(Gpu& gpu, const std::string& path, NSArray* tensors, int threads, bool eight_bit)
      : gpu_(gpu), eight_bit_(eight_bit) {
    File file(path);
    size_ = file.size;
    void* mapped = mmap(nullptr, size_, PROT_READ, MAP_SHARED, file.fd, 0);
    if (mapped == MAP_FAILED) fail("cannot map " + path);
    base_ = (const char*)mapped;
    if (!eight_bit) {
      data_ = gpu.alloc(size_);
      if (data_.buffer.length < size_) fail("Metal buffer is smaller than " + path);
      file.read_parallel(data_.buffer.contents, size_, 0, threads);
    }
    for (NSDictionary* item in tensors) {
      Tensor tensor;
      tensor.offset = [item[@"offset"] unsignedLongLongValue];
      tensor.length = [item[@"length"] unsignedLongLongValue];
      tensor.dtype = [item[@"dtype"] UTF8String];
      for (NSNumber* dimension in item[@"shape"]) tensor.shape.push_back(dimension.intValue);
      if (tensor.offset + tensor.length > size_) fail("tensor outside " + path);
      tensors_[[item[@"name"] UTF8String]] = tensor;
    }
  }
  ~Weights() {
    munmap((void*)base_, size_);
    gpu_.release(data_);
    gpu_.release(vectors_);
  }
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  const Tensor& tensor(const std::string& name) const {
    auto found = tensors_.find(name);
    if (found == tensors_.end()) fail("model has no tensor " + name);
    return found->second;
  }
  // BF16 matrix [O, I]. In 8-bit mode, matrices with at least 256 outputs are
  // stored as 8-bit unless `keep_bf16` (the routers).
  Mat matrix(const std::string& name, size_t O, size_t I, bool keep_bf16 = false) {
    const Tensor& t = tensor(name);
    if (t.dtype != "BF16" || t.length != O * I * 2) fail("unexpected shape for " + name);
    if (!eight_bit_) return {data_.at(t.offset), {}, false, (int)O, (int)I};
    if (O < (size_t)Q8_MIN_ROWS || keep_bf16 || I % Q8_GROUP != 0) {
      Buf copy = gpu_.alloc(t.length);
      memcpy(copy.buffer.contents, base_ + t.offset, t.length);
      return {copy, {}, false, (int)O, (int)I};
    }
    Mat out{gpu_.alloc(O * I), gpu_.alloc(O * (I / Q8_GROUP) * 8), true, (int)O, (int)I};
    const uint16_t* source = (const uint16_t*)(base_ + t.offset);
    uint8_t* q = (uint8_t*)out.w.buffer.contents;
    float* sb = (float*)out.sb.buffer.contents;
    size_t groups = I / Q8_GROUP;
    dispatch_apply((O + 63) / 64, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t block) {
      for (size_t row = block * 64; row < std::min(O, block * 64 + 64); ++row)
        for (size_t group = 0; group < groups; ++group) {
          const uint16_t* in = source + row * I + group * Q8_GROUP;
          float values[Q8_GROUP], low = INFINITY, high = -INFINITY;
          for (int i = 0; i < Q8_GROUP; ++i) {
            uint32_t bits = (uint32_t)in[i] << 16;
            memcpy(&values[i], &bits, 4);
            low = std::min(low, values[i]);
            high = std::max(high, values[i]);
          }
          float scale = high > low ? (high - low) / 255.0f : 1.0f;
          uint8_t* qo = q + row * I + group * Q8_GROUP;
          for (int i = 0; i < Q8_GROUP; ++i) qo[i] = (uint8_t)std::lround(std::min(255.0f, (values[i] - low) / scale));
          sb[(row * groups + group) * 2] = scale;
          sb[(row * groups + group) * 2 + 1] = low;
        }
    });
    return out;
  }
  // BF16 tensor used as raw bytes by a kernel (convolution weights).
  Buf raw(const std::string& name, size_t elements) {
    const Tensor& t = tensor(name);
    if (t.dtype != "BF16" || t.length != elements * 2) fail("unexpected shape for " + name);
    if (!eight_bit_) return data_.at(t.offset);
    Buf copy = gpu_.alloc(t.length);
    memcpy(copy.buffer.contents, base_ + t.offset, t.length);
    return copy;
  }
  // Small BF16 vector widened to FP32; returns its float index in vectors().
  size_t vector(const std::string& name, size_t elements, float add) {
    const Tensor& t = tensor(name);
    if (t.dtype != "BF16" || t.length != elements * 2) fail("unexpected shape for " + name);
    const uint16_t* source = (const uint16_t*)(base_ + t.offset);
    size_t start = floats_.size();
    for (size_t i = 0; i < elements; ++i) {
      uint32_t bits = (uint32_t)source[i] << 16;
      float value;
      memcpy(&value, &bits, 4);
      floats_.push_back(value + add);
    }
    return start;
  }
  // Append one float after the last vector (padding).
  void push(float value) { floats_.push_back(value); }
  // Mapped file bytes; valid for the lifetime of this object.
  const void* bytes(const std::string& name) const { return base_ + tensor(name).offset; }
  // Call once after every vector() call.
  void seal() {
    vectors_ = gpu_.alloc(floats_.size() * 4);
    memcpy(vectors_.buffer.contents, floats_.data(), floats_.size() * 4);
  }
  Buf vec(size_t index) const { return vectors_.at(index * 4); }
  float scalar(size_t index) const { return floats_[index]; }

 private:
  Gpu& gpu_;
  bool eight_bit_;
  const char* base_ = nullptr;
  size_t size_ = 0;
  Buf data_, vectors_;
  std::vector<float> floats_;
  std::unordered_map<std::string, Tensor> tensors_;
};

struct HyperW { size_t norm; Mat down, up, inject; bool combine; };
struct LayerW {
  bool linear = false, ple = false;
  HyperW attention, feed_forward;
  // linear attention
  Mat qkv, z, a, b, out;
  Buf conv;
  size_t a_log, dt_bias, gdn_norm;
  // QSA
  Mat q, k, v, o, index;
  size_t q_norm, k_norm, index_q_norm, index_k_norm;
  // MoE
  Mat router, shared_gate, shared_gate_proj, shared_up, shared_down;
  // PLE
  Mat ple_key, ple_value;
  Buf ple_conv;
  size_t ple_norm_key, ple_norm_query, ple_norm_conv;
};

// Every matrix of a layer, for passes over all of them.
std::vector<Mat*> matrices(LayerW& lw) {
  return {&lw.attention.down, &lw.attention.up, &lw.attention.inject, &lw.feed_forward.down, &lw.feed_forward.up,
          &lw.feed_forward.inject, &lw.qkv, &lw.z, &lw.a, &lw.b, &lw.out, &lw.q, &lw.k, &lw.v, &lw.o, &lw.index,
          &lw.router, &lw.shared_gate, &lw.shared_gate_proj, &lw.shared_up, &lw.shared_down, &lw.ple_key, &lw.ple_value};
}

HyperW hyper(Weights& w, const std::string& prefix, bool combine) {
  HyperW out;
  out.norm = w.vector(prefix + ".hc_norm.weight", W, 1.0f);
  out.down = w.matrix(prefix + ".input_mix_weight_down.weight", HC_RANK, W);
  out.up = w.matrix(prefix + ".input_mix_weight_up.weight", W, HC_RANK);
  out.combine = combine;
  if (combine) out.inject = w.matrix(prefix + ".block_inject_weight.weight", C, W);
  return out;
}

LayerW layer_weights(Weights& w, const std::string& p, bool linear, bool ple) {
  LayerW out;
  out.linear = linear;
  out.ple = ple;
  out.attention = hyper(w, p + ".attn_hyper_connection", true);
  out.feed_forward = hyper(w, p + ".mlp_hyper_connection", true);
  if (linear) {
    out.qkv = w.matrix(p + ".linear_attn.in_proj_qkv.weight", W, H);
    out.z = w.matrix(p + ".linear_attn.in_proj_z.weight", GDN_VALUE, H);
    out.a = w.matrix(p + ".linear_attn.in_proj_a.weight", GDN_HV, H);
    out.b = w.matrix(p + ".linear_attn.in_proj_b.weight", GDN_HV, H);
    out.conv = w.raw(p + ".linear_attn.conv1d.weight", (size_t)W * 4);
    out.out = w.matrix(p + ".linear_attn.out_proj.weight", H, GDN_VALUE);
    out.a_log = w.vector(p + ".linear_attn.A_log", GDN_HV, 0);
    out.dt_bias = w.vector(p + ".linear_attn.dt_bias", GDN_HV, 0);
    out.gdn_norm = w.vector(p + ".linear_attn.norm.weight", GDN_D, 0);
  } else {
    out.q = w.matrix(p + ".self_attn.q_proj.weight", HEADS * HEAD_DIM * 2, H);
    out.k = w.matrix(p + ".self_attn.k_proj.weight", KV_HEADS * HEAD_DIM, H);
    out.v = w.matrix(p + ".self_attn.v_proj.weight", KV_HEADS * HEAD_DIM, H);
    out.o = w.matrix(p + ".self_attn.o_proj.weight", H, HEADS * HEAD_DIM);
    out.index = w.matrix(p + ".self_attn.indexer.index_qk_proj.weight", (INDEX_HEADS + 1) * INDEX_DIM, H);
    out.q_norm = w.vector(p + ".self_attn.q_norm.weight", HEAD_DIM, 1.0f);
    out.k_norm = w.vector(p + ".self_attn.k_norm.weight", HEAD_DIM, 1.0f);
    out.index_q_norm = w.vector(p + ".self_attn.indexer.q_layernorm.weight", INDEX_DIM, 1.0f);
    out.index_k_norm = w.vector(p + ".self_attn.indexer.k_layernorm.weight", INDEX_DIM, 1.0f);
  }
  out.router = w.matrix(p + ".mlp.gate.weight", EXPERTS, H, true);
  out.shared_gate = w.matrix(p + ".mlp.shared_expert_gate.weight", 1, H);
  out.shared_gate_proj = w.matrix(p + ".mlp.shared_expert.gate_proj.weight", MOE_M, H);
  out.shared_up = w.matrix(p + ".mlp.shared_expert.up_proj.weight", MOE_M, H);
  out.shared_down = w.matrix(p + ".mlp.shared_expert.down_proj.weight", H, MOE_M);
  if (ple) {
    out.ple_key = w.matrix(p + ".ple.key_proj.weight", W, H);
    out.ple_value = w.matrix(p + ".ple.value_proj.weight", H, H);
    out.ple_conv = w.raw(p + ".ple.conv1d.weight", (size_t)W * 4);
    out.ple_norm_key = w.vector(p + ".ple.norm_key.weight", W, 1.0f);
    out.ple_norm_query = w.vector(p + ".ple.norm_query.weight", W, 1.0f);
    out.ple_norm_conv = w.vector(p + ".ple.norm_conv.weight", W, 1.0f);
  }
  return out;
}

// ------------------------------------------------------------------ expert arena
// Keeps recently used expert blobs in one Metal buffer and reads the rest from SSD.
class ExpertArena {
 public:
  ExpertArena(Gpu& gpu, int slots) : slots_(slots) {
    if (slots < TOP_K) fail("the expert cache needs at least 10 slots");
    buffer_ = gpu.alloc((size_t)slots * BLOB);
    owner_.assign(slots, -1);
    stamp_.assign(slots, 0);
    uses_.assign(slots, 0);
  }
  int add_file(const std::string& path) {
    files_.emplace_back(path);
    if (files_.back().size != (size_t)EXPERTS * BLOB) fail("unexpected expert file size: " + path);
    return (int)files_.size() - 1;
  }
  Buf buffer() const { return buffer_; }
  const File& file(int index) const { return files_[index]; }

  // Slots of these experts of one file, reading the missing ones. Everything
  // returned stays valid until the next acquire().
  void acquire(int file, const std::vector<int>& experts, std::vector<uint32_t>& slots) {
    std::vector<std::pair<int, int>> missing;
    plan(file, experts, slots, missing);
    load(file, missing);
  }

  // The slots of these experts of one file. Missing ones get a slot and are listed as
  // (expert, slot) in `missing`, to be filled by load(). Slots stay reserved until the
  // next plan() or acquire().
  void plan(int file, const std::vector<int>& experts, std::vector<uint32_t>& slots,
            std::vector<std::pair<int, int>>& missing) {
    wait();
    ++clock_;
    missing.clear();
    slots.resize(experts.size());
    for (size_t i = 0; i < experts.size(); ++i) {
      long key = (long)file * EXPERTS + experts[i];
      auto found = where_.find(key);
      touch();
      if (found != where_.end()) {
        slots[i] = (uint32_t)found->second;
        stamp_[found->second] = clock_;
        ++uses_[found->second];
        ++hits;
        continue;
      }
      int slot = victim();
      if (owner_[slot] >= 0) where_.erase(owner_[slot]);
      owner_[slot] = key;
      where_[key] = slot;
      stamp_[slot] = clock_;
      uses_[slot] = 1;
      slots[i] = (uint32_t)slot;
      missing.emplace_back(experts[i], slot);
      ++misses;
    }
  }

  // Read the experts plan() listed into their slots.
  void load(int file, const std::vector<std::pair<int, int>>& missing) {
    if (missing.empty()) return;
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    const File* source = &files_[file];
    char* base = (char*)buffer_.buffer.contents;
    const std::pair<int, int>* items = missing.data();
    __block bool failed = false;
    dispatch_apply(missing.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t index) {
      try { source->read(base + (size_t)items[index].second * BLOB, BLOB, (size_t)items[index].first * BLOB); }
      catch (...) { failed = true; }
    });
    read_seconds += CFAbsoluteTimeGetCurrent() - started;
    if (failed) fail("could not read an expert from SSD");
  }

  // Copy experts of one file from `source`, a whole expert file held in memory,
  // into the cache. uses[i] is how often the prompt used experts[i]. The copy
  // runs in the background: `source` must stay unchanged until wait() returns.
  void seed(int file, const std::vector<int>& experts, const std::vector<int>& uses, const char* source) {
    wait();
    ++clock_;
    char* base = (char*)buffer_.buffer.contents;
    for (size_t i = 0; i < experts.size(); ++i) {
      int expert = experts[i];
      long key = (long)file * EXPERTS + expert;
      auto found = where_.find(key);
      if (found != where_.end()) {
        stamp_[found->second] = clock_;
        uses_[found->second] += uses[i];
        continue;
      }
      int slot = victim();
      if (owner_[slot] >= 0) where_.erase(owner_[slot]);
      owner_[slot] = key;
      where_[key] = slot;
      stamp_[slot] = clock_;
      uses_[slot] = uses[i];
      ++seeded;
      dispatch_group_async(copies_, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
        memcpy(base + (size_t)slot * BLOB, source + (size_t)expert * BLOB, BLOB);
      });
    }
  }
  void wait() { dispatch_group_wait(copies_, DISPATCH_TIME_FOREVER); }
  int slots() const { return slots_; }
  int resident() const { return used_; }
  int files() const { return (int)files_.size(); }
  ~ExpertArena() { wait(); }

  long hits = 0, misses = 0, seeded = 0;
  double read_seconds = 0;

 private:
  // The least used slot, oldest first among equals, that the current call has not handed out.
  int victim() {
    if (used_ < slots_) return used_++;
    int best = -1;
    for (int slot = 0; slot < slots_; ++slot) {
      if (stamp_[slot] == clock_) continue;
      if (best < 0 || uses_[slot] < uses_[best] || (uses_[slot] == uses_[best] && stamp_[slot] < stamp_[best])) best = slot;
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
  long clock_ = 0;
  dispatch_group_t copies_ = dispatch_group_create();
  Buf buffer_;
  std::vector<File> files_;
  std::vector<long> owner_, stamp_, uses_;
  long touches_ = 0;
  std::unordered_map<long, int> where_;
};

// ------------------------------------------------------------------ N-gram rows
class NGramStore {
 public:
  NGramStore(const std::string& path, NSDictionary* descriptor, float scale, const int64_t* multipliers) : file_(path) {
    NSArray* offsets = descriptor[@"headOffsets"];
    NSArray* sizes = descriptor[@"headVocabSizes"];
    if (offsets.count != NGRAM_HEADS || sizes.count != NGRAM_HEADS || [descriptor[@"rowBytes"] intValue] != NGRAM_ROW)
      fail("unexpected N-gram layout");
    for (int i = 0; i < NGRAM_HEADS; ++i) {
      offsets_[i] = [offsets[i] longLongValue];
      sizes_[i] = [sizes[i] longLongValue];
    }
    rows_ = [descriptor[@"shardCount"] longLongValue] * [descriptor[@"shardRowCount"] longLongValue];
    if (file_.size != (size_t)rows_ * NGRAM_ROW) fail("N-gram file size does not match the manifest");
    memcpy(multipliers_, multipliers, sizeof(multipliers_));
    for (int code = 0; code < 256; ++code) {
      int exponent = (code >> 3) & 15, mantissa = code & 7;
      float value = exponent == 0 ? ldexpf((float)mantissa, -9) : ldexpf(1.0f + mantissa / 8.0f, exponent - 7);
      table_[code] = ((code & 0x80) ? -value : value) * scale;
    }
  }

  // history holds the two tokens before tokens[0]. Writes [count, 2560] floats.
  void lookup(const int32_t history[2], const int32_t* tokens, int count, float* out) const {
    const NGramStore* self = this;
    __block bool failed = false;
    dispatch_apply((size_t)count * NGRAM_HEADS, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t item) {
      int t = (int)(item / NGRAM_HEADS), head = (int)(item % NGRAM_HEADS);
      auto token = [&](int index) -> int64_t { return index >= 0 ? tokens[index] : history[2 + index]; };
      // Shifts never cross an EOS: an earlier token is replaced by EOS when an EOS lies between.
      int64_t t0 = token(t), t1 = token(t - 1), t2 = token(t - 2);
      if (t2 == EOS || t1 == EOS) t2 = EOS;
      uint64_t mixed = (uint64_t)t0 * (uint64_t)self->multipliers_[0] ^ (uint64_t)t1 * (uint64_t)self->multipliers_[1];
      if (head >= 8) mixed ^= (uint64_t)t2 * (uint64_t)self->multipliers_[2];
      int64_t remainder = (int64_t)mixed % self->sizes_[head];
      if (remainder < 0) remainder += self->sizes_[head];
      int64_t row = remainder + self->offsets_[head];
      uint8_t bytes[NGRAM_ROW];
      try { self->file_.read(bytes, NGRAM_ROW, (size_t)row * NGRAM_ROW); }
      catch (...) { failed = true; return; }
      float* destination = out + (size_t)t * H + head * NGRAM_ROW;
      for (int i = 0; i < NGRAM_ROW; ++i) destination[i] = self->table_[bytes[i]];
    });
    if (failed) fail("could not read N-gram rows");
  }

 private:
  File file_;
  int64_t offsets_[NGRAM_HEADS], sizes_[NGRAM_HEADS], multipliers_[3], rows_ = 0;
  float table_[256];
};

// ------------------------------------------------------------------ per-layer state
// Recurrent state is kept twice: a call reads `current` and writes the other
// copy, so the state before the latest call stays available for commit().
struct LinearState { Buf conv[2], state[2]; int current = 0; };
struct AttentionState { Buf keys, values, raw, pooled; int count = 0, pooled_blocks = 0, capacity = 0; };
// An image prompt: each prompt index's position on the time, height and width
// rotary axes, and the encoder features that replace the image tokens' embeddings.
// Indices past the prompt continue at index + delta on all three axes.
struct ImagePrompt {
  Buf map;               // int4 (t, h, w, 0) per prompt index
  uint32_t map_len = 0;  // 0: a text prompt
  int32_t delta = 0;
  std::vector<std::pair<int, int>> spans;  // (first index, token count) per image
  std::vector<float> features;             // [sum of counts, H]
  const float* feature(int index) const {
    size_t row = 0;
    for (const auto& span : spans) {
      if (index >= span.first && index < span.first + span.second) return features.data() + (row + index - span.first) * H;
      row += span.second;
    }
    return nullptr;
  }
};

// What one linear-attention layer needs to redo its state update for the first
// rows of a verification: raw and convolved qkv rows and the two gates.
struct LinearLog { Buf raw, rows, decay, beta; };
constexpr int VERIFY_ROWS = 8;

// ------------------------------------------------------------------ engine
class EngineImpl : public Engine {
 public:
  EngineImpl(const std::string& root, const Options& options) : options_(options) {
    NSDictionary* manifest = json(root + "/manifest.json");
    NSDictionary* config = json(root + "/config.json")[@"text_config"];
    check(config, manifest);
    contract_ = std::string("wq-state-1|") + [manifest[@"modelID"] UTF8String] + "@" + [manifest[@"revision"] UTF8String] +
                "|8bit=" + (options.eight_bit ? "1" : "0") +
                "|mtp=" + (options.mtp && [manifest[@"mtp"] isKindOfClass:[NSDictionary class]] ? "1" : "0");
    common_.reset(new Weights(gpu_, root + "/common.bin", manifest[@"commonTensors"], options.read_threads, options.eight_bit));
    const std::string base = "model.language_model.";
    NSArray* types = config[@"layer_types"];
    for (int layer = 0; layer < LAYERS; ++layer) {
      bool linear = [types[layer] isEqualToString:@"linear_attention"];
      layers_.push_back(layer_weights(*common_, base + "layers." + std::to_string(layer), linear, layer == 1));
    }
    mixer_ = hyper(*common_, base + "hyper_connection_mixer", false);
    embed_ = (const uint16_t*)common_->bytes(base + "embed_tokens.weight");
    lm_head_ = common_->matrix("lm_head.weight", VOCAB, H);
    size_t scale = common_->vector(base + "layers.1.ple.ple_embedding.ngram_embedding.weight_scale", 1, 0);
    int64_t multipliers[3];
    memcpy(multipliers, common_->bytes(base + "layers.1.ple.ple_embedding.layer_multipliers"), sizeof(multipliers));
    common_->seal();
    ngram_.reset(new NGramStore(root + "/ngram.bin", manifest[@"ngram"], common_->scalar(scale), multipliers));

    slots_ = options.slots > 0 ? options.slots : automatic_slots(gpu_.allocated_bytes);
    arena_.reset(new ExpertArena(gpu_, slots_));
    for (int layer = 0; layer < LAYERS; ++layer) {
      char name[64];
      snprintf(name, sizeof(name), "/experts/layer_%02d.bin", layer);
      arena_->add_file(root + name);
    }
    for (int layer = 0; layer < LAYERS; ++layer) {
      linear_.emplace_back();
      attention_.emplace_back();
      logs_.emplace_back();
      if (!layers_[layer].linear) continue;
      for (int copy = 0; copy < 2; ++copy) {
        linear_[layer].conv[copy] = gpu_.alloc((size_t)CONV_STATE * W * 4);
        linear_[layer].state[copy] = gpu_.alloc((size_t)GDN_HV * GDN_D * GDN_D * 4);
      }
      logs_[layer] = {gpu_.alloc((size_t)VERIFY_ROWS * W * 4), gpu_.alloc((size_t)VERIFY_ROWS * W * 4),
                      gpu_.alloc((size_t)VERIFY_ROWS * GDN_HV * 4), gpu_.alloc((size_t)VERIFY_ROWS * GDN_HV * 4)};
    }
    image_.map = gpu_.alloc(16);
    root_ = root;
    ple_conv_[0] = gpu_.alloc((size_t)PLE_STATE * W * 4);
    ple_conv_[1] = gpu_.alloc((size_t)PLE_STATE * W * 4);
    ple_log_ = gpu_.alloc((size_t)VERIFY_ROWS * W * 4);
    logits_ = gpu_.alloc((size_t)16 * VOCAB * 4);  // tiled products write whole 16-row tiles
    partial_ = gpu_.alloc((size_t)VERIFY_ROWS * 2048 * 8 * 4);  // split products: splits * rows * outputs
    boundary_.assign(W, 0.0f);

    NSDictionary* mtp = manifest[@"mtp"];
    if (options.mtp && [mtp isKindOfClass:[NSDictionary class]]) {
      mtp_.reset(new Weights(gpu_, root + "/mtp/common.bin", mtp[@"commonTensors"], options.read_threads, options.eight_bit));
      mtp_layer_ = layer_weights(*mtp_, "mtp.layers.0", false, false);
      mtp_mixer_ = hyper(*mtp_, "mtp.hyper_connection_mixer", false);
      mtp_fc_embedding_ = mtp_->matrix("mtp.fc_embedding.weight", H, H);
      mtp_fc_hidden_ = mtp_->matrix("mtp.fc_hidden.weight", H, H);
      mtp_norm_embedding_ = mtp_->vector("mtp.pre_fc_norm_embedding.weight", H, 1.0f);
      mtp_norm_hidden_ = mtp_->vector("mtp.pre_fc_norm_hidden.weight", W, 1.0f);
      mtp_->seal();
      mtp_file_ = arena_->add_file(root + "/mtp/experts/layer_00.bin");
    }
    reset();
    gpu_.settle();
  }

  ~EngineImpl() override {
    if (layer_read_ != nil) dispatch_group_wait(layer_read_, DISPATCH_TIME_FOREVER);
  }

  void reset() override {
    gpu_.flush();
    for (int layer = 0; layer < LAYERS; ++layer) {
      if (layers_[layer].linear) {
        memset(linear_[layer].conv[0].buffer.contents, 0, linear_[layer].conv[0].buffer.length);
        memset(linear_[layer].state[0].buffer.contents, 0, linear_[layer].state[0].buffer.length);
        linear_[layer].current = 0;
      } else {
        attention_[layer].count = 0;
        attention_[layer].pooled_blocks = 0;
      }
    }
    memset(ple_conv_[0].buffer.contents, 0, ple_conv_[0].buffer.length);
    ple_current_ = 0;
    history_[0] = history_[1] = EOS;
    position_ = 0;
    tokens_.clear();
    mtp_attention_.count = mtp_attention_.pooled_blocks = 0;
    image_.map_len = 0;
    image_.delta = 0;
    image_.spans.clear();
    image_.features.clear();
    pending_hidden_.clear();
    pending_tokens_.clear();
    drafted_ = 0;
    verified_.clear();
  }

  int position() const override { return position_; }
  std::string stats() const override {
    char text[256];
    snprintf(text, sizeof(text), "slots %d %swait %.3fs gpu %.3fs flushes %ld dispatches %ld expert hits %ld misses %ld seeded %ld read %.3fs",
             slots_, gpu_.tensor_ops ? "tensorops " : "", gpu_.wait_seconds, gpu_.gpu_seconds, gpu_.flushes, gpu_.dispatches, arena_->hits, arena_->misses, arena_->seeded, arena_->read_seconds);
    return text + gpu_.profile();
  }
  int vocab() const override { return VOCAB; }

  void set_cancel(std::function<bool()> check) override { cancel_ = std::move(check); }
  const std::string& contract() const override { return contract_; }
  Counters counters() const override {
    Counters c;
    c.expert_hits = arena_->hits;
    c.expert_misses = arena_->misses;
    c.experts_seeded = arena_->seeded;
    c.read_seconds = arena_->read_seconds;
    c.gpu_seconds = gpu_.gpu_seconds;
    c.gpu_bytes = gpu_.allocated_bytes;
    c.slots = slots_;
    c.resident = arena_->resident();
    c.tensor_ops = gpu_.tensor_ops;
    return c;
  }

  const float* eval(const int32_t* tokens, int count, int logit_rows) override {
    if (drafted_ != 0 || !verified_.empty()) fail("finish the draft round with verify and commit first");
    // A long call feeds the draft layer right away: start reading its experts early.
    forward(tokens, count, logit_rows, false, mtp_ != nullptr && count > WHOLE_LAYER_ROWS + 1 ? mtp_file_ : -1);
    // The draft layer follows the model one token behind: pair i is (hidden i, token i + 1).
    if (mtp_ != nullptr) {
      int first = position_ > count ? 0 : 1;  // row 0 is the hidden before this call
      int pairs = count + 1 - first - 1;
      if (pairs > PENDING_PAIRS) {
        catch_up();
        mtp_rows(all_.at((size_t)first * W * 4), tokens + first, pairs);
        gpu_.flush();
      } else if (pairs > 0) {
        const float* rows = (const float*)all_.buffer.contents + (size_t)first * W;
        pending_hidden_.insert(pending_hidden_.end(), rows, rows + (size_t)pairs * W);
        pending_tokens_.insert(pending_tokens_.end(), tokens + first, tokens + first + pairs);
      }
    }
    keep_boundary(count);
    return (const float*)logits_.buffer.contents;
  }

  bool has_mtp() const override { return mtp_ != nullptr; }

  std::vector<float> encode_image(const float* patches, int grid_h, int grid_w) override {
    if (grid_h < 2 || grid_w < 2 || grid_h % 2 || grid_w % 2 || grid_h * grid_w * 256 > 1048576)
      fail("unsupported image grid");
    gpu_.flush();
    return VisionRun(*this).encode(patches, grid_h, grid_w);
  }

  const float* eval_images(const int32_t* tokens, int count, const std::vector<ImageInput>& images, int logit_rows) override {
    if (position_ != 0 || drafted_ != 0 || !verified_.empty()) fail("an image prompt must start a new request (reset first)");
    if (images.empty()) fail("an image prompt needs at least one image");
    // Find each image's span: vision start, h * w / 4 image tokens, vision end.
    std::vector<std::pair<int, int>> spans;
    for (int i = 0; i < count; ++i) {
      if (tokens[i] != VISION_START) continue;
      int first = i + 1, length = 0;
      while (first + length < count && tokens[first + length] == IMAGE_TOKEN) ++length;
      if (first + length >= count || tokens[first + length] != VISION_END) fail("an image's tokens are not closed by its end token");
      spans.emplace_back(first, length);
      i = first + length;
    }
    if (spans.size() != images.size()) fail("the prompt and the images do not match in number");
    size_t merged = 0;
    for (size_t k = 0; k < images.size(); ++k) {
      const ImageInput& image = images[k];
      bool sized = image.features.empty() ? image.patches.size() == (size_t)image.grid_h * image.grid_w * V_PATCH
                                          : image.features.size() == (size_t)spans[k].second * H;
      if (spans[k].second != image.grid_h * image.grid_w / 4 || !sized) fail("an image's size does not match its tokens");
      merged += spans[k].second;
    }
    if (merged > 2048) fail("images exceed the 2048 image-token limit");
    // Positions (interleaved T/H/W): text advances one step per token; an image
    // takes (0, row, column) of its merged grid and advances by its longer side.
    std::vector<int32_t> map((size_t)count * 4, 0);
    int end = 0, current = 0, top = 0;
    auto text = [&](int from, int to) {
      for (int i = from; i < to; ++i) {
        int position = current + (i - from);
        map[(size_t)i * 4] = map[(size_t)i * 4 + 1] = map[(size_t)i * 4 + 2] = position;
        top = std::max(top, position);
      }
      current += to - from;
    };
    for (size_t k = 0; k < spans.size(); ++k) {
      int h = images[k].grid_h / 2, w = images[k].grid_w / 2;
      text(end, spans[k].first);
      for (int i = 0; i < spans[k].second; ++i) {
        int32_t* p = &map[(size_t)(spans[k].first + i) * 4];
        p[0] = current;
        p[1] = current + i / w;
        p[2] = current + i % w;
        top = std::max(top, std::max(p[1], p[2]));
      }
      current += std::max(h, w);
      end = spans[k].first + spans[k].second;
    }
    text(end, count);
    // Encode first: a failure leaves the engine as it was.
    std::vector<float> features;
    for (const ImageInput& image : images) {
      std::vector<float> one = image.features.empty() ? encode_image(image.patches.data(), image.grid_h, image.grid_w)
                                                      : image.features;
      for (float value : one)
        if (!std::isfinite(value)) fail("the image encoder produced invalid values");
      features.insert(features.end(), one.begin(), one.end());
    }
    gpu_.replace(image_.map, map.size() * 4);
    memcpy(image_.map.buffer.contents, map.data(), map.size() * 4);
    image_.map_len = (uint32_t)count;
    image_.delta = top + 1 - count;
    image_.spans = spans;
    image_.features = std::move(features);
    return eval(tokens, count, logit_rows);
  }

  const std::vector<int32_t>& tokens() const override { return tokens_; }

  std::shared_ptr<const State> snapshot() override {
    if (drafted_ != 0 || !verified_.empty()) fail("finish the draft round before taking a snapshot");
    // As in the MLX engine, image prompts bypass the prompt cache.
    if (image_.map_len > 0) fail("image prompts are not kept in the prompt cache");
    gpu_.flush();
    auto state = std::make_shared<State>();
    state->tokens = tokens_;
    std::vector<uint8_t>& out = state->data;
    auto put = [&](const void* data, size_t bytes) {
      out.insert(out.end(), (const uint8_t*)data, (const uint8_t*)data + bytes);
    };
    auto put_int = [&](int64_t value) { put(&value, sizeof(value)); };
    auto put_attention = [&](const AttentionState& a) {
      put_int(a.count);
      put_int(a.pooled_blocks);
      size_t kv = (size_t)a.count * KV_HEADS * HEAD_DIM * 2;
      if (a.count > 0) {
        put(a.keys.buffer.contents, kv);
        put(a.values.buffer.contents, kv);
        put(a.raw.buffer.contents, (size_t)a.count * INDEX_DIM * 2);
      }
      if (a.pooled_blocks > 0) put(a.pooled.buffer.contents, (size_t)a.pooled_blocks * INDEX_DIM * 4);
    };
    put_int(position_);
    put(history_, sizeof(history_));
    put(boundary_.data(), boundary_.size() * 4);
    for (int layer = 0; layer < LAYERS; ++layer) {
      if (layers_[layer].linear) {
        const LinearState& l = linear_[layer];
        put(l.conv[l.current].buffer.contents, (size_t)CONV_STATE * W * 4);
        put(l.state[l.current].buffer.contents, (size_t)GDN_HV * GDN_D * GDN_D * 4);
      } else {
        put_attention(attention_[layer]);
      }
    }
    put(ple_conv_[ple_current_].buffer.contents, (size_t)PLE_STATE * W * 4);
    put_int(mtp_ != nullptr ? 1 : 0);
    if (mtp_ != nullptr) {
      put_attention(mtp_attention_);
      put_int((int64_t)pending_tokens_.size());
      put(pending_tokens_.data(), pending_tokens_.size() * 4);
      put(pending_hidden_.data(), pending_hidden_.size() * 4);
    }
    return state;
  }

  void restore(const State& state) override {
    if (drafted_ != 0 || !verified_.empty()) fail("finish the draft round before restoring a snapshot");
    image_.map_len = 0;
    image_.delta = 0;
    image_.spans.clear();
    image_.features.clear();
    gpu_.flush();
    const uint8_t* at = state.data.data();
    const uint8_t* end = at + state.data.size();
    auto take = [&](void* data, size_t bytes) {
      if ((size_t)(end - at) < bytes) fail("the prompt cache state is incomplete");
      memcpy(data, at, bytes);
      at += bytes;
    };
    auto take_int = [&]() { int64_t value; take(&value, sizeof(value)); return value; };
    auto take_attention = [&](AttentionState& a) {
      int64_t count = take_int(), pooled = take_int();
      if (count < 0 || pooled < 0 || pooled > count / RATIO) fail("the prompt cache state is invalid");
      a.count = a.pooled_blocks = 0;
      grow(a, (int)count);
      size_t kv = (size_t)count * KV_HEADS * HEAD_DIM * 2;
      if (count > 0) {
        take(a.keys.buffer.contents, kv);
        take(a.values.buffer.contents, kv);
        take(a.raw.buffer.contents, (size_t)count * INDEX_DIM * 2);
      }
      if (pooled > 0) take(a.pooled.buffer.contents, (size_t)pooled * INDEX_DIM * 4);
      a.count = (int)count;
      a.pooled_blocks = (int)pooled;
    };
    int64_t position = take_int();
    if (position != (int64_t)state.tokens.size()) fail("the prompt cache state does not match its tokens");
    take(history_, sizeof(history_));
    take(boundary_.data(), boundary_.size() * 4);
    for (int layer = 0; layer < LAYERS; ++layer) {
      if (layers_[layer].linear) {
        LinearState& l = linear_[layer];
        take(l.conv[l.current].buffer.contents, (size_t)CONV_STATE * W * 4);
        take(l.state[l.current].buffer.contents, (size_t)GDN_HV * GDN_D * GDN_D * 4);
      } else {
        take_attention(attention_[layer]);
      }
    }
    take(ple_conv_[ple_current_].buffer.contents, (size_t)PLE_STATE * W * 4);
    if (take_int() != (mtp_ != nullptr ? 1 : 0)) fail("the prompt cache state was taken with another draft layer setting");
    if (mtp_ != nullptr) {
      take_attention(mtp_attention_);
      int64_t pairs = take_int();
      if (pairs < 0 || pairs > position) fail("the prompt cache state is invalid");
      pending_tokens_.resize(pairs);
      pending_hidden_.resize((size_t)pairs * W);
      take(pending_tokens_.data(), pending_tokens_.size() * 4);
      take(pending_hidden_.data(), pending_hidden_.size() * 4);
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
    if (memcmp(magic, kStateMagic, 8) != 0 || sizes[0] > 4096 || sizes[1] > (1u << 24) ||
        32 + sizes[0] + sizes[1] * 4 + sizes[2] != file.size)
      fail(path + " is not a prompt cache file");
    std::string contract(sizes[0], '\0');
    file.read(contract.data(), sizes[0], 32);
    if (contract != contract_) fail(path + " was written for another model or other settings");
    auto state = std::make_shared<State>();
    state->tokens.resize(sizes[1]);
    state->data.resize(sizes[2]);
    file.read(state->tokens.data(), sizes[1] * 4, 32 + sizes[0]);
    file.read(state->data.data(), sizes[2], 32 + sizes[0] + sizes[1] * 4);
    return state;
  }

  const float* draft(int32_t token) override {
    if (mtp_ == nullptr) fail("this model was opened without its draft layer");
    if (position_ < 1 || !verified_.empty()) fail("draft needs a consumed prompt and no open verification");
    if (token < 0 || token >= VOCAB) fail("token outside the vocabulary");
    if (drafted_ >= VERIFY_ROWS) fail("too many draft tokens in one round");
    gpu_.flush();
    if (drafted_ == 0) {
      catch_up();
      if (mtp_attention_.count != position_ - 1) fail("the draft layer is out of step with the model");
      mtp_checkpoint_ = mtp_attention_.count;
      if (mtp_in_.buffer == nil) gpu_.replace(mtp_in_, (size_t)16 * W * 4);
      memcpy(mtp_in_.buffer.contents, boundary_.data(), (size_t)W * 4);
      mtp_rows(mtp_in_, &token, 1);
    } else {
      mtp_rows(mtp_h_, &token, 1);  // the draft layer's own hidden state continues the chain
    }
    h_ = mtp_h_;
    hyper_mix(*mtp_, mtp_mixer_, 1);
    linear(s_.mixed, lm_head_, logits_, 1, H, VOCAB);
    gpu_.flush();
    ++drafted_;
    return (const float*)logits_.buffer.contents;
  }

  const float* verify(const int32_t* tokens, int count) override {
    if (!verified_.empty()) fail("commit the open verification first");
    if (count < 1 || count > VERIFY_ROWS) fail("verify takes 1 to 8 tokens");
    if (mtp_ != nullptr && drafted_ != 0 && drafted_ != count && drafted_ != count - 1)
      fail("verify must cover the anchor and the drafted tokens");
    verify_history_[0] = history_[0];
    verify_history_[1] = history_[1];
    forward(tokens, count, count, true);
    verified_.assign(tokens, tokens + count);
    return (const float*)logits_.buffer.contents;
  }

  void commit(int keep) override {
    int count = (int)verified_.size();
    if (count == 0 || keep < 1 || keep > count) fail("commit does not match the verification");
    gpu_.flush();
    int rejected = count - keep;
    if (rejected > 0) {
      for (int layer = 0; layer < LAYERS; ++layer) {
        if (layers_[layer].linear) {
          // Redo the state update for the kept rows, from the state before the verification.
          LinearState& state = linear_[layer];
          int before = 1 - state.current;
          ConvP conv{(uint32_t)keep, W, 4, 1, CONV_STATE, 1};
          gpu_.run("conv_state_next", {state.conv[before], logs_[layer].raw, state.conv[state.current]}, conv, W, CONV_STATE);
          GdnP p{(uint32_t)keep, GDN_HK, GDN_HV, GDN_D, GDN_D, W, GDN_KEY};
          gpu_.run("gdn_step", {logs_[layer].rows, logs_[layer].decay, logs_[layer].beta, state.state[before],
                                state.state[state.current], s_.y}, p, 32, GDN_D, GDN_HV, MTLSizeMake(32, 4, 1));
        } else {
          trim(attention_[layer], attention_[layer].count - rejected);
        }
      }
      ConvP ple{(uint32_t)keep, W, 4, 3, PLE_STATE, 1};
      gpu_.run("conv_state_next", {ple_conv_[1 - ple_current_], ple_log_, ple_conv_[ple_current_]}, ple, W, PLE_STATE);
      gpu_.flush();
      position_ -= rejected;
      tokens_.resize(position_);
      history_[0] = keep >= 2 ? verified_[keep - 2] : verify_history_[1];
      history_[1] = verified_[keep - 1];
    }
    keep_boundary(keep);
    if (mtp_ != nullptr && drafted_ != 0) {
      // The draft layer keeps one pair per kept token.
      if (keep <= drafted_) {
        trim(mtp_attention_, mtp_checkpoint_ + keep);
      } else {
        int32_t last = verified_.back();  // drafted but not yet consumed by the draft layer
        mtp_rows(mtp_h_, &last, 1);
        gpu_.flush();
      }
    } else if (mtp_ != nullptr) {
      // No draft ran: the kept tokens become pairs like any consumed token.
      int first = position_ > keep ? 0 : 1;
      int pairs = keep + 1 - first - 1;
      const float* rows = (const float*)all_.buffer.contents + (size_t)first * W;
      pending_hidden_.insert(pending_hidden_.end(), rows, rows + (size_t)pairs * W);
      pending_tokens_.insert(pending_tokens_.end(), verified_.begin() + first, verified_.begin() + first + pairs);
    }
    drafted_ = 0;
    verified_.clear();
  }

 private:
  void check_cancel() {
    if (cancel_ && cancel_()) throw Cancelled();
  }

  // Expert cache size when Options::slots is 0, following Strata's limits for streaming
  // Macs: the engine's whole footprint stays under 60% of the RAM, and at least 7 GB or
  // 15% of the RAM (whichever is more) of the memory macOS can give without swapping
  // stays free. `resident` is what the engine holds already; the two whole-layer buffers,
  // prompt scratch and attention state are counted on top (about 6 GB at 4,096 rows).
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
    double later = 6.0 * gib;
    double keep = std::max(7.0 * gib, 0.15 * (double)ram);
    double budget = std::min(0.6 * (double)ram - (double)resident - later, (double)available - keep - later);
    int slots = (int)std::max(0.0, budget / (double)BLOB);
    return std::max(MIN_AUTOMATIC_SLOTS, std::min(MAX_AUTOMATIC_SLOTS, slots));
  }

  // The image encoder of one encode_image() call. Its weights (about 0.9 GB) are
  // loaded for the call and released after it, as in the MLX engine.
  class VisionRun {
   public:
    explicit VisionRun(EngineImpl& engine) : e_(engine) {
      NSString* path = [NSString stringWithUTF8String:engine.options_.vision_catalog.c_str()];
      NSData* data = engine.options_.vision_catalog.empty() ? nil : [NSData dataWithContentsOfFile:path];
      NSDictionary* catalog = data == nil ? nil : [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
      if (![catalog isKindOfClass:[NSDictionary class]]) fail("image input needs the vision catalog (QwenVision.json)");
      std::string file = engine.root_ + "/vision/common.bin";
      struct stat info;
      if (stat(file.c_str(), &info) != 0) fail("the image weights are missing: " + file);
      for (NSDictionary* model in catalog[@"models"])
        if ([model[@"file"][@"path"] isEqualToString:@"vision/common.bin"] &&
            [model[@"file"][@"size"] unsignedLongLongValue] != (unsigned long long)info.st_size)
          fail("the image weights do not have the expected size");
      w_.reset(new Weights(engine.gpu_, file, catalog[@"tensors"], engine.options_.read_threads, false));
      const std::string p = "model.visual.";
      patch_ = w_->matrix(p + "patch_embed.proj.weight", V_HIDDEN, V_PATCH);
      patch_bias_ = w_->vector(p + "patch_embed.proj.bias", V_HIDDEN, 0);
      table_ = (const uint16_t*)w_->bytes(p + "pos_embed.weight");
      if (w_->tensor(p + "pos_embed.weight").length != (size_t)V_SIDE * V_SIDE * V_HIDDEN * 2) fail("unexpected image position table");
      for (int b = 0; b < V_DEPTH; ++b) {
        std::string q = p + "blocks." + std::to_string(b) + ".";
        Block block;
        block.norm1 = w_->vector(q + "norm1.weight", V_HIDDEN, 0);
        block.norm1_bias = w_->vector(q + "norm1.bias", V_HIDDEN, 0);
        block.norm2 = w_->vector(q + "norm2.weight", V_HIDDEN, 0);
        block.norm2_bias = w_->vector(q + "norm2.bias", V_HIDDEN, 0);
        block.qkv = w_->matrix(q + "attn.qkv.weight", 3 * V_HIDDEN, V_HIDDEN);
        block.qkv_bias = w_->vector(q + "attn.qkv.bias", 3 * V_HIDDEN, 0);
        block.proj = w_->matrix(q + "attn.proj.weight", V_HIDDEN, V_HIDDEN);
        block.proj_bias = w_->vector(q + "attn.proj.bias", V_HIDDEN, 0);
        // fc1 gets 16 zero rows and fc2 16 zero columns, so both widths are tile multiples.
        block.fc1 = padded_rows(w_->matrix(q + "mlp.linear_fc1.weight", V_MLP, V_HIDDEN));
        block.fc1_bias = w_->vector(q + "mlp.linear_fc1.bias", V_MLP, 0);
        for (int i = V_MLP; i < V_MLP_PADDED; ++i) w_->push(0.0f);
        block.fc2 = padded_columns(w_->matrix(q + "mlp.linear_fc2.weight", V_HIDDEN, V_MLP));
        block.fc2_bias = w_->vector(q + "mlp.linear_fc2.bias", V_HIDDEN, 0);
        blocks_.push_back(block);
      }
      merge_norm_ = w_->vector(p + "merger.norm.weight", V_HIDDEN, 0);
      merge_norm_bias_ = w_->vector(p + "merger.norm.bias", V_HIDDEN, 0);
      merge1_ = w_->matrix(p + "merger.linear_fc1.weight", V_MERGED, V_MERGED);
      merge1_bias_ = w_->vector(p + "merger.linear_fc1.bias", V_MERGED, 0);
      merge2_ = w_->matrix(p + "merger.linear_fc2.weight", H, V_MERGED);
      merge2_bias_ = w_->vector(p + "merger.linear_fc2.bias", H, 0);
      w_->seal();
    }

    // patches: [h * w, 1536] in the processor's merge-block order. Returns [h * w / 4, 2560].
    std::vector<float> encode(const float* patches, int grid_h, int grid_w) {
      Gpu& gpu = e_.gpu_;
      int N = grid_h * grid_w, rows = (N + 15) / 16 * 16, merged = N / 4, merged_rows = (merged + 15) / 16 * 16;
      auto alloc = [&](size_t floats) { return gpu.alloc(floats * 4); };
      Buf in = alloc((size_t)rows * V_PATCH), x = alloc((size_t)rows * V_HIDDEN), y = alloc((size_t)rows * V_HIDDEN);
      Buf qkv = alloc((size_t)rows * 3 * V_HIDDEN), attended = alloc((size_t)rows * V_HIDDEN);
      Buf mlp = alloc((size_t)rows * V_MLP_PADDED), position = alloc((size_t)N * V_HIDDEN), angles = alloc((size_t)N * V_HEAD_DIM / 2);
      Buf wide = alloc((size_t)merged_rows * V_MERGED), out = alloc((size_t)merged_rows * H), scores;
      // Released on every exit, a cancellation or error included.
      struct Release {
        Gpu& gpu;
        std::vector<Buf*> buffers;
        ~Release() { for (Buf* b : buffers) gpu.release(*b); }
      } release{gpu, {&in, &x, &y, &qkv, &attended, &mlp, &position, &angles, &wide, &out, &scores}};
      memcpy(in.buffer.contents, patches, (size_t)N * V_PATCH * 4);
      // Patch n sits at row (2 * block_row + sub_row), column (2 * block_column + sub_column).
      float* pos = (float*)position.buffer.contents;
      float* angle = (float*)angles.buffer.contents;
      int quarter = V_HEAD_DIM / 4;
      for (int n = 0; n < N; ++n) {
        int sub_column = n % 2, sub_row = (n / 2) % 2, block = n / 4;
        int row = (block / (grid_w / 2)) * 2 + sub_row, column = (block % (grid_w / 2)) * 2 + sub_column;
        // Learned positions, bilinear with corners aligned to the 48 x 48 table.
        float ys = (float)((double)row * (V_SIDE - 1) / (grid_h - 1)), xs = (float)((double)column * (V_SIDE - 1) / (grid_w - 1));
        int y0 = (int)std::floor(ys), x0 = (int)std::floor(xs), y1 = std::min(y0 + 1, V_SIDE - 1), x1 = std::min(x0 + 1, V_SIDE - 1);
        float dy = ys - y0, dx = xs - x0;
        int corners[4] = {y0 * V_SIDE + x0, y0 * V_SIDE + x1, y1 * V_SIDE + x0, y1 * V_SIDE + x1};
        float weights[4] = {(1 - dy) * (1 - dx), (1 - dy) * dx, dy * (1 - dx), dy * dx};
        for (int i = 0; i < V_HIDDEN; ++i) {
          float total = 0;
          for (int c = 0; c < 4; ++c) {
            uint32_t bits = (uint32_t)table_[(size_t)corners[c] * V_HIDDEN + i] << 16;
            float value;
            memcpy(&value, &bits, 4);
            total += value * weights[c];
          }
          pos[(size_t)n * V_HIDDEN + i] = total;
        }
        for (int j = 0; j < quarter; ++j) {
          float frequency = 1.0f / std::pow(10000.0f, (float)(2 * j) / (float)(V_HEAD_DIM / 2));
          angle[(size_t)n * (V_HEAD_DIM / 2) + j] = row * frequency;
          angle[(size_t)n * (V_HEAD_DIM / 2) + quarter + j] = column * frequency;
        }
      }
      const Weights& w = *w_;
      e_.linear(in, patch_, x, N, V_PATCH, V_HIDDEN);
      bias(x, w.vec(patch_bias_), N, V_HIDDEN, 0);
      gpu.run("add_in_place", {x, position}, CountP{(uint32_t)(N * V_HIDDEN), 1.0f}, (size_t)N * V_HIDDEN);
      // Attention scores for a block of queries at a time, up to 64 MB.
      int chunk = std::max(1, std::min(N, (int)((64u << 20) / ((size_t)V_HEADS * N * 4))));
      scores = alloc((size_t)chunk * V_HEADS * N);
      for (const Block& b : blocks_) {
        e_.check_cancel();
        gpu.run("layer_norm", {x, w.vec(b.norm1), w.vec(b.norm1_bias), y}, VisionNormP{(uint32_t)N, V_HIDDEN, 1e-6f}, 32, N, 1, kSimdRows);
        e_.linear(y, b.qkv, qkv, N, V_HIDDEN, 3 * V_HIDDEN);
        bias(qkv, w.vec(b.qkv_bias), N, 3 * V_HIDDEN, 0);
        VisionAttnP a{(uint32_t)N, 0, (uint32_t)N, V_HEADS, V_HEAD_DIM, 1.0f / std::sqrt((float)V_HEAD_DIM)};
        gpu.run("vision_rope", {qkv, angles}, a, V_HEAD_DIM / 2, 2 * V_HEADS, N);
        for (int q0 = 0; q0 < N; q0 += chunk) {
          a.Q0 = (uint32_t)q0;
          a.Qn = (uint32_t)std::min(chunk, N - q0);
          gpu.run("vision_scores", {qkv, scores}, a, N, V_HEADS, a.Qn);
          gpu.run("vision_softmax", {scores}, a, 32, V_HEADS, a.Qn, kSimdRows);
          gpu.run("vision_output", {scores, qkv, attended}, a, 32, V_HEADS * V_HEAD_DIM, a.Qn, kSimdRows);
        }
        e_.linear(attended, b.proj, y, N, V_HIDDEN, V_HIDDEN);
        bias(y, w.vec(b.proj_bias), N, V_HIDDEN, 0);
        gpu.run("add_in_place", {x, y}, CountP{(uint32_t)(N * V_HIDDEN), 1.0f}, (size_t)N * V_HIDDEN);
        gpu.run("layer_norm", {x, w.vec(b.norm2), w.vec(b.norm2_bias), y}, VisionNormP{(uint32_t)N, V_HIDDEN, 1e-6f}, 32, N, 1, kSimdRows);
        e_.linear(y, b.fc1, mlp, N, V_HIDDEN, V_MLP_PADDED);
        bias(mlp, w.vec(b.fc1_bias), N, V_MLP_PADDED, 1);
        e_.linear(mlp, b.fc2, y, N, V_MLP_PADDED, V_HIDDEN);
        bias(y, w.vec(b.fc2_bias), N, V_HIDDEN, 0);
        gpu.run("add_in_place", {x, y}, CountP{(uint32_t)(N * V_HIDDEN), 1.0f}, (size_t)N * V_HIDDEN);
        gpu.flush();
      }
      // Merger: the four patches of each 2 x 2 block are adjacent rows, so [N, 1152] is [N / 4, 4608].
      gpu.run("layer_norm", {x, w.vec(merge_norm_), w.vec(merge_norm_bias_), y}, VisionNormP{(uint32_t)N, V_HIDDEN, 1e-6f}, 32, N, 1, kSimdRows);
      e_.linear(y, merge1_, wide, merged, V_MERGED, V_MERGED);
      bias(wide, w.vec(merge1_bias_), merged, V_MERGED, 2);
      e_.linear(wide, merge2_, out, merged, V_MERGED, H);
      bias(out, w.vec(merge2_bias_), merged, H, 0);
      gpu.flush();
      const float* result = (const float*)out.buffer.contents;
      return std::vector<float>(result, result + (size_t)merged * H);
    }

    ~VisionRun() {
      for (Block& b : blocks_) {
        e_.gpu_.release(b.fc1.w);
        e_.gpu_.release(b.fc2.w);
      }
    }

   private:
    struct Block {
      size_t norm1, norm1_bias, norm2, norm2_bias, qkv_bias, proj_bias, fc1_bias, fc2_bias;
      Mat qkv, proj, fc1, fc2;
    };
    void bias(Buf y, Buf b, int N, int O, uint32_t act) {
      e_.gpu_.run("bias_act", {y, b}, BiasP{(uint32_t)N, (uint32_t)O, act}, O, N);
    }
    Mat padded_rows(const Mat& m) {
      Buf out = e_.gpu_.alloc((size_t)V_MLP_PADDED * V_HIDDEN * 2);
      memset(out.buffer.contents, 0, out.buffer.length);
      memcpy(out.buffer.contents, (const char*)m.w.buffer.contents + m.w.offset, (size_t)V_MLP * V_HIDDEN * 2);
      return {out, {}, false, V_MLP_PADDED, V_HIDDEN};
    }
    Mat padded_columns(const Mat& m) {
      Buf out = e_.gpu_.alloc((size_t)V_HIDDEN * V_MLP_PADDED * 2);
      memset(out.buffer.contents, 0, out.buffer.length);
      for (int r = 0; r < V_HIDDEN; ++r)
        memcpy((char*)out.buffer.contents + (size_t)r * V_MLP_PADDED * 2,
               (const char*)m.w.buffer.contents + m.w.offset + (size_t)r * V_MLP * 2, (size_t)V_MLP * 2);
      return {out, {}, false, V_HIDDEN, V_MLP_PADDED};
    }

    EngineImpl& e_;
    std::unique_ptr<Weights> w_;
    Mat patch_, merge1_, merge2_;
    size_t patch_bias_ = 0, merge_norm_ = 0, merge_norm_bias_ = 0, merge1_bias_ = 0, merge2_bias_ = 0;
    const uint16_t* table_ = nullptr;
    std::vector<Block> blocks_;
  };

  // Draft pairs from short calls wait here until the draft layer is needed.
  static constexpr int PENDING_PAIRS = 16;

  // Runs the 48 layers over `count` tokens. all_ row 0 holds the hidden state
  // before the call and rows 1..count the new ones. With `verifying`, the
  // linear-attention layers log what commit() needs to keep only some rows.
  // `then_file` names an expert file to start reading while the last layer runs.
  void forward(const int32_t* tokens, int count, int logit_rows, bool verifying, int then_file = -1) {
    if (count < 1 || logit_rows < 0 || logit_rows > std::min(count, VERIFY_ROWS)) fail("invalid eval request");
    for (int i = 0; i < count; ++i)
      if (tokens[i] < 0 || tokens[i] >= VOCAB) fail("token outside the vocabulary");
    gpu_.flush();
    if (layer_read_ != nil) dispatch_group_wait(layer_read_, DISPATCH_TIME_FOREVER);  // a read left by a failed call
    layer_read_failed_ = layer_started_ = false;
    size_t rows = ((size_t)count + 1 + 15) / 16 * 16;
    if (rows * W * 4 > all_bytes_) {
      all_bytes_ = rows * W * 4;
      gpu_.replace(all_, all_bytes_);
    }
    float* hidden = (float*)all_.buffer.contents;
    memcpy(hidden, boundary_.data(), (size_t)W * 4);
    for (int t = 0; t < count; ++t) {
      float* row = hidden + (size_t)(t + 1) * W;
      const float* feature = image_.feature(position_ + t);
      if (feature == nullptr) {
        embed(tokens[t], row, C);
        continue;
      }
      for (int copy = 0; copy < C; ++copy) memcpy(row + copy * H, feature, H * 4);
    }
    // Layer-major: every layer runs over the whole call before the next layer
    // starts, so a long prompt reads each layer's expert file once.
    bool whole_layer = count > WHOLE_LAYER_ROWS;
    int chunk = std::max(1, std::min(count, options_.chunk));
    std::vector<int32_t> padded{history_[0], history_[1]};  // N-gram rows look two tokens back
    padded.insert(padded.end(), tokens, tokens + count);
    reserve(chunk);
    logging_ = verifying;
    for (int layer = 0; layer < LAYERS; ++layer) {
      check_cancel();
      if (!options_.dump_dir.empty() && count == 1) dump(layer);
      if (!options_.dump_dir.empty() && count > 1 && layer < 9) {  // debug: every row's input to the first layers
        gpu_.flush();
        char name[48];
        snprintf(name, sizeof(name), "/rows_l%02d.bin", layer);
        FILE* file = fopen((options_.dump_dir + name).c_str(), "wb");
        if (file != nullptr) {
          fwrite((const float*)all_.buffer.contents + W, 4, (size_t)count * W, file);
          fclose(file);
        }
      }
      if (whole_layer) wait_layer(layer, layer + 1 < LAYERS ? layer + 1 : then_file);
      const LayerW& lw = whole_layer ? bf16_layer(layers_[layer]) : layers_[layer];
      for (int start = 0; start < count; start += chunk) {
        h_ = all_.at((size_t)(start + 1) * W * 4);
        run_layer(*common_, lw, &linear_[layer], &attention_[layer], &logs_[layer], layer,
                  padded.data() + 2 + start, std::min(chunk, count - start), whole_layer, padded.data() + start);
      }
      if (whole_layer && options_.seed) seed_layer(layer);
    }
    logging_ = false;
    if (!options_.dump_dir.empty() && count == 1) dump(LAYERS);
    if (logit_rows > 0) {
      h_ = all_.at((size_t)(count - logit_rows + 1) * W * 4);
      hyper_mix(*common_, mixer_, logit_rows);
      linear(s_.mixed, lm_head_, logits_, logit_rows, H, VOCAB);
    }
    gpu_.flush();
    history_[0] = count >= 2 ? tokens[count - 2] : history_[1];
    history_[1] = tokens[count - 1];
    position_ += count;
    tokens_.insert(tokens_.end(), tokens, tokens + count);
  }

  // Token embedding repeated `copies` times.
  void embed(int32_t token, float* out, int copies) const {
    const uint16_t* row = embed_ + (size_t)token * H;
    for (int i = 0; i < H; ++i) {
      uint32_t bits = (uint32_t)row[i] << 16;
      memcpy(out + i, &bits, 4);
    }
    for (int copy = 1; copy < copies; ++copy) memcpy(out + copy * H, out, H * 4);
  }

  // Remember the hidden state after the n-th token of the latest forward call.
  void keep_boundary(int n) {
    memcpy(boundary_.data(), (const float*)all_.buffer.contents + (size_t)n * W, (size_t)W * 4);
  }

  static void trim(AttentionState& state, int count) {
    state.count = count;
    state.pooled_blocks = std::min(state.pooled_blocks, count / RATIO);
  }

  // The layer with its 8-bit matrices replaced by BF16 copies in a scratch buffer,
  // as the MLX engine does for long products: the tiled BF16 kernel is faster
  // there, and one copy per layer is cheap next to a whole-layer pass. The copy
  // stays valid until the next call.
  const LayerW& bf16_layer(const LayerW& source) {
    if (!options_.eight_bit) return source;
    dense_layer_ = source;
    size_t bytes = 0;
    for (Mat* m : matrices(dense_layer_)) if (m->q8) bytes += (size_t)m->O * m->I * 2;
    if (dense_.buffer == nil || dense_.buffer.length < bytes) {
      gpu_.flush();
      gpu_.replace(dense_, bytes);
    }
    size_t at = 0;
    for (Mat* m : matrices(dense_layer_)) {
      if (!m->q8) continue;
      Buf out = dense_.at(at);
      gpu_.run("dequant_q8", {m->w, m->sb, out}, DequantP{(uint32_t)m->O, (uint32_t)m->I}, m->I / 4, m->O);
      *m = {out, {}, false, m->O, m->I};
      at += (size_t)m->O * m->I * 2;
    }
    return dense_layer_;
  }

  // Wait for a whole expert file read in the background and start the next one.
  // The GPU works on one layer while the next file is read into the other
  // buffer; everything that used that buffer has finished.
  void wait_layer(int file, int next) {
    gpu_.flush();
    if (!layer_started_) read_layer(file);
    CFAbsoluteTime started = CFAbsoluteTimeGetCurrent();
    dispatch_group_wait(layer_read_, DISPATCH_TIME_FOREVER);
    arena_->read_seconds += CFAbsoluteTimeGetCurrent() - started;
    layer_started_ = false;
    if (layer_read_failed_) fail("could not read a layer of experts from SSD");
    layer_blobs_ = layer_buffers_[layer_slot_];
    if (next >= 0) read_layer(next);
    seed_counts_.assign(EXPERTS, 0);
  }

  // The prompt just went through this whole expert file. Keep the experts it
  // used most in the cache, so the tokens that follow do not read them from SSD.
  void seed_layer(int file) {
    int keep = arena_->slots() / arena_->files();
    std::vector<int> order(EXPERTS);
    for (int i = 0; i < EXPERTS; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + std::min(keep, EXPERTS), order.end(),
                      [&](int a, int b) { return seed_counts_[a] > seed_counts_[b]; });
    order.resize(std::min(keep, EXPERTS));
    while (!order.empty() && seed_counts_[order.back()] == 0) order.pop_back();
    std::vector<int> uses;
    for (int expert : order) uses.push_back(seed_counts_[expert]);
    arena_->seed(file, order, uses, (const char*)layer_blobs_.buffer.contents);
  }

  // Feed the draft layer the pairs that short calls left behind.
  void catch_up() {
    int pairs = (int)pending_tokens_.size();
    if (pairs == 0) return;
    gpu_.flush();
    size_t bytes = (size_t)(pairs + 15) / 16 * 16 * W * 4;
    if (mtp_in_.buffer == nil || mtp_in_.buffer.length < bytes) gpu_.replace(mtp_in_, bytes);
    memcpy(mtp_in_.buffer.contents, pending_hidden_.data(), (size_t)pairs * W * 4);
    mtp_rows(mtp_in_, pending_tokens_.data(), pairs);
    gpu_.flush();
    pending_hidden_.clear();
    pending_tokens_.clear();
  }

  // The draft layer over `count` pairs (hidden, next token). Leaves the last
  // chunk's wide output in mtp_h_.
  void mtp_rows(Buf hidden, const int32_t* tokens, int count) {
    // Pair k pairs a hidden state with the token at this prompt index; image
    // tokens use their encoder features, as the target model did.
    int first_index = mtp_attention_.count + 1;
    bool whole_layer = count > WHOLE_LAYER_ROWS;
    int chunk = std::max(1, std::min(count, options_.chunk));
    gpu_.flush();
    reserve(chunk);
    size_t bytes = (size_t)(chunk + 15) / 16 * 16 * W * 4;
    if (mtp_h_.buffer == nil || mtp_h_.buffer.length < bytes) gpu_.replace(mtp_h_, bytes);
    if (whole_layer) wait_layer(mtp_file_, -1);
    const Weights& w = *mtp_;
    const LayerW& lw = whole_layer ? bf16_layer(mtp_layer_) : mtp_layer_;
    for (int start = 0; start < count; start += chunk) {
      check_cancel();
      int T = std::min(chunk, count - start);
      gpu_.flush();  // the embedding scratch is written from the CPU
      float* embedded = (float*)s_.embedding.buffer.contents;
      for (int t = 0; t < T; ++t) {
        const float* feature = image_.feature(first_index + start + t);
        if (feature != nullptr) memcpy(embedded + (size_t)t * H, feature, H * 4);
        else embed(tokens[start + t], embedded + (size_t)t * H, 1);
      }
      norm(s_.embedding, w.vec(mtp_norm_embedding_), s_.embedding, T, 1, H, H, 0, 1, true, 1.0f);
      linear(s_.embedding, mtp_fc_embedding_, s_.result, T, H, H);
      norm(hidden.at((size_t)start * W * 4), w.vec(mtp_norm_hidden_), s_.normed, T, 1, W, W, 0, 1, true, 1.0f);
      linear(s_.normed, mtp_fc_hidden_, mtp_h_, T * C, H, H);  // each of the four streams is one row
      gpu_.run("add_broadcast", {mtp_h_, s_.result}, HcP{(uint32_t)T, H, C}, W, T);
      h_ = mtp_h_;
      run_layer(w, lw, nullptr, &mtp_attention_, nullptr, mtp_file_, nullptr, T, whole_layer, nullptr);
    }
    if (whole_layer && options_.seed) seed_layer(mtp_file_);
  }

  // -------------------------------------------------------------- loading helpers
  static NSDictionary* json(const std::string& path) {
    NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
    if (data == nil) fail("cannot read " + path);
    NSDictionary* parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    if (![parsed isKindOfClass:[NSDictionary class]]) fail("invalid JSON in " + path);
    return parsed;
  }

  static void check(NSDictionary* config, NSDictionary* manifest) {
    auto same = [&](NSString* key, int expected) {
      if ([config[key] intValue] != expected) fail(std::string("unsupported model: ") + key.UTF8String);
    };
    same(@"hidden_size", H); same(@"hc_count", C); same(@"num_hidden_layers", LAYERS); same(@"vocab_size", VOCAB);
    same(@"hc_lowrank", HC_RANK); same(@"num_attention_heads", HEADS); same(@"num_key_value_heads", KV_HEADS);
    same(@"head_dim", HEAD_DIM); same(@"indexer_n_heads", INDEX_HEADS); same(@"indexer_head_dim", INDEX_DIM);
    same(@"indexer_budget", INDEX_BUDGET); same(@"indexer_compress_ratio", RATIO);
    same(@"linear_num_key_heads", GDN_HK); same(@"linear_num_value_heads", GDN_HV);
    same(@"linear_key_head_dim", GDN_D); same(@"linear_value_head_dim", GDN_D); same(@"linear_conv_kernel_dim", 4);
    same(@"num_experts", EXPERTS); same(@"num_experts_per_tok", TOP_K); same(@"moe_intermediate_size", MOE_M);
    same(@"shared_expert_intermediate_size", MOE_M); same(@"ngram_size", 3); same(@"ple_conv_kernel_size", 4);
    same(@"eos_token_id", EOS);
    if (HEADS / KV_HEADS != 12) fail("unsupported model: query heads per KV head");
    NSArray* ple = config[@"ple_layer_ids"];
    if (ple.count != 1 || [ple[0] intValue] != 2) fail("unsupported model: ple_layer_ids");
    if ([config[@"rope_parameters"][@"rope_theta"] doubleValue] != 1e7) fail("unsupported model: rope_theta");
    if ([manifest[@"expertBlobSize"] unsignedLongLongValue] != BLOB || [manifest[@"expertCount"] intValue] != EXPERTS)
      fail("unsupported model: expert blob layout");
    NSDictionary* quantization = manifest[@"expertQuantization"];
    if (![quantization[@"mode"] isEqualToString:@"mxfp4"] || [quantization[@"groupSize"] intValue] != 32)
      fail("unsupported model: experts are not MXFP4");
  }

  // -------------------------------------------------------------- scratch
  struct Scratch {
    int rows = 0;
    Buf normed, down, mid, up, mixed, inject, result;
    Buf qkv, conv, z, a, b, decay, beta, y, gated;
    Buf q_proj, k_proj, v_proj, index_proj, query, index_query, attended, selection;
    Buf router, shared_gate, gate_proj, up_proj, shared_act, shared, pairs, score, gate_up, act, routed;
    Buf embedding, value, packed, tiles, tile_rows, tile_pairs, expert_packed, tile_gate_up, tile_act, tile_routed, bf16_rows;
    Buf weights; size_t weights_bytes = 0;
    Buf block_scores; size_t block_bytes = 0;
  } s_;

  void reserve(int rows) {
    if (rows <= s_.rows) return;
    gpu_.flush();
    for (Buf* b : {&s_.normed, &s_.down, &s_.mid, &s_.up, &s_.mixed, &s_.inject, &s_.result, &s_.qkv, &s_.conv, &s_.z,
                   &s_.a, &s_.b, &s_.decay, &s_.beta, &s_.y, &s_.gated, &s_.q_proj, &s_.k_proj, &s_.v_proj, &s_.index_proj,
                   &s_.query, &s_.index_query, &s_.attended, &s_.selection, &s_.router, &s_.shared_gate, &s_.gate_proj,
                   &s_.up_proj, &s_.shared_act, &s_.shared, &s_.pairs, &s_.score, &s_.gate_up, &s_.act, &s_.routed,
                   &s_.embedding, &s_.value, &s_.packed, &s_.tiles, &s_.tile_rows, &s_.tile_pairs, &s_.expert_packed,
                   &s_.tile_gate_up, &s_.tile_act, &s_.tile_routed, &s_.bf16_rows})
      gpu_.release(*b);
    size_t n = ((size_t)rows + 15) / 16 * 16;  // tiled products read and write whole tiles
    auto f = [&](size_t per_row) { return gpu_.alloc(n * per_row * 4); };
    s_.normed = f(W); s_.down = f(HC_RANK); s_.mid = f(HC_RANK); s_.up = f(W);
    s_.mixed = f(H); s_.inject = f(C); s_.result = f(H);
    s_.qkv = f(W); s_.conv = f(W); s_.z = f(GDN_VALUE); s_.a = f(GDN_HV); s_.b = f(GDN_HV);
    s_.decay = f(GDN_HV); s_.beta = f(GDN_HV); s_.y = f(GDN_VALUE); s_.gated = f(GDN_VALUE);
    s_.q_proj = f(HEADS * HEAD_DIM * 2); s_.k_proj = f(KV_HEADS * HEAD_DIM); s_.v_proj = f(KV_HEADS * HEAD_DIM);
    s_.index_proj = f((INDEX_HEADS + 1) * INDEX_DIM); s_.query = f(HEADS * HEAD_DIM);
    s_.index_query = f(INDEX_HEADS * INDEX_DIM); s_.attended = f(HEADS * HEAD_DIM); s_.selection = f(SELECTED);
    s_.router = f(EXPERTS); s_.shared_gate = f(1); s_.gate_proj = f(MOE_M); s_.up_proj = f(MOE_M);
    s_.shared_act = f(MOE_M); s_.shared = f(H); s_.pairs = f(TOP_K * 3); s_.score = f(TOP_K);  // a Pair is three uints
    s_.gate_up = f((size_t)TOP_K * 2 * MOE_M); s_.act = f((size_t)TOP_K * MOE_M); s_.routed = f((size_t)TOP_K * H);
    s_.embedding = f(H); s_.value = f(W);
    s_.bf16_rows = gpu_.alloc(gpu_.tensor_ops ? n * W * 2 : 16);  // BF16 copy of the widest product input
    bool batched = rows >= TILED_EXPERT_ROWS;
    size_t blocks = (n + 7) / 8, tiles = (n * TOP_K + 7) / 8 + EXPERTS;
    s_.tiles = gpu_.alloc(tiles * sizeof(Tile));
    s_.tile_rows = gpu_.alloc(tiles * 16 * 4);
    s_.tile_pairs = gpu_.alloc(tiles * 8 * 4);
    if (gpu_.simdgroup_matrices) {
      // Experts run on 16-row tiles kept in tile order from the gather to the combine.
      size_t tile_rows = batched ? (n * TOP_K / 16 + EXPERTS) * 16 : 16;
      s_.expert_packed = gpu_.alloc(tile_rows * H * 4);
      s_.tile_gate_up = gpu_.alloc(tile_rows * 2 * MOE_M * 4);
      s_.tile_act = gpu_.alloc(tile_rows * MOE_M * 4);
      s_.tile_routed = gpu_.alloc(tile_rows * H * 4);
    } else {
      s_.packed = gpu_.alloc(rows >= GEMM_ROWS ? blocks * W * 8 * 4 : 16);  // widest product: W inputs
      s_.expert_packed = gpu_.alloc(batched ? tiles * H * 8 * 4 : 16);
    }
    s_.rows = rows;
  }

  // -------------------------------------------------------------- building blocks
  // y[T, O] = x[T, I] . w[O, I]. Buffers passed here hold rows up to the next multiple of 16.
  // Several products of the same x in a row pass `reuse` after the first call.
  void linear(Buf x, const Mat& m, Buf y, int T, int I, int O, bool reuse = false) {
    LinearP rows{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)I, (uint32_t)O};
    if (m.q8) {
      linear_q8(x, m, y, T, I, O);
      return;
    }
    Buf w = m.w;
    if (T < GEMM_ROWS) {
      if (O < 2048 && I >= 1024) {
        // Few outputs over long rows: split each row so enough simdgroups share the work.
        int splits = std::min(I / 512, (2048 + O - 1) / O);
        if (splits > 1) {
          int span = (I / splits + 3) / 4 * 4;
          splits = (I + span - 1) / span;
          SplitP sp{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)I, (uint32_t)span, (uint32_t)splits};
          gpu_.run("linear_bf16_split", {x, w, partial_}, sp, 32, O, (size_t)((T + 3) / 4) * splits, kSimdRows);
          gpu_.run("split_sum", {partial_, y}, sp, O, T);
          return;
        }
      }
      // Two outputs per simdgroup when O is even: 1.2-1.5 times faster on M2 Max, same sums.
      if (O % 2 == 0) {
        if (T == 1) gpu_.run("linear_bf16_2o", {x, w, y}, rows, 32, O / 2, 1, kSimdRows);
        else gpu_.run("linear_bf16_4_2o", {x, w, y}, rows, 32, O / 2, (T + 3) / 4, kSimdRows);
      } else if (T == 1) {
        gpu_.run("linear_bf16", {x, w, y}, rows, 32, O, 1, kSimdRows);
      } else {
        gpu_.run("linear_bf16_4", {x, w, y}, rows, 32, O, (T + 3) / 4, kSimdRows);
      }
      return;
    }
    if (gpu_.simdgroup_matrices && O % 32 != 0) {
      gpu_.run("linear_bf16", {x, w, y}, rows, 32, O, T, kSimdRows);
      return;
    }
    if (gpu_.tensor_ops && T >= TENSOR_OPS_ROWS && I % 64 == 0 && T <= s_.rows) {
      // BF16 rows for the accelerators; the scratch holds the widest product (W inputs).
      gpu_.run("to_bf16_rows", {x, s_.bf16_rows}, TopsCopyP{(uint32_t)T, (uint32_t)I, (uint32_t)I}, I, T);
      gpu_.run_groups("tops_gemm_bf16", {s_.bf16_rows, w, y}, TopsGemmP{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)O},
                      MTLSizeMake((O + 63) / 64, (T + 63) / 64, 1), MTLSizeMake(128, 1, 1));
      return;
    }
    if (gpu_.simdgroup_matrices && T >= STAGED_GEMM_ROWS && O % 64 == 0 && I % 16 == 0 && !gpu_.no_staged) {
      gpu_.run_groups("gemm_bf16_tg", {x, w, y}, GemmP{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)O},
                      MTLSizeMake(O / 64, (T + 63) / 64, 1), MTLSizeMake(128, 1, 1));
      return;
    }
    if (gpu_.simdgroup_matrices) {
      gpu_.run("gemm_bf16_sg", {x, w, y}, GemmP{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)O}, 32, O / 32, (T + 15) / 16,
               MTLSizeMake(32, 4, 1));
      return;
    }
    if (T > s_.rows) {  // the image encoder's products are larger than the packing scratch
      gpu_.run("linear_bf16", {x, w, y}, rows, 32, O, T, kSimdRows);
      return;
    }
    uint32_t blocks = (uint32_t)(T + 7) / 8;
    if (!reuse) gpu_.run("pack_rows8", {x, s_.packed}, PackP{(uint32_t)T, (uint32_t)I, (uint32_t)I, blocks}, I, blocks);
    gpu_.run("gemm_bf16", {s_.packed, w, y}, GemmP{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)O}, (O + 3) / 4, blocks);
  }
  // linear() for 8-bit weights: the same choice of kernels, with the weights
  // dequantized where they are read.
  void linear_q8(Buf x, const Mat& m, Buf y, int T, int I, int O) {
    LinearP rows{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)I, (uint32_t)O};
    if (T < GEMM_ROWS || !gpu_.simdgroup_matrices || O % 32 != 0) {
      if (T < GEMM_ROWS && O < 2048 && I >= 1024) {
        int splits = std::min(I / 512, (2048 + O - 1) / O);
        if (splits > 1) {
          int span = (I / splits + Q8_GROUP - 1) / Q8_GROUP * Q8_GROUP;
          splits = (I + span - 1) / span;
          SplitP sp{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)I, (uint32_t)span, (uint32_t)splits};
          gpu_.run("linear_q8_split", {x, m.w, m.sb, partial_}, sp, 32, O, (size_t)((T + 3) / 4) * splits, kSimdRows);
          gpu_.run("split_sum", {partial_, y}, sp, O, T);
          return;
        }
      }
      if (O % 2 == 0) {
        if (T == 1) gpu_.run("linear_q8_2o", {x, m.w, m.sb, y}, rows, 32, O / 2, 1, kSimdRows);
        else gpu_.run("linear_q8_4_2o", {x, m.w, m.sb, y}, rows, 32, O / 2, (T + 3) / 4, kSimdRows);
      } else if (T == 1) {
        gpu_.run("linear_q8", {x, m.w, m.sb, y}, rows, 32, O, 1, kSimdRows);
      } else {
        gpu_.run("linear_q8_4", {x, m.w, m.sb, y}, rows, 32, O, (T + 3) / 4, kSimdRows);
      }
      return;
    }
    gpu_.run("gemm_q8_sg", {x, m.w, m.sb, y}, GemmP{(uint32_t)T, (uint32_t)I, (uint32_t)O, (uint32_t)O}, 32, O / 32, (T + 15) / 16,
             MTLSizeMake(32, 4, 1));
  }

  void norm(Buf x, Buf w, Buf y, int T, int groups, int D, int row_stride, int base, int w_period, bool has_w, float scale) {
    NormP p{(uint32_t)T, (uint32_t)groups, (uint32_t)D, (uint32_t)row_stride, (uint32_t)base, (uint32_t)w_period,
            has_w ? 1u : 0u, EPS, scale};
    gpu_.run("rms_norm", {x, has_w ? w : x, y}, p, 32, groups, T, kSimdRows);
  }

  // h_ -> s_.mixed (and s_.inject when the block combines).
  void hyper_mix(const Weights& w, const HyperW& hw, int T) {
    norm(h_, w.vec(hw.norm), s_.normed, T, C, H, W, 0, C, true, 1.0f);
    linear(s_.normed, hw.down, s_.down, T, W, HC_RANK);
    if (hw.combine) linear(s_.normed, hw.inject, s_.inject, T, W, C, true);
    gpu_.run("silu_scaled", {s_.down, s_.mid}, CountP{(uint32_t)(T * HC_RANK), 1.0f / C}, (size_t)T * HC_RANK);
    linear(s_.mid, hw.up, s_.up, T, HC_RANK, W);
    gpu_.run("hc_mix", {s_.up, s_.normed, s_.mixed}, HcP{(uint32_t)T, H, C}, H, T);
  }
  // h_ += s_.result through the gates in s_.inject.
  void hyper_inject(int T) { gpu_.run("hc_inject", {h_, s_.result, s_.inject}, HcP{(uint32_t)T, H, C}, W, T); }

  // One decoder layer over the T rows at h_. `before` points at the two tokens
  // ahead of this chunk. `file` is the layer's expert file in the arena.
  void run_layer(const Weights& w, const LayerW& lw, LinearState* linear_state, AttentionState* attention_state,
                 LinearLog* log, int file, const int32_t* tokens, int T, bool whole_layer, const int32_t* before) {
    if (lw.ple) ple(w, lw, tokens, T, before);
    hyper_mix(w, lw.attention, T);
    if (lw.linear) delta_net(w, lw, *linear_state, logging_ ? log : nullptr, T);
    else attention(w, lw, *attention_state, T);
    hyper_inject(T);
    hyper_mix(w, lw.feed_forward, T);
    experts(lw, file, T, whole_layer);
    hyper_inject(T);
  }

  // -------------------------------------------------------------- Gated DeltaNet: s_.mixed -> s_.result
  void delta_net(const Weights& w, const LayerW& lw, LinearState& state, LinearLog* log, int T) {
    // A verification writes these rows where commit() can find them again.
    Buf raw = log != nullptr ? log->raw : s_.qkv, rows = log != nullptr ? log->rows : s_.conv;
    Buf decay = log != nullptr ? log->decay : s_.decay, beta = log != nullptr ? log->beta : s_.beta;
    int next = 1 - state.current;
    linear(s_.mixed, lw.qkv, raw, T, H, W);
    linear(s_.mixed, lw.z, s_.z, T, H, GDN_VALUE, true);
    linear(s_.mixed, lw.a, s_.a, T, H, GDN_HV, true);
    linear(s_.mixed, lw.b, s_.b, T, H, GDN_HV, true);
    ConvP conv{(uint32_t)T, W, 4, 1, CONV_STATE, 1};
    gpu_.run("conv_depthwise", {state.conv[state.current], raw, lw.conv, rows}, conv, W, T);
    gpu_.run("conv_state_next", {state.conv[state.current], raw, state.conv[next]}, conv, W, CONV_STATE);
    norm(rows, rows, rows, T, GDN_HK, GDN_D, W, 0, 1, false, 1.0f / GDN_D);
    norm(rows, rows, rows, T, GDN_HK, GDN_D, W, GDN_KEY, 1, false, 1.0f / std::sqrt((float)GDN_D));
    GdnP p{(uint32_t)T, GDN_HK, GDN_HV, GDN_D, GDN_D, W, GDN_KEY};
    gpu_.run("gdn_gates", {s_.a, s_.b, w.vec(lw.a_log), w.vec(lw.dt_bias), decay, beta}, p, GDN_HV, T);
    gpu_.run("gdn_step", {rows, decay, beta, state.state[state.current], state.state[next], s_.y},
             p, 32, GDN_D, GDN_HV, MTLSizeMake(32, 4, 1));
    state.current = next;
    gpu_.run("gated_rms_norm", {s_.y, w.vec(lw.gdn_norm), s_.z, s_.gated}, GatedNormP{(uint32_t)(T * GDN_HV), GDN_D, EPS},
             (size_t)T * GDN_HV);
    linear(s_.gated, lw.out, s_.result, T, GDN_VALUE, H);
  }

  // -------------------------------------------------------------- QSA: s_.mixed -> s_.result
  void grow(AttentionState& state, int needed) {
    if (needed <= state.capacity) return;
    gpu_.flush();
    int capacity = std::max(4096, state.capacity * 2);
    while (capacity < needed) capacity *= 2;
    auto resize = [&](Buf& buffer, size_t old_bytes, size_t new_bytes) {
      Buf next = gpu_.alloc(new_bytes);
      if (buffer.buffer != nil) memcpy(next.buffer.contents, buffer.buffer.contents, old_bytes);
      gpu_.release(buffer);
      buffer = next;
    };
    size_t kv = (size_t)KV_HEADS * HEAD_DIM * 2;
    resize(state.keys, state.count * kv, capacity * kv);
    resize(state.values, state.count * kv, capacity * kv);
    resize(state.raw, (size_t)state.count * INDEX_DIM * 2, (size_t)capacity * INDEX_DIM * 2);
    resize(state.pooled, (size_t)state.pooled_blocks * INDEX_DIM * 4, (size_t)(capacity / RATIO) * INDEX_DIM * 4);
    state.capacity = capacity;
  }

  void attention(const Weights& w, const LayerW& lw, AttentionState& state, int T) {
    const float ln_theta = std::log(1e7f);
    int pos0 = state.count;
    grow(state, pos0 + T);
    linear(s_.mixed, lw.q, s_.q_proj, T, H, HEADS * HEAD_DIM * 2);
    linear(s_.mixed, lw.k, s_.k_proj, T, H, KV_HEADS * HEAD_DIM, true);
    linear(s_.mixed, lw.v, s_.v_proj, T, H, KV_HEADS * HEAD_DIM, true);
    linear(s_.mixed, lw.index, s_.index_proj, T, H, (INDEX_HEADS + 1) * INDEX_DIM, true);
    size_t kv_row = (size_t)KV_HEADS * HEAD_DIM * 2;
    RopeP query{(uint32_t)T, HEADS, HEAD_DIM, HEADS * HEAD_DIM * 2, 0, HEAD_DIM * 2, ROTARY, (uint32_t)pos0, EPS, ln_theta, 1,
                image_.map_len, image_.delta};
    gpu_.run("norm_rope_f32", {s_.q_proj, w.vec(lw.q_norm), s_.query, image_.map}, query, HEADS, T);
    RopeP key{(uint32_t)T, KV_HEADS, HEAD_DIM, KV_HEADS * HEAD_DIM, 0, HEAD_DIM, ROTARY, (uint32_t)pos0, EPS, ln_theta, 1,
              image_.map_len, image_.delta};
    gpu_.run("norm_rope_f16", {s_.k_proj, w.vec(lw.k_norm), state.keys.at(pos0 * kv_row), image_.map}, key, KV_HEADS, T);
    gpu_.run("copy_f16", {s_.v_proj, state.values.at(pos0 * kv_row)}, CopyP{(uint32_t)T, KV_HEADS * HEAD_DIM, KV_HEADS * HEAD_DIM, 0},
             KV_HEADS * HEAD_DIM, T);
    RopeP index_query{(uint32_t)T, INDEX_HEADS, INDEX_DIM, (INDEX_HEADS + 1) * INDEX_DIM, 0, INDEX_DIM, ROTARY, (uint32_t)pos0, EPS,
                      ln_theta, 1, image_.map_len, image_.delta};
    gpu_.run("norm_rope_f32", {s_.index_proj, w.vec(lw.index_q_norm), s_.index_query, image_.map}, index_query, INDEX_HEADS, T);
    gpu_.run("copy_f16", {s_.index_proj, state.raw.at((size_t)pos0 * INDEX_DIM * 2)},
             CopyP{(uint32_t)T, INDEX_DIM, (INDEX_HEADS + 1) * INDEX_DIM, INDEX_HEADS * INDEX_DIM}, INDEX_DIM, T);
    state.count += T;
    int blocks = state.count / RATIO;
    if (blocks > state.pooled_blocks) {
      PoolP pool{(uint32_t)(blocks - state.pooled_blocks), (uint32_t)state.pooled_blocks, RATIO, INDEX_DIM, ROTARY, EPS, ln_theta,
                 image_.map_len, image_.delta};
      gpu_.run("qsa_pool", {state.raw, w.vec(lw.index_k_norm), state.pooled, image_.map}, pool, pool.blocks);
      state.pooled_blocks = blocks;
    }
    // K is padded to a multiple of 8; keys past the query's position are masked anyway.
    AttentionP p{(uint32_t)T, (uint32_t)(state.count + 7) / 8 * 8, HEADS, KV_HEADS, HEAD_DIM, (uint32_t)pos0, 0, 1.0f / std::sqrt((float)HEAD_DIM)};
    if (state.count > INDEX_BUDGET) {
      select_blocks(state, T, pos0, blocks);
      p.K = SELECTED;
      p.use_selection = 1;
    }
    size_t bytes = (size_t)T * HEADS * p.K * 4;
    if (bytes > s_.weights_bytes) {
      gpu_.flush();
      gpu_.replace(s_.weights, bytes + bytes / 2);
      s_.weights_bytes = bytes + bytes / 2;
    }
    gpu_.run("qsa_scores", {s_.query, state.keys, s_.selection, s_.weights}, p, p.K, KV_HEADS, T);
    gpu_.run("qsa_softmax", {s_.weights}, p, 32, HEADS, T, kSimdRows);
    if (gpu_.simdgroup_matrices) {
      gpu_.run("qsa_output_sg", {s_.weights, state.values, s_.selection, s_.attended}, p, 32, HEAD_DIM / 32, KV_HEADS * T, MTLSizeMake(32, 4, 1));
      gpu_.run("qsa_gate", {s_.attended, s_.q_proj}, p, HEADS * HEAD_DIM, T);
    } else {
      gpu_.run("qsa_output", {s_.weights, state.values, s_.selection, s_.q_proj, s_.attended}, p, 32, HEAD_DIM / 4, HEADS * T, kSimdRows);
    }
    linear(s_.attended, lw.o, s_.result, T, HEADS * HEAD_DIM, H);
  }

  // Each query keeps the best-scoring complete blocks within the budget plus its unfinished tail.
  void select_blocks(AttentionState& state, int T, int pos0, int blocks) {
    size_t bytes = (size_t)T * blocks * 4;
    if (bytes > s_.block_bytes) {
      gpu_.flush();
      gpu_.replace(s_.block_scores, bytes * 2);
      s_.block_bytes = bytes * 2;
    }
    BlockScoreP p{(uint32_t)T, (uint32_t)blocks, INDEX_HEADS, INDEX_DIM, (uint32_t)pos0, RATIO};
    gpu_.run("qsa_block_scores", {s_.index_query, state.pooled, s_.block_scores}, p, blocks, T);
    gpu_.flush();
    const float* scores = (const float*)s_.block_scores.buffer.contents;
    int32_t* selection = (int32_t*)s_.selection.buffer.contents;
    const int budget = INDEX_BUDGET / RATIO;
    dispatch_apply((size_t)T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
      int position = pos0 + (int)t;
      int visible = std::min(blocks, (position + 1) / RATIO);
      const float* row = scores + t * (size_t)blocks;
      int32_t* out = selection + t * SELECTED;
      std::vector<int> order(visible);
      for (int i = 0; i < visible; ++i) order[i] = i;
      int keep = std::min(budget, visible);
      if (keep < visible)
        std::nth_element(order.begin(), order.begin() + keep, order.end(), [&](int a, int b) { return row[a] > row[b]; });
      int written = 0;
      for (int i = 0; i < keep; ++i)
        for (int j = 0; j < RATIO; ++j) out[written++] = order[i] * RATIO + j;
      for (int key = visible * RATIO; key <= position; ++key) out[written++] = key;
      while (written < SELECTED) out[written++] = -1;
    });
  }

  // -------------------------------------------------------------- MoE: s_.mixed -> s_.result
  void experts(const LayerW& lw, int layer, int T, bool whole_layer) {
    linear(s_.mixed, lw.router, s_.router, T, H, EXPERTS);
    linear(s_.mixed, lw.shared_gate, s_.shared_gate, T, H, 1, true);
    linear(s_.mixed, lw.shared_gate_proj, s_.gate_proj, T, H, MOE_M, true);
    linear(s_.mixed, lw.shared_up, s_.up_proj, T, H, MOE_M, true);
    gpu_.run("mul_silu", {s_.gate_proj, s_.up_proj, s_.shared_act}, CountP{(uint32_t)(T * MOE_M), 1.0f}, (size_t)T * MOE_M);
    linear(s_.shared_act, lw.shared_down, s_.shared, T, MOE_M, H);
    gpu_.flush();  // the router's choice decides which experts to read

    const float* logits = (const float*)s_.router.buffer.contents;
    Pair* pairs = (Pair*)s_.pairs.buffer.contents;
    float* score = (float*)s_.score.buffer.contents;
    std::vector<int> chosen((size_t)T * TOP_K);
    for (int t = 0; t < T; ++t) {
      const float* row = logits + (size_t)t * EXPERTS;
      int order[EXPERTS];
      for (int i = 0; i < EXPERTS; ++i) order[i] = i;
      std::partial_sort(order, order + TOP_K, order + EXPERTS, [&](int a, int b) { return row[a] > row[b]; });
      // softmax over all experts, then renormalise the kept ones: the total cancels.
      double top = row[order[0]], kept = 0, weights[TOP_K];
      for (int k = 0; k < TOP_K; ++k) { weights[k] = std::exp((double)row[order[k]] - top); kept += weights[k]; }
      for (int k = 0; k < TOP_K; ++k) {
        chosen[(size_t)t * TOP_K + k] = order[k];
        score[(size_t)t * TOP_K + k] = (float)(weights[k] / kept);
        pairs[(size_t)t * TOP_K + k].x_row = (uint32_t)t;
        pairs[(size_t)t * TOP_K + k].out = (uint32_t)(t * TOP_K + k);
      }
    }
    uint32_t count = (uint32_t)(T * TOP_K);
    if (whole_layer)
      for (int expert : chosen) ++seed_counts_[expert];
    if ((whole_layer || T >= TILED_EXPERT_ROWS) && gpu_.simdgroup_matrices) {
      // Group the (token, expert) pairs by expert, 16 rows to a tile. Every later
      // step keeps this order; s_.pairs maps each pair to its tile row.
      std::vector<uint32_t> order(count);
      for (uint32_t i = 0; i < count; ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return chosen[a] < chosen[b]; });
      uint32_t* slots = (uint32_t*)s_.tiles.buffer.contents;
      uint32_t* rows = (uint32_t*)s_.tile_rows.buffer.contents;
      uint32_t* pair_row = (uint32_t*)s_.pairs.buffer.contents;
      uint32_t tile_count = 0, filled = 16;
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t pair = order[i];
        if (filled == 16 || slots[tile_count - 1] != (uint32_t)chosen[pair]) {
          slots[tile_count] = (uint32_t)chosen[pair];
          for (int r = 0; r < 16; ++r) rows[tile_count * 16 + r] = 0xFFFFFFFFu;
          ++tile_count;
          filled = 0;
        }
        rows[(tile_count - 1) * 16 + filled] = pair / TOP_K;
        pair_row[pair] = (tile_count - 1) * 16 + filled;
        ++filled;
      }
      uint32_t tile_rows = tile_count * 16;
      Buf blobs = layer_blobs_;
      if (!whole_layer) {
        // Tiles name experts; read the chosen ones into the cache and name their slots instead.
        std::vector<int> unique;
        for (uint32_t t = 0; t < tile_count; ++t)
          if (unique.empty() || unique.back() != (int)slots[t]) unique.push_back((int)slots[t]);
        std::vector<uint32_t> cached;
        arena_->acquire(layer, unique, cached);
        for (uint32_t t = 0, u = 0; t < tile_count; ++t) {
          while (unique[u] != (int)slots[t]) ++u;
          slots[t] = cached[u];
        }
        blobs = arena_->buffer();
      }
      gpu_.run("gather_rows16", {s_.mixed, s_.tile_rows, s_.expert_packed}, GatherP{tile_count, H, H}, H, tile_rows);
      ExpertGemmP gate_up{tile_count, H, 2 * MOE_M, 0, (uint32_t)GATE_UP_SCALE, (uint32_t)BLOB};
      if (gpu_.tensor_ops && tile_count * 16 >= TENSOR_OPS_ROWS) {
        gpu_.run_groups("tops_expert", {s_.expert_packed, blobs, s_.tiles, s_.tile_gate_up},
                        TopsExpertP{tile_count, H, 2 * MOE_M, 0, (uint32_t)GATE_UP_SCALE, (uint32_t)BLOB},
                        MTLSizeMake(2 * MOE_M / 64, tile_count, 1), MTLSizeMake(128, 1, 1));
      } else {
        gpu_.run("expert_gemm_sg", {s_.expert_packed, blobs, s_.tiles, s_.tile_gate_up}, gate_up, 32, 2 * MOE_M / 32, tile_count,
                 MTLSizeMake(32, 4, 1));
      }
      gpu_.run("expert_act", {s_.tile_gate_up, s_.tile_act}, ActP{tile_rows, MOE_M}, MOE_M, tile_rows);
      ExpertGemmP down{tile_count, MOE_M, H, (uint32_t)DOWN_WEIGHT, (uint32_t)DOWN_SCALE, (uint32_t)BLOB};
      if (gpu_.tensor_ops && tile_count * 16 >= TENSOR_OPS_ROWS) {
        gpu_.run_groups("tops_expert", {s_.tile_act, blobs, s_.tiles, s_.tile_routed},
                        TopsExpertP{tile_count, MOE_M, H, (uint32_t)DOWN_WEIGHT, (uint32_t)DOWN_SCALE, (uint32_t)BLOB},
                        MTLSizeMake(H / 64, tile_count, 1), MTLSizeMake(128, 1, 1));
      } else {
        gpu_.run("expert_gemm_sg", {s_.tile_act, blobs, s_.tiles, s_.tile_routed}, down, 32, H / 32, tile_count,
                 MTLSizeMake(32, 4, 1));
      }
      gpu_.run("moe_combine_rows", {s_.tile_routed, s_.score, s_.shared, s_.shared_gate, s_.pairs, s_.result},
               CombineRowsP{(uint32_t)T, H, TOP_K}, H, T);
      return;
    }
    if (whole_layer) {
      // Group the (token, expert) pairs by expert, 8 to a tile: a tile reads its
      // expert's weights once for all of its rows.
      std::vector<uint32_t> order(count);
      for (uint32_t i = 0; i < count; ++i) order[i] = i;
      std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return chosen[a] < chosen[b]; });
      Tile* tiles = (Tile*)s_.tiles.buffer.contents;
      uint32_t* rows = (uint32_t*)s_.tile_rows.buffer.contents;
      uint32_t* tile_pairs = (uint32_t*)s_.tile_pairs.buffer.contents;
      uint32_t tile_count = 0, filled = 8;
      for (uint32_t i = 0; i < count; ++i) {
        uint32_t pair = order[i];
        if (filled == 8 || tiles[tile_count - 1].slot != (uint32_t)chosen[pair]) {
          tiles[tile_count].slot = (uint32_t)chosen[pair];
          for (int r = 0; r < 8; ++r)
            tiles[tile_count].out[r] = rows[tile_count * 8 + r] = tile_pairs[tile_count * 8 + r] = 0xFFFFFFFFu;
          ++tile_count;
          filled = 0;
        }
        tiles[tile_count - 1].out[filled] = tile_pairs[(tile_count - 1) * 8 + filled] = pair;
        rows[(tile_count - 1) * 8 + filled] = pair / TOP_K;
        ++filled;
      }
      gpu_.run("pack_gather8", {s_.mixed, s_.tile_rows, s_.expert_packed}, PackP{(uint32_t)T, H, H, tile_count}, H, tile_count);
      ExpertGemmP gate_up{tile_count, H, 2 * MOE_M, 0, (uint32_t)GATE_UP_SCALE, (uint32_t)BLOB};
      gpu_.run("expert_gemm", {s_.expert_packed, layer_blobs_, s_.tiles, s_.gate_up}, gate_up, 2 * MOE_M / 4, tile_count);
      gpu_.run("expert_act", {s_.gate_up, s_.act}, ActP{count, MOE_M}, MOE_M, count);
      gpu_.run("pack_gather8", {s_.act, s_.tile_pairs, s_.expert_packed}, PackP{count, MOE_M, MOE_M, tile_count}, MOE_M, tile_count);
      ExpertGemmP down{tile_count, MOE_M, H, (uint32_t)DOWN_WEIGHT, (uint32_t)DOWN_SCALE, (uint32_t)BLOB};
      gpu_.run("expert_gemm", {s_.expert_packed, layer_blobs_, s_.tiles, s_.routed}, down, H / 4, tile_count);
      gpu_.run("moe_combine", {s_.routed, s_.score, s_.shared, s_.shared_gate, s_.result}, CombineP{(uint32_t)T, H, TOP_K}, H, T);
      return;
    }
    std::vector<int> unique(chosen);
    std::sort(unique.begin(), unique.end());
    unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
    std::vector<uint32_t> slots;
    std::vector<std::pair<int, int>> missing;
    arena_->plan(layer, unique, slots, missing);
    // Pairs whose expert is cached go first: the GPU computes them while the missing
    // experts are read (as the MLX engine's ready-expert decode does).
    std::vector<char> late(unique.size(), 0);
    for (const auto& item : missing) late[std::lower_bound(unique.begin(), unique.end(), item.first) - unique.begin()] = 1;
    std::vector<Pair> ordered(count);
    uint32_t ready = 0, tail = count;
    for (uint32_t i = 0; i < count; ++i) {
      size_t u = std::lower_bound(unique.begin(), unique.end(), chosen[i]) - unique.begin();
      Pair pair{(uint32_t)(i / TOP_K), slots[u], i};
      if (late[u]) ordered[--tail] = pair;
      else ordered[ready++] = pair;
    }
    memcpy(pairs, ordered.data(), count * sizeof(Pair));
    Buf blobs = arena_->buffer();
    auto run_pairs = [&](uint32_t first, uint32_t n) {
      if (n == 0) return;
      Buf list = s_.pairs.at(first * sizeof(Pair));
      ExpertP gate_up{n, H, 2 * MOE_M, H, 0, (uint32_t)GATE_UP_SCALE, (uint32_t)BLOB, 0};
      gpu_.run("expert_linear4", {s_.mixed, blobs, list, s_.gate_up}, gate_up, 32, 2 * MOE_M / 4, n, kSimdRows);
      gpu_.run("expert_act_pairs", {s_.gate_up, s_.act, list}, ActP{n, MOE_M}, MOE_M, n);
      ExpertP down{n, MOE_M, H, MOE_M, (uint32_t)DOWN_WEIGHT, (uint32_t)DOWN_SCALE, (uint32_t)BLOB, 1};
      gpu_.run("expert_linear4", {s_.act, blobs, list, s_.routed}, down, 32, H / 4, n, kSimdRows);
    };
    run_pairs(0, ready);
    if (!missing.empty()) {
      // Starting the cached pairs first costs one more submission (about 0.14 ms), which
      // pays only when enough experts are read. The reads fill other slots than these.
      if (ready > 0 && missing.size() >= OVERLAP_MIN_READS) gpu_.commit();
      arena_->load(layer, missing);
      run_pairs(ready, count - ready);
    }
    gpu_.run("moe_combine", {s_.routed, s_.score, s_.shared, s_.shared_gate, s_.result}, CombineP{(uint32_t)T, H, TOP_K}, H, T);
  }

  // -------------------------------------------------------------- PLE: adds the N-gram term to h_
  void ple(const Weights& w, const LayerW& lw, const int32_t* tokens, int T, const int32_t* history) {
    ngram_->lookup(history, tokens, T, (float*)s_.embedding.buffer.contents);
    linear(s_.embedding, lw.ple_key, s_.up, T, H, W);
    norm(s_.up, w.vec(lw.ple_norm_key), s_.up, T, C, H, W, 0, C, true, 1.0f);
    norm(h_, w.vec(lw.ple_norm_query), s_.normed, T, C, H, W, 0, C, true, 1.0f);
    gpu_.run("ple_gate", {s_.up, s_.normed, s_.inject}, PleP{(uint32_t)T, H, C}, C, T);
    linear(s_.embedding, lw.ple_value, s_.mixed, T, H, H, true);
    gpu_.run("ple_value", {s_.inject, s_.mixed, s_.value}, PleP{(uint32_t)T, H, C}, W, T);
    Buf normed = logging_ ? ple_log_ : s_.normed;  // commit() redoes the state update from these rows
    norm(s_.value, w.vec(lw.ple_norm_conv), normed, T, C, H, W, 0, C, true, 1.0f);
    ConvP conv{(uint32_t)T, W, 4, 3, PLE_STATE, 1};
    gpu_.run("conv_depthwise", {ple_conv_[ple_current_], normed, lw.ple_conv, s_.up}, conv, W, T);
    gpu_.run("conv_state_next", {ple_conv_[ple_current_], normed, ple_conv_[1 - ple_current_]}, conv, W, PLE_STATE);
    ple_current_ = 1 - ple_current_;
    gpu_.run("add2", {h_, s_.value, s_.up}, CountP{(uint32_t)(T * W), 1.0f}, (size_t)T * W);
  }

  // Start reading one whole expert file in the background, into the buffer the GPU is not using.
  void read_layer(int file) {
    arena_->wait();  // seeding may still be copying out of the buffer this read overwrites
    layer_slot_ = 1 - layer_slot_;
    Buf& target = layer_buffers_[layer_slot_];
    if (target.buffer == nil) target = gpu_.alloc((size_t)EXPERTS * BLOB);
    if (layer_read_ == nil) layer_read_ = dispatch_group_create();
    void* destination = target.buffer.contents;
    const File* source = &arena_->file(file);
    int threads = options_.read_threads;
    bool* failed = &layer_read_failed_;
    layer_started_ = true;
    dispatch_group_async(layer_read_, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      try { source->read_parallel(destination, (size_t)EXPERTS * BLOB, 0, threads); }
      catch (...) { *failed = true; }
    });
  }

  void dump(int layer) {
    gpu_.flush();
    char name[32];
    snprintf(name, sizeof(name), "/p%05d_l%02d.bin", position_, layer);
    FILE* file = fopen((options_.dump_dir + name).c_str(), "wb");
    if (file == nullptr) fail("cannot write to the dump directory");
    fwrite((const float*)all_.buffer.contents + W, 4, W, file);
    fclose(file);
  }

  Options options_;
  Gpu gpu_;
  std::unique_ptr<Weights> common_;
  std::vector<LayerW> layers_;
  HyperW mixer_;
  const uint16_t* embed_ = nullptr;
  Mat lm_head_;
  Buf logits_, ple_conv_[2], ple_log_, layer_blobs_, layer_buffers_[2];
  int ple_current_ = 0;
  dispatch_group_t layer_read_ = nil;
  bool layer_read_failed_ = false, layer_started_ = false;
  int layer_slot_ = 0;
  bool logging_ = false;
  std::vector<int> seed_counts_ = std::vector<int>(EXPERTS, 0);
  LayerW dense_layer_;
  Buf dense_;
  std::vector<LinearLog> logs_;
  // Draft layer (MTP)
  std::unique_ptr<Weights> mtp_;
  LayerW mtp_layer_;
  HyperW mtp_mixer_;
  Mat mtp_fc_embedding_, mtp_fc_hidden_;
  Buf mtp_h_, mtp_in_;
  size_t mtp_norm_embedding_ = 0, mtp_norm_hidden_ = 0;
  int mtp_file_ = -1, mtp_checkpoint_ = 0, drafted_ = 0;
  AttentionState mtp_attention_;
  std::vector<float> boundary_, pending_hidden_;
  std::vector<int32_t> pending_tokens_, verified_;
  int32_t verify_history_[2] = {EOS, EOS};
  Buf all_, h_, partial_;  // wide hidden rows of the current call, and the chunk being processed
  size_t all_bytes_ = 0;
  std::unique_ptr<NGramStore> ngram_;
  std::unique_ptr<ExpertArena> arena_;
  std::vector<LinearState> linear_;
  std::vector<AttentionState> attention_;
  int32_t history_[2] = {EOS, EOS};
  int position_ = 0;
  int slots_ = 0;
  std::function<bool()> cancel_;
  ImagePrompt image_;
  std::string root_;
  std::vector<int32_t> tokens_;
  std::string contract_;
};

}  // namespace

std::unique_ptr<Engine> Engine::open(const std::string& model_dir, const Options& options) {
  @autoreleasepool {
    return std::unique_ptr<Engine>(new EngineImpl(model_dir, options));
  }
}

}  // namespace wq
