import DeepSeekRepack
import Foundation

enum ExpertMemory {
  static let gib = 1_073_741_824.0

  static func bytes(gib value: Double) throws -> UInt64 {
    guard value.isFinite, value > 0, value <= 1_024, value * gib >= 1 else {
      throw ConfigurationError(L10n.string("Expert cache memory must be greater than 0 and at most 1024 GiB."))
    }
    return UInt64((value * gib).rounded(.down))
  }

  // Pinned installation contracts. A loaded manifest is authoritative.
  static func blobBytes(for kind: ModelKind) -> UInt64 {
    switch kind {
    case .deepSeekV4: 13_369_344
    case .deepSeekV41: 18_800_640
    case .qwen3_8FlashNext: 2_611_200
    default: 0
    }
  }

  static func capacity(gib: Double, blobBytes: UInt64, minimum: Int) throws -> Int {
    let budget = try bytes(gib: gib)
    guard blobBytes > 0, budget / blobBytes >= minimum else {
      throw ConfigurationError(L10n.string("Expert cache memory is too small for this model."))
    }
    return Int(budget / blobBytes)
  }

  static func sliderRange(physicalMemory: UInt64) -> ClosedRange<Double> {
    // Stay on the 0.1 GiB grid without exceeding installed physical memory.
    0.1...max(0.1, (Double(physicalMemory) / gib * 10).rounded(.down) / 10)
  }

  static func sliderValue(_ value: Double, in range: ClosedRange<Double>) -> Double {
    guard value.isFinite else { return range.lowerBound }
    return min(range.upperBound, max(range.lowerBound, roundedGiB(value)))
  }

  static func roundedGiB(_ value: Double) -> Double {
    (value * 10).rounded() / 10
  }

  static func defaultGiB(slots: Int, blobBytes: UInt64) -> Double {
    roundedGiB(legacyGiB(slots: slots, blobBytes: blobBytes))
  }

  static func legacyGiB(slots: Int, blobBytes: UInt64) -> Double {
    Double(slots) * Double(blobBytes) / gib
  }
}

/// Planning arithmetic only: no model loading, GPU allocations, or measured claims.
struct MemoryPlanningProfile {
  let kind: ModelKind
  let manifest: InstalledManifest
  let config: [String: Any]

  static func load(at url: URL, kind: ModelKind) -> MemoryPlanningProfile? {
    guard let manifest = try? InstalledModel.loadManifest(at: url),
      let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
      let raw = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    let config = raw["text_config"] as? [String: Any] ?? raw
    guard let hidden = (config["hidden_size"] ?? config["dim"]) as? Int, hidden > 0,
      let heads = (config["num_attention_heads"] ?? config["n_heads"]) as? Int, heads > 0,
      let dim = config["head_dim"] as? Int, dim > 0,
      manifest.expertBlobSize > 0,
      manifest.files.contains(where: { $0.path == "common.bin" && $0.size > 0 })
    else { return nil }
    return MemoryPlanningProfile(kind: kind, manifest: manifest, config: config)
  }

  private func number(_ name: String, _ fallback: Double) -> Double {
    let aliases = ["hidden_size": "dim", "num_attention_heads": "n_heads",
      "sliding_window": "window_size", "vocab_size": "vocab_size"]
    return (config[name] as? NSNumber)?.doubleValue
      ?? aliases[name].flatMap { (config[$0] as? NSNumber)?.doubleValue } ?? fallback
  }

  private func integers(_ name: String, alias: String) -> [Int]? {
    config[name] as? [Int] ?? config[alias] as? [Int]
  }

  /// Estimate allocations that can coexist, not the sum of every execution phase.
  /// Capacity planning with retained prompt caches, not a prediction of cache occupancy.
  /// Fixed context lengths assume input-heavy requests. This is not a measured peak or hard limit.
  func estimate(_ s: ModelAdvancedSettings, mtpAvailable: Bool, dsparkAvailable: Bool,
    contextTokens: Int? = nil
  ) -> MemoryEstimate? {
    let input = contextTokens ?? s.estimateInputTokens ?? 4_096
    let output = contextTokens == nil ? s.defaultMaxTokens : 0
    let maximum = manifest.maximumContext ?? Int(number("max_position_embeddings", 1_048_576))
    let ratio = s.anePrefillRatio ?? 0
    guard [.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext].contains(kind),
      input > 0, output >= 0, (contextTokens != nil || output > 0),
      input <= maximum, output <= maximum - input,
      s.readWorkers > 0, (s.prefetchReadWorkers ?? 2) > 0,
      s.prefillStepSize >= 0, (s.moePrefillStepSize ?? 0) >= 0,
      s.promptCacheMemoryGiB >= 1, s.promptCacheEntries >= 1,
      ratio.isFinite, (0...1).contains(ratio)
    else { return nil }
    let blob = manifest.expertBlobSize
    func payload(_ budget: Double?, _ slots: Int, _ minimum: Int) -> Double? {
      let value = budget ?? ExpertMemory.legacyGiB(slots: slots, blobBytes: blob)
      guard let count = try? ExpertMemory.capacity(gib: value, blobBytes: blob, minimum: minimum) else { return nil }
      return Double(count) * Double(blob)
    }
    func file(_ name: String) -> Double { Double(manifest.files.first { $0.path == name }?.size ?? 0) }
    guard let expert = payload(s.expertCacheGiB, s.slots, manifest.selectedExpertCount), file("common.bin") > 0 else { return nil }
    let qwen = kind == .qwen3_8FlashNext
    if qwen {
      do { try s.validateQwenFlashSettings() } catch { return nil }
    }
    let flashWaves = qwen && s.qwenFlashWavesEnabled
    // Packed LRU payload only; Python objects and OS pages are outside this estimate.
    let ngramCache = qwen ? Double(s.effectiveQwenNgramCacheBytes) : 0
    let v41 = kind == .deepSeekV41
    let mtp = qwen && s.mtpEnabled == true && mtpAvailable
    let dspark = !qwen && s.dsparkEnabled && dsparkAvailable
    if mtp && manifest.mtp == nil || dspark && manifest.dspark == nil { return nil }

    let hidden = number("hidden_size", qwen ? 2_560 : v41 ? 5_120 : 4_096)
    let dim = number("head_dim", qwen ? 256 : 512)
    let heads = number("num_attention_heads", qwen ? 24 : 64)
    let hc = number("hc_count", number("hc_mult", 4))
    let window = number("sliding_window", 128)
    let indexDim = number("indexer_head_dim", number("index_head_dim", 128))
    let indexHeads = number("indexer_n_heads", number("index_n_heads", qwen ? 4 : 32))
    let selected = Double(manifest.selectedExpertCount)
    let topk = number("indexer_budget", number("index_topk", qwen ? 2_048 : 512))
    let intermediate = number("moe_intermediate_size", number("moe_inter_dim", qwen ? 640 : v41 ? 2_304 : 2_048))
    let vocab = number("vocab_size", qwen ? 248_320 : 129_280)
    guard [hidden, dim, heads, hc, window, indexDim, indexHeads, topk, intermediate, vocab]
      .allSatisfy({ $0.isFinite && $0 > 0 }) else { return nil }
    let tokens = Double(input + output)
    let step = min(Double(input), Double(s.prefillStepSize == 0
      ? (input < 1_024 ? 128 : input < 4_096 ? 256 : 1_024) : s.prefillStepSize))
    // V4.1 can seed DSpark from layer-major prefill; V4 still uses chunks.
    let threshold = qwen ? 128 : s.layerMajorPrefillThreshold ?? 1_024
    let layerMajor = s.layerMajorPrefill && (!dspark || v41) && input >= threshold
    let batched = layerMajor && s.batchedExpertPrefill == true && !flashWaves
    let moeStep = v41 && layerMajor ? min(Double(input), 4_096)
      : !qwen && layerMajor
        ? min(Double(input), Double((s.moePrefillStepSize ?? 0) == 0 ? 4_096 : s.moePrefillStepSize!)) : step
    guard let cache = cacheFootprint(s, tokens: tokens, step: step, layerMajor: layerMajor,
      dim: dim, hidden: hidden, window: window, indexDim: indexDim) else { return nil }

    // Native V4.1 expands per-32-output-row FP8 scales to per-output-row scales.
    let scaleExpansion = v41 ? manifest.commonTensors.filter { $0.name.hasSuffix(".scale") }
      .reduce(0.0) { $0 + Double($1.length) * 31 } : 0
    let weights = file("common.bin") + scaleExpansion
    let largestTensor = manifest.commonTensors.map { Double($0.length) }.max() ?? 0
    let embedding = manifest.commonTensors.first {
      $0.name.hasSuffix("embed_tokens.weight") || $0.name == "embed.weight"
    }
    let activationBytes = embedding?.dtype == "F32" ? 4.0 : 2.0

    // ANE projections remain compiled while the model is loaded. Count selected
    // FP16 rows and input/output surfaces; compilation overhead is in the margin.
    let aneLayers = qwen ? number("num_hidden_layers", Double(manifest.layerCount)) / 4 : Double(manifest.layerCount)
    let aneOutput = heads * dim * (qwen ? 2 : 1)
    let aneRows = min(aneOutput, (aneOutput * ratio / 256).rounded() * 256)
    let aneInput = qwen ? hidden : number("q_lora_rank", 1_280)
    let ane = (qwen || s.deepSeekANEPrefill == true) && aneRows > 0
      ? aneLayers * (aneRows * aneInput * 2 + 1_024 * (aneInput + aneRows) * 2) : 0
    var auxiliaryWeights = 0.0
    var auxiliaryExperts = 0.0
    var auxiliaryState = 0.0
    if mtp {
      guard let budget = payload(s.mtpCacheGiB, s.mtpSlots ?? 32, 10), file("mtp/common.bin") > 0 else { return nil }
      auxiliaryWeights = file("mtp/common.bin")
      auxiliaryExperts = budget
      // MTP's own attention is uncompressed. Keep a full-context allowance even
      // though layer-major prefill initially supplies only the final paired window.
      auxiliaryState = ceil(tokens / 256) * 256 * (2 * number("num_key_value_heads", 2) * dim * 2 + 2 * indexDim * 2)
    }
    if dspark {
      guard let descriptor = manifest.dspark,
        let budget = payload(s.dsparkCacheGiB, s.dsparkSlots, max(30, manifest.selectedExpertCount * descriptor.blockSize)),
        file("dspark/common.bin") > 0 else { return nil }
      auxiliaryWeights = file("dspark/common.bin")
      auxiliaryExperts = budget
      auxiliaryState = Double(descriptor.layerCount) * window * dim * (v41 ? 4 : 2 * activationBytes)
    }
    let auxiliary = auxiliaryWeights + auxiliaryExperts + auxiliaryState
    // App disables ordinary Prompt Cache reuse for MTP/DSpark. Otherwise reserve
    // only snapshots that fit the entry count and budget, not an empty 8 GiB arena.
    // Runtime always keeps its newest entry, even when that entry exceeds the budget.
    let retained = s.promptCacheMode == .off || mtp || dspark ? 0
      : min(Double(s.promptCacheEntries) * cache.bytes, max(cache.bytes, Double(s.promptCacheMemoryGiB) * ExpertMemory.gib))
    let conversation = cache.bytes + retained
    let allocator = ExpertMemory.gib // runtime's actual mx.set_cache_limit
    // Aligned prefill staging is retained by reader workers until unload.
    let extraStaging = qwen && s.qwenWholeLayerExperimentsActive
      ? Double(blob) * Double((s.qwenPrefillReadExperts ?? 1) - 1) * Double(s.readWorkers) : 0
    let reads = Double(blob) * Double(s.readWorkers + (s.prefetchReadWorkers ?? 2)) + extraStaging
    // V4 pipelines the next expert layer itself; Qwen/V4.1 expose this as a switch.
    let layerCopies = kind == .deepSeekV4 || s.nextLayerPrefetch == true ? 2.0 : 1.0
    let seeded = qwen && batched
      ? min(expert, Double(blob) * Double(manifest.layerCount) * Double(s.qwenPrefillSeedExperts ?? 0)) : 0
    let prefillExperts = batched ? Double(blob) * Double(manifest.expertCount) * layerCopies + seeded : expert

    func attentionWork(_ queries: Double) -> Double {
      if qwen {
        // Structural upper estimate; runtime may reduce chunks for workspace.
        let micro = min(queries, Double(s.qwenQSAQueryChunk ?? 4))
        if s.qwenSparseSDPA == true && s.qwenQSAIndexed == true && queries <= 8 && tokens > topk {
          let partials = micro * heads * 32 * (dim + 2) * 4
          let indexScores = micro * ceil(tokens / number("indexer_compress_ratio", 4)) * indexHeads * 8
          return partials + indexScores + queries * heads * dim * activationBytes * 3
        }
        let gathered = micro * min(tokens, topk + number("indexer_compress_ratio", 4))
          * (2 * number("num_key_value_heads", 2) * dim * activationBytes + heads * 8)
        let scores = micro * ceil(tokens / number("indexer_compress_ratio", 4)) * indexHeads * 8
        return max(gathered, scores) + queries * heads * dim * activationBytes * 3
      }
      // V4.1 sparse attention gathers 256 queries at a time; its indexer still
      // scores a full prefill chunk. Candidate filtering does not shrink the source's work.
      let gatherQueries = v41 ? min(queries, 256) : queries
      let gather = gatherQueries * (min(cache.maxHistoryRows, topk) + window) * (dim * 4 + heads * 8)
      let indexScores = queries * cache.maxHistoryRows * indexHeads * 8
      return max(gather, indexScores) + queries * heads * dim * activationBytes * 3
    }
    func moeWork(_ count: Double) -> Double {
      count * selected * (hidden + 2 * intermediate) * activationBytes
        + count * hidden * hc * 4 * 2 // FP32 hyper-connection mixing
    }
    // Full-prompt hidden arrays exist only in layer-major prefill. Include input,
    // chunk outputs and concatenation; both DeepSeek models retain attention results.
    let hiddenCopies = qwen ? 3.0 : 4.0
    let capturedHidden = v41 && dspark && layerMajor
      ? Double(input) * hidden * Double(manifest.dspark?.targetLayerIDs.count ?? 3) * 4 * 2 : 0
    let promptHidden = (layerMajor ? Double(input) * hidden * hc * activationBytes * hiddenCopies : 0) + capturedHidden
    let logits = layerMajor ? 0 : step * vocab * (v41 ? 4 : activationBytes)
    let prefillWork = max(attentionWork(step), moeWork(moeStep), logits) + cache.growth
    // Each layer-major chunk retains one SharedState. Its arrays use that
    // chunk's visible history, not the final cache's reserved capacity.
    var sharedIndices = 0.0
    if v41 {
      let ratios = config["compress_ratios"] as? [Int] ?? []
      let source = Int(number("candidate_source_layer_id", number("candidate_source_layer", -1)))
      let candidateRatio = ratios.indices.contains(source) ? max(1, ratios[source]) : 1
      let blockSize = max(1, number("candidate_block_size", 1))
      let blockCount = max(0, number("candidate_topk_blocks", 0))
      let begins = layerMajor ? stride(from: 0, to: input, by: Int(step))
        : stride(from: max(0, input - Int(step)), to: input, by: Int(step))
      for begin in begins {
        let end = min(input, begin + Int(step))
        let queries = Double(end - begin)
        let rows = floor(Double(end) / Double(candidateRatio))
        sharedIndices += queries * min(topk, rows) * 4 // top-k is explicitly int32
        if source >= 0 {
          // _candidate_blocks combines uint32 argpartition with int32 arange,
          // so retained candidate positions are int64. Masks remain bool.
          let positions = min(blockCount, ceil(rows / blockSize)) * blockSize
          sharedIndices += queries * (rows + positions * 8)
        }
      }

      // The old measured allowance described interleaved attention/FFN chunks.
      // Split attention/FFN and pooled layer buffers need new peak measurements;
      // use the structural allowance above instead of reusing that calibration.

    }
    let decodeQueries = mtp ? 6.0 : dspark ? Double(manifest.dspark?.blockSize ?? 5) : 1.0
    // Verification forks main state only in speculative generation, not all phases.
    let verification = mtp || dspark ? cache.bytes : 0
    let decodeWork = max(attentionWork(decodeQueries), moeWork(decodeQueries), decodeQueries * vocab * 4) + cache.growth
    // Plain chunked prefill retains a first-chunk and a last-chunk checkpoint.
    // One full snapshot is included in retained; keep only the small first one here.
    let checkpoint = retained > 0 && !layerMajor ? min(cache.bytes, cache.bytes * step / tokens) : 0
    let auxiliaryTensor = mtp ? manifest.mtp?.commonTensors : dspark ? manifest.dspark?.commonTensors : nil
    let loadingCopy = max(largestTensor, auxiliaryTensor?.map { Double($0.length) }.max() ?? 0)
    let loading = MemoryStageEstimate(model: weights, conversation: 0, auxiliary: auxiliaryWeights,
      temporary: loadingCopy + ane + allocator)
    let prefill = MemoryStageEstimate(model: weights + prefillExperts, conversation: conversation,
      auxiliary: auxiliary, temporary: promptHidden + sharedIndices + prefillWork + checkpoint + reads + ane + allocator + ngramCache)
    // Qwen MTP retains prefilled_hidden through its generation iterator.
    let retainedHidden = mtp && layerMajor ? Double(input) * hidden * hc * activationBytes : 0
    let decoding = MemoryStageEstimate(model: weights + expert, conversation: conversation,
      auxiliary: auxiliary + verification, temporary: decodeWork + retainedHidden + checkpoint + reads + ane + allocator + ngramCache)
    let result = MemoryEstimate(loading: loading, prefill: prefill, decoding: decoding,
      inputTokens: input, outputTokens: output)
    return result.total.isFinite ? result : nil
  }

  /// Retained arrays plus a single active layer's growth/unpacking temporary.
  /// V4.1 cache owners and storage formats mirror deepseek_v41/cache.py.
  private func cacheFootprint(_ s: ModelAdvancedSettings, tokens: Double, step: Double,
    layerMajor: Bool, dim: Double, hidden: Double, window: Double, indexDim: Double
  ) -> CacheFootprint? {
    let layers = Double(manifest.layerCount)
    if kind == .qwen3_8FlashNext {
      let types = config["layer_types"] as? [String]
      let full = types.map { Double($0.filter { $0 == "full_attention" }.count) }
        ?? ceil(layers / number("full_attention_interval", 4))
      let capacity = ceil(tokens / 256) * 256
      let kvHeads = number("num_key_value_heads", 2)
      let kvWidth = 2 * kvHeads * dim
      let indexWidth = 2 * indexDim
      let perLayer = capacity * (kvWidth * (s.packedKVCache == true ? 1.125 : 2)
        + indexWidth * (s.packedIndexCache == true ? 0.625 : 2))
      let recurrent = max(0, layers - full) * number("linear_num_value_heads", 48)
        * number("linear_key_head_dim", 128) * number("linear_value_head_dim", 128) * 4
      let convWidth = 2 * number("linear_num_key_heads", 16) * number("linear_key_head_dim", 128)
        + number("linear_num_value_heads", 48) * number("linear_value_head_dim", 128)
      let conv = max(0, layers - full) * convWidth * number("linear_conv_kernel_dim", 4) * 2
      let ple = number("ngram_size", 3) * number("ple_conv_kernel_size", 4) * hidden * number("hc_count", 4) * 2
      let unpack = capacity * ((s.packedKVCache == true ? kvWidth * 2 : 0)
        + (s.packedIndexCache == true ? indexWidth * 2 : 0))
      return CacheFootprint(bytes: full * perLayer + recurrent + conv + ple,
        growth: perLayer + unpack, maxHistoryRows: capacity)
    }
    let v41 = kind == .deepSeekV41
    let ratios = config["compress_ratios"] as? [Int]
      ?? (0..<manifest.layerCount).map { $0 == 0 || $0 == manifest.layerCount - 1 ? 0 : ($0 % 2 == 0 ? 4 : 128) }
    let sources = v41 ? integers("kv_source_layer_ids", alias: "kv_source_layers") ?? [] : Array(0..<manifest.layerCount)
    let indexSources = Set(integers("index_source_layer_ids", alias: "index_source_layers") ?? sources)
    guard ratios.count >= manifest.layerCount, !sources.isEmpty,
      sources.allSatisfy({ $0 >= 0 && $0 < manifest.layerCount && ratios[$0] >= (v41 ? 1 : 0) }) else { return nil }
    // V4.1 grows capacity by doubling; layer-major ensures the full input in one call.
    var capacity = max(4_096.0, layerMajor ? tokens : 4_096)
    if v41 && !layerMajor { while capacity < tokens { capacity *= 2 } }
    capacity = min(capacity, Double(manifest.maximumContext ?? 1_048_576))
    // DeepSeekV41ForCausalLM.make_cache explicitly selects BF16 (not the native default FP32).
    let windowBytes = v41 ? (s.packedKVCache == true ? 1.03125 : 2.0) : 4.0
    var state = layers * window * dim * windowBytes
    var growth = 0.0
    var maxRows = 0.0
    for layer in sources where ratios[layer] > 0 {
      let ratio = Double(ratios[layer])
      let rows = v41 ? floor(capacity / ratio) : ceil(tokens / ratio)
      maxRows = max(maxRows, v41 ? floor(tokens / ratio) : rows)
      let hasIndex = v41 ? indexSources.contains(layer) : ratios[layer] == 4
      let width = dim + (hasIndex ? indexDim : 0)
      let stored: Double
      if v41 {
        stored = rows * (dim * (s.packedKVCache == true ? 0.5625 : 2)
          + (hasIndex ? indexDim * (s.packedIndexCache == true ? 0.53125 : 2) : 0))
        // Both compression accumulators use FP32, even with packed storage.
        state += ratio > 1 ? ratio * dim * 8 : 0
      } else {
        // MXFP8PoolingCache keeps 8-bit and 4-bit chunks. It also materializes
        // concatenated packed rows when accessed; BF16 keeps one pooled array.
        stored = rows * width * (s.bf16KVCache ? 2 : (1.03125 + 0.53125) * 2)
        state += ratio * width * 8 * 2 + 64 * width * 2
      }
      state += stored
      growth = max(growth, stored + rows * width * 4)
    }
    if v41 { state += capacity * 8 } // Engram token history, not the mapped embedding tables.
    return CacheFootprint(bytes: state, growth: growth, maxHistoryRows: maxRows)
  }
}

private struct CacheFootprint {
  let bytes: Double
  let growth: Double
  let maxHistoryRows: Double
}

struct MemoryStageEstimate {
  let model: Double
  let conversation: Double
  let auxiliary: Double
  let temporary: Double
  var total: Double { model + conversation + auxiliary + temporary }
}

struct MemoryEstimate {
  let loading: MemoryStageEstimate
  let prefill: MemoryStageEstimate
  let decoding: MemoryStageEstimate
  let inputTokens: Int
  let outputTokens: Int
  var peak: MemoryStageEstimate { [loading, prefill, decoding].max { $0.total < $1.total }! }
  // One explicitly uncalibrated allowance, applied after selecting the largest phase.
  var margin: Double { peak.total * 0.10 + 0.5 * ExpertMemory.gib }
  var total: Double { peak.total + margin }
  var model: Double { peak.model }
  var conversation: Double { peak.conversation }
  var auxiliary: Double { peak.auxiliary }
}
