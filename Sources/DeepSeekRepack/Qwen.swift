import Foundation

enum QwenContract {
  static let modelID = "Qwen/Qwen3.8-Flash-Next-FP8"
  static let revision = "bcd9f01ddc9cff2316eb84281bebcd5b058bddce"
  static let layerCount = 48
  static let expertCount = 512
  static let selectedExpertCount = 10
  static let hiddenSize = 2_560
  static let expertIntermediateSize = 640
  static let maximumContext = 262_144
  static let mtpLayerCount = 1
  static let mtpCommonTensorCount = 29
  static let ngramShardCount = 128
  static let ngramShardRowCount = 2_500_012
  static let ngramRowBytes = 160
  static let commonAlignment: UInt64 = 256
  static let companions = [
    CompanionFile(source: "config.json", destination: "config.json"),
    CompanionFile(source: "generation_config.json", destination: "generation_config.json"),
    CompanionFile(source: "tokenizer.json", destination: "tokenizer/tokenizer.json"),
    CompanionFile(source: "tokenizer_config.json", destination: "tokenizer/tokenizer_config.json"),
    CompanionFile(source: "chat_template.jinja", destination: "tokenizer/chat_template.jinja"),
    CompanionFile(source: "vocab.json", destination: "tokenizer/vocab.json"),
    CompanionFile(source: "merges.txt", destination: "tokenizer/merges.txt"),
  ]

  static let quantization = ExpertQuantizationDescriptor(
    mode: "mxfp4", bits: 4, groupSize: 32, conversionVersion: 2)
  static let ngram = NGramDescriptor(
    file: "ngram.bin",
    dtype: "F8_E4M3",
    rowBytes: ngramRowBytes,
    shardCount: ngramShardCount,
    shardRowCount: ngramShardRowCount,
    headOffsets: [
      0, 20_000_003, 40_000_026, 60_000_059, 80_000_106, 100_000_165,
      120_000_228, 140_000_297, 160_000_374, 180_000_455, 200_000_548,
      220_000_655, 240_000_802, 260_000_955, 280_001_114, 300_001_275,
    ],
    headVocabSizes: [
      20_000_003, 20_000_023, 20_000_033, 20_000_047, 20_000_059, 20_000_063,
      20_000_069, 20_000_077, 20_000_081, 20_000_093, 20_000_107, 20_000_147,
      20_000_153, 20_000_159, 20_000_161, 20_000_171,
    ]
  )
  static let ngramScaleName =
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"

  static let expertRegions: [ExpertRegion] = {
    var offset: UInt64 = 0
    func region(_ name: String, _ dtype: String, _ shape: [Int], _ bytes: UInt64)
      -> ExpertRegion
    {
      defer { offset += bytes }
      return ExpertRegion(name: name, dtype: dtype, shape: shape, offset: offset, length: bytes)
    }
    return [
      region("gate_up.weight", "U32", [1_280, 320], 1_638_400),
      region("gate_up.scale", "U8", [1_280, 80], 102_400),
      region("down.weight", "U32", [2_560, 80], 819_200),
      region("down.scale", "U8", [2_560, 20], 51_200),
    ]
  }()
  static let expertBlobSize = expertRegions.reduce(UInt64(0)) { $0 + $1.length }

  static func validate(_ config: QwenConfig) throws {
    let text = config.textConfig
    let actual: [(String, String, String)] = [
      ("architecture", config.architectures.joined(separator: ","), "Qwen4ExpForConditionalGeneration"),
      ("model_type", config.modelType, "qwen4_exp"),
      ("text_config.model_type", text.modelType, "qwen4_exp_text"),
      ("text_config.dtype", text.dtype, "bfloat16"),
      ("quantization_config.quant_method", config.quantizationConfig.method, "fp8"),
      ("quantization_config.activation_scheme", config.quantizationConfig.activationScheme, "dynamic"),
      (
        "quantization_config.weight_block_size",
        config.quantizationConfig.weightBlockSize.map(String.init).joined(separator: ","),
        "128,128"
      ),
      ("text_config.hidden_size", String(text.hiddenSize), String(hiddenSize)),
      ("text_config.moe_intermediate_size", String(text.moeIntermediateSize), String(expertIntermediateSize)),
      ("text_config.num_experts", String(text.expertCount), String(expertCount)),
      ("text_config.num_experts_per_tok", String(text.selectedExpertCount), String(selectedExpertCount)),
      ("text_config.num_hidden_layers", String(text.hiddenLayerCount), String(layerCount)),
      ("text_config.max_position_embeddings", String(text.maximumContext), String(maximumContext)),
      ("text_config.mtp_num_hidden_layers", String(text.mtpLayerCount), String(mtpLayerCount)),
      ("text_config.mtp_use_dedicated_embeddings", String(text.mtpUsesDedicatedEmbeddings), "false"),
      ("text_config.split_ngram_parts", String(text.ngramShardCount), String(ngramShardCount)),
      ("text_config.num_attention_heads", String(text.attentionHeadCount), "24"),
      ("text_config.num_key_value_heads", String(text.keyValueHeadCount), "2"),
      ("text_config.head_dim", String(text.headDimension), "256"),
      ("text_config.linear_num_value_heads", String(text.linearValueHeadCount), "48"),
      ("text_config.linear_num_key_heads", String(text.linearKeyHeadCount), "16"),
      ("text_config.linear_key_head_dim", String(text.linearKeyHeadDimension), "128"),
      ("text_config.linear_value_head_dim", String(text.linearValueHeadDimension), "128"),
      ("text_config.output_gate_type", text.outputGateType, "sigmoid"),
      ("text_config.shared_expert_intermediate_size", String(text.sharedExpertIntermediateSize), "640"),
      ("text_config.indexer_budget", String(text.indexerBudget), "2048"),
      ("text_config.indexer_compress_ratio", String(text.indexerCompressRatio), "4"),
      ("text_config.hc_count", String(text.hyperConnectionCount), "4"),
      ("text_config.hc_lowrank", String(text.hyperConnectionLowRank), "320"),
      ("text_config.ngram_size", String(text.ngramSize), "3"),
      ("text_config.heads_per_ngram", String(text.headsPerNGram), "8"),
      ("text_config.ple_embed_dim", String(text.pleEmbeddingDimension), "2560"),
      ("text_config.ple_layer_ids", text.pleLayerIDs.map(String.init).joined(separator: ","), "2"),
      (
        "text_config.layer_types",
        text.layerTypes.joined(separator: ","),
        (0..<layerCount).map { ($0 + 1).isMultiple(of: 4) ? "full_attention" : "linear_attention" }.joined(separator: ",")
      ),
    ]
    if let mismatch = actual.first(where: { $0.1 != $0.2 }) {
      throw RepackError.incompatibleModel(
        "\(mismatch.0) is \(mismatch.1); expected \(mismatch.2)")
    }
  }
}

struct QwenConfig: Decodable, Sendable {
  let architectures: [String]
  let modelType: String
  let textConfig: TextConfig
  let quantizationConfig: QuantizationConfig

  enum CodingKeys: String, CodingKey {
    case architectures
    case modelType = "model_type"
    case textConfig = "text_config"
    case quantizationConfig = "quantization_config"
  }

  struct QuantizationConfig: Decodable, Sendable {
    let method: String
    let activationScheme: String
    let weightBlockSize: [Int]

    enum CodingKeys: String, CodingKey {
      case method = "quant_method"
      case activationScheme = "activation_scheme"
      case weightBlockSize = "weight_block_size"
    }
  }

  struct TextConfig: Decodable, Sendable {
    let modelType: String
    let dtype: String
    let hiddenSize: Int
    let moeIntermediateSize: Int
    let expertCount: Int
    let selectedExpertCount: Int
    let hiddenLayerCount: Int
    let maximumContext: Int
    let mtpLayerCount: Int
    let mtpUsesDedicatedEmbeddings: Bool
    let ngramShardCount: Int
    let attentionHeadCount: Int
    let keyValueHeadCount: Int
    let headDimension: Int
    let linearValueHeadCount: Int
    let linearKeyHeadCount: Int
    let linearKeyHeadDimension: Int
    let linearValueHeadDimension: Int
    let outputGateType: String
    let sharedExpertIntermediateSize: Int
    let indexerBudget: Int
    let indexerCompressRatio: Int
    let hyperConnectionCount: Int
    let hyperConnectionLowRank: Int
    let ngramSize: Int
    let headsPerNGram: Int
    let pleEmbeddingDimension: Int
    let pleLayerIDs: [Int]
    let layerTypes: [String]

    enum CodingKeys: String, CodingKey {
      case modelType = "model_type"
      case dtype
      case hiddenSize = "hidden_size"
      case moeIntermediateSize = "moe_intermediate_size"
      case expertCount = "num_experts"
      case selectedExpertCount = "num_experts_per_tok"
      case hiddenLayerCount = "num_hidden_layers"
      case maximumContext = "max_position_embeddings"
      case mtpLayerCount = "mtp_num_hidden_layers"
      case mtpUsesDedicatedEmbeddings = "mtp_use_dedicated_embeddings"
      case ngramShardCount = "split_ngram_parts"
      case attentionHeadCount = "num_attention_heads"
      case keyValueHeadCount = "num_key_value_heads"
      case headDimension = "head_dim"
      case linearValueHeadCount = "linear_num_value_heads"
      case linearKeyHeadCount = "linear_num_key_heads"
      case linearKeyHeadDimension = "linear_key_head_dim"
      case linearValueHeadDimension = "linear_value_head_dim"
      case outputGateType = "output_gate_type"
      case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
      case indexerBudget = "indexer_budget"
      case indexerCompressRatio = "indexer_compress_ratio"
      case hyperConnectionCount = "hc_count"
      case hyperConnectionLowRank = "hc_lowrank"
      case ngramSize = "ngram_size"
      case headsPerNGram = "heads_per_ngram"
      case pleEmbeddingDimension = "ple_embed_dim"
      case pleLayerIDs = "ple_layer_ids"
      case layerTypes = "layer_types"
    }
  }
}

enum QwenMTPPlanner {
  static let commonPath = "mtp/common.bin"
  static let expertPath = "mtp/experts/layer_00.bin"

  static func makePlan(index: CheckpointIndex, tensors: [String: SafeTensor]) throws
    -> RepackPlan
  {
    guard tensors.count == index.weightMap.count,
      tensors.keys.allSatisfy({ $0.hasPrefix("mtp.") })
    else {
      throw RepackError.invalidIndex("MTP tensor index is incomplete")
    }

    let experts = try expertConversions(tensors)
    let excluded = Set(experts.flatMap { [$0.tensor, $0.sourceScaleTensor] })
    var copies: [TensorCopy] = []
    var commonTensors: [InstalledTensor] = []
    var commonOffset: UInt64 = 0
    for tensor in tensors.values.sorted(by: { $0.name < $1.name })
    where !excluded.contains(tensor.name)
    {
      guard !tensor.name.contains(".experts.") else {
        throw RepackError.invalidPlan("unexpected MTP expert tensor \(tensor.name)")
      }
      commonOffset = aligned(commonOffset, to: QwenContract.commonAlignment)
      commonTensors.append(
        InstalledTensor(
          name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
          offset: commonOffset, length: tensor.length))
      copies.append(
        TensorCopy(
          tensor: tensor.name, sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset, length: tensor.length,
          destinationFile: commonPath, destinationOffset: commonOffset))
      commonOffset += tensor.length
    }
    guard commonTensors.count == QwenContract.mtpCommonTensorCount else {
      throw RepackError.invalidPlan(
        "MTP has \(commonTensors.count) common tensors; expected \(QwenContract.mtpCommonTensorCount)")
    }

    let layerSize = UInt64(QwenContract.expertCount) * QwenContract.expertBlobSize
    return RepackPlan(
      formatVersion: 2,
      modelID: QwenContract.modelID,
      revision: QwenContract.revision,
      layerCount: QwenContract.layerCount,
      expertCount: QwenContract.expertCount,
      selectedExpertCount: QwenContract.selectedExpertCount,
      expertBlobSize: QwenContract.expertBlobSize,
      checkpointTensorBytes: copies.reduce(UInt64(0)) { $0 + $1.length }
        + experts.reduce(UInt64(0)) { $0 + sourceBytes($1) },
      files: [
        PlannedFile(path: commonPath, size: commonOffset),
        PlannedFile(path: expertPath, size: layerSize),
      ],
      commonTensors: commonTensors,
      expertRegions: QwenContract.expertRegions,
      copies: copies,
      modelKind: .qwen3_8FlashNext,
      maximumContext: QwenContract.maximumContext,
      expertQuantization: QwenContract.quantization,
      expertConversions: experts)
  }

  private static func expertConversions(_ tensors: [String: SafeTensor]) throws
    -> [ExpertConversion]
  {
    var result: [ExpertConversion] = []
    for expert in 0..<QwenContract.expertCount {
      for projection in ["gate_proj", "up_proj", "down_proj"] {
        let rows = projection == "down_proj"
          ? QwenContract.hiddenSize : QwenContract.expertIntermediateSize
        let columns = projection == "down_proj"
          ? QwenContract.expertIntermediateSize : QwenContract.hiddenSize
        let prefix = "mtp.layers.0.mlp.experts.\(expert).\(projection)"
        let name = "\(prefix).weight"
        let scaleName = "\(prefix).weight_scale_inv"
        guard let tensor = tensors[name], let scale = tensors[scaleName] else {
          throw RepackError.invalidPlan("missing MTP routed expert tensor \(prefix)")
        }
        guard tensor.dtype == "F8_E4M3", tensor.shape == [rows, columns],
          scale.dtype == "BF16", scale.shape == [rows / 128, columns / 128]
        else {
          throw RepackError.invalidPlan("invalid MTP FP8 layout for \(prefix)")
        }
        result.append(
          ExpertConversion(
            tensor: name, sourceFile: tensor.sourceFile, sourceOffset: tensor.sourceOffset,
            sourceDType: tensor.dtype, sourceShape: tensor.shape,
            sourceScaleTensor: scaleName, sourceScaleFile: scale.sourceFile,
            sourceScaleOffset: scale.sourceOffset, sourceScaleDType: scale.dtype,
            sourceScaleShape: scale.shape, destinationFile: expertPath,
            expert: expert,
            destinationRow: projection == "up_proj" ? QwenContract.expertIntermediateSize : 0,
            weightRegion: projection == "down_proj" ? "down.weight" : "gate_up.weight",
            scaleRegion: projection == "down_proj" ? "down.scale" : "gate_up.scale"))
      }
    }
    return result
  }

  private static func sourceBytes(_ conversion: ExpertConversion) -> UInt64 {
    conversion.sourceShape.reduce(UInt64(1)) { $0 * UInt64($1) }
      + conversion.sourceScaleShape.reduce(UInt64(2)) { $0 * UInt64($1) }
  }
}

enum QwenPlanner {
  static func makePlan(index: CheckpointIndex, tensors: [String: SafeTensor]) throws
    -> RepackPlan
  {
    guard tensors.count == index.weightMap.count else {
      throw RepackError.invalidIndex(
        "resolved \(tensors.count) tensors; index contains \(index.weightMap.count)")
    }

    let experts = try expertConversions(tensors)
    let ngramTensors = try (0..<QwenContract.ngramShardCount).map { shard in
      let name = ngramName(shard)
      guard let tensor = tensors[name] else {
        throw RepackError.invalidPlan("missing N-gram tensor \(name)")
      }
      guard tensor.dtype == QwenContract.ngram.dtype,
        tensor.shape == [QwenContract.ngramShardRowCount, QwenContract.ngramRowBytes]
      else {
        throw RepackError.invalidPlan("invalid N-gram tensor layout for \(name)")
      }
      return tensor
    }
    guard let ngramScale = tensors[QwenContract.ngramScaleName],
      ngramScale.dtype == "BF16", ngramScale.shape == [1]
    else {
      throw RepackError.invalidPlan("missing or invalid Qwen N-gram weight scale")
    }

    var copies: [TensorCopy] = []
    var commonTensors: [InstalledTensor] = []
    var commonOffset: UInt64 = 0
    let excluded = Set(experts.flatMap { [$0.tensor, $0.sourceScaleTensor] })
      .union(ngramTensors.map(\.name))
    for tensor in tensors.values.sorted(by: { $0.name < $1.name }) {
      guard !excluded.contains(tensor.name), !isExcluded(tensor.name) else { continue }
      commonOffset = aligned(commonOffset, to: QwenContract.commonAlignment)
      commonTensors.append(
        InstalledTensor(
          name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
          offset: commonOffset, length: tensor.length))
      copies.append(
        TensorCopy(
          tensor: tensor.name, sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset, length: tensor.length,
          destinationFile: "common.bin", destinationOffset: commonOffset))
      commonOffset += tensor.length
    }
    guard !commonTensors.isEmpty else {
      throw RepackError.invalidPlan("Qwen text model has no common tensors")
    }

    var ngramOffset: UInt64 = 0
    for tensor in ngramTensors {
      copies.append(
        TensorCopy(
          tensor: tensor.name, sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset, length: tensor.length,
          destinationFile: QwenContract.ngram.file, destinationOffset: ngramOffset))
      ngramOffset += tensor.length
    }

    let layerSize = UInt64(QwenContract.expertCount) * QwenContract.expertBlobSize
    let files = [
      PlannedFile(path: "common.bin", size: commonOffset),
      PlannedFile(path: QwenContract.ngram.file, size: ngramOffset),
    ] + (0..<QwenContract.layerCount).map {
      PlannedFile(path: layerPath($0), size: layerSize)
    }
    return RepackPlan(
      formatVersion: 2,
      modelID: QwenContract.modelID,
      revision: QwenContract.revision,
      layerCount: QwenContract.layerCount,
      expertCount: QwenContract.expertCount,
      selectedExpertCount: QwenContract.selectedExpertCount,
      expertBlobSize: QwenContract.expertBlobSize,
      checkpointTensorBytes: copies.reduce(UInt64(0)) { $0 + $1.length }
        + experts.reduce(UInt64(0)) { $0 + sourceBytes($1) },
      files: files,
      commonTensors: commonTensors,
      expertRegions: QwenContract.expertRegions,
      copies: copies,
      modelKind: .qwen3_8FlashNext,
      maximumContext: QwenContract.maximumContext,
      expertQuantization: QwenContract.quantization,
      ngram: QwenContract.ngram,
      expertConversions: experts)
  }

  private static func expertConversions(_ tensors: [String: SafeTensor]) throws
    -> [ExpertConversion]
  {
    var result: [ExpertConversion] = []
    for layer in 0..<QwenContract.layerCount {
      for expert in 0..<QwenContract.expertCount {
        for projection in ["gate_proj", "up_proj", "down_proj"] {
          let rows = projection == "down_proj"
            ? QwenContract.hiddenSize : QwenContract.expertIntermediateSize
          let columns = projection == "down_proj"
            ? QwenContract.expertIntermediateSize : QwenContract.hiddenSize
          let prefix = "model.language_model.layers.\(layer).mlp.experts.\(expert).\(projection)"
          let name = "\(prefix).weight"
          let scaleName = "\(prefix).weight_scale_inv"
          guard let tensor = tensors[name], let scale = tensors[scaleName] else {
            throw RepackError.invalidPlan("missing routed expert tensor \(prefix)")
          }
          guard tensor.dtype == "F8_E4M3", tensor.shape == [rows, columns],
            scale.dtype == "BF16",
            scale.shape == [rows / 128, columns / 128]
          else {
            throw RepackError.invalidPlan(
              "invalid FP8 layout for \(prefix): \(tensor.dtype) \(tensor.shape), "
                + "\(scale.dtype) \(scale.shape)")
          }
          result.append(
            ExpertConversion(
              tensor: name, sourceFile: tensor.sourceFile, sourceOffset: tensor.sourceOffset,
              sourceDType: tensor.dtype, sourceShape: tensor.shape,
              sourceScaleTensor: scaleName, sourceScaleFile: scale.sourceFile,
              sourceScaleOffset: scale.sourceOffset, sourceScaleDType: scale.dtype,
              sourceScaleShape: scale.shape, destinationFile: layerPath(layer),
              expert: expert,
              destinationRow: projection == "up_proj" ? QwenContract.expertIntermediateSize : 0,
              weightRegion: projection == "down_proj" ? "down.weight" : "gate_up.weight",
              scaleRegion: projection == "down_proj" ? "down.scale" : "gate_up.scale"))
        }
      }
    }
    return result
  }

  private static func sourceBytes(_ conversion: ExpertConversion) -> UInt64 {
    conversion.sourceShape.reduce(UInt64(1)) { $0 * UInt64($1) }
      + conversion.sourceScaleShape.reduce(UInt64(2)) { $0 * UInt64($1) }
  }

  private static func isExcluded(_ name: String) -> Bool {
    name.hasPrefix("model.visual.") || name.hasPrefix("mtp.")
      || name.hasSuffix("ngram_heads_offsets") || name.hasSuffix("ngram_heads_vocab_sizes")
  }

  private static func ngramName(_ shard: Int) -> String {
    "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard).weight"
  }

  private static func layerPath(_ layer: Int) -> String {
    String(format: "experts/layer_%02d.bin", layer)
  }
}

public struct QwenFlashNextCheckpoint: Sendable {
  private let source: any CheckpointSource

  public init() {
    source = HuggingFaceSource(modelID: QwenContract.modelID, revision: QwenContract.revision)
  }

  init(source: any CheckpointSource) {
    self.source = source
  }

  public func makeRepackPlan() async throws -> RepackPlan {
    let config: QwenConfig
    do {
      config = try JSONDecoder().decode(QwenConfig.self, from: try await source.data(path: "config.json"))
    } catch {
      throw RepackError.incompatibleModel("cannot decode Qwen model config: \(error)")
    }
    try QwenContract.validate(config)
    let index = try CheckpointIndex.decode(try await source.data(path: "model.safetensors.index.json"))
    return try QwenPlanner.makePlan(index: index, tensors: try await readTensors(index: index))
  }

  public func makeMTPRepackPlan() async throws -> RepackPlan {
    let config: QwenConfig
    do {
      config = try JSONDecoder().decode(
        QwenConfig.self, from: try await source.data(path: "config.json"))
    } catch {
      throw RepackError.incompatibleModel("cannot decode Qwen model config: \(error)")
    }
    try QwenContract.validate(config)
    let index = try CheckpointIndex.decode(
      try await source.data(path: "model.safetensors.index.json"))
    let mtpWeightMap = index.weightMap.filter { $0.key.hasPrefix("mtp.") }
    guard !mtpWeightMap.isEmpty else {
      throw RepackError.invalidIndex("checkpoint index has no MTP tensors")
    }
    let mtpIndex = CheckpointIndex(totalSize: index.totalSize, weightMap: mtpWeightMap)
    return try QwenMTPPlanner.makePlan(
      index: mtpIndex, tensors: try await readTensors(index: mtpIndex))
  }

  public func installMTP(
    at output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let output = output.standardizedFileURL
    let current = try InstalledModel.loadManifest(at: output)
    guard current.modelKind == .qwen3_8FlashNext else {
      throw RepackError.incompatibleModel("installed model is not Qwen3.8-Flash-Next")
    }
    if current.mtp != nil {
      return try verifyInstalledMTP(current, at: output)
    }

    let fileManager = FileManager.default
    let installedMTP = output.appendingPathComponent("mtp")
    guard !fileManager.fileExists(atPath: installedMTP.path) else {
      throw RepackError.destinationExists(installedMTP.path)
    }
    let stage = output.deletingLastPathComponent().appendingPathComponent(
      output.lastPathComponent + ".mtp-install")
    guard !fileManager.fileExists(atPath: stage.path) else {
      throw RepackError.destinationExists(stage.path)
    }

    let plan = try await makeMTPRepackPlan()
    let staged = try await Repacker(source: source).run(
      plan: plan, output: stage, progress: progress, companions: [])
    let mtpFiles = staged.files.filter { $0.path.hasPrefix("mtp/") }
    guard mtpFiles.count == 2 else {
      throw RepackError.invalidPlan("staged MTP file set is incomplete")
    }
    for file in mtpFiles {
      let url = try safeFileURL(root: stage, path: file.path)
      guard try fileSize(url) == file.size, try sha256(url) == file.sha256 else {
        throw RepackError.invalidPlan("staged MTP file failed verification: \(file.path)")
      }
    }

    let stagedMTP = stage.appendingPathComponent("mtp")
    try fileManager.moveItem(at: stagedMTP, to: installedMTP)
    do {
      let manifest = InstalledManifest(
        formatVersion: current.formatVersion,
        modelID: current.modelID,
        revision: current.revision,
        layerCount: current.layerCount,
        expertCount: current.expertCount,
        selectedExpertCount: current.selectedExpertCount,
        expertBlobSize: current.expertBlobSize,
        files: current.files + mtpFiles,
        commonTensors: current.commonTensors,
        expertRegions: current.expertRegions,
        dspark: current.dspark,
        mtp: MTPDescriptor(
          layerCount: QwenContract.mtpLayerCount,
          useDedicatedEmbeddings: false,
          commonTensors: plan.commonTensors),
        modelKind: current.modelKind,
        maximumContext: current.maximumContext,
        expertQuantization: current.expertQuantization,
        ngram: current.ngram)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(manifest).write(
        to: output.appendingPathComponent("manifest.json"), options: .atomic)
      try? fileManager.removeItem(at: stage)
      return try verifyInstalledMTP(
        InstalledModel.loadManifest(at: output), at: output)
    } catch {
      try? fileManager.moveItem(at: installedMTP, to: stagedMTP)
      throw error
    }
  }

  public func repack(
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    try await repack(plan: makeRepackPlan(), to: output, progress: progress)
  }

  public func repack(
    plan: RepackPlan,
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    guard plan.formatVersion == 2,
      plan.modelKind == .qwen3_8FlashNext,
      plan.modelID == QwenContract.modelID,
      plan.revision == QwenContract.revision,
      plan.layerCount == QwenContract.layerCount,
      plan.expertCount == QwenContract.expertCount,
      plan.selectedExpertCount == QwenContract.selectedExpertCount,
      plan.expertBlobSize == QwenContract.expertBlobSize,
      plan.maximumContext == QwenContract.maximumContext,
      plan.expertRegions == QwenContract.expertRegions,
      plan.expertQuantization == QwenContract.quantization,
      plan.ngram == QwenContract.ngram,
      plan.expertConversions?.count
        == QwenContract.layerCount * QwenContract.expertCount * 3
    else {
      throw RepackError.incompatibleModel("repack plan does not match the pinned Qwen model contract")
    }
    let manifest = try await Repacker(source: source).run(plan: plan, output: output, progress: progress)
    try await QwenVisionArtifact.install(at: output, progress: progress)
    return manifest
  }

  public func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let manifest = try InstalledModel.loadManifest(at: output)
    guard manifest.modelKind == .qwen3_8FlashNext else {
      throw RepackError.incompatibleModel("installed model is not Qwen3.8-Flash-Next")
    }
    guard manifest.mtp == nil else {
      throw RepackError.invalidPlan(
        "Qwen main-model repair is not available while MTP is installed")
    }
    return try await Repacker(source: source).repair(
      plan: makeRepackPlan(), output: output, invalidFiles: invalidFiles, progress: progress)
  }

  private func readTensors(index: CheckpointIndex) async throws -> [String: SafeTensor] {
    var headers: [String: (base: UInt64, header: SafeTensorsHeader)] = [:]
    let shards = Set(index.weightMap.values).sorted()
    try await withThrowingTaskGroup(
      of: (String, UInt64, SafeTensorsHeader).self
    ) { group in
      var iterator = shards.makeIterator()
      for _ in 0..<min(8, shards.count) {
        guard let shard = iterator.next() else { break }
        group.addTask { try await readHeader(shard) }
      }
      while let (shard, base, header) = try await group.next() {
        headers[shard] = (base, header)
        if let next = iterator.next() {
          group.addTask { try await readHeader(next) }
        }
      }
    }
    var tensors: [String: SafeTensor] = [:]
    for (name, shard) in index.weightMap {
      guard let sourceHeader = headers[shard], let entry = sourceHeader.header.entries[name] else {
        throw RepackError.invalidIndex("index tensor \(name) is missing from \(shard)")
      }
      tensors[name] = SafeTensor(
        name: name, sourceFile: shard, dtype: entry.dtype, shape: entry.shape,
        sourceOffset: sourceHeader.base + entry.dataStart,
        length: entry.dataEnd - entry.dataStart)
    }
    return tensors
  }

  private func verifyInstalledMTP(
    _ manifest: InstalledManifest,
    at output: URL
  ) throws -> InstalledManifest {
    let files = manifest.files.filter { $0.path.hasPrefix("mtp/") }
    guard manifest.mtp != nil, files.count == 2 else {
      throw RepackError.invalidPlan("installed MTP file set is incomplete")
    }
    for file in files {
      let url = try safeFileURL(root: output, path: file.path)
      guard try fileSize(url) == file.size else {
        throw RepackError.invalidPlan("installed size mismatch for \(file.path)")
      }
      guard try sha256(url) == file.sha256 else {
        throw RepackError.invalidPlan("installed SHA-256 mismatch for \(file.path)")
      }
    }
    return manifest
  }

  private func readHeader(_ shard: String) async throws
    -> (String, UInt64, SafeTensorsHeader)
  {
    let prefix = try await source.data(path: shard, range: 0..<8)
    let headerLength = try littleEndianUInt64(prefix)
    guard headerLength > 1, headerLength <= 64 * 1_024 * 1_024 else {
      throw RepackError.invalidSafeTensors("invalid header length \(headerLength) in \(shard)")
    }
    let data = try await source.data(path: shard, range: 8..<(8 + headerLength))
    return (shard, 8 + headerLength, try SafeTensorsHeader.decode(data))
  }
}

public struct QwenInstalledModelArtifact: Sendable {
  public static let repository = "Yanun/Qwen3.8-Flash-Next-MXFP4"
  public static let revision = "753d0aa57059fad70a5f7e6cc249f25df56bbd34"

  private let downloader: InstalledArtifactDownloader

  public init() {
    let source = HuggingFaceSource(modelID: Self.repository, revision: Self.revision)
    downloader = InstalledArtifactDownloader(
      repository: Self.repository, revision: Self.revision, source: source)
  }

  init(source: any CheckpointSource) {
    downloader = InstalledArtifactDownloader(
      repository: Self.repository, revision: Self.revision, source: source)
  }

  public func installedBytes() async throws -> UInt64 {
    let manifest = try await downloader.manifest().manifest
    return try InstalledModel.verificationFiles(for: manifest).reduce(UInt64(0)) { total, file in
      let result = total.addingReportingOverflow(file.size)
      guard !result.overflow else {
        throw RepackError.invalidPlan("installed file sizes overflow")
      }
      return result.partialValue
    }
  }

  public func install(
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    try await downloader.install(to: output, progress: progress)
  }

  public func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    try await downloader.repair(
      at: output, invalidFiles: invalidFiles, progress: progress)
  }
}
