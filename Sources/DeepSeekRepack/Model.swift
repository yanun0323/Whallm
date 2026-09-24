import Foundation

public struct CompanionFile: Sendable {
  let source: String
  let destination: String
}

public struct ModelKind: RawRepresentable, Codable, Hashable, Sendable {
  public let rawValue: String

  public init?(rawValue: String) {
    guard ModelPackages.descriptors.contains(where: { $0.kind == rawValue }) else { return nil }
    self.rawValue = rawValue
  }

  public init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    let value = try container.decode(String.self)
    guard let kind = Self(rawValue: value) else {
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unknown model kind")
    }
    self = kind
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(rawValue)
  }

  public var descriptor: ModelPackageDescriptor {
    ModelPackages.descriptors.first(where: { $0.kind == rawValue })!
  }

  public static let deepSeekV4 = Self(rawValue: "deepseek-v4")!
  public static let deepSeekV41 = Self(rawValue: "deepseek-v4.1")!
  public static let qwen3_8FlashNext = Self(rawValue: "qwen3.8-flash-next")!
  public static let mimoV26FlashRL = Self(rawValue: "mimo-v2.6-flash-rl")!
}

enum ModelContract {
  static let modelID = "deepseek-ai/DeepSeek-V4-Flash-0731"
  static let revision = "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
  static let layerCount = 43
  static let expertCount = 256
  static let selectedExpertCount = 6
  static let hiddenSize = 4_096
  static let expertIntermediateSize = 2_048
  static let hashLayerCount = 3
  static let maximumContext = 1_048_576
  static let dsparkLayerCount = 3
  static let dsparkBlockSize = 5
  static let dsparkNoiseTokenID = 128_799
  static let dsparkTargetLayerIDs = [40, 41, 42]
  static let dsparkMarkovRank = 256
  static let commonAlignment: UInt64 = 256
  static let companions = [
    CompanionFile(source: "config.json", destination: "config.json"),
    CompanionFile(source: "generation_config.json", destination: "generation_config.json"),
    CompanionFile(source: "inference/config.json", destination: "inference/config.json"),
    CompanionFile(source: "tokenizer.json", destination: "tokenizer/tokenizer.json"),
    CompanionFile(
      source: "tokenizer_config.json", destination: "tokenizer/tokenizer_config.json"),
    CompanionFile(source: "encoding/encoding_dsv4.py", destination: "encoding/encoding_dsv4.py"),
  ]
  static let companionPaths = companions.map(\.destination)

  static let expertRegions: [ExpertRegion] = {
    var offset: UInt64 = 0
    return [
      makeRegion("w1.weight", dtype: "I8", shape: [2_048, 2_048], offset: &offset),
      makeRegion("w1.scale", dtype: "F8_E8M0", shape: [2_048, 128], offset: &offset),
      makeRegion("w2.weight", dtype: "I8", shape: [4_096, 1_024], offset: &offset),
      makeRegion("w2.scale", dtype: "F8_E8M0", shape: [4_096, 64], offset: &offset),
      makeRegion("w3.weight", dtype: "I8", shape: [2_048, 2_048], offset: &offset),
      makeRegion("w3.scale", dtype: "F8_E8M0", shape: [2_048, 128], offset: &offset),
    ]
  }()

  static let expertBlobSize = expertRegions.reduce(0) { $0 + $1.length }

  private static func makeRegion(
    _ name: String,
    dtype: String,
    shape: [Int],
    offset: inout UInt64
  ) -> ExpertRegion {
    let length = shape.reduce(UInt64(1)) { $0 * UInt64($1) }
    let region = ExpertRegion(
      name: name, dtype: dtype, shape: shape, offset: offset, length: length)
    offset += length
    return region
  }

  static func validate(_ config: ModelConfig) throws {
    let actual: [(String, String, String)] = [
      ("architecture", config.architectures.joined(separator: ","), "DeepseekV4ForCausalLM"),
      ("expert_dtype", config.expertDType, "fp4"),
      ("hidden_size", String(config.hiddenSize), String(hiddenSize)),
      ("moe_intermediate_size", String(config.moeIntermediateSize), String(expertIntermediateSize)),
      ("n_routed_experts", String(config.routedExpertCount), String(expertCount)),
      ("n_shared_experts", String(config.sharedExpertCount), "1"),
      ("num_experts_per_tok", String(config.selectedExpertCount), String(selectedExpertCount)),
      ("num_hidden_layers", String(config.hiddenLayerCount), String(layerCount)),
      ("num_hash_layers", String(config.hashLayerCount), String(hashLayerCount)),
      ("max_position_embeddings", String(config.maximumContext), String(maximumContext)),
    ]
    if let mismatch = actual.first(where: { $0.1 != $0.2 }) {
      throw RepackError.incompatibleModel("\(mismatch.0) is \(mismatch.1); expected \(mismatch.2)")
    }
  }

  static func validate(_ config: DSparkConfig) throws {
    let actual: [(String, String, String)] = [
      ("dim", String(config.hiddenSize), String(hiddenSize)),
      ("n_mtp_layers", String(config.layerCount), String(dsparkLayerCount)),
      ("dspark_block_size", String(config.blockSize), String(dsparkBlockSize)),
      ("dspark_noise_token_id", String(config.noiseTokenID), String(dsparkNoiseTokenID)),
      (
        "dspark_target_layer_ids",
        config.targetLayerIDs.map(String.init).joined(separator: ","),
        dsparkTargetLayerIDs.map(String.init).joined(separator: ",")
      ),
      ("dspark_markov_rank", String(config.markovRank), String(dsparkMarkovRank)),
    ]
    if let mismatch = actual.first(where: { $0.1 != $0.2 }) {
      throw RepackError.incompatibleModel("\(mismatch.0) is \(mismatch.1); expected \(mismatch.2)")
    }
  }
}

struct ModelConfig: Decodable, Sendable {
  let architectures: [String]
  let expertDType: String
  let hiddenSize: Int
  let moeIntermediateSize: Int
  let routedExpertCount: Int
  let sharedExpertCount: Int
  let selectedExpertCount: Int
  let hiddenLayerCount: Int
  let hashLayerCount: Int
  let maximumContext: Int

  enum CodingKeys: String, CodingKey {
    case architectures
    case expertDType = "expert_dtype"
    case hiddenSize = "hidden_size"
    case moeIntermediateSize = "moe_intermediate_size"
    case routedExpertCount = "n_routed_experts"
    case sharedExpertCount = "n_shared_experts"
    case selectedExpertCount = "num_experts_per_tok"
    case hiddenLayerCount = "num_hidden_layers"
    case hashLayerCount = "num_hash_layers"
    case maximumContext = "max_position_embeddings"
  }
}

struct DSparkConfig: Decodable, Sendable {
  let hiddenSize: Int
  let layerCount: Int
  let blockSize: Int
  let noiseTokenID: Int
  let targetLayerIDs: [Int]
  let markovRank: Int

  enum CodingKeys: String, CodingKey {
    case hiddenSize = "dim"
    case layerCount = "n_mtp_layers"
    case blockSize = "dspark_block_size"
    case noiseTokenID = "dspark_noise_token_id"
    case targetLayerIDs = "dspark_target_layer_ids"
    case markovRank = "dspark_markov_rank"
  }
}

struct SafeTensor: Equatable, Sendable {
  let name: String
  let sourceFile: String
  let dtype: String
  let shape: [Int]
  let sourceOffset: UInt64
  let length: UInt64
}

public struct ExpertRegion: Codable, Equatable, Sendable {
  public let name: String
  public let dtype: String
  public let shape: [Int]
  public let offset: UInt64
  public let length: UInt64
}

public struct InstalledTensor: Codable, Equatable, Sendable {
  public let name: String
  public let dtype: String
  public let shape: [Int]
  public let offset: UInt64
  public let length: UInt64
}

public struct PlannedFile: Codable, Equatable, Sendable {
  public let path: String
  public let size: UInt64
}

public struct TensorCopy: Codable, Equatable, Sendable {
  public let tensor: String
  public let sourceFile: String
  public let sourceOffset: UInt64
  public let length: UInt64
  public let destinationFile: String
  public let destinationOffset: UInt64
}

public struct DSparkDescriptor: Codable, Equatable, Sendable {
  public let layerCount: Int
  public let blockSize: Int
  public let noiseTokenID: Int
  public let targetLayerIDs: [Int]
  public let markovRank: Int
  public let commonTensors: [InstalledTensor]
}

public struct MTPDescriptor: Codable, Equatable, Sendable {
  public let layerCount: Int
  public let useDedicatedEmbeddings: Bool
  public let commonTensors: [InstalledTensor]

  public init(
    layerCount: Int,
    useDedicatedEmbeddings: Bool,
    commonTensors: [InstalledTensor]
  ) {
    self.layerCount = layerCount
    self.useDedicatedEmbeddings = useDedicatedEmbeddings
    self.commonTensors = commonTensors
  }
}

public struct ExpertQuantizationDescriptor: Codable, Equatable, Sendable {
  public let mode: String
  public let bits: Int
  public let groupSize: Int
  public let conversionVersion: Int

  public init(mode: String, bits: Int, groupSize: Int, conversionVersion: Int) {
    self.mode = mode
    self.bits = bits
    self.groupSize = groupSize
    self.conversionVersion = conversionVersion
  }
}

public struct NGramDescriptor: Codable, Equatable, Sendable {
  public let file: String
  public let dtype: String
  public let rowBytes: Int
  public let shardCount: Int
  public let shardRowCount: Int
  public let headOffsets: [Int64]
  public let headVocabSizes: [Int64]

  public init(
    file: String,
    dtype: String,
    rowBytes: Int,
    shardCount: Int,
    shardRowCount: Int,
    headOffsets: [Int64],
    headVocabSizes: [Int64]
  ) {
    self.file = file
    self.dtype = dtype
    self.rowBytes = rowBytes
    self.shardCount = shardCount
    self.shardRowCount = shardRowCount
    self.headOffsets = headOffsets
    self.headVocabSizes = headVocabSizes
  }
}

public struct EngramTableDescriptor: Codable, Equatable, Sendable {
  public let layer: Int
  public let weightFile: String
  public let scaleFile: String
  public let rows: Int
  public let dimension: Int
  public let blockSize: Int

  public init(
    layer: Int,
    weightFile: String,
    scaleFile: String,
    rows: Int,
    dimension: Int,
    blockSize: Int
  ) {
    self.layer = layer
    self.weightFile = weightFile
    self.scaleFile = scaleFile
    self.rows = rows
    self.dimension = dimension
    self.blockSize = blockSize
  }
}

public struct EngramDescriptor: Codable, Equatable, Sendable {
  public let tables: [EngramTableDescriptor]

  public init(tables: [EngramTableDescriptor]) {
    self.tables = tables
  }
}

public struct ExpertConversion: Codable, Equatable, Sendable {
  public let tensor: String
  public let sourceFile: String
  public let sourceOffset: UInt64
  public let sourceDType: String
  public let sourceShape: [Int]
  public let sourceScaleTensor: String
  public let sourceScaleFile: String
  public let sourceScaleOffset: UInt64
  public let sourceScaleDType: String
  public let sourceScaleShape: [Int]
  public let destinationFile: String
  public let expert: Int
  public let destinationRow: Int
  public let weightRegion: String
  public let scaleRegion: String

  public init(
    tensor: String,
    sourceFile: String,
    sourceOffset: UInt64,
    sourceDType: String,
    sourceShape: [Int],
    sourceScaleTensor: String,
    sourceScaleFile: String,
    sourceScaleOffset: UInt64,
    sourceScaleDType: String,
    sourceScaleShape: [Int],
    destinationFile: String,
    expert: Int,
    destinationRow: Int,
    weightRegion: String,
    scaleRegion: String
  ) {
    self.tensor = tensor
    self.sourceFile = sourceFile
    self.sourceOffset = sourceOffset
    self.sourceDType = sourceDType
    self.sourceShape = sourceShape
    self.sourceScaleTensor = sourceScaleTensor
    self.sourceScaleFile = sourceScaleFile
    self.sourceScaleOffset = sourceScaleOffset
    self.sourceScaleDType = sourceScaleDType
    self.sourceScaleShape = sourceScaleShape
    self.destinationFile = destinationFile
    self.expert = expert
    self.destinationRow = destinationRow
    self.weightRegion = weightRegion
    self.scaleRegion = scaleRegion
  }
}

public struct RepackPlan: Codable, Equatable, Sendable {
  public let formatVersion: Int
  public let modelID: String
  public let revision: String
  public let layerCount: Int
  public let expertCount: Int
  public let selectedExpertCount: Int
  public let expertBlobSize: UInt64
  public let checkpointTensorBytes: UInt64
  public let files: [PlannedFile]
  public let commonTensors: [InstalledTensor]
  public let expertRegions: [ExpertRegion]
  public let dspark: DSparkDescriptor?
  public let copies: [TensorCopy]
  public let modelKind: ModelKind?
  public let maximumContext: Int?
  public let expertQuantization: ExpertQuantizationDescriptor?
  public let ngram: NGramDescriptor?
  public let engram: EngramDescriptor?
  public let expertConversions: [ExpertConversion]?

  public var installedBytes: UInt64 { files.reduce(0) { $0 + $1.size } }

  public init(
    formatVersion: Int,
    modelID: String,
    revision: String,
    layerCount: Int,
    expertCount: Int,
    selectedExpertCount: Int,
    expertBlobSize: UInt64,
    checkpointTensorBytes: UInt64,
    files: [PlannedFile],
    commonTensors: [InstalledTensor],
    expertRegions: [ExpertRegion],
    dspark: DSparkDescriptor? = nil,
    copies: [TensorCopy],
    modelKind: ModelKind? = nil,
    maximumContext: Int? = nil,
    expertQuantization: ExpertQuantizationDescriptor? = nil,
    ngram: NGramDescriptor? = nil,
    engram: EngramDescriptor? = nil,
    expertConversions: [ExpertConversion]? = nil
  ) {
    self.formatVersion = formatVersion
    self.modelID = modelID
    self.revision = revision
    self.layerCount = layerCount
    self.expertCount = expertCount
    self.selectedExpertCount = selectedExpertCount
    self.expertBlobSize = expertBlobSize
    self.checkpointTensorBytes = checkpointTensorBytes
    self.files = files
    self.commonTensors = commonTensors
    self.expertRegions = expertRegions
    self.dspark = dspark
    self.copies = copies
    self.modelKind = modelKind
    self.maximumContext = maximumContext
    self.expertQuantization = expertQuantization
    self.ngram = ngram
    self.engram = engram
    self.expertConversions = expertConversions
  }
}

public struct InstalledFile: Codable, Equatable, Sendable {
  public let path: String
  public let size: UInt64
  public let sha256: String
}

public struct InstalledManifest: Codable, Equatable, Sendable {
  public let formatVersion: Int
  public let modelID: String
  public let revision: String
  public let layerCount: Int
  public let expertCount: Int
  public let selectedExpertCount: Int
  public let expertBlobSize: UInt64
  public let files: [InstalledFile]
  public let commonTensors: [InstalledTensor]
  public let expertRegions: [ExpertRegion]
  public let dspark: DSparkDescriptor?
  public let mtp: MTPDescriptor?
  public let modelKind: ModelKind?
  public let maximumContext: Int?
  public let expertQuantization: ExpertQuantizationDescriptor?
  public let ngram: NGramDescriptor?
  public let engram: EngramDescriptor?

  public init(
    formatVersion: Int,
    modelID: String,
    revision: String,
    layerCount: Int,
    expertCount: Int,
    selectedExpertCount: Int,
    expertBlobSize: UInt64,
    files: [InstalledFile],
    commonTensors: [InstalledTensor],
    expertRegions: [ExpertRegion],
    dspark: DSparkDescriptor? = nil,
    mtp: MTPDescriptor? = nil,
    modelKind: ModelKind? = nil,
    maximumContext: Int? = nil,
    expertQuantization: ExpertQuantizationDescriptor? = nil,
    ngram: NGramDescriptor? = nil,
    engram: EngramDescriptor? = nil
  ) {
    self.formatVersion = formatVersion
    self.modelID = modelID
    self.revision = revision
    self.layerCount = layerCount
    self.expertCount = expertCount
    self.selectedExpertCount = selectedExpertCount
    self.expertBlobSize = expertBlobSize
    self.files = files
    self.commonTensors = commonTensors
    self.expertRegions = expertRegions
    self.dspark = dspark
    self.mtp = mtp
    self.modelKind = modelKind
    self.maximumContext = maximumContext
    self.expertQuantization = expertQuantization
    self.ngram = ngram
    self.engram = engram
  }
}

public struct RepackProgress: Sendable {
  public let copiedBytes: UInt64
  public let downloadedBytes: UInt64
  public let totalBytes: UInt64
}

public enum InstalledFileIssueKind: String, Equatable, Sendable {
  case missing
  case sizeMismatch
  case checksumMismatch
}

public struct InstalledFileIssue: Equatable, Sendable {
  public let path: String
  public let kind: InstalledFileIssueKind

  public init(path: String, kind: InstalledFileIssueKind) {
    self.path = path
    self.kind = kind
  }
}

public struct InstalledModelVerification: Sendable {
  public let manifest: InstalledManifest
  public let issues: [InstalledFileIssue]

  public var isValid: Bool { issues.isEmpty }
}

public struct VerificationProgress: Sendable {
  public let checkedBytes: UInt64
  public let totalBytes: UInt64
}

public enum RepackError: Error, CustomStringConvertible, Equatable {
  case badResponse(String)
  case incompatibleModel(String)
  case invalidIndex(String)
  case invalidSafeTensors(String)
  case invalidPlan(String)
  case destinationExists(String)
  case insufficientStorage(required: UInt64, available: Int64)

  public var description: String {
    switch self {
    case .badResponse(let message), .incompatibleModel(let message), .invalidIndex(let message),
      .invalidSafeTensors(let message), .invalidPlan(let message):
      return message
    case .destinationExists(let path):
      return "destination already exists: \(path)"
    case .insufficientStorage(let required, let available):
      return "insufficient storage: need \(required) bytes; \(available) bytes are available"
    }
  }
}

func aligned(_ value: UInt64, to alignment: UInt64) -> UInt64 {
  (value + alignment - 1) / alignment * alignment
}
