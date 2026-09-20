import Foundation

enum DeepSeekV41Contract {
  static let modelID = "deepseek-ai/DeepSeek-V4.1-Flash"
  static let revision = "dba1be0a40aa45a94ad051997016db3960a90277"
  static let layerCount = 40
  static let expertCount = 384
  static let selectedExpertCount = 6
  static let hiddenSize = 5_120
  static let expertIntermediateSize = 2_304
  static let maximumContext = 1_048_576
  static let commonAlignment: UInt64 = 256
  static let engramLayers = [1, 14]
  static let engramRows = [384_006_168, 384_016_682]
  static let engramDimension = 256
  static let engramBlockSize = 32
  static let kvSourceLayers = [2, 8, 14, 20]
  static let indexSourceLayers = [2, 8, 14, 20, 24, 28, 32, 36]
  static let companions = [
    CompanionFile(source: "config.json", destination: "config.json"),
    CompanionFile(source: "tokenizer.json", destination: "tokenizer/tokenizer.json"),
    CompanionFile(
      source: "tokenizer_config.json", destination: "tokenizer/tokenizer_config.json"),
    CompanionFile(source: "encoding/encoding.py", destination: "encoding/encoding.py"),
  ]
  static let companionPaths = companions.map(\.destination)

  static let expertRegions: [ExpertRegion] = {
    var offset: UInt64 = 0
    return [
      makeRegion("w1.weight", dtype: "I8", shape: [2_304, 2_560], offset: &offset),
      makeRegion("w1.scale", dtype: "F8_E8M0", shape: [2_304, 160], offset: &offset),
      makeRegion("w2.weight", dtype: "I8", shape: [5_120, 1_152], offset: &offset),
      makeRegion("w2.scale", dtype: "F8_E8M0", shape: [5_120, 72], offset: &offset),
      makeRegion("w3.weight", dtype: "I8", shape: [2_304, 2_560], offset: &offset),
      makeRegion("w3.scale", dtype: "F8_E8M0", shape: [2_304, 160], offset: &offset),
    ]
  }()
  static let expertBlobSize = expertRegions.reduce(0) { $0 + $1.length }
  static let engram = EngramDescriptor(
    tables: zip(engramLayers, engramRows).map { layer, rows in
      EngramTableDescriptor(
        layer: layer,
        weightFile: engramWeightPath(layer),
        scaleFile: engramScalePath(layer),
        rows: rows,
        dimension: engramDimension,
        blockSize: engramBlockSize)
    })

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

  static func validate(_ config: DeepSeekV41Config) throws {
    let text = config.textConfig
    let actual: [(String, String, String)] = [
      ("architecture", config.architectures.joined(separator: ","), "DeepseekV41ForCausalLM"),
      ("model_type", config.modelType, "deepseek_v41"),
      ("text_config.model_type", text.modelType, "deepseek_v41_text"),
      ("hidden_size", String(text.hiddenSize), String(hiddenSize)),
      ("moe_intermediate_size", String(text.moeIntermediateSize), String(expertIntermediateSize)),
      ("n_routed_experts", String(text.routedExpertCount), String(expertCount)),
      ("n_shared_experts", String(text.sharedExpertCount), "1"),
      ("num_experts_per_tok", String(text.selectedExpertCount), String(selectedExpertCount)),
      ("num_hidden_layers", String(text.hiddenLayerCount), String(layerCount)),
      ("max_position_embeddings", String(text.maximumContext), String(maximumContext)),
      ("engram_layer_ids", text.engramLayers.map(String.init).joined(separator: ","), engramLayers.map(String.init).joined(separator: ",")),
      ("engram_num_embeddings", text.engramRows.map(String.init).joined(separator: ","), engramRows.map(String.init).joined(separator: ",")),
      ("engram_head_dim", String(text.engramDimension), String(engramDimension)),
      ("kv_source_layer_ids", text.kvSourceLayers.map(String.init).joined(separator: ","), kvSourceLayers.map(String.init).joined(separator: ",")),
      ("index_source_layer_ids", text.indexSourceLayers.map(String.init).joined(separator: ","), indexSourceLayers.map(String.init).joined(separator: ",")),
      ("quant_method", config.quantization.quantMethod, "fp8"),
      ("expert_dtype", config.quantization.expertDType, "fp4"),
      ("scale_fmt", config.quantization.scaleFormat, "ue8m0"),
      ("weight_block_size", config.quantization.weightBlockSize.map(String.init).joined(separator: ","), "32,32"),
    ]
    if let mismatch = actual.first(where: { $0.1 != $0.2 }) {
      throw RepackError.incompatibleModel("\(mismatch.0) is \(mismatch.1); expected \(mismatch.2)")
    }
  }

  static func validate(_ plan: RepackPlan) throws {
    guard plan.formatVersion == 3,
      plan.modelKind == .deepSeekV41,
      plan.modelID == modelID,
      plan.revision == revision,
      plan.layerCount == layerCount,
      plan.expertCount == expertCount,
      plan.selectedExpertCount == selectedExpertCount,
      plan.expertBlobSize == expertBlobSize,
      plan.expertRegions == expertRegions,
      plan.maximumContext == maximumContext,
      plan.engram == engram,

      plan.ngram == nil,
      plan.expertQuantization == nil
    else {
      throw RepackError.incompatibleModel("repack plan does not match the pinned V4.1 model contract")
    }
    try validateDSpark(plan.dspark, files: plan.files.map { InstalledFile(path: $0.path, size: $0.size, sha256: "") })
  }

  static func validate(_ manifest: InstalledManifest) throws -> InstalledManifest {
    guard manifest.formatVersion == 3,
      manifest.modelKind == .deepSeekV41,
      manifest.modelID == modelID,
      manifest.revision == revision,
      manifest.layerCount == layerCount,
      manifest.expertCount == expertCount,
      manifest.selectedExpertCount == selectedExpertCount,
      manifest.expertBlobSize == expertBlobSize,
      manifest.expertRegions == expertRegions,
      manifest.maximumContext == maximumContext,
      manifest.engram == engram,

      manifest.mtp == nil,
      manifest.ngram == nil,
      manifest.expertQuantization == nil
    else {
      throw RepackError.incompatibleModel(
        "installed manifest does not match the pinned V4.1 model contract")
    }

    try validateDSpark(manifest.dspark, files: manifest.files)
    var required: Set<String> = ["common.bin"]
    required.formUnion(companionPaths)
    required.formUnion((0..<layerCount).map(expertLayerPath))
    for table in engram.tables {
      required.insert(table.weightFile)
      required.insert(table.scaleFile)
    }
    if manifest.dspark != nil {
      required.formUnion(["dspark/common.bin", "inference/config.json"])
      required.formUnion((0..<3).map { String(format: "dspark/experts/layer_%02d.bin", $0) })
    }
    let paths = manifest.files.map(\.path)
    guard Set(paths) == required, Set(paths).count == paths.count else {
      throw RepackError.invalidPlan("installed V4.1 files do not match the pinned layout")
    }
    let sizes = Dictionary(uniqueKeysWithValues: manifest.files.map { ($0.path, $0.size) })
    let expertLayerSize = UInt64(expertCount) * expertBlobSize
    for layer in 0..<layerCount where sizes[expertLayerPath(layer)] != expertLayerSize {
      throw RepackError.invalidPlan("installed V4.1 expert layer has an invalid size")
    }
    for table in engram.tables {
      guard sizes[table.weightFile] == UInt64(table.rows * table.dimension),
        sizes[table.scaleFile] == UInt64(table.rows * (table.dimension / table.blockSize))
      else {
        throw RepackError.invalidPlan("installed V4.1 engram table has an invalid size")
      }
    }
    guard let commonSize = sizes["common.bin"], !manifest.commonTensors.isEmpty else {
      throw RepackError.invalidPlan("installed V4.1 common tensor file is missing")
    }
    var names = Set<String>()
    for tensor in manifest.commonTensors {
      guard names.insert(tensor.name).inserted,
        !isExpert(tensor.name), !isEngramTable(tensor.name), !isMTP(tensor.name),
        !isVision(tensor.name), isTextCommon(tensor.name), tensor.offset <= commonSize,
        tensor.length <= commonSize - tensor.offset
      else {
        throw RepackError.invalidPlan("invalid V4.1 common tensor \(tensor.name)")
      }
    }
    return manifest
  }

  static func validateDSpark(_ descriptor: DSparkDescriptor?, files: [InstalledFile]) throws {
    guard let value = descriptor else { return }
    guard Set(files.map(\.path)).count == files.count else {
      throw RepackError.invalidPlan("duplicate V4.1 DSpark file path")
    }
    let sizes = Dictionary(uniqueKeysWithValues: files.map { ($0.path, $0.size) })
    guard value.layerCount == 3, value.blockSize == 5, value.noiseTokenID == 128799,
      value.targetLayerIDs == [37, 38, 39], value.markovRank == 256,
      value.commonTensors.count == 97, let size = sizes["dspark/common.bin"],
      Set(value.commonTensors.map(\.name)).count == 97,
      value.commonTensors.allSatisfy({ $0.name.hasPrefix("mtp.") && !$0.name.contains(".experts.") && $0.offset <= size && $0.length <= size - $0.offset }),
      (0..<3).allSatisfy({ sizes[String(format: "dspark/experts/layer_%02d.bin", $0)] == UInt64(128) * expertBlobSize })
    else { throw RepackError.invalidPlan("invalid V4.1 DSpark contract") }
  }

  static func expertLayerPath(_ layer: Int) -> String {
    String(format: "experts/layer_%02d.bin", layer)
  }

  static func engramWeightPath(_ layer: Int) -> String {
    String(format: "engram/layer_%02d.weight.bin", layer)
  }

  static func engramScalePath(_ layer: Int) -> String {
    String(format: "engram/layer_%02d.scale.bin", layer)
  }

  static func isExpert(_ name: String) -> Bool {
    name.hasPrefix("layers.") && name.contains(".ffn.experts.")
  }

  static func isEngramTable(_ name: String) -> Bool {
    name.hasSuffix(".engram.embed.weight") || name.hasSuffix(".engram.embed.scale")
  }

  static func isMTP(_ name: String) -> Bool {
    name.hasPrefix("mtp.")
  }

  static func isVision(_ name: String) -> Bool {
    name.hasPrefix("vision.") || name.hasPrefix("aligner.") || name == "image_start"
      || name == "image_end" || name == "image_newline"
  }

  static func isTextCommon(_ name: String) -> Bool {
    name.hasPrefix("layers.") || name == "embed.weight" || name == "head.weight"
      || name == "norm.weight"
  }
}

struct DeepSeekV41Config: Decodable, Sendable {
  struct TextConfig: Decodable, Sendable {
    let modelType: String
    let hiddenSize: Int
    let moeIntermediateSize: Int
    let routedExpertCount: Int
    let sharedExpertCount: Int
    let selectedExpertCount: Int
    let hiddenLayerCount: Int
    let maximumContext: Int
    let engramLayers: [Int]
    let engramRows: [Int]
    let engramDimension: Int
    let kvSourceLayers: [Int]
    let indexSourceLayers: [Int]

    enum CodingKeys: String, CodingKey {
      case modelType = "model_type"
      case hiddenSize = "hidden_size"
      case moeIntermediateSize = "moe_intermediate_size"
      case routedExpertCount = "n_routed_experts"
      case sharedExpertCount = "n_shared_experts"
      case selectedExpertCount = "num_experts_per_tok"
      case hiddenLayerCount = "num_hidden_layers"
      case maximumContext = "max_position_embeddings"
      case engramLayers = "engram_layer_ids"
      case engramRows = "engram_num_embeddings"
      case engramDimension = "engram_head_dim"
      case kvSourceLayers = "kv_source_layer_ids"
      case indexSourceLayers = "index_source_layer_ids"
    }
  }

  struct Quantization: Decodable, Sendable {
    let quantMethod: String
    let expertDType: String
    let scaleFormat: String
    let weightBlockSize: [Int]

    enum CodingKeys: String, CodingKey {
      case quantMethod = "quant_method"
      case expertDType = "expert_dtype"
      case scaleFormat = "scale_fmt"
      case weightBlockSize = "weight_block_size"
    }
  }

  let architectures: [String]
  let modelType: String
  let textConfig: TextConfig
  let quantization: Quantization

  enum CodingKeys: String, CodingKey {
    case architectures
    case modelType = "model_type"
    case textConfig = "text_config"
    case quantization = "quantization_config"
  }
}

enum DeepSeekV41Planner {
  static func makePlan(index: CheckpointIndex, tensors: [String: SafeTensor], includeDSpark: Bool = false) throws
    -> RepackPlan
  {
    guard Set(index.weightMap.keys) == Set(tensors.keys) else {
      throw RepackError.invalidPlan("checkpoint tensor table does not match the index")
    }

    var copies: [TensorCopy] = []
    let expectedExpertNames = Set(expectedExperts().map(\.name))
    let actualExpertNames = Set(tensors.keys.filter(DeepSeekV41Contract.isExpert))
    if let missing = expectedExpertNames.subtracting(actualExpertNames).sorted().first {
      throw RepackError.invalidPlan("missing V4.1 expert tensor \(missing)")
    }
    if let unexpected = actualExpertNames.subtracting(expectedExpertNames).sorted().first {
      throw RepackError.invalidPlan("unexpected V4.1 expert tensor \(unexpected)")
    }
    if let unexpected = tensors.keys.filter({ name in
      !DeepSeekV41Contract.isExpert(name)
        && !DeepSeekV41Contract.isEngramTable(name)
        && !DeepSeekV41Contract.isMTP(name)
        && !DeepSeekV41Contract.isVision(name)
        && !DeepSeekV41Contract.isTextCommon(name)
    }).sorted().first {
      throw RepackError.invalidPlan("unexpected V4.1 tensor family \(unexpected)")
    }

    for expected in expectedExperts() {
      guard let tensor = tensors[expected.name],
        tensor.dtype == expected.region.dtype,
        tensor.shape == expected.region.shape,
        tensor.length == expected.region.length
      else {
        throw RepackError.invalidPlan("invalid V4.1 expert tensor \(expected.name)")
      }
      copies.append(
        TensorCopy(
          tensor: tensor.name,
          sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset,
          length: tensor.length,
          destinationFile: DeepSeekV41Contract.expertLayerPath(expected.layer),
          destinationOffset: UInt64(expected.expert) * DeepSeekV41Contract.expertBlobSize
            + expected.region.offset))
    }

    var plannedFiles: [PlannedFile] = []
    for table in DeepSeekV41Contract.engram.tables {
      let weightName = "layers.\(table.layer).engram.embed.weight"
      let scaleName = "layers.\(table.layer).engram.embed.scale"
      guard let weight = tensors[weightName], weight.dtype == "F8_E4M3",
        weight.shape == [table.rows, table.dimension],
        weight.length == UInt64(table.rows) * UInt64(table.dimension),
        let scale = tensors[scaleName], scale.dtype == "F8_E8M0",
        scale.shape == [table.rows, table.dimension / table.blockSize],
        scale.length == UInt64(table.rows) * UInt64(table.dimension / table.blockSize)
      else {
        throw RepackError.invalidPlan("invalid V4.1 engram table for layer \(table.layer)")
      }
      copies.append(
        TensorCopy(
          tensor: weight.name, sourceFile: weight.sourceFile, sourceOffset: weight.sourceOffset,
          length: weight.length, destinationFile: table.weightFile, destinationOffset: 0))
      copies.append(
        TensorCopy(
          tensor: scale.name, sourceFile: scale.sourceFile, sourceOffset: scale.sourceOffset,
          length: scale.length, destinationFile: table.scaleFile, destinationOffset: 0))
      plannedFiles.append(PlannedFile(path: table.weightFile, size: weight.length))
      plannedFiles.append(PlannedFile(path: table.scaleFile, size: scale.length))
    }

    var commonTensors: [InstalledTensor] = []
    var commonOffset: UInt64 = 0
    for tensor in tensors.values.sorted(by: { $0.name < $1.name })
    where !DeepSeekV41Contract.isExpert(tensor.name)
      && !DeepSeekV41Contract.isEngramTable(tensor.name)
      && !DeepSeekV41Contract.isMTP(tensor.name)
      && !DeepSeekV41Contract.isVision(tensor.name)
    {
      commonOffset = aligned(commonOffset, to: DeepSeekV41Contract.commonAlignment)
      commonTensors.append(
        InstalledTensor(
          name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
          offset: commonOffset, length: tensor.length))
      copies.append(
        TensorCopy(
          tensor: tensor.name, sourceFile: tensor.sourceFile, sourceOffset: tensor.sourceOffset,
          length: tensor.length, destinationFile: "common.bin",
          destinationOffset: commonOffset))
      commonOffset += tensor.length
    }
    guard !commonTensors.isEmpty else {
      throw RepackError.invalidPlan("V4.1 checkpoint has no text common tensors")
    }

    let expertLayerSize = UInt64(DeepSeekV41Contract.expertCount)
      * DeepSeekV41Contract.expertBlobSize
    plannedFiles.insert(PlannedFile(path: "common.bin", size: commonOffset), at: 0)
    plannedFiles.append(
      contentsOf: (0..<DeepSeekV41Contract.layerCount).map {
        PlannedFile(path: DeepSeekV41Contract.expertLayerPath($0), size: expertLayerSize)
      })

    var dspark: DSparkDescriptor?
    if includeDSpark {
      let expectedNames = Set((0..<3).flatMap { layer in
        (0..<128).flatMap { expert in
          DeepSeekV41Contract.expertRegions.map { "mtp.\(layer).ffn.experts.\(expert).\($0.name)" }
        }
      })
      guard Set(tensors.keys.filter { $0.hasPrefix("mtp.") && $0.contains(".experts.") }) == expectedNames else {
        throw RepackError.invalidPlan("V4.1 DSpark expert set is incomplete")
      }
      for layer in 0..<3 {
        let path = String(format: "dspark/experts/layer_%02d.bin", layer)
        plannedFiles.append(PlannedFile(path: path, size: UInt64(128) * DeepSeekV41Contract.expertBlobSize))
        for expert in 0..<128 {
          for region in DeepSeekV41Contract.expertRegions {
            let name = "mtp.\(layer).ffn.experts.\(expert).\(region.name)"
            guard let tensor = tensors[name], tensor.shape == region.shape,
              tensor.dtype == region.dtype, tensor.length == region.length else {
              throw RepackError.invalidPlan("invalid V4.1 DSpark tensor \(name)")
            }
            copies.append(TensorCopy(tensor: name, sourceFile: tensor.sourceFile,
              sourceOffset: tensor.sourceOffset, length: tensor.length, destinationFile: path,
              destinationOffset: UInt64(expert) * DeepSeekV41Contract.expertBlobSize + region.offset))
          }
        }
      }
      var common: [InstalledTensor] = []
      var offset: UInt64 = 0
      for tensor in tensors.values.sorted(by: { $0.name < $1.name })
      where tensor.name.hasPrefix("mtp.") && !tensor.name.contains(".experts.") {
        offset = aligned(offset, to: DeepSeekV41Contract.commonAlignment)
        common.append(InstalledTensor(name: tensor.name, dtype: tensor.dtype, shape: tensor.shape,
          offset: offset, length: tensor.length))
        copies.append(TensorCopy(tensor: tensor.name, sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset, length: tensor.length,
          destinationFile: "dspark/common.bin", destinationOffset: offset))
        offset += tensor.length
      }
      guard common.count == 97 else { throw RepackError.invalidPlan("incomplete V4.1 DSpark common tensors") }
      plannedFiles.append(PlannedFile(path: "dspark/common.bin", size: offset))
      dspark = DSparkDescriptor(layerCount: 3, blockSize: 5, noiseTokenID: 128799,
        targetLayerIDs: [37, 38, 39], markovRank: 256, commonTensors: common)
    }

    return RepackPlan(
      formatVersion: 3,
      modelID: DeepSeekV41Contract.modelID,
      revision: DeepSeekV41Contract.revision,
      layerCount: DeepSeekV41Contract.layerCount,
      expertCount: DeepSeekV41Contract.expertCount,
      selectedExpertCount: DeepSeekV41Contract.selectedExpertCount,
      expertBlobSize: DeepSeekV41Contract.expertBlobSize,
      checkpointTensorBytes: copies.reduce(UInt64(0)) { $0 + $1.length },
      files: plannedFiles,
      commonTensors: commonTensors,
      expertRegions: DeepSeekV41Contract.expertRegions,
      dspark: dspark,
      copies: copies,
      modelKind: .deepSeekV41,
      maximumContext: DeepSeekV41Contract.maximumContext,
      engram: DeepSeekV41Contract.engram)
  }

  private struct ExpectedExpert {
    let name: String
    let layer: Int
    let expert: Int
    let region: ExpertRegion
  }

  private static func expectedExperts() -> [ExpectedExpert] {
    var result: [ExpectedExpert] = []
    result.reserveCapacity(
      DeepSeekV41Contract.layerCount * DeepSeekV41Contract.expertCount
        * DeepSeekV41Contract.expertRegions.count)
    for layer in 0..<DeepSeekV41Contract.layerCount {
      for expert in 0..<DeepSeekV41Contract.expertCount {
        for region in DeepSeekV41Contract.expertRegions {
          result.append(
            ExpectedExpert(
              name: "layers.\(layer).ffn.experts.\(expert).\(region.name)",
              layer: layer,
              expert: expert,
              region: region))
        }
      }
    }
    return result
  }
}

public struct DeepSeekV41Checkpoint: Sendable {
  private let source: any CheckpointSource

  public init() {
    source = HuggingFaceSource(
      modelID: DeepSeekV41Contract.modelID, revision: DeepSeekV41Contract.revision)
  }

  init(source: any CheckpointSource) {
    self.source = source
  }

  public func makeRepackPlan(includeDSpark: Bool = false) async throws -> RepackPlan {
    let configData = try await source.data(path: "config.json")
    let config: DeepSeekV41Config
    do {
      config = try JSONDecoder().decode(DeepSeekV41Config.self, from: configData)
    } catch {
      throw RepackError.incompatibleModel("cannot decode V4.1 model config: \(error)")
    }
    try DeepSeekV41Contract.validate(config)
    let indexData = try await source.data(path: "model.safetensors.index.json")
    let index = try CheckpointIndex.decode(indexData)
    return try DeepSeekV41Planner.makePlan(index: index, tensors: try await readTensors(index: index), includeDSpark: includeDSpark)
  }

  public func repack(
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let plan = try await makeRepackPlan()
    return try await repack(plan: plan, to: output, progress: progress)
  }

  public func repack(
    plan: RepackPlan,
    to output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    try DeepSeekV41Contract.validate(plan)
    return try await Repacker(source: source).run(plan: plan, output: output, progress: progress)
  }

  public func repair(
    at output: URL,
    invalidFiles: Set<String>,
    progress: (@Sendable (RepackProgress) -> Void)? = nil
  ) async throws -> InstalledManifest {
    let installed = try InstalledModel.loadManifest(at: output)
    let plan = try await makeRepackPlan(includeDSpark: installed.dspark != nil)
    return try await Repacker(source: source).repair(
      plan: plan, output: output, invalidFiles: invalidFiles, progress: progress)
  }

  public func installDSpark(at output: URL,
    progress: (@Sendable (RepackProgress) -> Void)? = nil) async throws -> InstalledManifest {
    let manifest = try DeepSeekV41Contract.validate(InstalledModel.loadManifest(at: output))
    if manifest.dspark != nil { return try InstalledModel.verify(at: output) }
    let plan = try await makeRepackPlan(includeDSpark: true)
    return try await Repacker(source: source).repair(plan: plan, output: output,
      invalidFiles: Set(plan.files.map(\.path).filter { $0.hasPrefix("dspark/") } + ["inference/config.json"]), progress: progress)
  }

  private func readTensors(index: CheckpointIndex) async throws -> [String: SafeTensor] {
    var headers: [String: (base: UInt64, header: SafeTensorsHeader)] = [:]
    for shard in Set(index.weightMap.values).sorted() {
      let prefix = try await source.data(path: shard, range: 0..<8)
      let headerLength = try littleEndianUInt64(prefix)
      guard headerLength > 1, headerLength <= 64 * 1_024 * 1_024 else {
        throw RepackError.invalidSafeTensors("invalid header length \(headerLength) in \(shard)")
      }
      let data = try await source.data(path: shard, range: 8..<(8 + headerLength))
      headers[shard] = (8 + headerLength, try SafeTensorsHeader.decode(data))
    }

    var tensors: [String: SafeTensor] = [:]
    tensors.reserveCapacity(index.weightMap.count)
    for (name, shard) in index.weightMap {
      guard let sourceHeader = headers[shard], let entry = sourceHeader.header.entries[name] else {
        throw RepackError.invalidIndex("index tensor \(name) is missing from \(shard)")
      }
      tensors[name] = SafeTensor(
        name: name,
        sourceFile: shard,
        dtype: entry.dtype,
        shape: entry.shape,
        sourceOffset: sourceHeader.base + entry.dataStart,
        length: entry.dataEnd - entry.dataStart)
    }
    return tensors
  }
}
