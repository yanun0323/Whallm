import Foundation

enum RepackPlanner {
  static func dsparkInstalledBytes(tensors: [String: SafeTensor]) throws -> UInt64 {
    try validateExperts(
      expected: expectedDSparkExperts(),
      actual: tensors.keys.filter(isDSparkExpert),
      label: "DSpark routed expert"
    )
    var copies: [TensorCopy] = []
    let common = appendCommonTensors(
      tensors.values.filter { $0.name.hasPrefix("mtp.") && !isDSparkExpert($0.name) },
      destination: "dspark/common.bin", copies: &copies)
    guard !common.tensors.isEmpty else {
      throw RepackError.invalidPlan("DSpark has no common tensors")
    }
    try appendExperts(expectedDSparkExperts(), tensors: tensors, path: dsparkLayerPath,
      label: "DSpark routed expert", copies: &copies)
    return common.size + UInt64(ModelContract.dsparkLayerCount * ModelContract.expertCount)
      * ModelContract.expertBlobSize
  }

  static func makePlan(
    index: CheckpointIndex,
    tensors: [String: SafeTensor],
    includeDSpark: Bool = false
  ) throws -> RepackPlan {
    guard tensors.count == index.weightMap.count else {
      throw RepackError.invalidIndex(
        "resolved \(tensors.count) tensors; index contains \(index.weightMap.count)"
      )
    }

    try validateExperts(
      expected: expectedExperts(),
      actual: tensors.keys.filter(isMainExpert),
      label: "routed expert"
    )
    if includeDSpark {
      try validateExperts(
        expected: expectedDSparkExperts(),
        actual: tensors.keys.filter(isDSparkExpert),
        label: "DSpark routed expert"
      )
    }

    var copies: [TensorCopy] = []
    copies.reserveCapacity(tensors.count)
    let mainCommon = appendCommonTensors(
      tensors.values.filter { !$0.name.hasPrefix("mtp.") && !isMainExpert($0.name) },
      destination: "common.bin",
      copies: &copies
    )
    guard !mainCommon.tensors.isEmpty else {
      throw RepackError.invalidPlan("main model has no common tensors")
    }

    let dsparkCommon =
      includeDSpark
      ? appendCommonTensors(
        tensors.values.filter { $0.name.hasPrefix("mtp.") && !isDSparkExpert($0.name) },
        destination: "dspark/common.bin",
        copies: &copies
      ) : (tensors: [], size: 0)
    if includeDSpark && dsparkCommon.tensors.isEmpty {
      throw RepackError.invalidPlan("DSpark has no common tensors")
    }

    try appendExperts(
      expectedExperts(),
      tensors: tensors,
      path: layerPath,
      label: "routed expert",
      copies: &copies
    )
    if includeDSpark {
      try appendExperts(
        expectedDSparkExperts(),
        tensors: tensors,
        path: dsparkLayerPath,
        label: "DSpark routed expert",
        copies: &copies
      )
    }

    let layerSize = UInt64(ModelContract.expertCount) * ModelContract.expertBlobSize
    var files =
      [PlannedFile(path: "common.bin", size: mainCommon.size)]
      + (0..<ModelContract.layerCount).map { PlannedFile(path: layerPath($0), size: layerSize) }
    if includeDSpark {
      files.append(PlannedFile(path: "dspark/common.bin", size: dsparkCommon.size))
      files += (0..<ModelContract.dsparkLayerCount).map {
        PlannedFile(path: dsparkLayerPath($0), size: layerSize)
      }
    }

    let dspark =
      includeDSpark
      ? DSparkDescriptor(
        layerCount: ModelContract.dsparkLayerCount,
        blockSize: ModelContract.dsparkBlockSize,
        noiseTokenID: ModelContract.dsparkNoiseTokenID,
        targetLayerIDs: ModelContract.dsparkTargetLayerIDs,
        markovRank: ModelContract.dsparkMarkovRank,
        commonTensors: dsparkCommon.tensors
      ) : nil
    return RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: copies.reduce(UInt64(0)) { $0 + $1.length },
      files: files,
      commonTensors: mainCommon.tensors,
      expertRegions: ModelContract.expertRegions,
      dspark: dspark,
      copies: copies
    )
  }

  private struct ExpectedExpert {
    let name: String
    let layer: Int
    let expert: Int
    let region: ExpertRegion
  }

  private static func validateExperts(
    expected: [ExpectedExpert],
    actual: [String],
    label: String
  ) throws {
    let expectedNames = Set(expected.map(\.name))
    let actualNames = Set(actual)
    if let name = expectedNames.subtracting(actualNames).sorted().first {
      throw RepackError.invalidPlan("missing \(label) tensor \(name)")
    }
    if let name = actualNames.subtracting(expectedNames).sorted().first {
      throw RepackError.invalidPlan("unexpected \(label) tensor \(name)")
    }
  }

  private static func appendCommonTensors(
    _ source: some Sequence<SafeTensor>,
    destination: String,
    copies: inout [TensorCopy]
  ) -> (tensors: [InstalledTensor], size: UInt64) {
    var installed: [InstalledTensor] = []
    var offset: UInt64 = 0
    for tensor in source.sorted(by: { $0.name < $1.name }) {
      offset = aligned(offset, to: ModelContract.commonAlignment)
      installed.append(
        InstalledTensor(
          name: tensor.name,
          dtype: tensor.dtype,
          shape: tensor.shape,
          offset: offset,
          length: tensor.length
        ))
      copies.append(
        TensorCopy(
          tensor: tensor.name,
          sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset,
          length: tensor.length,
          destinationFile: destination,
          destinationOffset: offset
        ))
      offset += tensor.length
    }
    return (installed, offset)
  }

  private static func appendExperts(
    _ expected: [ExpectedExpert],
    tensors: [String: SafeTensor],
    path: (Int) -> String,
    label: String,
    copies: inout [TensorCopy]
  ) throws {
    for expected in expected {
      guard let tensor = tensors[expected.name] else {
        throw RepackError.invalidPlan("missing \(label) tensor \(expected.name)")
      }
      guard tensor.dtype == expected.region.dtype,
        tensor.shape == expected.region.shape,
        tensor.length == expected.region.length
      else {
        throw RepackError.invalidPlan(
          "invalid layout for \(tensor.name): \(tensor.dtype) \(tensor.shape) \(tensor.length) bytes"
        )
      }
      copies.append(
        TensorCopy(
          tensor: tensor.name,
          sourceFile: tensor.sourceFile,
          sourceOffset: tensor.sourceOffset,
          length: tensor.length,
          destinationFile: path(expected.layer),
          destinationOffset: UInt64(expected.expert) * ModelContract.expertBlobSize
            + expected.region.offset
        ))
    }
  }

  private static func expectedExperts() -> [ExpectedExpert] {
    expectedExperts(prefix: "layers", layers: ModelContract.layerCount)
  }

  private static func expectedDSparkExperts() -> [ExpectedExpert] {
    expectedExperts(prefix: "mtp", layers: ModelContract.dsparkLayerCount)
  }

  private static func expectedExperts(prefix: String, layers: Int) -> [ExpectedExpert] {
    var result: [ExpectedExpert] = []
    result.reserveCapacity(
      layers * ModelContract.expertCount * ModelContract.expertRegions.count)
    for layer in 0..<layers {
      for expert in 0..<ModelContract.expertCount {
        for region in ModelContract.expertRegions {
          result.append(
            ExpectedExpert(
              name: "\(prefix).\(layer).ffn.experts.\(expert).\(region.name)",
              layer: layer,
              expert: expert,
              region: region
            ))
        }
      }
    }
    return result
  }

  private static func isMainExpert(_ name: String) -> Bool {
    name.hasPrefix("layers.") && name.contains(".ffn.experts.")
  }

  private static func isDSparkExpert(_ name: String) -> Bool {
    name.hasPrefix("mtp.") && name.contains(".ffn.experts.")
  }

  private static func layerPath(_ layer: Int) -> String {
    String(format: "experts/layer_%02d.bin", layer)
  }

  private static func dsparkLayerPath(_ layer: Int) -> String {
    String(format: "dspark/experts/layer_%02d.bin", layer)
  }
}
