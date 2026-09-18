import Foundation
import XCTest

@testable import DeepSeekRepack

final class DeepSeekRepackTests: XCTestCase {
  func testDeepSeekV41ContractDecodesPinnedArchitecture() throws {
    let config = try JSONDecoder().decode(
      DeepSeekV41Config.self,
      from: Data(
        """
        {
          "architectures": ["DeepseekV41ForCausalLM"],
          "model_type": "deepseek_v41",
          "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 5120,
            "moe_intermediate_size": 2304,
            "n_routed_experts": 384,
            "n_shared_experts": 1,
            "num_experts_per_tok": 6,
            "num_hidden_layers": 40,
            "max_position_embeddings": 1048576,
            "engram_layer_ids": [1, 14],
            "engram_num_embeddings": [384006168, 384016682],
            "engram_head_dim": 256,
            "kv_source_layer_ids": [2, 8, 14, 20],
            "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]
          },
          "quantization_config": {
            "quant_method": "fp8",
            "expert_dtype": "fp4",
            "scale_fmt": "ue8m0",
            "weight_block_size": [32, 32]
          }
        }
        """.utf8
      )
    )

    XCTAssertNoThrow(try DeepSeekV41Contract.validate(config))
    XCTAssertEqual(DeepSeekV41Contract.expertBlobSize, 18_800_640)
    XCTAssertEqual(DeepSeekV41Contract.engram.tables.map(\.layer), [1, 14])
    XCTAssertEqual(
      DeepSeekV41Contract.companionPaths,
      [
        "config.json", "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json",
        "encoding/encoding.py",
      ]
    )
    XCTAssertTrue(DeepSeekV41Contract.isTextCommon("layers.0.attn.wq_a.weight"))
    XCTAssertTrue(DeepSeekV41Contract.isTextCommon("embed.weight"))
    XCTAssertFalse(DeepSeekV41Contract.isTextCommon("visual.encoder.weight"))
  }

  func testDSparkInstallRejectsDeepSeekV41BeforeRepair() throws {
    let manifest = InstalledManifest(
      formatVersion: 3,
      modelID: DeepSeekV41Contract.modelID,
      revision: DeepSeekV41Contract.revision,
      layerCount: DeepSeekV41Contract.layerCount,
      expertCount: DeepSeekV41Contract.expertCount,
      selectedExpertCount: DeepSeekV41Contract.selectedExpertCount,
      expertBlobSize: DeepSeekV41Contract.expertBlobSize,
      files: [],
      commonTensors: [],
      expertRegions: DeepSeekV41Contract.expertRegions,
      modelKind: .deepSeekV41,
      maximumContext: DeepSeekV41Contract.maximumContext,
      engram: DeepSeekV41Contract.engram
    )

    XCTAssertThrowsError(try DeepSeekV4Checkpoint.validateDSparkInstallTarget(manifest)) {
      XCTAssertEqual(
        $0 as? RepackError,
        .incompatibleModel("installed model is not DeepSeek-V4-Flash-0731")
      )
    }
  }

  func testPlannerCreatesCanonicalExpertLayout() throws {
    let fixture = makePlannerFixture()
    let plan = try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors)

    XCTAssertEqual(plan.expertBlobSize, 13_369_344)
    XCTAssertEqual(plan.files.count, 44)
    XCTAssertEqual(plan.copies.count, 43 * 256 * 6 + 1)
    XCTAssertEqual(plan.files[1].path, "experts/layer_00.bin")
    XCTAssertEqual(plan.files[1].size, 256 * 13_369_344)

    let first = try XCTUnwrap(
      plan.copies.first {
        $0.tensor == "layers.0.ffn.experts.0.w1.weight"
      })
    XCTAssertEqual(first.destinationFile, "experts/layer_00.bin")
    XCTAssertEqual(first.destinationOffset, 0)

    let secondExpert = try XCTUnwrap(
      plan.copies.first {
        $0.tensor == "layers.0.ffn.experts.1.w1.weight"
      })
    XCTAssertEqual(secondExpert.destinationOffset, 13_369_344)
  }

  func testFormatOnePlanDecodesWithoutFormatTwoFields() throws {
    let fixture = makePlannerFixture()
    let plan = try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors)
    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(plan)) as? [String: Any]
    )
    for key in [
      "modelKind", "maximumContext", "expertQuantization", "ngram", "expertConversions",
    ] {
      object.removeValue(forKey: key)
    }

    let decoded = try JSONDecoder().decode(
      RepackPlan.self, from: JSONSerialization.data(withJSONObject: object))

    XCTAssertEqual(decoded.formatVersion, 1)
    XCTAssertNil(decoded.modelKind)
    XCTAssertNil(decoded.expertQuantization)
    XCTAssertNil(decoded.ngram)
  }

  func testPlannerRejectsMissingExpertTensor() throws {
    var fixture = makePlannerFixture()
    let missing = "layers.42.ffn.experts.255.w3.scale"
    fixture.tensors.removeValue(forKey: missing)
    fixture.index = CheckpointIndex(
      totalSize: fixture.index.totalSize,
      weightMap: fixture.index.weightMap.filter { $0.key != missing }
    )

    XCTAssertThrowsError(try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors))
    {
      XCTAssertTrue(String(describing: $0).contains("missing routed expert tensor"))
    }
  }

  func testPlannerRejectsInvalidExpertShape() throws {
    var fixture = makePlannerFixture()
    let name = "layers.0.ffn.experts.0.w1.weight"
    let original = try XCTUnwrap(fixture.tensors[name])
    fixture.tensors[name] = SafeTensor(
      name: name,
      sourceFile: original.sourceFile,
      dtype: original.dtype,
      shape: [1, 1],
      sourceOffset: original.sourceOffset,
      length: original.length
    )

    XCTAssertThrowsError(try RepackPlanner.makePlan(index: fixture.index, tensors: fixture.tensors))
    {
      XCTAssertTrue(String(describing: $0).contains("invalid layout"))
    }
  }

  func testPlannerAddsOptionalDSparkLayout() throws {
    let fixture = makePlannerFixture(includeDSpark: true)
    let plan = try RepackPlanner.makePlan(
      index: fixture.index,
      tensors: fixture.tensors,
      includeDSpark: true
    )

    XCTAssertEqual(plan.files.count, 48)
    XCTAssertEqual(plan.dspark?.layerCount, 3)
    XCTAssertEqual(plan.dspark?.blockSize, 5)
    XCTAssertEqual(plan.dspark?.targetLayerIDs, [40, 41, 42])
    XCTAssertEqual(plan.dspark?.commonTensors.map(\.name), ["mtp.0.main_proj.weight"])
    XCTAssertEqual(
      plan.files.first { $0.path == "dspark/experts/layer_00.bin" }?.size,
      256 * 13_369_344
    )
    let expert = try XCTUnwrap(
      plan.copies.first { $0.tensor == "mtp.2.ffn.experts.255.w3.scale" }
    )
    XCTAssertEqual(expert.destinationFile, "dspark/experts/layer_02.bin")
  }

  func testSafeTensorsHeaderDecodesTensorMetadata() throws {
    let data = Data(
      #"{"tensor":{"dtype":"I8","shape":[2,3],"data_offsets":[4,10]},"__metadata__":{"format":"pt"}}"#
        .utf8)
    let header = try SafeTensorsHeader.decode(data)
    XCTAssertEqual(
      header.entries["tensor"],
      SafeTensorsHeader.Entry(dtype: "I8", shape: [2, 3], dataStart: 4, dataEnd: 10)
    )
  }

  func testRepackerCopiesOnlyPlannedByteRanges() async throws {
    var files = companionFiles()
    files["shard"] = Data([0, 1, 2, 3, 4, 5, 6, 7])
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 4,
      files: [PlannedFile(path: "common.bin", size: 6)],
      commonTensors: [InstalledTensor(name: "test", dtype: "I8", shape: [4], offset: 1, length: 4)],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "test",
          sourceFile: "shard",
          sourceOffset: 2,
          length: 4,
          destinationFile: "common.bin",
          destinationOffset: 1
        )
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    let manifest = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
    XCTAssertEqual(manifest.files.count, 1 + ModelContract.companionPaths.count)
    XCTAssertEqual(
      try Data(contentsOf: output.appendingPathComponent("common.bin")),
      Data([0, 2, 3, 4, 5, 0])
    )
  }

  func testRepackerResumesValidatedChunks() async throws {
    var files = companionFiles()
    var shard = Data(repeating: 0, count: 100_004)
    shard.replaceSubrange(0..<4, with: [1, 2, 3, 4])
    shard.replaceSubrange(100_000..<100_004, with: [5, 6, 7, 8])
    files["shard"] = shard
    let source = FailingCheckpointSource(files: files, failAt: 100_000, failures: 3)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 8,
      files: [PlannedFile(path: "common.bin", size: 8)],
      commonTensors: [
        InstalledTensor(
          name: "test", dtype: "I8", shape: [8], offset: 0, length: 8
        )
      ],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "a", sourceFile: "shard", sourceOffset: 0, length: 4,
          destinationFile: "common.bin", destinationOffset: 0),
        TensorCopy(
          tensor: "b", sourceFile: "shard", sourceOffset: 100_000, length: 4,
          destinationFile: "common.bin", destinationOffset: 4),
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    do {
      _ = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
      XCTFail("first repack must fail")
    } catch {
      XCTAssertTrue(FileManager.default.fileExists(atPath: output.path + ".partial"))
    }

    _ = try await Repacker(source: source).run(plan: plan, output: output, progress: nil)
    XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("common.bin")), Data(1...8))
    let firstRangeReads = await source.readCount(at: 0)
    XCTAssertEqual(firstRangeReads, 1)
  }

  func testAuditFindsChecksumFailureAndRepairDownloadsInvalidFile() async throws {
    var files = companionFiles()
    files["shard"] = Data([1, 2, 3, 4])
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 1,
      modelID: ModelContract.modelID,
      revision: ModelContract.revision,
      layerCount: ModelContract.layerCount,
      expertCount: ModelContract.expertCount,
      selectedExpertCount: ModelContract.selectedExpertCount,
      expertBlobSize: ModelContract.expertBlobSize,
      checkpointTensorBytes: 4,
      files: [PlannedFile(path: "common.bin", size: 4)],
      commonTensors: [InstalledTensor(name: "test", dtype: "I8", shape: [4], offset: 0, length: 4)],
      expertRegions: ModelContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "test",
          sourceFile: "shard",
          sourceOffset: 0,
          length: 4,
          destinationFile: "common.bin",
          destinationOffset: 0
        )
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("model.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    let manifest = try await Repacker(source: source).run(
      plan: plan, output: output, progress: nil)
    try Data([9, 9, 9, 9]).write(to: output.appendingPathComponent("common.bin"))

    let failed = try InstalledModel.audit(manifest: manifest, at: output)
    XCTAssertEqual(
      failed.issues,
      [InstalledFileIssue(path: "common.bin", kind: .checksumMismatch)]
    )

    let repaired = try await Repacker(source: source).repair(
      plan: plan,
      output: output,
      invalidFiles: ["common.bin"],
      progress: nil
    )
    XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("common.bin")), Data(1...4))
    XCTAssertTrue(try InstalledModel.audit(manifest: repaired, at: output).isValid)
  }

  func testMXFP4MatchesMLXFixedVector() throws {
    let values: [Float] = [
      0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -1, -1.5, -2, -3, -4, -6,
    ] + Array(repeating: 0, count: 18)
    let source = Data(values.flatMap { value -> [UInt8] in
      let bits = UInt16(value.bitPattern >> 16)
      return [UInt8(bits & 0xff), UInt8(bits >> 8)]
    })

    let result = try MXFP4.quantizeBF16(source, rows: 1, columns: 32)

    XCTAssertEqual(
      [UInt8](result.weights),
      [0x21, 0x43, 0x65, 0x97, 0xba, 0xdc, 0xfe] + Array(repeating: 0, count: 9))
    XCTAssertEqual([UInt8](result.scales), [127])
  }

  func testMXFP4MatchesMLXBF16Rows() throws {
    let bits: [UInt16] = [
      49381, 49134, 16475, 49335, 48859, 16539, 49289, 16256,
      16585, 49207, 16411, 49367, 49079, 16503, 49321, 0,
      16553, 49271, 16311, 16599, 49179, 16439, 49353, 49024,
      16521, 49307, 16091, 16567, 49243, 16366, 16613, 49152,
      16466, 49339, 48914, 16535, 49294, 16219, 16581, 49216,
      16402, 49371, 49097, 16494, 49326, 48658, 16549, 49280,
      16293, 16594, 49189, 16430, 49358, 49042, 16517, 49312,
      16018, 16562, 49253, 16347, 16608, 49161, 16457, 49344,
    ]
    let source = Data(bits.flatMap { [UInt8($0 & 0xff), UInt8($0 >> 8)] })

    let result = try MXFP4.quantizeBF16(source, rows: 2, columns: 32)

    XCTAssertEqual(
      [UInt8](result.weights),
      [
        207, 245, 105, 46, 215, 244, 107, 15, 231, 115, 92, 175, 230, 113, 77, 199,
        245, 105, 46, 215, 244, 107, 143, 231, 115, 93, 175, 230, 113, 62, 199, 245,
      ])
    XCTAssertEqual([UInt8](result.scales), [127, 127])
  }

  func testMXFP4ConvertsOfficialFP8Blocks() throws {
    let firstRow: [UInt8] = [
      0x30, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x4c,
      0xb0, 0xb8, 0xbc, 0xc0, 0xc4, 0xc8, 0xcc,
    ] + Array(repeating: 0, count: 114)
    let source = Data(firstRow + Array(repeating: 0, count: 127 * 128))
    let result = try MXFP4.quantizeFP8(
      source,
      inverseScales: Data([0x80, 0x3f]),
      rows: 128,
      columns: 128)

    XCTAssertEqual(
      [UInt8](result.weights.prefix(16)),
      [0x21, 0x43, 0x65, 0x97, 0xba, 0xdc, 0xfe] + Array(repeating: 0, count: 9))
    XCTAssertEqual([UInt8](result.scales.prefix(4)), [127, 0, 0, 0])
  }

  func testMXFP4FP8ConversionMatchesMLXFixedVector() throws {
    let source = Data((0..<(128 * 128)).map { index -> UInt8 in
      let value = UInt8(truncatingIfNeeded: index * 37 + 11)
      return value & 0x7f == 0x7f ? value ^ 1 : value
    })
    let result = try MXFP4.quantizeFP8(
      source,
      inverseScales: Data([0x40, 0x3f]),
      rows: 128,
      columns: 128)

    XCTAssertEqual(
      sha256(result.weights),
      "7c332f3b69c7e580b8f3dd7c89bf44d18f81debfb17a648a7d131e28907a3b71")
    XCTAssertEqual(
      sha256(result.scales),
      "0fecfb3515a322ce43dca1e7d6115fcf2c1f34d474774097fb4eededdcc6cf49")
  }

  func testInstalledArtifactFileDownloadResumesExistingBytes() async throws {
    let sourceData = Data("0123456789".utf8)
    let source = MemoryCheckpointSource(files: ["weights.bin": sourceData])
    let file = InstalledFile(
      path: "weights.bin", size: UInt64(sourceData.count), sha256: sha256(sourceData))
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("weights.bin")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }
    try Data(sourceData.prefix(4)).write(to: output)
    let counter = DownloadByteCounter()

    let digest = try await InstalledArtifactFileDownloader(source: source, chunkSize: 3).run(
      file: file, to: output
    ) { copiedBytes, downloadedBytes in
      await counter.add(copiedBytes: copiedBytes, downloadedBytes: downloadedBytes)
    }

    XCTAssertEqual(try Data(contentsOf: output), sourceData)
    XCTAssertEqual(digest, file.sha256)
    let counts = await counter.counts
    XCTAssertEqual(counts.copied, 10)
    XCTAssertEqual(counts.downloaded, 6)
  }

  func testQwenPlannerCreatesFormatTwoAndExcludesVisionAndMTP() throws {
    let fixture = makeQwenPlannerFixture()
    let plan = try QwenPlanner.makePlan(index: fixture.index, tensors: fixture.tensors)

    XCTAssertEqual(plan.formatVersion, 2)
    XCTAssertEqual(plan.modelKind, .qwen3_8FlashNext)
    XCTAssertEqual(plan.revision, QwenContract.revision)
    XCTAssertEqual(plan.maximumContext, 262_144)
    XCTAssertEqual(plan.expertBlobSize, 2_611_200)
    XCTAssertEqual(plan.files.count, 50)
    XCTAssertEqual(plan.expertConversions?.count, 48 * 512 * 3)
    XCTAssertEqual(plan.ngram?.dtype, "F8_E4M3")
    XCTAssertEqual(plan.ngram?.rowBytes, 160)
    XCTAssertEqual(plan.ngram?.headOffsets.count, 16)
    XCTAssertEqual(
      plan.commonTensors.map(\.name),
      [
        "model.language_model.embed_tokens.weight",
        QwenContract.ngramScaleName,
      ])
    XCTAssertFalse(plan.copies.contains { $0.tensor.hasPrefix("model.visual.") })
    XCTAssertFalse(plan.copies.contains { $0.tensor.hasPrefix("mtp.") })
  }

  func testQwenMTPPlannerCreatesSidecarLayout() throws {
    let fixture = makeQwenMTPPlannerFixture()
    let plan = try QwenMTPPlanner.makePlan(index: fixture.index, tensors: fixture.tensors)

    XCTAssertEqual(plan.files, [
      PlannedFile(path: "mtp/common.bin", size: 7_169),
      PlannedFile(
        path: "mtp/experts/layer_00.bin",
        size: UInt64(QwenContract.expertCount) * QwenContract.expertBlobSize),
    ])
    XCTAssertEqual(plan.commonTensors.count, 29)
    XCTAssertEqual(plan.expertConversions?.count, 512 * 3)
    XCTAssertTrue(plan.commonTensors.allSatisfy { $0.name.hasPrefix("mtp.") })
    XCTAssertTrue(plan.copies.allSatisfy { $0.destinationFile == "mtp/common.bin" })
  }

  func testQwenConversionCanRepairOneDamagedExpertLayer() async throws {
    let gateBytes = QwenContract.expertIntermediateSize * QwenContract.hiddenSize
    let downBytes = QwenContract.hiddenSize * QwenContract.expertIntermediateSize
    let scaleBytes = 5 * 20 * 2
    var files = qwenCompanionFiles()
    files["common"] = Data([7])
    files["gate"] = Data(repeating: 0, count: gateBytes)
    files["up"] = Data(repeating: 0, count: gateBytes)
    files["down"] = Data(repeating: 0, count: downBytes)
    files["gate-scale"] = Data(repeating: 0, count: scaleBytes)
    files["up-scale"] = Data(repeating: 0, count: scaleBytes)
    files["down-scale"] = Data(repeating: 0, count: scaleBytes)
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 2,
      modelID: QwenContract.modelID,
      revision: QwenContract.revision,
      layerCount: 1,
      expertCount: 1,
      selectedExpertCount: 1,
      expertBlobSize: QwenContract.expertBlobSize,
      checkpointTensorBytes: UInt64(1 + gateBytes * 2 + downBytes + scaleBytes * 3),
      files: [
        PlannedFile(path: "common.bin", size: 1),
        PlannedFile(path: "experts/layer_00.bin", size: QwenContract.expertBlobSize),
      ],
      commonTensors: [
        InstalledTensor(name: "fixture", dtype: "U8", shape: [1], offset: 0, length: 1)
      ],
      expertRegions: QwenContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "fixture", sourceFile: "common", sourceOffset: 0, length: 1,
          destinationFile: "common.bin", destinationOffset: 0)
      ],
      modelKind: .qwen3_8FlashNext,
      maximumContext: QwenContract.maximumContext,
      expertQuantization: QwenContract.quantization,
      expertConversions: [
        ExpertConversion(
          tensor: "gate", sourceFile: "gate", sourceOffset: 0, sourceDType: "F8_E4M3",
          sourceShape: [QwenContract.expertIntermediateSize, QwenContract.hiddenSize],
          sourceScaleTensor: "gate-scale", sourceScaleFile: "gate-scale",
          sourceScaleOffset: 0, sourceScaleDType: "BF16", sourceScaleShape: [5, 20],
          destinationFile: "experts/layer_00.bin", expert: 0, destinationRow: 0,
          weightRegion: "gate_up.weight",
          scaleRegion: "gate_up.scale"),
        ExpertConversion(
          tensor: "up", sourceFile: "up", sourceOffset: 0, sourceDType: "F8_E4M3",
          sourceShape: [QwenContract.expertIntermediateSize, QwenContract.hiddenSize],
          sourceScaleTensor: "up-scale", sourceScaleFile: "up-scale",
          sourceScaleOffset: 0, sourceScaleDType: "BF16", sourceScaleShape: [5, 20],
          destinationFile: "experts/layer_00.bin", expert: 0,
          destinationRow: QwenContract.expertIntermediateSize,
          weightRegion: "gate_up.weight", scaleRegion: "gate_up.scale"),
        ExpertConversion(
          tensor: "down", sourceFile: "down", sourceOffset: 0, sourceDType: "F8_E4M3",
          sourceShape: [QwenContract.hiddenSize, QwenContract.expertIntermediateSize],
          sourceScaleTensor: "down-scale", sourceScaleFile: "down-scale",
          sourceScaleOffset: 0, sourceScaleDType: "BF16", sourceScaleShape: [20, 5],
          destinationFile: "experts/layer_00.bin", expert: 0, destinationRow: 0,
          weightRegion: "down.weight",
          scaleRegion: "down.scale"),
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("qwen.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    let manifest = try await Repacker(source: source).run(
      plan: plan, output: output, progress: nil)
    let layer = output.appendingPathComponent("experts/layer_00.bin")
    XCTAssertEqual(try Data(contentsOf: layer), Data(repeating: 0, count: 2_611_200))
    let handle = try FileHandle(forWritingTo: layer)
    try handle.write(contentsOf: Data([1]))
    try handle.close()
    XCTAssertEqual(
      try InstalledModel.audit(manifest: manifest, at: output).issues.first?.kind,
      .checksumMismatch)

    let repaired = try await Repacker(source: source).repair(
      plan: plan, output: output, invalidFiles: ["experts/layer_00.bin"], progress: nil)
    XCTAssertTrue(try InstalledModel.audit(manifest: repaired, at: output).isValid)
  }

  func testQwenConversionRejectsPlanWithoutConversionVersion() async throws {
    let gateBytes = QwenContract.expertIntermediateSize * QwenContract.hiddenSize
    let downBytes = QwenContract.hiddenSize * QwenContract.expertIntermediateSize
    let scaleBytes = 5 * 20 * 2
    var files = qwenCompanionFiles()
    files["common"] = Data([7])
    files["gate"] = Data(repeating: 0, count: gateBytes)
    files["up"] = Data(repeating: 0, count: gateBytes)
    files["down"] = Data(repeating: 0, count: downBytes)
    files["gate-scale"] = Data(repeating: 0, count: scaleBytes)
    files["up-scale"] = Data(repeating: 0, count: scaleBytes)
    files["down-scale"] = Data(repeating: 0, count: scaleBytes)
    let source = MemoryCheckpointSource(files: files)
    let plan = RepackPlan(
      formatVersion: 2,
      modelID: QwenContract.modelID,
      revision: QwenContract.revision,
      layerCount: 1,
      expertCount: 1,
      selectedExpertCount: 1,
      expertBlobSize: QwenContract.expertBlobSize,
      checkpointTensorBytes: UInt64(1 + gateBytes * 2 + downBytes + scaleBytes * 3),
      files: [
        PlannedFile(path: "common.bin", size: 1),
        PlannedFile(path: "experts/layer_00.bin", size: QwenContract.expertBlobSize),
      ],
      commonTensors: [
        InstalledTensor(name: "fixture", dtype: "U8", shape: [1], offset: 0, length: 1)
      ],
      expertRegions: QwenContract.expertRegions,
      copies: [
        TensorCopy(
          tensor: "fixture", sourceFile: "common", sourceOffset: 0, length: 1,
          destinationFile: "common.bin", destinationOffset: 0)
      ],
      modelKind: .qwen3_8FlashNext,
      maximumContext: QwenContract.maximumContext,
      expertQuantization: nil,
      expertConversions: [
        ExpertConversion(
          tensor: "gate", sourceFile: "gate", sourceOffset: 0, sourceDType: "F8_E4M3",
          sourceShape: [QwenContract.expertIntermediateSize, QwenContract.hiddenSize],
          sourceScaleTensor: "gate-scale", sourceScaleFile: "gate-scale",
          sourceScaleOffset: 0, sourceScaleDType: "BF16", sourceScaleShape: [5, 20],
          destinationFile: "experts/layer_00.bin", expert: 0, destinationRow: 0,
          weightRegion: "gate_up.weight",
          scaleRegion: "gate_up.scale"),
      ]
    )
    let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let output = parent.appendingPathComponent("qwen.dsv4")
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: parent) }

    do {
      _ = try await Repacker(source: source).run(
        plan: plan, output: output, progress: nil)
      XCTExpectFailure("a plan with conversions but no quantization must not install")
    } catch let RepackError.invalidPlan(message) {
      XCTAssertTrue(
        message.contains("invalid MXFP4 conversion for gate")
          || message.contains("missing expert quantization conversion version"),
        "unexpected message: \(message)")
    }
  }
}

private actor DownloadByteCounter {
  private(set) var counts: (copied: UInt64, downloaded: UInt64) = (0, 0)

  func add(copiedBytes: UInt64, downloadedBytes: UInt64) {
    counts.copied += copiedBytes
    counts.downloaded += downloadedBytes
  }
}

private struct PlannerFixture {
  var index: CheckpointIndex
  var tensors: [String: SafeTensor]
}

private func makePlannerFixture(includeDSpark: Bool = false) -> PlannerFixture {
  var tensors: [String: SafeTensor] = [:]
  var weightMap: [String: String] = [:]
  var offset: UInt64 = 0

  let common = SafeTensor(
    name: "embed.weight",
    sourceFile: "common.safetensors",
    dtype: "BF16",
    shape: [2, 2],
    sourceOffset: 100,
    length: 8
  )
  tensors[common.name] = common
  weightMap[common.name] = common.sourceFile

  for layer in 0..<ModelContract.layerCount {
    for expert in 0..<ModelContract.expertCount {
      for region in ModelContract.expertRegions {
        let name = "layers.\(layer).ffn.experts.\(expert).\(region.name)"
        let tensor = SafeTensor(
          name: name,
          sourceFile: "experts.safetensors",
          dtype: region.dtype,
          shape: region.shape,
          sourceOffset: offset,
          length: region.length
        )
        tensors[name] = tensor
        weightMap[name] = tensor.sourceFile
        offset += tensor.length
      }
    }
  }
  if includeDSpark {
    let dsparkCommon = SafeTensor(
      name: "mtp.0.main_proj.weight",
      sourceFile: "dspark.safetensors",
      dtype: "BF16",
      shape: [2, 2],
      sourceOffset: offset,
      length: 8
    )
    tensors[dsparkCommon.name] = dsparkCommon
    weightMap[dsparkCommon.name] = dsparkCommon.sourceFile
    offset += dsparkCommon.length
    for layer in 0..<ModelContract.dsparkLayerCount {
      for expert in 0..<ModelContract.expertCount {
        for region in ModelContract.expertRegions {
          let name = "mtp.\(layer).ffn.experts.\(expert).\(region.name)"
          let tensor = SafeTensor(
            name: name,
            sourceFile: "dspark.safetensors",
            dtype: region.dtype,
            shape: region.shape,
            sourceOffset: offset,
            length: region.length
          )
          tensors[name] = tensor
          weightMap[name] = tensor.sourceFile
          offset += tensor.length
        }
      }
    }
  }
  return PlannerFixture(
    index: CheckpointIndex(totalSize: offset + common.length, weightMap: weightMap),
    tensors: tensors
  )
}

private func makeQwenPlannerFixture() -> PlannerFixture {
  var tensors: [String: SafeTensor] = [:]
  var weightMap: [String: String] = [:]
  var offset: UInt64 = 0

  func add(_ name: String, shape: [Int], dtype: String = "BF16") {
    let itemSize: UInt64 = dtype == "I64" ? 8 : (dtype == "F8_E4M3" ? 1 : 2)
    let length = shape.reduce(itemSize) { $0 * UInt64($1) }
    let tensor = SafeTensor(
      name: name, sourceFile: "fixture.safetensors", dtype: dtype, shape: shape,
      sourceOffset: offset, length: length)
    tensors[name] = tensor
    weightMap[name] = tensor.sourceFile
    offset += length
  }

  add("model.language_model.embed_tokens.weight", shape: [2, 2])
  add("model.visual.blocks.0.weight", shape: [2, 2])
  add("mtp.layers.0.weight", shape: [2, 2])
  add(
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets",
    shape: [16], dtype: "I64")
  add(
    "model.language_model.layers.1.ple.ple_embedding.ngram_heads_vocab_sizes",
    shape: [16], dtype: "I64")
  for shard in 0..<QwenContract.ngramShardCount {
    add(
      "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_\(shard).weight",
      shape: [QwenContract.ngramShardRowCount, QwenContract.ngramRowBytes],
      dtype: "F8_E4M3")
  }
  add(QwenContract.ngramScaleName, shape: [1])
  for layer in 0..<QwenContract.layerCount {
    for expert in 0..<QwenContract.expertCount {
      for projection in ["gate_proj", "up_proj", "down_proj"] {
        let rows = projection == "down_proj"
          ? QwenContract.hiddenSize : QwenContract.expertIntermediateSize
        let columns = projection == "down_proj"
          ? QwenContract.expertIntermediateSize : QwenContract.hiddenSize
        let prefix =
          "model.language_model.layers.\(layer).mlp.experts.\(expert).\(projection)"
        add("\(prefix).weight", shape: [rows, columns], dtype: "F8_E4M3")
        add("\(prefix).weight_scale_inv", shape: [rows / 128, columns / 128])
      }
    }
  }
  return PlannerFixture(
    index: CheckpointIndex(totalSize: offset, weightMap: weightMap), tensors: tensors)
}

private func makeQwenMTPPlannerFixture() -> PlannerFixture {
  var tensors: [String: SafeTensor] = [:]
  var weightMap: [String: String] = [:]
  var offset: UInt64 = 0

  func add(_ name: String, shape: [Int], dtype: String = "BF16") {
    let itemSize: UInt64 = dtype == "F8_E4M3" ? 1 : 2
    let length = shape.reduce(itemSize) { $0 * UInt64($1) }
    let tensor = SafeTensor(
      name: name, sourceFile: "mtp.safetensors", dtype: dtype, shape: shape,
      sourceOffset: offset, length: length)
    tensors[name] = tensor
    weightMap[name] = tensor.sourceFile
    offset += length
  }

  for index in 0..<QwenContract.mtpCommonTensorCount {
    add("mtp.fixture_\(index)", shape: [1], dtype: "F8_E4M3")
  }
  for expert in 0..<QwenContract.expertCount {
    for projection in ["gate_proj", "up_proj", "down_proj"] {
      let rows = projection == "down_proj"
        ? QwenContract.hiddenSize : QwenContract.expertIntermediateSize
      let columns = projection == "down_proj"
        ? QwenContract.expertIntermediateSize : QwenContract.hiddenSize
      let prefix = "mtp.layers.0.mlp.experts.\(expert).\(projection)"
      add("\(prefix).weight", shape: [rows, columns], dtype: "F8_E4M3")
      add("\(prefix).weight_scale_inv", shape: [rows / 128, columns / 128])
    }
  }
  return PlannerFixture(
    index: CheckpointIndex(totalSize: offset, weightMap: weightMap), tensors: tensors)
}

private struct MemoryCheckpointSource: CheckpointSource {
  let files: [String: Data]

  func data(path: String) async throws -> Data {
    guard let data = files[path] else { throw RepackError.badResponse("missing \(path)") }
    return data
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    guard let data = files[path], range.upperBound <= UInt64(data.count) else {
      throw RepackError.badResponse("invalid fixture range for \(path)")
    }
    return Data(data[Int(range.lowerBound)..<Int(range.upperBound)])
  }
}

private actor FailingCheckpointSource: CheckpointSource {
  let files: [String: Data]
  let failAt: UInt64
  var failures: Int
  var reads: [UInt64: Int] = [:]

  init(files: [String: Data], failAt: UInt64, failures: Int) {
    self.files = files
    self.failAt = failAt
    self.failures = failures
  }

  func data(path: String) async throws -> Data {
    guard let data = files[path] else { throw RepackError.badResponse("missing \(path)") }
    return data
  }

  func data(path: String, range: Range<UInt64>) async throws -> Data {
    reads[range.lowerBound, default: 0] += 1
    if range.lowerBound == failAt, failures > 0 {
      failures -= 1
      throw RepackError.badResponse("fixture interruption")
    }
    guard let data = files[path], range.upperBound <= UInt64(data.count) else {
      throw RepackError.badResponse("invalid fixture range for \(path)")
    }
    return Data(data[Int(range.lowerBound)..<Int(range.upperBound)])
  }

  func readCount(at offset: UInt64) -> Int {
    reads[offset, default: 0]
  }
}

private func companionFiles() -> [String: Data] {
  Dictionary(
    uniqueKeysWithValues: ModelContract.companions.map {
      ($0.source, Data($0.source.utf8))
    })
}

private func qwenCompanionFiles() -> [String: Data] {
  Dictionary(
    uniqueKeysWithValues: QwenContract.companions.map {
      ($0.source, Data($0.source.utf8))
    })
}
