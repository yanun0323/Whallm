import Foundation
import XCTest
@testable import DeepSeekRepack

final class DeepSeekV41DSparkTests: XCTestCase {
  func testOptionalPlanSeparates128ExpertDraftStagesFrom384ExpertMainModel() throws {
    var tensors: [String: SafeTensor] = [:]
    func add(_ name: String, _ dtype: String, _ shape: [Int], _ bytes: UInt64) {
      tensors[name] = SafeTensor(name: name, sourceFile: "fixture.safetensors", dtype: dtype, shape: shape,
        sourceOffset: 0, length: bytes)
    }
    for layer in 0..<40 {
      for expert in 0..<384 {
        for region in DeepSeekV41Contract.expertRegions {
          add("layers.\(layer).ffn.experts.\(expert).\(region.name)", region.dtype, region.shape, region.length)
        }
      }
    }
    for table in DeepSeekV41Contract.engram.tables {
      add("layers.\(table.layer).engram.embed.weight", "F8_E4M3", [table.rows, 256], UInt64(table.rows) * 256)
      add("layers.\(table.layer).engram.embed.scale", "F8_E8M0", [table.rows, 8], UInt64(table.rows) * 8)
    }
    add("norm.weight", "BF16", [5120], 10240)
    for layer in 0..<3 {
      for expert in 0..<128 {
        for region in DeepSeekV41Contract.expertRegions {
          add("mtp.\(layer).ffn.experts.\(expert).\(region.name)", region.dtype, region.shape, region.length)
        }
      }
    }
    for i in 0..<97 { add("mtp.0.fixture.\(i).weight", "BF16", [32], 64) }
    let index = CheckpointIndex(totalSize: 0, weightMap: Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, "fixture.safetensors") }))
    let main = try DeepSeekV41Planner.makePlan(index: index, tensors: tensors)
    XCTAssertNil(main.dspark)
    XCTAssertFalse(main.copies.contains { $0.tensor.hasPrefix("mtp.") })
    let plan = try DeepSeekV41Planner.makePlan(index: index, tensors: tensors, includeDSpark: true)
    try DeepSeekV41Contract.validate(plan)
    XCTAssertEqual(plan.dspark?.targetLayerIDs, [37, 38, 39])
    XCTAssertEqual(plan.dspark?.commonTensors.count, 97)
    XCTAssertEqual(plan.files.first { $0.path == "dspark/experts/layer_02.bin" }?.size, 128 * DeepSeekV41Contract.expertBlobSize)
    XCTAssertEqual(plan.copies.filter { $0.destinationFile.hasPrefix("dspark/experts/") }.count, 3 * 128 * 6)
    XCTAssertEqual(plan.commonTensors.count, 1)
    for includeDSpark in [false, true] {
      let october = try DeepSeekV41Planner.makePlan(
        index: index, tensors: tensors, includeDSpark: includeDSpark, kind: .deepSeekV41_1017)
      XCTAssertEqual(october.modelKind, .deepSeekV41_1017)
      XCTAssertEqual(october.modelID, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
      XCTAssertEqual(october.revision, "1bae0e1088993b9b82de08f59de5be5a34e42170")
      XCTAssertEqual(october.files, includeDSpark ? plan.files : main.files)
      XCTAssertEqual(october.copies, includeDSpark ? plan.copies : main.copies)
      XCTAssertNoThrow(try DeepSeekV41Contract.validate(october, kind: .deepSeekV41_1017))
      XCTAssertThrowsError(try DeepSeekV41Contract.validate(october))
      var raw = try XCTUnwrap(JSONSerialization.jsonObject(
        with: JSONEncoder().encode(october)) as? [String: Any])
      for (field, value) in [("modelID", DeepSeekV41Contract.modelID),
                             ("revision", DeepSeekV41Contract.revision),
                             ("modelKind", "deepseek-v4.1")] {
        var crossed = raw
        crossed[field] = value
        let invalid = try JSONDecoder().decode(RepackPlan.self,
          from: JSONSerialization.data(withJSONObject: crossed))
        XCTAssertThrowsError(try DeepSeekV41Contract.validate(invalid, kind: .deepSeekV41_1017))
      }
      raw.removeValue(forKey: "copies")
      raw.removeValue(forKey: "checkpointTensorBytes")
      raw["files"] = october.files.map { ["path": $0.path, "size": $0.size, "sha256": String(repeating: "0", count: 64)] as [String: Any] }
      var files = try XCTUnwrap(raw["files"] as? [[String: Any]])
      let companions = DeepSeekV41Contract.companionPaths + (includeDSpark ? ["inference/config.json"] : [])
      files += companions.map { ["path": $0, "size": 1, "sha256": String(repeating: "0", count: 64)] }
      raw["files"] = files
      let manifest = try JSONDecoder().decode(InstalledManifest.self,
        from: JSONSerialization.data(withJSONObject: raw))
      XCTAssertNoThrow(try ModelPackages.package(for: .deepSeekV41_1017).validate(manifest))
      XCTAssertThrowsError(try ModelPackages.package(for: .deepSeekV41).validate(manifest))
    }
    XCTAssertThrowsError(try DeepSeekV41Contract.validate(main, kind: .deepSeekV41_1017))
    tensors["mtp.2.ffn.experts.127.w1.weight"] = nil
    let brokenIndex = CheckpointIndex(totalSize: 0, weightMap: Dictionary(uniqueKeysWithValues: tensors.keys.map { ($0, "fixture.safetensors") }))
    XCTAssertThrowsError(try DeepSeekV41Planner.makePlan(index: brokenIndex, tensors: tensors, includeDSpark: true))
  }
}
