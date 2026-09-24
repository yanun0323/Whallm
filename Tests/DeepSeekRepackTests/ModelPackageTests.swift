import Foundation
import XCTest

@testable import DeepSeekRepack

final class ModelPackageTests: XCTestCase {
  func testCatalogMatchesPinnedInstallationContracts() throws {
    let expected = [
      (ModelKind.deepSeekV4, ModelContract.modelID, ModelContract.revision),
      (.deepSeekV41, DeepSeekV41Contract.modelID, DeepSeekV41Contract.revision),
      (.qwen3_8FlashNext, QwenContract.modelID, QwenContract.revision),
      (.mimoV26FlashRL, MiMoContract.modelID, MiMoContract.revision),
    ]
    XCTAssertEqual(ModelPackages.descriptors.count, expected.count)
    for (kind, modelID, revision) in expected {
      XCTAssertEqual(kind.descriptor.checkpointModelID, modelID)
      XCTAssertEqual(kind.descriptor.checkpointRevision, revision)
      XCTAssertFalse(ModelPackages.package(for: kind).companions.isEmpty)
      let encoded = try JSONEncoder().encode(kind)
      XCTAssertEqual(try JSONDecoder().decode(String.self, from: encoded), kind.rawValue)
      XCTAssertEqual(try JSONDecoder().decode(ModelKind.self, from: encoded), kind)
    }
    XCTAssertNil(ModelKind(rawValue: "unregistered-model"))
    XCTAssertThrowsError(try JSONDecoder().decode(ModelKind.self, from: Data("\"unregistered-model\"".utf8)))
  }

  func testCatalogAllowsSharedSchemaButRejectsDuplicateIdentityAndUnsafePaths() throws {
    let url = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
      .deletingLastPathComponent().appending(path: "Sources/DeepSeekRepack/Resources/ModelPackages.json")
    let original = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    var models = try XCTUnwrap(original["models"] as? [[String: Any]])
    var fourth = models[0]
    fourth["kind"] = "fixture-fourth"
    fourth["apiModelID"] = "fixture-fourth"
    models.append(fourth)
    var catalog = original
    catalog["models"] = models
    XCTAssertEqual(try ModelPackages.decodeDescriptors(JSONSerialization.data(withJSONObject: catalog)).count, models.count)
    for (field, value) in [("kind", "deepseek-v4"), ("directoryName", "../outside")] {
      var invalid = models
      invalid[invalid.count - 1][field] = value
      catalog["models"] = invalid
      XCTAssertThrowsError(try ModelPackages.decodeDescriptors(JSONSerialization.data(withJSONObject: catalog)))
    }
  }
}
