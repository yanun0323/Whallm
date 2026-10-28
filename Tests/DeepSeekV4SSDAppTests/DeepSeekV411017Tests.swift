import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

final class DeepSeekV411017Tests: XCTestCase {
  func testBothCheckpointsUseTheSameMemoryRules() throws {
    for kind in [ModelKind.deepSeekV41, .deepSeekV41_1017] {
      XCTAssertEqual(ExpertMemory.blobBytes(for: kind), 18_800_640)
      XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: kind, manifest: nil), 40 * 384)
      let settings = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(settings.expertCacheAutomatic, true)
      XCTAssertNoThrow(try settings.validate(for: kind))
    }
  }

  @MainActor
  func testBothCheckpointsHaveIndependentSettingsAliasesAndServerEntries() throws {
    let suite = "DeepSeek1017.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    let library = ModelLibrary(defaults: store)
    var settings: [ModelKind: ModelAdvancedSettings] = [:]
    var models: [InstalledModelInfo] = []
    for (index, kind) in [ModelKind.deepSeekV41, .deepSeekV41_1017].enumerated() {
      var saved = ModelAdvancedSettings.defaults(for: kind)
      saved.defaultTemperature = index == 0 ? 0.3 : 0.7
      store.set(try JSONEncoder().encode(saved), forKey: "modelAdvancedSettings.\(kind.rawValue)")
      settings[kind] = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store)
      try library.saveAlias("alias-" + kind.rawValue, for: kind)
      models.append(InstalledModelInfo(
        url: URL(fileURLWithPath: "/fixture/" + kind.descriptor.directoryName),
        size: 0, quickIssues: [], hasMTP: false, hasDSpark: false, modelKind: kind,
        modelID: kind.descriptor.checkpointModelID))
    }
    XCTAssertEqual(settings[.deepSeekV41]?.defaultTemperature, 0.3)
    XCTAssertEqual(settings[.deepSeekV41_1017]?.defaultTemperature, 0.7)
    let catalog = try ModelLibrary.makeServerCatalog(models: models, aliases: library.aliases,
      settings: settings, powerSavingLimitGBps: nil)
    XCTAssertEqual(Set(catalog.models.map(\.id)), ["deepseek-v4.1-flash", "deepseek-v4.1-flash-1017"])
    XCTAssertEqual(Set(catalog.models.map(\.path)).count, 2)
    for entry in catalog.models {
      XCTAssertTrue(entry.runtime.nativeAutomaticExpertCache)
      XCTAssertFalse(entry.runtime.v41PackedKV)
      XCTAssertFalse(entry.runtime.v41PackedIndex)
      XCTAssertEqual(entry.alias, "alias-" + entry.modelKind)
    }
  }
}
