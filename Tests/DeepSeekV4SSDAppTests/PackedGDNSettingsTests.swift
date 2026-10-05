import AppKit
import DeepSeekRepack
import Foundation
import SwiftUI
import XCTest
@testable import DeepSeekV4SSDApp

final class PackedGDNSettingsTests: XCTestCase {
  private let swift = ModelKind.swift1_5Qwen3_8FlashNext

  func testDefaultLegacyPersistenceAndModelIsolation() throws {
    let suite = "PackedGDNSettingsTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    var settings = ModelAdvancedSettings.defaults(for: swift)
    XCTAssertEqual(settings.qwenPackedGDNPrefill, false)
    var legacy = try object(settings)
    legacy.removeValue(forKey: "qwenPackedGDNPrefill")
    let old = try JSONDecoder().decode(ModelAdvancedSettings.self,
      from: JSONSerialization.data(withJSONObject: legacy)).normalized(for: swift)
    XCTAssertEqual(old.qwenPackedGDNPrefill, false)
    settings.qwenPackedGDNPrefill = true
    settings.save(for: swift, defaults: store)
    XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: swift, defaults: store).qwenPackedGDNPrefill, true)
    for kind in [ModelKind.qwen3_8FlashNext, .deepSeekV4, .deepSeekV41, .mimoV26FlashRL] {
      XCTAssertEqual(ModelAdvancedSettings.defaults(for: kind).qwenPackedGDNPrefill, false)
      XCTAssertEqual(settings.normalized(for: kind).qwenPackedGDNPrefill, false)
      XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store).qwenPackedGDNPrefill, false)
    }
    var reset = ModelSettingsResetConfirmation()
    reset.begin(for: swift, locked: false); reset.advance()
    XCTAssertEqual(reset.confirm(for: swift, locked: false)?.qwenPackedGDNPrefill, false)
  }

  @MainActor
  func testCatalogAndMTPKeepSavedChoiceWithoutActivatingConflict() throws {
    for (name, kind, saved, mtp, expected) in [
      ("swift-packed-off", swift, false, false, false),
      ("swift-packed-on", swift, true, false, true),
      ("swift-packed-mtp", swift, true, true, false),
      ("qwen-packed-inactive", ModelKind.qwen3_8FlashNext, true, false, false),
    ] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.qwenPackedGDNPrefill = saved; settings.mtpEnabled = mtp
      let original = settings
      let result = try catalog(settings, kind: kind)
      let runtime = try object(result.models[0].runtime)
      XCTAssertEqual(runtime["qwen_packed_gdn_prefill"] as? Bool, expected)
      XCTAssertEqual(settings, original, "Catalog generation must not erase an inactive saved choice")
      if kind == swift {
        XCTAssertEqual(settings.normalized(for: swift).qwenPackedGDNPrefill, saved)
        settings.mtpEnabled = false
        XCTAssertEqual(settings.effectiveQwenPackedGDNPrefill, saved)
      }
      if let path = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
        let folder = URL(fileURLWithPath: path).appendingPathComponent("catalogs")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try result.encoded().write(to: folder.appendingPathComponent("\(name).json"))
      }
    }
  }

  @MainActor
  func testLegacyCatalogEmitsFalseAndLoadedModelsStayLocked() throws {
    var raw = try object(catalog(.defaults(for: swift), kind: swift).models[0].runtime)
    raw.removeValue(forKey: "qwen_packed_gdn_prefill")
    let old = try JSONDecoder().decode(ModelCatalog.Entry.Runtime.self,
      from: JSONSerialization.data(withJSONObject: raw))
    XCTAssertEqual(try object(old)["qwen_packed_gdn_prefill"] as? Bool, false)
    let id = swift.apiModelID
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: id, loadingModel: nil, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: id, modelActionID: nil))
    XCTAssertTrue(modelAdvancedSettingsAreLocked(modelID: id, loadedModel: nil, loadingModel: nil, modelActionID: id))
  }

  @MainActor
  func testSwiftExperimentRendersThreeLanguagesAndAllControlStates() throws {
    _ = NSApplication.shared
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for state in ["off", "on", "locked", "mtp"] {
        var settings = ModelAdvancedSettings.defaults(for: swift)
        settings.qwenPackedGDNPrefill = state != "off"
        settings.mtpEnabled = state == "mtp"
        let host = NSHostingView(rootView: QwenFlashSettingsSection(settings: .constant(settings),
          modelKind: swift, settingsLocked: state == "locked", language: language)
          .padding(24).frame(width: 760)
          .background(Color(nsColor: .windowBackgroundColor)).preferredColorScheme(.dark))
        let size = host.fittingSize
        XCTAssertGreaterThan(size.height, 300)
        XCTAssertLessThan(size.height, 1_900, "Help text must fit the scrollable settings card")
        host.frame = NSRect(origin: .zero, size: size)
        let window = NSWindow(contentRect: NSRect(x: -10_000, y: -10_000, width: size.width, height: size.height),
          styleMask: [.borderless], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = host
        defer { window.close() }
        window.orderBack(nil); RunLoop.main.run(until: Date().addingTimeInterval(0.05))
        host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        XCTAssertGreaterThan(png.count, 1_000)
        if let path = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
          let folder = URL(fileURLWithPath: path).appendingPathComponent("screenshots")
          try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
          try png.write(to: folder.appendingPathComponent("packed-gdn-\(language.rawValue)-\(state).png"))
        }
      }
    }
  }

  private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
  }

  @MainActor
  private func catalog(_ settings: ModelAdvancedSettings, kind: ModelKind) throws -> ModelCatalog {
    let model = InstalledModelInfo(url: URL(fileURLWithPath: "/tmp/packed-gdn-ui-\(kind.rawValue)"),
      size: 1, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: kind,
      modelID: kind.descriptor.checkpointModelID)
    return try ModelLibrary.makeServerCatalog(models: [model], aliases: [:], settings: [kind: settings],
      powerSavingLimitGBps: nil)
  }
}
