import AppKit
import DeepSeekRepack
import Foundation
import SwiftUI
import XCTest
@testable import DeepSeekV4SSDApp

final class SortedExpertSettingsTests: XCTestCase {
  private let qwenKinds = [ModelKind.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext]

  func testDefaultLegacyPersistenceAndNonQwenNeutrality() throws {
    let suite = "SortedExpertSettingsTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { store.removePersistentDomain(forName: suite) }
    for kind in qwenKinds {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(settings.qwenSortedExpertPrefill, false)
      XCTAssertFalse(settings.effectiveQwenSortedExpertPrefill)
      var legacy = try object(settings)
      legacy.removeValue(forKey: "qwenSortedExpertPrefill")
      let old = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONSerialization.data(withJSONObject: legacy)).normalized(for: kind)
      XCTAssertEqual(old.qwenSortedExpertPrefill, false)
      settings.qwenSortedExpertPrefill = true
      settings.save(for: kind, defaults: store)
      XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: kind, defaults: store).qwenSortedExpertPrefill, true)
      var reset = ModelSettingsResetConfirmation()
      reset.begin(for: kind, locked: false); reset.advance()
      XCTAssertEqual(reset.confirm(for: kind, locked: false)?.qwenSortedExpertPrefill, false)
    }
    var enabled = ModelAdvancedSettings.defaults(for: .swift1_5Qwen3_8FlashNext)
    enabled.qwenSortedExpertPrefill = true
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .mimoV26FlashRL] {
      XCTAssertEqual(ModelAdvancedSettings.defaults(for: kind).qwenSortedExpertPrefill, false)
      XCTAssertEqual(enabled.normalized(for: kind).qwenSortedExpertPrefill, false)
    }
  }

  @MainActor
  func testCatalogSendsOnlyAnActiveChoiceAndKeepsTheSavedOne() throws {
    let swift = ModelKind.swift1_5Qwen3_8FlashNext
    for (name, kind, saved, grouped, waves, mtp, expected) in [
      ("swift-sorted-off", swift, false, true, 0, false, false),
      ("swift-sorted-on", swift, true, true, 0, false, true),
      ("qwen-sorted-on", ModelKind.qwen3_8FlashNext, true, true, 0, false, true),
      ("swift-sorted-mtp", swift, true, true, 0, true, true),
      ("swift-sorted-no-grouping", swift, true, false, 0, false, false),
      ("swift-sorted-waves", swift, true, true, 8, false, false),
    ] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.qwenSortedExpertPrefill = saved
      settings.qwenGroupedExperts = grouped
      settings.qwenExpertWaveSlots = waves
      settings.mtpEnabled = mtp
      let original = settings
      let result = try catalog(settings, kind: kind)
      let runtime = try object(result.models[0].runtime)
      XCTAssertEqual(runtime["qwen_sorted_expert_prefill"] as? Bool, expected, name)
      if expected {
        XCTAssertEqual(runtime["qwen_grouped_experts"] as? Bool, true, name)
      }
      XCTAssertEqual(settings, original, "Catalog generation must not erase an inactive saved choice")
      XCTAssertEqual(settings.normalized(for: kind).qwenSortedExpertPrefill, saved)
      if let path = ProcessInfo.processInfo.environment["WHALLM_QWEN_UI_ARTIFACTS"] {
        let folder = URL(fileURLWithPath: path).appendingPathComponent("catalogs")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try result.encoded().write(to: folder.appendingPathComponent("\(name).json"))
      }
    }
  }

  @MainActor
  func testMaskedQSAPrefillDefaultsOffAndReachesOnlyQwen() throws {
    for kind in qwenKinds {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(settings.qwenQSAMaskedPrefill, false)
      XCTAssertEqual(try object(catalog(settings, kind: kind).models[0].runtime)["qwen_qsa_masked_prefill"] as? Bool, false)
      settings.qwenQSAMaskedPrefill = true
      XCTAssertEqual(try object(catalog(settings, kind: kind).models[0].runtime)["qwen_qsa_masked_prefill"] as? Bool, true)
      var legacy = try object(settings)
      legacy.removeValue(forKey: "qwenQSAMaskedPrefill")
      let old = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONSerialization.data(withJSONObject: legacy)).normalized(for: kind)
      XCTAssertEqual(old.qwenQSAMaskedPrefill, false)
    }
    var enabled = ModelAdvancedSettings.defaults(for: .swift1_5Qwen3_8FlashNext)
    enabled.qwenQSAMaskedPrefill = true
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .mimoV26FlashRL] {
      XCTAssertEqual(enabled.normalized(for: kind).qwenQSAMaskedPrefill, false)
    }
  }

  @MainActor
  func testLegacyCatalogEmitsFalse() throws {
    let kind = ModelKind.qwen3_8FlashNext
    var raw = try object(catalog(.defaults(for: kind), kind: kind).models[0].runtime)
    raw.removeValue(forKey: "qwen_sorted_expert_prefill")
    let old = try JSONDecoder().decode(ModelCatalog.Entry.Runtime.self,
      from: JSONSerialization.data(withJSONObject: raw))
    XCTAssertEqual(try object(old)["qwen_sorted_expert_prefill"] as? Bool, false)
  }

  @MainActor
  func testRendersThreeLanguagesAndAllControlStates() throws {
    _ = NSApplication.shared
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      for state in ["off", "on", "locked", "inactive"] {
        var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
        settings.qwenSortedExpertPrefill = state != "off"
        settings.qwenGroupedExperts = state != "inactive"
        let host = NSHostingView(rootView: QwenFlashSettingsSection(settings: .constant(settings),
          modelKind: .qwen3_8FlashNext, settingsLocked: state == "locked", language: language)
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
          try png.write(to: folder.appendingPathComponent("sorted-experts-\(language.rawValue)-\(state).png"))
        }
      }
    }
  }

  private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
    try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
  }

  @MainActor
  private func catalog(_ settings: ModelAdvancedSettings, kind: ModelKind) throws -> ModelCatalog {
    let model = InstalledModelInfo(url: URL(fileURLWithPath: "/tmp/sorted-experts-ui-\(kind.rawValue)"),
      size: 1, quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: kind,
      modelID: kind.descriptor.checkpointModelID)
    return try ModelLibrary.makeServerCatalog(models: [model], aliases: [:], settings: [kind: settings],
      powerSavingLimitGBps: nil)
  }
}
