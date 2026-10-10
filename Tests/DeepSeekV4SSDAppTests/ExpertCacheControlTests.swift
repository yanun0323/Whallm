import Foundation
import AppKit
import SwiftUI
import DeepSeekRepack
import XCTest
@testable import DeepSeekV4SSDApp

final class ExpertCacheControlTests: XCTestCase {
  @MainActor
  func testCacheSliderRendersWithoutFooterIndicators() throws {
    _ = NSApplication.shared
    let host = NSHostingView(rootView: CacheBudgetSlider(value: .constant(7.5), range: 0.1...64)
      .padding(24).frame(width: 520, height: 90))
    let window = NSWindow(contentRect: NSRect(x: -10000, y: -10000, width: 520, height: 90),
                          styleMask: [.borderless], backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    defer { window.close() }
    host.layoutSubtreeIfNeeded()
    RunLoop.main.run(until: Date().addingTimeInterval(0.05))
    // Modern SwiftUI draws its slider directly; there need not be an NSSlider
    // subview. Check the rendered strip below the track rather than AppKit internals.
    let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
    host.cacheDisplay(in: host.bounds, to: bitmap)
    var maximumFooterAlpha: CGFloat = 0
    for y in Int(Double(bitmap.pixelsHigh) * 0.70)..<Int(Double(bitmap.pixelsHigh) * 0.85) {
      for x in 0..<bitmap.pixelsWide {
        maximumFooterAlpha = max(maximumFooterAlpha, try XCTUnwrap(bitmap.colorAt(x: x, y: y)).alphaComponent)
      }
    }
    // The system thumb shadow can leave a one-byte alpha tail, not a tick.
    XCTAssertLessThanOrEqual(maximumFooterAlpha, 1.0 / 255.0 + 0.000001)
    if let path = ProcessInfo.processInfo.environment["WHALLM_CAPTURE_CACHE_SLIDER"] {
      try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
        .write(to: URL(fileURLWithPath: path))
    }
  }

  @MainActor
  func testTicklessSliderAdjustmentKeepsOneDecimalAndBounds() {
    let range = 0.1...64.0
    XCTAssertEqual(CacheBudgetSlider.adjusted(7.5, by: 0.1, in: range), 7.6)
    XCTAssertEqual(CacheBudgetSlider.adjusted(7.5, by: -0.1, in: range), 7.4)
    XCTAssertEqual(CacheBudgetSlider.adjusted(64, by: 0.1, in: range), 64)
    XCTAssertEqual(CacheBudgetSlider.adjusted(0.1, by: -0.1, in: range), 0.1)
  }

  func testEveryControlUsesItsOwnBudgetAndClearsItWhenSlotsAreEdited() throws {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    for control in ExpertCacheControl.allCases {
      var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
      settings[keyPath: control.budgetKey] = 1.2
      let before = settings
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob),
        try ExpertMemory.capacity(gib: 1.2, blobBytes: blob, minimum: 1))
      XCTAssertEqual(settings, before) // Switching presentation only reads the value.
      control.setSlots(123, in: &settings)
      XCTAssertNil(settings[keyPath: control.budgetKey])
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob), 123)
      XCTAssertEqual(control.legacySlots(in: settings), 123)
      for other in ExpertCacheControl.allCases where other != control {
        XCTAssertEqual(settings[keyPath: other.budgetKey], before[keyPath: other.budgetKey])
        XCTAssertEqual(other.legacySlots(in: settings), other.legacySlots(in: before))
      }
      let decoded = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONEncoder().encode(settings))
      XCTAssertEqual(control.slots(in: decoded, blobBytes: blob), 123)
      XCTAssertNil(decoded[keyPath: control.budgetKey])
    }
  }

  func testEverySliderClampsRoundsAndReplacesOldSlotCapacity() throws {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    let range = ExpertMemory.sliderRange(physicalMemory: 64 * 1_073_741_824)
    for control in ExpertCacheControl.allCases {
      var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
      control.setSlots(-1, in: &settings)
      XCTAssertThrowsError(try settings.validate(for: .qwen3_8FlashNext))
      control.setGiB(1.26, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 1.3)
      let expected = try ExpertMemory.capacity(gib: 1.3, blobBytes: blob, minimum: 1)
      XCTAssertEqual(control.slots(in: settings, blobBytes: blob), expected)
      XCTAssertEqual(control.legacySlots(in: settings), expected)
      XCTAssertNoThrow(try settings.validate(for: .qwen3_8FlashNext))
      control.setGiB(100, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 64)
      control.setGiB(0, in: &settings, blobBytes: blob, range: range)
      XCTAssertEqual(settings[keyPath: control.budgetKey], 0.1)
    }
  }

  func testPercentTotalsAndDefaultsPerModel() {
    XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: .swift1_5Qwen3_8FlashNext, manifest: nil), 24_576)
    XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: .deepSeekV4, manifest: nil), 11_008)
    XCTAssertEqual(ExpertMemory.totalExperts(for: .expert, kind: .deepSeekV41, manifest: nil), 15_360)
    XCTAssertEqual(ExpertMemory.totalExperts(for: .mtp, kind: .qwen3_8FlashNext, manifest: nil), 512)
    XCTAssertEqual(ExpertMemory.totalExperts(for: .dspark, kind: .deepSeekV4, manifest: nil), 768)
    // Both Qwen kinds default to 8.5% of 24576 experts and to every MTP expert.
    for kind in [ModelKind.qwen3_8FlashNext, .swift1_5Qwen3_8FlashNext] {
      let blob = ExpertMemory.blobBytes(for: kind)
      let defaults = ModelAdvancedSettings.defaults(for: kind)
      let slots = ExpertCacheControl.expert.slots(in: defaults, blobBytes: blob)
      XCTAssertEqual(slots, 2_089)
      XCTAssertEqual(defaults.slots, 2_089)
      XCTAssertEqual(ExpertMemory.percent(slots: slots, totalExperts: 24_576), 8.5, accuracy: 0.01)
      XCTAssertEqual(ExpertCacheControl.mtp.slots(in: defaults, blobBytes: blob), 512)
      XCTAssertEqual(defaults.mtpSlots, 512)
      XCTAssertNoThrow(try defaults.validate(for: kind))
    }
  }

  func testPercentRangeKeepsTheMinimumAndThisMacsMemory() {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    let gib: UInt64 = 1_073_741_824
    XCTAssertEqual(ExpertMemory.percentRange(totalExperts: 24_576, blobBytes: blob,
      physicalMemory: 128 * gib, minimum: 10), 0.5...100)
    let small = ExpertMemory.percentRange(totalExperts: 24_576, blobBytes: blob, physicalMemory: 16 * gib, minimum: 10)
    XCTAssertEqual(small.upperBound, 26.5) // 6579 slots fit in 16 GiB: 26.77% rounded down to 0.5
    XCTAssertEqual(ExpertMemory.percentRange(totalExperts: 512, blobBytes: blob,
      physicalMemory: 64 * gib, minimum: 10).lowerBound, 2) // 10 of 512 is 1.95%
    XCTAssertEqual(CacheBudgetSlider.adjusted(12.5, by: 0.5, in: 0.5...100, step: 0.5), 13)
    XCTAssertEqual(CacheBudgetSlider.adjusted(100, by: 0.5, in: 0.5...100, step: 0.5), 100)
  }

  func testPercentStoresAGiBBudgetThatResolvesToTheSameSlots() throws {
    let blob = ExpertMemory.blobBytes(for: .qwen3_8FlashNext)
    let range = 0.5...100.0
    for control in ExpertCacheControl.allCases {
      for (percent, total) in [(12.5, 24_576), (0.5, 24_576), (100, 24_576), (8, 512), (33.3, 15_360)] {
        var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
        control.setPercent(percent, in: &settings, blobBytes: blob, totalExperts: total, range: range)
        let snapped = ExpertMemory.snapped(percent, step: ExpertMemory.percentStep, in: range)
        let expected = ExpertMemory.slots(percent: snapped, totalExperts: total)
        XCTAssertEqual(control.legacySlots(in: settings), expected)
        let budget = try XCTUnwrap(settings[keyPath: control.budgetKey])
        // The catalog floors this budget to whole experts, as the runtime does.
        XCTAssertEqual(try ExpertMemory.bytes(gib: budget) / blob, UInt64(expected))
        XCTAssertEqual(control.slots(in: settings, blobBytes: blob), expected)
        let decoded = try JSONDecoder().decode(ModelAdvancedSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(control.slots(in: decoded, blobBytes: blob), expected)
      }
    }
  }

  func testPreferenceIsOffByDefaultPersistsAndDoesNotChangeModelSettings() throws {
    let name = "ExpertCacheControlTests.\(UUID().uuidString)"
    let store = try XCTUnwrap(UserDefaults(suiteName: name))
    defer { store.removePersistentDomain(forName: name) }
    let settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    settings.save(for: .qwen3_8FlashNext, defaults: store)
    let saved = ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store)
    XCTAssertFalse(store.bool(forKey: ExpertCacheControl.slotsPreferenceKey))
    store.set(true, forKey: ExpertCacheControl.slotsPreferenceKey)
    XCTAssertTrue(try XCTUnwrap(UserDefaults(suiteName: name)).bool(forKey: ExpertCacheControl.slotsPreferenceKey))
    XCTAssertEqual(ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store), saved)
    store.set(false, forKey: ExpertCacheControl.slotsPreferenceKey)
    XCTAssertEqual(ModelAdvancedSettings.load(for: .qwen3_8FlashNext, defaults: store), saved)
  }

  func testPreferenceCopyIsLocalized() {
    for language in [AppLanguage.traditionalChinese, .simplifiedChinese] {
      for key in ["Edit expert caches in slots",
                  "Use integer slot inputs instead of percentage sliders for expert, MTP, and DSpark caches. Switching does not change saved capacity.",
                  "Enter the number of experts to retain. Applies on next model load."] {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }
}
