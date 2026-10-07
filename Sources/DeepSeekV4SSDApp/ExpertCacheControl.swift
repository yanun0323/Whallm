import Foundation

enum ExpertCacheControl: CaseIterable {
  static let slotsPreferenceKey = "editExpertCachesInSlots"
  case expert, mtp, dspark

  var title: String {
    switch self {
    case .expert: "Expert cache GiB"
    case .mtp: "MTP expert cache GiB"
    case .dspark: "DSpark expert cache GiB"
    }
  }

  var percentTitle: String {
    switch self {
    case .expert: "Experts in memory"
    case .mtp: "MTP experts in memory"
    case .dspark: "DSpark experts in memory"
    }
  }

  var slotsTitle: String {
    switch self {
    case .expert: "Slots"
    case .mtp: "MTP slots"
    case .dspark: "DSpark slots"
    }
  }

  var budgetKey: WritableKeyPath<ModelAdvancedSettings, Double?> {
    switch self {
    case .expert: \.expertCacheGiB
    case .mtp: \.mtpCacheGiB
    case .dspark: \.dsparkCacheGiB
    }
  }

  func legacySlots(in settings: ModelAdvancedSettings) -> Int {
    switch self {
    case .expert: settings.slots
    case .mtp: settings.mtpSlots ?? 32
    case .dspark: settings.dsparkSlots
    }
  }

  func slots(in settings: ModelAdvancedSettings, blobBytes: UInt64) -> Int {
    guard let budget = settings[keyPath: budgetKey] else { return legacySlots(in: settings) }
    return (try? ExpertMemory.capacity(gib: budget, blobBytes: blobBytes, minimum: 1)) ?? 0
  }

  func setSlots(_ count: Int, in settings: inout ModelAdvancedSettings) {
    switch self {
    case .expert: settings.slots = count
    case .mtp: settings.mtpSlots = count
    case .dspark: settings.dsparkSlots = count
    }
    // Slot counts must not be overridden by an older GiB budget.
    settings[keyPath: budgetKey] = nil
  }

  /// Store the slot count and a GiB budget that resolves to it; the backend keeps GiB.
  func setPercent(_ value: Double, in settings: inout ModelAdvancedSettings,
                  blobBytes: UInt64, totalExperts: Int, range: ClosedRange<Double>) {
    let percent = ExpertMemory.snapped(value, step: ExpertMemory.percentStep, in: range)
    let count = ExpertMemory.slots(percent: percent, totalExperts: totalExperts)
    setSlots(count, in: &settings)
    settings[keyPath: budgetKey] = ExpertMemory.exactGiB(slots: count, blobBytes: blobBytes)
  }

  func setGiB(_ value: Double, in settings: inout ModelAdvancedSettings,
              blobBytes: UInt64, range: ClosedRange<Double>) {
    let budget = ExpertMemory.sliderValue(value, in: range)
    let count = (try? ExpertMemory.capacity(gib: budget, blobBytes: blobBytes, minimum: 1)) ?? 0
    setSlots(count, in: &settings)
    settings[keyPath: budgetKey] = budget
  }
}
