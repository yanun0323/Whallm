import SwiftUI

/// Continuous native track: `Slider(step:)` adds automatic tick dots on macOS.
/// Quantize in the binding instead, retaining explicit 0.1 GiB keyboard steps.
struct CacheBudgetSlider: View {
  @Binding var value: Double
  let range: ClosedRange<Double>
  var step = 0.1
  @Environment(\.isEnabled) private var isEnabled

  static func adjusted(_ value: Double, by delta: Double, in range: ClosedRange<Double>,
                       step: Double = 0.1) -> Double {
    ExpertMemory.snapped(value + delta, step: step, in: range)
  }

  var body: some View {
    Slider(value: Binding(
      get: { value },
      set: { value = ExpertMemory.snapped($0, step: step, in: range) }
    ), in: range)
    .onKeyPress(.leftArrow) { adjust(-step) }
    .onKeyPress(.rightArrow) { adjust(step) }
    .onKeyPress(.downArrow) { adjust(-step) }
    .onKeyPress(.upArrow) { adjust(step) }
    .accessibilityAdjustableAction { direction in
      switch direction {
      case .increment: _ = adjust(step)
      case .decrement: _ = adjust(-step)
      @unknown default: break
      }
    }
  }

  private func adjust(_ delta: Double) -> KeyPress.Result {
    guard isEnabled else { return .ignored }
    value = Self.adjusted(value, by: delta, in: range, step: step)
    return .handled
  }
}
