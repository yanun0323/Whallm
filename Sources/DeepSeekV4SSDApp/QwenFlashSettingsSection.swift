import DeepSeekRepack
import SwiftUI

struct QwenFlashSettingsSection: View {
  @Binding var settings: ModelAdvancedSettings
  let modelKind: ModelKind
  let settingsLocked: Bool
  let language: AppLanguage

  private func text(_ key: String) -> String { L10n.string(key, language: language) }

  var body: some View {
    if modelKind == .qwen3_8FlashNext {
      VStack(alignment: .leading, spacing: 12) {
        Text(text(QwenFlashCopy.title))
          .font(.headline)
          .accessibilityAddTraits(.isHeader)
        Text(text(QwenFlashCopy.warning))
          .font(.callout)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)
        Text(text(settingsLocked ? QwenFlashCopy.locked : QwenFlashCopy.lifecycle))
          .font(.caption)
          .foregroundStyle(.secondary)
          .fixedSize(horizontal: false, vertical: true)

        VStack(spacing: 0) {
          numberRow(QwenFlashCopy.waves, hint: QwenFlashCopy.wavesHint,
            key: \.qwenExpertWaveSlots, identifier: "qwen-flash-waves")
          if settings.qwenFlashWavesEnabled {
            Text(text(QwenFlashCopy.wavesConflict))
              .font(.caption)
              .foregroundStyle(.secondary)
              .fixedSize(horizontal: false, vertical: true)
              .frame(maxWidth: .infinity, alignment: .leading)
              .padding(.bottom, 8)
          }
          Divider()
          SettingRow(QwenFlashCopy.backend, hint: text(QwenFlashCopy.backendHint), language: language) {
            Picker(text(QwenFlashCopy.backend), selection: Binding(
              get: { settings.qwenNgramIO ?? "mmap" },
              set: { settings.qwenNgramIO = $0 }
            )) {
              Text(text(QwenFlashCopy.mmap)).tag("mmap")
              Text(text(QwenFlashCopy.pread)).tag("pread")
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 220)
            .accessibilityLabel(text(QwenFlashCopy.backend))
            .accessibilityHint(text(QwenFlashCopy.backendHint))
            .accessibilityIdentifier("qwen-flash-ngram-backend")
          }
          Divider()
          numberRow(QwenFlashCopy.cache, hint: QwenFlashCopy.cacheHint,
            key: \.qwenNgramCacheMiB, identifier: "qwen-flash-ngram-cache")
            .disabled(settings.qwenNgramIO != "pread")
          Divider()
          SettingRow(QwenFlashCopy.sdpa, hint: text(QwenFlashCopy.sdpaHint), language: language) {
            Toggle(text(QwenFlashCopy.sdpa), isOn: Binding(
              get: { settings.qwenSparseSDPA ?? false },
              set: { settings.qwenSparseSDPA = $0 }
            ))
            .labelsHidden()
            .accessibilityLabel(text(QwenFlashCopy.sdpa))
            .accessibilityHint(text(QwenFlashCopy.sdpaHint))
            .accessibilityIdentifier("qwen-flash-sparse-sdpa")
          }
          Divider()
          numberRow(QwenFlashCopy.queryChunk, hint: QwenFlashCopy.queryChunkHint,
            key: \.qwenQSAQueryChunk, identifier: "qwen-qsa-query-chunk",
            limits: 1...128, defaultValue: 4)
          Divider()
          SettingRow(QwenFlashCopy.indexed, hint: text(QwenFlashCopy.indexedHint), language: language) {
            Toggle(text(QwenFlashCopy.indexed), isOn: Binding(
              get: { settings.qwenSparseSDPA == true && settings.qwenQSAIndexed == true },
              set: { settings.qwenQSAIndexed = $0 }
            ))
            .labelsHidden()
            .disabled(settings.qwenSparseSDPA != true)
            .accessibilityLabel(text(QwenFlashCopy.indexed))
            .accessibilityHint(text(QwenFlashCopy.indexedHint))
            .accessibilityIdentifier("qwen-qsa-indexed")
          }
        }
        .appCard()
        .disabled(settingsLocked)

        if let validationError {
          Text(validationError)
            .font(.caption)
            .foregroundStyle(.red)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
      .padding(.top, 12)
      .accessibilityIdentifier("qwen-flash-experiments")
    }
  }

  private var validationError: String? {
    do {
      try settings.validateQwenFlashSettings()
      return nil
    } catch {
      return error.localizedDescription
    }
  }

  private func numberRow(_ title: String, hint: String,
    key: WritableKeyPath<ModelAdvancedSettings, Int?>, identifier: String,
    limits: ClosedRange<Int> = 0...512, defaultValue: Int = 0
  ) -> some View {
    let value = Binding<Int>(
      get: { settings[keyPath: key] ?? defaultValue },
      set: { settings[keyPath: key] = min(limits.upperBound, max(limits.lowerBound, $0)) }
    )
    return SettingRow(title, hint: text(hint), language: language) {
      HStack(spacing: 8) {
        TextField(text(title), value: value, format: .number.grouping(.never))
          .appInput(width: 100)
          .accessibilityLabel(text(title))
          .accessibilityHint(text(hint))
          .accessibilityIdentifier(identifier)
        Stepper(text(title), value: value, in: limits)
          .labelsHidden()
          .accessibilityLabel(text(title))
          .accessibilityValue(String(value.wrappedValue))
          .accessibilityIdentifier("\(identifier)-stepper")
      }
    }
  }
}
