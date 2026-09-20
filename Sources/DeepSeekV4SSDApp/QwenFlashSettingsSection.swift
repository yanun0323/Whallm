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
    key: WritableKeyPath<ModelAdvancedSettings, Int?>, identifier: String
  ) -> some View {
    let value = Binding<Int>(
      get: { settings[keyPath: key] ?? 0 },
      set: { settings[keyPath: key] = min(512, max(0, $0)) }
    )
    return SettingRow(title, hint: text(hint), language: language) {
      HStack(spacing: 8) {
        TextField(text(title), value: value, format: .number.grouping(.never))
          .appInput(width: 100)
          .accessibilityLabel(text(title))
          .accessibilityHint(text(hint))
          .accessibilityIdentifier(identifier)
        Stepper(text(title), value: value, in: 0...512)
          .labelsHidden()
          .accessibilityLabel(text(title))
          .accessibilityValue(String(value.wrappedValue))
          .accessibilityIdentifier("\(identifier)-stepper")
      }
    }
  }
}
