import SwiftUI

/// Composer-only controls; transcript attachments reuse ChatAttachmentPreview.
/// Neither document nor audio content is rendered or played inside the App.
struct ChatAttachmentsView: View {
  let attachments: [ChatAttachment]
  let history: [ChatAttachment]
  let allowedKinds: Set<ChatAttachmentKind>
  var hasSelectedModel = true
  let language: AppLanguage
  let disabled: Bool
  let choose: (Set<ChatAttachmentKind>) -> Void
  let addFiles: ([URL]) -> Bool
  let remove: (ChatAttachment) -> Void
  @State private var showingLimits = false
  @State private var dropTargeted = false
  @FocusState private var addFocused: Bool
  @FocusState private var removeFocused: UUID?

  private var all: [ChatAttachment] { history + attachments }
  private var remaining: Int { max(0, 8 - all.count) }
  private var kinds: [ChatAttachmentKind] { ChatAttachmentKind.allCases.filter(allowedKinds.contains) }
  private func localized(_ key: String) -> String { L10n.string(key, language: language) }

  var body: some View {
    VStack(alignment: .leading, spacing: 8) {
      if allowedKinds.isEmpty {
        Label(localized(hasSelectedModel ? "This model accepts text only." : "Start the server and select a model to see supported inputs."), systemImage: "text.alignleft")
          .font(.callout).foregroundStyle(.secondary)
      } else {
        ViewThatFits(in: .horizontal) {
          HStack(spacing: 12) { addMenu; capabilityLabels; Spacer(minLength: 0); limitsButton }
          VStack(alignment: .leading, spacing: 8) {
            HStack { addMenu; Spacer(); limitsButton }
            capabilityLabels
          }
        }
        Text(localized(dropTargeted && !disabled ? "Release to attach files" : "Drop local files here, or choose Attach files."))
          .font(.caption).foregroundStyle(.secondary)
        Text(usageLabel)
          .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
          .accessibilityIdentifier("chat.attachmentUsage")
        if remaining == 0 {
          Text(localized("Attachment limit reached. Remove pending files or clear chat history."))
            .font(.caption).foregroundStyle(.secondary)
        }
      }
      if !attachments.isEmpty {
        ScrollViewReader { scroll in
          ScrollView(.horizontal) {
            HStack(spacing: 8) {
              ForEach(attachments) { attachment in
                HStack(spacing: 4) {
                  ChatAttachmentPreview(attachment: attachment, language: language)
                  Button {
                    let next = attachments.first { $0.id != attachment.id }?.id
                    remove(attachment)
                    removeFocused = next
                    if next == nil { addFocused = true }
                  } label: {
                    Image(systemName: "xmark").frame(width: 24, height: 28)
                  }
                  .buttonStyle(.borderless)
                  .help(L10n.string("Remove attachment %@", language: language, attachment.name))
                  .accessibilityLabel(L10n.string("Remove attachment %@", language: language, attachment.name))
                  .accessibilityIdentifier("chat.removeAttachment.\(attachment.id)")
                  .focused($removeFocused, equals: attachment.id)
                  .disabled(disabled)
                }
                .padding(8)
                .frame(width: 280)
                .background(AppTheme.fieldBackground, in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.primary.opacity(0.1)))
                .accessibilityElement(children: .contain)
                .id(attachment.id)
              }
            }
            .padding(2)
          }
          .scrollIndicators(.visible)
          .onChange(of: removeFocused) {
            if let id = removeFocused { scroll.scrollTo(id, anchor: .center) }
          }
        }
        // A single native scrolling strip keeps the editor and send controls
        // visible even with eight files or a short window.
        .frame(height: 76)
        .accessibilityLabel(localized("Pending attachments"))
      }
    }
    .padding(10)
    .background(dropTargeted && !disabled ? Color.accentColor.opacity(0.08) : Color.clear,
                in: RoundedRectangle(cornerRadius: 10))
    .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(
      dropTargeted && !disabled ? Color.accentColor : Color.primary.opacity(0.16),
      style: StrokeStyle(lineWidth: 1, dash: [4, 3])))
    .dropDestination(for: URL.self) { urls, _ in
      guard !disabled, !allowedKinds.isEmpty else { return false }
      return addFiles(urls)
    } isTargeted: { dropTargeted = $0 }
  }

  private var addMenu: some View {
    Menu {
      Button(localized("Choose files…")) { choose(allowedKinds) }
        .keyboardShortcut("a", modifiers: [.command, .shift])
      Divider()
      ForEach(kinds) { kind in
        Button { choose([kind]) } label: {
          Label(localized(kind.actionKey), systemImage: kind.symbol)
        }
        .disabled(all.filter { $0.kind == kind }.count >= kind.maximumCount)
      }
    } label: {
      Label(localized("Attach files"), systemImage: "paperclip")
    }
    .fixedSize()
    .focused($addFocused)
    .disabled(disabled || remaining == 0)
    .accessibilityIdentifier("chat.attachFiles")
  }

  private var capabilityLabels: some View {
    HStack(spacing: 12) {
      ForEach(kinds) { kind in
        Label(localized(kind.titleKey), systemImage: kind.symbol)
          .font(.caption).foregroundStyle(.secondary).fixedSize()
      }
    }
    .accessibilityElement(children: .combine)
  }

  private var usageLabel: String {
    let sizes = all.map(\.storedByteCount)
    let bytes = sizes.contains(nil) ? "—" : (Double(sizes.compactMap { $0 }.reduce(0, +)) / 1_048_576)
      .formatted(.number.locale(language.locale).precision(.fractionLength(1)))
    return L10n.string("%lld / 8 files · %@ / 32 MiB (including history)", language: language, Int64(all.count), bytes)
  }

  private var limitsButton: some View {
    Button { showingLimits.toggle() } label: {
      Label(localized("Attachment limits"), systemImage: "info.circle")
    }
    .fixedSize()
    .popover(isPresented: $showingLimits) {
      VStack(alignment: .leading, spacing: 12) {
        Text(localized("Attachment limits")).font(.headline)
        Text(localized("Up to 8 files and 32 MiB per request, including chat history. Each file: 8 MiB."))
        ForEach(kinds) { kind in
          VStack(alignment: .leading, spacing: 4) {
            HStack {
              Label(localized(kind.titleKey), systemImage: kind.symbol).font(.callout.bold())
              Spacer()
              Text(L10n.string("%@ / %@", language: language,
                String(all.filter { $0.kind == kind }.count), String(kind.maximumCount)))
                .monospacedDigit()
            }
            .accessibilityElement(children: .combine)
            Text(localized(kind.limitKey))
          }
        }
        Text(localized("Up to 4 documents and 2 audio clips. Images and audio share a 2048-token budget; the server checks expanded inputs."))
        Text(localized("Attachments are saved on this Mac and sent to the configured server with each message."))
          .foregroundStyle(.secondary)
        Button(localized("Done")) { showingLimits = false }
          .keyboardShortcut(.defaultAction)
      }
      .font(.callout)
      .fixedSize(horizontal: false, vertical: true)
      .padding(20)
      .frame(width: 380)
      .onExitCommand { showingLimits = false }
    }
  }
}
