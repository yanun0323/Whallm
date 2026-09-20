import AppKit
import DeepSeekRepack
import SwiftUI

struct ContentView: View {
  @ObservedObject var server: ServerController
  @ObservedObject var appUpdater: AppUpdater
  @StateObject private var modelLibrary = ModelLibrary()
  @StateObject private var chatSession = ChatSession()
  @StateObject private var throughputSession = ThroughputSession()
  // SwiftUI can reconstruct this value repeatedly; defer credential access to .task.
  @State private var configuration = ServerConfiguration.load(defaults: .standard, apiKey: "")
  @State private var didLoadConfiguration = false
  @State private var persistedAPIKey = ""
  @State private var advancedSettings = ModelAdvancedSettings.defaults(for: .deepSeekV4)
  @State private var advancedSettingsModelKind: ModelKind?
  @State private var aliasDraft = ""
  @State private var aliasError: String?
  @State private var modelNavigationPath: [String] = []
  @AppStorage("selectedAppPage") private var selectedPage = AppPage.server
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue

  var body: some View {
    NavigationSplitView {
      AppSidebar(selection: $selectedPage, language: selectedLanguage)
        .navigationSplitViewColumnWidth(min: 190, ideal: 220, max: 250)
    } detail: {
      NavigationStack(path: $modelNavigationPath) {
        VStack(spacing: 0) {
          HStack {
            Text(selectedPage.title(language: selectedLanguage))
              .font(.title2.bold())
            Spacer()
          }
          .frame(maxWidth: AppLayout.contentWidth)
          .frame(maxWidth: .infinity)
          .padding(.horizontal, 40)
          .padding(.vertical, 16)

          Divider()

          selectedPageView
        }
        .background(AppTheme.pageBackground)
        .navigationDestination(for: String.self) { rawValue in
          if let modelKind = ModelKind(rawValue: rawValue) {
            modelAdvancedPage(for: modelKind)
          }
        }
      }
    }
    .navigationSplitViewStyle(.balanced)
    .background(AppTheme.pageBackground)
    .preferredColorScheme(.dark)
    .environment(\.locale, selectedLanguage.locale)
    .onDisappear {
      chatSession.stopGenerating()
      throughputSession.cancel()
    }
    .task {
      if !didLoadConfiguration {
        let loaded = ServerConfiguration.localDefault
        persistedAPIKey = loaded.apiKey
        configuration = loaded
        didLoadConfiguration = true
      }
      await modelLibrary.scan()
      activateSelectedModel()
      modelLibrary.resumeDownloadIfNeeded()
    }
    .onChange(of: modelLibrary.models) { activateSelectedModel() }
    .onChange(of: modelLibrary.selectedModelKind) { activateSelectedModel() }
    .onChange(of: configuration) {
      configuration.save()
    }
    .onChange(of: configuration.apiKey) {
      guard didLoadConfiguration, configuration.apiKey != persistedAPIKey else { return }
      AppKeychain.saveAPIKey(configuration.apiKey)
      persistedAPIKey = configuration.apiKey
    }
    .onChange(of: advancedSettings) {
      if let advancedSettingsModelKind {
        advancedSettings.save(for: advancedSettingsModelKind)
        syncAdvancedSettings(for: advancedSettingsModelKind)
      }
    }
    .onChange(of: aliasDraft) {
      guard let advancedSettingsModelKind else { return }
      do {
        _ = try modelLibrary.saveAlias(aliasDraft, for: advancedSettingsModelKind)
        aliasError = nil
        syncAdvancedSettings(for: advancedSettingsModelKind)
      } catch {
        aliasError = error.localizedDescription
      }
    }
    .onChange(of: selectedPage) {
      if selectedPage != .model {
        modelNavigationPath.removeAll()
      }
    }
    .onChange(of: languageCode) { modelLibrary.refreshPreflight() }
  }

  private var selectedLanguage: AppLanguage {
    (AppLanguage(rawValue: languageCode) ?? .appDefault).resolved
  }

  @ViewBuilder
  private var selectedPageView: some View {
    switch selectedPage {
    case .server:
      ServerView(
        page: .server,
        configuration: $configuration,
        server: server,
        modelLibrary: modelLibrary,
        language: selectedLanguage,
        showAdvancedSettings: showModelAdvancedSettings
      )
    case .model:
      ServerView(
        page: .model,
        configuration: $configuration,
        server: server,
        modelLibrary: modelLibrary,
        language: selectedLanguage,
        showAdvancedSettings: showModelAdvancedSettings
      )
    case .advanced:
      AdvancedView(
        configuration: $configuration,
        serverActive: server.isActive,
        language: selectedLanguage
      )
    case .chat:
      ChatView(
        configuration: configuration,
        server: server,
        session: chatSession,
        language: selectedLanguage
      )
    case .metric:
      MetricView(
        state: server.state,
        performance: server.performance,
        history: server.performanceHistory,
        language: selectedLanguage,
        clearHistory: server.clearPerformanceHistory
      )
    case .throughput:
      ThroughputView(configuration: configuration, server: server, modelLibrary: modelLibrary,
        session: throughputSession, language: selectedLanguage)
    case .logs:
      LogsView(server: server, configuration: $configuration, language: selectedLanguage)
    case .about:
      AboutView(language: selectedLanguage)
    case .settings:
      SettingsView(
        languageCode: $languageCode,
        language: selectedLanguage,
        appUpdater: appUpdater
      )
    }
  }

  private func activateSelectedModel() {
    let modelKind = modelLibrary.selectedModelKind
    if advancedSettingsModelKind != modelKind {
      if let advancedSettingsModelKind {
        advancedSettings.save(for: advancedSettingsModelKind)
      }
      advancedSettingsModelKind = modelKind
      advancedSettings = ModelAdvancedSettings.loadOrDefault(for: modelKind)
      aliasDraft = modelLibrary.alias(for: modelKind)
      aliasError = nil
    }
  }

  private func showModelAdvancedSettings(_ modelKind: ModelKind) {
    if modelLibrary.selectedModelKind != modelKind {
      modelLibrary.selectedModelKind = modelKind
    }
    activateSelectedModel()
    modelNavigationPath = [modelKind.rawValue]
  }

  private func modelAdvancedPage(for modelKind: ModelKind) -> some View {
    ModelAdvancedView(
      settings: $advancedSettings,
      alias: $aliasDraft,
      aliasError: aliasError,
      settingsLocked: modelAdvancedSettingsAreLocked(
        modelID: modelKind.apiModelID,
        loadedModel: server.performance.loadedModel,
        loadingModel: server.performance.loadingModel,
        modelActionID: server.modelAction?.modelID
      ),
      modelURL: modelLibrary.usableModel(for: modelKind)?.url,
      mtpAvailable: modelLibrary.usableModel(for: modelKind)?.hasMTP == true,
      dsparkAvailable: modelLibrary.usableModel(for: modelKind)?.hasDSpark == true,
      modelKind: modelKind,
      language: selectedLanguage,
      onBack: { modelNavigationPath.removeAll() }
    )
    .background(AppTheme.pageBackground)
    .navigationBarBackButtonHidden()
  }

  private func syncAdvancedSettings(for modelKind: ModelKind) {
    guard server.canManageModels,
      !modelAdvancedSettingsAreLocked(
        modelID: modelKind.apiModelID,
        loadedModel: server.performance.loadedModel,
        loadingModel: server.performance.loadingModel,
        modelActionID: server.modelAction?.modelID
      ),
      let catalog = try? modelLibrary.makeServerCatalog(
        powerSavingLimitGBps: configuration.powerSavingLimitGBps),
      let entry = catalog.models.first(where: { $0.id == modelKind.apiModelID })
    else { return }
    server.configureModel(entry)
  }
}

enum AppPage: String, CaseIterable, Identifiable {
  case server
  case model
  case advanced
  case chat
  case metric // Preserve the saved page identifier; the visible title is Status.
  case throughput
  case logs
  case settings
  case about

  var id: String { rawValue }

  static let primaryPages: [AppPage] = [.server, .model, .metric, .advanced, .logs]
  static let playgroundPages: [AppPage] = [.chat, .throughput]
  static let generalPages: [AppPage] = [.settings, .about]

  var icon: String {
    switch self {
    case .server: "externaldrive"
    case .model: "shippingbox"
    case .advanced: "slider.horizontal.3"
    case .chat: "bubble"
    case .metric: "gauge.with.dots.needle.50percent"
    case .throughput: "speedometer"
    case .logs: "doc.text"
    case .settings: "gearshape"
    case .about: "info.circle"
    }
  }

  func title(language: AppLanguage) -> String {
    switch self {
    case .server: L10n.string("Server", language: language)
    case .model: L10n.string("Model", language: language)
    case .advanced: L10n.string("Advance", language: language)
    case .chat: L10n.string("Chat", language: language)
    case .metric: L10n.string("Status", language: language)
    case .throughput: L10n.string("Throughput", language: language)
    case .logs: L10n.string("Logs", language: language)
    case .settings: L10n.string("Settings", language: language)
    case .about: L10n.string("About", language: language)
    }
  }
}

struct AppSidebar: View {
  @Binding var selection: AppPage
  let language: AppLanguage

  var body: some View {
    List(selection: $selection) {
      Section { rows(AppPage.primaryPages) }
      Section(L10n.string("Playground", language: language)) { rows(AppPage.playgroundPages) }
      Section(L10n.string("General", language: language)) { rows(AppPage.generalPages) }
    }
    .listStyle(.sidebar)
    .scrollContentBackground(.hidden)
    .background(AppTheme.sidebarBackground)
  }

  private func rows(_ pages: [AppPage]) -> some View {
    ForEach(pages) { page in
      Label(page.title(language: language), systemImage: page.icon)
        .padding(.vertical, 6)
        .tag(page)
    }
  }
}

enum AppLayout {
  // Change this value to set the visible page content width.
  static let contentWidth: CGFloat = 880
}

enum AppTheme {
  static let pageBackground = Color(red: 0.095, green: 0.095, blue: 0.1)
  static let sidebarBackground = Color(red: 0.12, green: 0.12, blue: 0.125)
  static let cardBackground = Color(red: 0.15, green: 0.15, blue: 0.155)
  static let fieldBackground = Color(red: 0.075, green: 0.075, blue: 0.08)
  static let cardRadius: CGFloat = 16
  static let fieldRadius: CGFloat = 8
}

private struct TertiaryIconButtonStyle: ButtonStyle {
  var color = Color.secondary

  func makeBody(configuration: Configuration) -> some View {
    TertiaryIconButtonBody(
      label: configuration.label,
      isPressed: configuration.isPressed,
      color: color
    )
  }
}

private struct TertiaryIconButtonBody<Label: View>: View {
  let label: Label
  let isPressed: Bool
  let color: Color
  @Environment(\.isEnabled) private var isEnabled
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var isHovered = false

  var body: some View {
    label
      .labelStyle(.iconOnly)
      .font(.system(size: 14, weight: .semibold))
      .foregroundStyle(color)
      .opacity(iconOpacity)
      .frame(width: 32, height: 32)
      .background {
        RoundedRectangle(cornerRadius: 8, style: .continuous)
          .fill(color.opacity(backgroundOpacity))
      }
      .frame(width: 40, height: 40)
      .contentShape(Rectangle())
      .scaleEffect(reduceMotion ? 1 : scale)
      .onHover { isHovered = $0 }
      .animation(reduceMotion ? nil : .easeOut(duration: 0.12), value: isHovered)
      .animation(reduceMotion ? nil : .easeOut(duration: 0.12), value: isPressed)
  }

  private var iconOpacity: Double {
    if !isEnabled { return 0.35 }
    if isPressed { return 0.55 }
    return isHovered ? 1 : 0.78
  }

  private var backgroundOpacity: Double {
    if !isEnabled { return 0 }
    if isPressed { return 0.18 }
    return isHovered ? 0.12 : 0
  }

  private var scale: CGFloat {
    isEnabled && isPressed ? 0.96 : 1
  }
}

private struct SectionHeader: View {
  let title: String

  var body: some View {
    Text(title.uppercased())
      .font(.callout.weight(.semibold))
      .tracking(1.1)
      .foregroundStyle(.secondary)
      .accessibilityAddTraits(.isHeader)
  }
}

private struct AppCardModifier: ViewModifier {
  let padding: CGFloat

  func body(content: Content) -> some View {
    content
      .padding(padding)
      .background(AppTheme.cardBackground, in: RoundedRectangle(cornerRadius: AppTheme.cardRadius))
      .overlay(
        RoundedRectangle(cornerRadius: AppTheme.cardRadius)
          .stroke(Color.primary.opacity(0.06))
      )
  }
}

private struct AppInputModifier: ViewModifier {
  let width: CGFloat?

  func body(content: Content) -> some View {
    content
      .textFieldStyle(.plain)
      .padding(.horizontal, 11)
      .frame(width: width)
      .frame(minHeight: 34)
      .background(
        AppTheme.fieldBackground, in: RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
      )
      .overlay(
        RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
          .stroke(Color.primary.opacity(0.12))
      )
  }
}

extension View {
  func appCard(padding: CGFloat = 16) -> some View {
    modifier(AppCardModifier(padding: padding))
  }

  func appInput(width: CGFloat? = nil) -> some View {
    modifier(AppInputModifier(width: width))
  }
}

private enum ServerViewPage {
  case server
  case model
}

private struct ServerView: View {
  let page: ServerViewPage
  @Binding var configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  @ObservedObject var modelLibrary: ModelLibrary
  let language: AppLanguage
  let showAdvancedSettings: (ModelKind) -> Void
  @State private var confirmsDownload = false
  @State private var downloadTarget: ModelKind?
  @State private var confirmsMTPDownload = false
  @State private var mtpDownloadTarget: InstalledModelInfo?
  @State private var confirmsRepair = false
  @State private var repairTarget: InstalledModelInfo?
  @State private var confirmsReinstall = false
  @State private var reinstallTarget: URL?
  @State private var startError: String?

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        if page == .server {
          serverSummaryCard

          SectionHeader(title: L10n.string("System Check", language: language))
            .padding(.top, 10)
          systemCheckPanel

          SectionHeader(title: L10n.string("Server", language: language))
            .padding(.top, 10)
          serverPanel
        } else {
          if let loadedModelKind {
            SectionHeader(title: L10n.string("Loaded", language: language))
            loadedModelPanel(loadedModelKind)
          }

          HStack(spacing: 16) {
            SectionHeader(title: L10n.string("Model", language: language))
            Spacer()
            Button {
              chooseModelDirectory()
            } label: {
              Label(
                L10n.string("Select Model Folder", language: language),
                systemImage: "folder"
              )
            }
            .buttonStyle(.bordered)
            .frame(minHeight: 40)
            .contentShape(Rectangle())
            .help(L10n.string("Select Model Folder", language: language))
            .disabled(modelLibrary.isBusy)
          }

          modelPanel

          if let modelActionError = server.modelActionError {
            Label(modelActionError, systemImage: "exclamationmark.triangle.fill")
              .font(.callout)
              .foregroundStyle(.red)
              .textSelection(.enabled)
              .accessibilityLabel(
                L10n.string("Error: %@", language: language, modelActionError))
          }

          if modelLibrary.isBusy && modelLibrary.downloadModelKind == nil {
            operationPanel
          }

          if let message = modelLibrary.message {
            Label(
              message,
              systemImage: message.hasPrefix("無法")
                ? "exclamationmark.triangle.fill" : "info.circle"
            )
            .foregroundStyle(message.hasPrefix("無法") ? Color.red : Color.secondary)
            .textSelection(.enabled)
            .accessibilityLabel(L10n.string("Model status: %@", language: language, message))
          }

          if !modelLibrary.damagedModels.isEmpty || !modelLibrary.invalidModelURLs.isEmpty {
            damagedModelsPanel
          }
        }
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
    .confirmationDialog(
      L10n.string("Download and install the model?"),
      isPresented: $confirmsDownload,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Download Model", language: language)) {
        if let downloadTarget { modelLibrary.startDownload(for: downloadTarget) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will install the model in %@. You can resume an interrupted download.",
          language: language,
          modelLibrary.rootURL.path))
    }
    .confirmationDialog(
      L10n.string("Verify and repair the model?"),
      isPresented: $confirmsRepair,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Verify and Repair")) {
        if let repairTarget { modelLibrary.startRepair(repairTarget) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will verify the complete model. It will download only missing or damaged data."))
    }
    .confirmationDialog(
      L10n.string("Download the model again?"),
      isPresented: $confirmsReinstall,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Download Again"), role: .destructive) {
        if let reinstallTarget { modelLibrary.reinstall(reinstallTarget) }
      }
      Button(L10n.string("Cancel"), role: .cancel) {}
    } message: {
      Text(
        L10n.string(
          "The app will move the damaged model to Trash. It will then download the complete model.")
      )
    }
    .confirmationDialog(
      L10n.string("Download and install MTP?", language: language),
      isPresented: $confirmsMTPDownload,
      titleVisibility: .visible
    ) {
      Button(L10n.string("Download MTP", language: language)) {
        if let mtpDownloadTarget { modelLibrary.startMTPInstallation(mtpDownloadTarget) }
      }
      Button(L10n.string("Cancel", language: language), role: .cancel) {}
    } message: {
      if let mtpDownloadTarget {
        Text(
          L10n.string(
            "The app will add MTP to %@. The existing Qwen model will remain installed.",
            language: language,
            mtpDownloadTarget.url.path
          )
        )
      }
    }
  }

  private var serverSummaryCard: some View {
    VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 18) {
        Image(nsImage: NSImage(named: NSImage.applicationIconName) ?? NSImage())
          .resizable()
          .interpolation(.high)
          .frame(width: 56, height: 56)
          .accessibilityHidden(true)

        VStack(alignment: .leading, spacing: 5) {
          HStack(spacing: 10) {
            Text("Whallm")
              .font(.title3.bold())
              .lineLimit(1)
            Label(summaryStatusLabel, systemImage: summaryStatusSymbol)
              .font(.callout.weight(.semibold))
              .foregroundStyle(summaryStatusColor)
              .padding(.horizontal, 10)
              .padding(.vertical, 4)
              .background(summaryStatusColor.opacity(0.12), in: Capsule())
          }
          Text(summaryDetail)
            .font(.callout.monospaced())
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
        }
        .accessibilityElement(children: .combine)

        Spacer(minLength: 20)

        Button {
          if server.isActive {
            server.stop()
          } else {
            do {
              let catalog = try modelLibrary.makeServerCatalog(
                powerSavingLimitGBps: configuration.powerSavingLimitGBps)
              startError = nil
              server.start(configuration, catalog: catalog)
            } catch {
              startError = error.localizedDescription
            }
          }
        } label: {
          Label(
            L10n.string(server.isActive ? "Stop Server" : "Start Server", language: language),
            systemImage: server.isActive ? "stop.fill" : "play.fill"
          )
        }
        .buttonStyle(.borderedProminent)
        .tint(.blue)
        .controlSize(.large)
        .keyboardShortcut(server.isActive ? "." : "\r", modifiers: .command)
      }

      if let startError {
        Text(startError)
          .font(.caption)
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", language: language, startError))
      }
    }
    .appCard(padding: 22)
  }

  private var serverPanel: some View {
    VStack(alignment: .leading, spacing: 0) {
      if case .failed(let message) = server.state {
        Label(message, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", language: language, message))
          .padding(.bottom, 12)
      }

      SettingRow(
        "Listen Address",
        hint: "Choose which devices can connect. From another device, use this Mac’s LAN IP address—not 0.0.0.0.",
        language: language
      ) {
        Picker(L10n.string("Listen Address", language: language), selection: $configuration.host) {
          Text(L10n.string("127.0.0.1 (Local only)", language: language))
            .tag("127.0.0.1")
          Text(L10n.string("0.0.0.0 (All networks)", language: language))
            .tag("0.0.0.0")
        }
        .labelsHidden()
        .frame(width: 290, alignment: .trailing)
      }
      .disabled(server.isActive)

      Divider()

      SettingRow(
        "Port",
        hint: "Default 11434. Restart the server after changing it.",
        language: language
      ) {
        TextField("11434", value: $configuration.port, format: .number.grouping(.never))
          .appInput(width: 120)
          .accessibilityLabel(L10n.string("Port", language: language))
      }
      .disabled(server.isActive)

      Divider()

      SettingRow(
        "API key",
        hint: "Optional for local use",
        language: language
      ) {
        SecureField(L10n.string("Optional for local use"), text: $configuration.apiKey)
          .appInput(width: 320)
      }
      .disabled(server.isActive)

    }
    .appCard()
  }

  private var systemCheckPanel: some View {
    HStack(spacing: 0) {
      ForEach(modelLibrary.preflightChecks) { check in
        HStack(spacing: 10) {
          Image(systemName: preflightCheckSymbol(check.id))
            .font(.system(size: 20, weight: .regular))
            .foregroundStyle(.secondary)
            .frame(width: 24)
            .accessibilityHidden(true)

          Text(check.title)
            .font(.callout.weight(.medium))
            .lineLimit(1)

          Spacer(minLength: 8)

          Image(systemName: preflightSymbol(check.status))
            .font(.system(size: 16, weight: .semibold))
            .foregroundStyle(preflightColor(check.status))
            .accessibilityHidden(true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(check.detail)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(preflightAccessibilityLabel(check))

        if check.id != modelLibrary.preflightChecks.last?.id {
          Divider()
            .frame(height: 28)
            .padding(.horizontal, 16)
        }
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
    .appCard()
  }

  private var modelPanel: some View {
    VStack(alignment: .leading, spacing: 14) {
      List(selection: selectedModelKind) {
        ForEach(selectableModelKinds, id: \.rawValue) { modelKind in
          modelRow(modelKind)
          .contentShape(Rectangle())
          .tag(modelKind.rawValue)
          .selectionDisabled(modelSelectionLocked)
          .task { await modelLibrary.refreshInstallationPlan(for: modelKind) }
        }
      }
      .listStyle(.inset)
      .scrollContentBackground(.hidden)
      .frame(height: modelListHeight)
      .accessibilityLabel(L10n.string("Model", language: language))
      .accessibilityHint(
        L10n.string(
          "Select a model. You can select it before it is installed.", language: language))

      if let selectedModel, selectableModelKinds.contains(selectedModel.modelKind) {
        Divider()

        HStack(spacing: 10) {
          Button(L10n.string("Show in Finder")) {
            modelLibrary.reveal(selectedModel.url)
          }
          Button(L10n.string("Verify Complete Model")) {
            modelLibrary.startVerification(selectedModel)
          }
          .disabled(server.isActive || modelLibrary.isBusy)
          if selectedModel.modelKind.descriptor.supports("dspark"), !selectedModel.hasDSpark {
            Button(L10n.string("Install DSpark (10.12 GiB)")) {
              modelLibrary.startDSparkInstallation(selectedModel)
            }
            .disabled(server.isActive || modelLibrary.isBusy)
          }
          Spacer()
        }
      }

      Divider()
      modelFolderSummary

      if modelLibrary.verificationModelPath == selectedModel?.url.path,
        let issues = modelLibrary.verificationIssues
      {
        if issues.isEmpty {
          Label(L10n.string("Complete verification passed"), systemImage: "checkmark.seal.fill")
            .foregroundStyle(.green)
        } else {
          Label(
            L10n.string("%lld files need repair", Int64(issues.count)),
            systemImage: "exclamationmark.triangle.fill"
          )
          .foregroundStyle(.red)
          Button(L10n.string("Verify and Repair")) {
            repairTarget = selectedModel
            confirmsRepair = true
          }
          .disabled(modelLibrary.isBusy || server.isActive)
        }
      }
    }
    .appCard()
  }

  private func loadedModelPanel(_ modelKind: ModelKind) -> some View {
    modelRow(modelKind)
      .task { await modelLibrary.refreshInstallationPlan(for: modelKind) }
      .appCard()
  }

  private func modelRow(_ modelKind: ModelKind) -> some View {
    let model = modelLibrary.usableModel(for: modelKind)
    let downloadBlock = modelLibrary.downloadBlock(for: modelKind)
    let isDownloading = modelLibrary.downloadModelKind == modelKind
    let downloadReason = visibleDownloadReason(modelKind, block: downloadBlock)

    return VStack(alignment: .leading, spacing: 10) {
      HStack(spacing: 12) {
        if modelLibrary.isScanning
          || (model == nil && modelLibrary.isPlanningInstallation(for: modelKind))
        {
          ProgressView()
            .controlSize(.small)
            .accessibilityHidden(true)
        } else {
          Image(
            systemName: isModelLoading(modelKind)
              ? "clock.fill" : (model == nil ? "arrow.down.circle" : "checkmark.circle.fill")
          )
            .foregroundStyle(
              isModelLoading(modelKind) ? Color.orange : (model == nil ? Color.orange : Color.green)
            )
            .accessibilityHidden(true)
        }

        VStack(alignment: .leading, spacing: 3) {
          Text(modelKind.displayName)
            .font(.body.weight(.medium))
          modelStatus(modelKind, model: model)
        }

        Spacer(minLength: 16)

        if isDownloading {
          Button(L10n.string("Stop Download", language: language)) {
            modelLibrary.cancelOperation()
          }
          .buttonStyle(.bordered)
          .disabled(modelLibrary.operationPhase == .cancelling)
        } else if model == nil {
          modelDownloadButton(modelKind, block: downloadBlock)
        } else if let model, shouldShowMTPDownloadButton(model) {
          mtpDownloadButton(model)
        }

        modelLifecycleButton(modelKind, model: model)

        Button {
          showAdvancedSettings(modelKind)
        } label: {
          Label(
            L10n.string("Advanced Settings", language: language),
            systemImage: "slider.horizontal.3"
          )
        }
        .buttonStyle(TertiaryIconButtonStyle())
        .accessibilityLabel(
          L10n.string(
            "Advanced Settings for %@",
            language: language,
            modelKind.displayName
          )
        )
        .help(
          L10n.string(
            "Advanced Settings for %@",
            language: language,
            modelKind.displayName
          )
        )
        .disabled(modelSelectionLocked && modelKind != modelLibrary.selectedModelKind)
      }
      .frame(minHeight: 48)

      if shouldShowModelDownloadReason(
        modelIsInstalled: model != nil,
        modelIsDownloading: isDownloading,
        hasReason: downloadReason != nil
      ), let downloadReason
      {
        Label(downloadReason, systemImage: "exclamationmark.triangle.fill")
          .font(.caption)
          .foregroundStyle(.secondary)
          .padding(.leading, 32)
      }

      if isDownloading {
        modelDownloadProgress
          .padding(.leading, 32)
          .padding(.bottom, 8)
      }
    }
  }

  @ViewBuilder
  private func modelStatus(_ modelKind: ModelKind, model: InstalledModelInfo?) -> some View {
    if let model {
      let status = L10n.string(
        isModelLoading(modelKind)
          ? "Loading" : (isModelLoaded(modelKind) ? "Loaded" : "Installed"),
        language: language
      )
      Text(
        L10n.string(
          shouldShowMTPDownloadButton(model) ? "%@ · %@ · MTP download available" : "%@ · %@",
          language: language,
          status,
          formattedBytes(model.size)
        )
      )
      .font(.callout)
      .foregroundStyle(.secondary)
    } else if modelLibrary.isPlanningInstallation(for: modelKind) {
      Text(L10n.string("Not installed · Checking download size…", language: language))
        .font(.callout)
        .foregroundStyle(.secondary)
    } else if let bytes = modelLibrary.plannedInstalledBytes(for: modelKind) {
      Text(
        L10n.string(
          "Not installed · Download size: %@",
          language: language,
          formattedBytes(bytes)
        )
      )
      .font(.callout)
      .foregroundStyle(.secondary)
    } else if case .some(.installationPlanUnavailable) = modelLibrary.downloadBlock(
      for: modelKind)
    {
      Text(L10n.string("Not installed · Download size unavailable", language: language))
        .font(.callout)
        .foregroundStyle(.secondary)
    } else {
      Text(L10n.string("Not installed", language: language))
        .font(.callout)
        .foregroundStyle(.secondary)
    }
  }

  private func modelLifecycleButton(
    _ modelKind: ModelKind,
    model: InstalledModelInfo?
  ) -> some View {
    let isLoaded = isModelLoaded(modelKind)
    let label = L10n.string(
      isLoaded ? "Unload %@" : "Load %@",
      language: language,
      modelKind.displayName
    )
    let disabledReason = modelLifecycleDisabledReason(modelKind, model: model)

    return Button {
      Task {
        if isLoaded {
          await server.unloadModel(modelKind.apiModelID)
        } else {
          await server.loadModel(modelKind.apiModelID) {
            try modelLibrary.makeServerCatalog(
              powerSavingLimitGBps: configuration.powerSavingLimitGBps)
          }
        }
      }
    } label: {
      Label(
        label,
        systemImage: isLoaded ? "eject.fill" : "play.fill"
      )
    }
    .fontWeight(isLoaded ? .thin : .regular)
    .buttonStyle(TertiaryIconButtonStyle(color: isLoaded ? .secondary : .accentColor))
    .accessibilityLabel(label)
    .accessibilityHint(disabledReason ?? "")
    .help(disabledReason ?? label)
    .disabled(disabledReason != nil)
    .overlay {
      if let disabledReason {
        Color.clear
          .contentShape(Rectangle())
          .help(disabledReason)
          .accessibilityHidden(true)
      }
    }
  }

  private func modelLifecycleDisabledReason(
    _ modelKind: ModelKind,
    model: InstalledModelInfo?
  ) -> String? {
    if model == nil {
      return L10n.string("Install the model first.", language: language)
    }
    if !server.canManageModels {
      return L10n.string("Start the server first", language: language)
    }
    if server.performance.generating {
      return L10n.string("Wait for the current response to finish.", language: language)
    }
    if server.modelAction != nil || server.performance.loadingModel != nil {
      return L10n.string("Another model action is in progress.", language: language)
    }
    if !server.catalogModels.contains(where: { $0.id == modelKind.apiModelID }) {
      return L10n.string(
        "The model is not available to this server. Restart the server.",
        language: language
      )
    }
    return nil
  }

  @ViewBuilder
  private func modelDownloadButton(
    _ modelKind: ModelKind,
    block: ModelDownloadBlock?
  ) -> some View {
    let isDisabled = modelDownloadIsDisabled(
      serverIsActive: server.isActive,
      operationIsBusy: modelLibrary.isBusy,
      canStartDownload: modelLibrary.canStartDownload(modelKind),
      hasPartialDownload: modelLibrary.hasPartialDownload,
      targetHasPartialDownload: modelLibrary.hasPartialDownload(for: modelKind)
    )
    let label =
      modelLibrary.hasPartialDownload(for: modelKind) ? "Resume" : "Download"
    let help = downloadHelp(modelKind, block: block)

    let button = Button {
      requestDownload(modelKind)
    } label: {
      Label(L10n.string(label, language: language), systemImage: "arrow.down.circle")
    }
    .frame(minHeight: 40)
    .contentShape(Rectangle())
    .disabled(isDisabled)
    .overlay {
      if isDisabled {
        Color.clear
          .contentShape(Rectangle())
          .help(help)
          .accessibilityHidden(true)
      }
    }
    .help(help)
    .accessibilityHint(help)

    if modelLibrary.selectedModelKind == modelKind {
      button
        .buttonStyle(.borderedProminent)
        .tint(isDisabled ? .gray : .blue)
    } else {
      button
        .buttonStyle(.bordered)
        .tint(isDisabled ? .gray : .secondary)
    }
  }

  private func mtpDownloadButton(_ model: InstalledModelInfo) -> some View {
    let block = modelLibrary.mtpDownloadBlock(for: model)
    let disabledReason =
      modelLibrary.isBusy
      ? L10n.string("Wait for the current model operation to finish.", language: language)
      : block?.message
    let label = L10n.string(
      modelLibrary.hasPartialMTPInstallation(for: model)
        ? "Resume MTP download for %@" : "Download MTP for %@",
      language: language,
      model.modelKind.displayName
    )

    return Button {
      requestMTPDownload(model)
    } label: {
      Label(label, systemImage: "arrow.down.circle")
    }
    .buttonStyle(TertiaryIconButtonStyle())
    .accessibilityLabel(label)
    .accessibilityHint(disabledReason ?? "")
    .help(label)
    .disabled(disabledReason != nil)
    .overlay {
      if let disabledReason {
        Color.clear
          .contentShape(Rectangle())
          .help(disabledReason)
          .accessibilityHidden(true)
      }
    }
  }

  private var modelDownloadProgress: some View {
    VStack(alignment: .leading, spacing: 6) {
      if let progress = modelLibrary.operationProgress, let fraction = progress.fraction {
        let fractionText = formattedProgress(fraction)
        HStack(spacing: 12) {
          Text(modelLibrary.operationPhase.label)
            .font(.callout.weight(.medium))
          Spacer()
          Text(fractionText)
            .font(.callout.weight(.semibold).monospacedDigit())
        }
        RoundedRectangle(cornerRadius: 3, style: .continuous)
          .fill(Color.primary.opacity(0.12))
          .overlay {
            RoundedRectangle(cornerRadius: 3, style: .continuous)
              .fill(Color.accentColor)
              .scaleEffect(x: max(0, min(1, fraction)), anchor: .leading)
          }
          .frame(height: 6)
          .accessibilityElement()
          .accessibilityLabel(modelLibrary.operationPhase.label)
          .accessibilityValue(fractionText)
        modelDownloadMetadata(progress)
      } else {
        HStack(spacing: 10) {
          ProgressView()
            .controlSize(.small)
            .accessibilityHidden(true)
          Text(modelLibrary.operationPhase.label)
            .font(.callout.weight(.medium))
        }
        .accessibilityElement(children: .combine)
      }
    }
  }

  private var modelListHeight: CGFloat {
    var height = CGFloat(selectableModelKinds.count) * 62
    if modelLibrary.downloadModelKind != nil {
      height += modelDownloadProgressExtraHeight(
        hasProgressFraction: modelLibrary.operationProgress?.fraction != nil)
    }
    let visibleDownloadReasonCount = selectableModelKinds.filter {
      shouldShowModelDownloadReason(
        modelIsInstalled: modelLibrary.usableModel(for: $0) != nil,
        modelIsDownloading: modelLibrary.downloadModelKind == $0,
        hasReason: visibleDownloadReason($0, block: modelLibrary.downloadBlock(for: $0)) != nil
      )
    }.count
    height += CGFloat(visibleDownloadReasonCount) * 32
    return min(height, 320)
  }

  private var selectedModelKind: Binding<String> {
    Binding(
      get: { modelLibrary.selectedModelKind.rawValue },
      set: { value in
        guard !modelSelectionLocked, let modelKind = ModelKind(rawValue: value),
          modelKind != modelLibrary.selectedModelKind
        else { return }
        modelLibrary.selectedModelKind = modelKind
      }
    )
  }

  private var modelSelectionLocked: Bool {
    modelSelectionIsLocked(
      serverIsActive: server.isActive,
      operationIsBusy: modelLibrary.isBusy,
      downloadIsActive: modelLibrary.downloadModelKind != nil,
      hasPartialDownload: modelLibrary.hasPartialDownload
    )
  }

  private var operationPanel: some View {
    VStack(alignment: .leading, spacing: 10) {
      Text(modelLibrary.operationPhase.label).font(.headline)
      if let progress = modelLibrary.operationProgress, let fraction = progress.fraction {
        ProgressView(value: fraction)
          .accessibilityLabel(modelLibrary.operationPhase.label)
          .accessibilityValue(formattedProgress(fraction))
        modelDownloadMetadata(progress)
      } else {
        ProgressView()
          .accessibilityLabel(modelLibrary.operationPhase.label)
      }
      Button(L10n.string("Stop Current Operation")) { modelLibrary.cancelOperation() }
        .disabled(modelLibrary.operationPhase == .cancelling)
    }
    .appCard()
  }

  private var damagedModelsPanel: some View {
    VStack(alignment: .leading, spacing: 16) {
      Text(L10n.string("Models That Need Attention")).font(.headline)
      ForEach(modelLibrary.damagedModels) { model in
        HStack(alignment: .top, spacing: 12) {
          Label(
            L10n.string(
              "%@: %lld files are missing or have the wrong size", model.name,
              Int64(model.quickIssues.count)),
            systemImage: "exclamationmark.triangle.fill"
          )
          .foregroundStyle(.red)
          Spacer()
          Button(L10n.string("Show in Finder")) { modelLibrary.reveal(model.url) }
          Button(L10n.string("Verify and Repair")) {
            repairTarget = model
            confirmsRepair = true
          }
          .disabled(modelLibrary.isBusy || server.isActive)
        }
      }
      ForEach(modelLibrary.invalidModelURLs, id: \.path) { url in
        HStack(alignment: .top, spacing: 12) {
          Label(
            L10n.string("%@: The manifest cannot be read", url.lastPathComponent),
            systemImage: "xmark.octagon.fill"
          )
          .foregroundStyle(.red)
          Spacer()
          Button(L10n.string("Show in Finder")) { modelLibrary.reveal(url) }
          Button(L10n.string("Download Again")) {
            reinstallTarget = url
            confirmsReinstall = true
          }
          .disabled(modelLibrary.isBusy || server.isActive || !modelLibrary.canDownload)
        }
      }
    }
    .appCard()
  }

  private var selectedModel: InstalledModelInfo? {
    modelLibrary.usableModel(for: modelLibrary.selectedModelKind)
  }

  private var loadedModelKind: ModelKind? {
    modelKind(withAPIModelID: server.performance.loadedModel)
  }

  private var selectableModelKinds: [ModelKind] {
    ModelLibrary.supportedModelKinds.filter { $0 != loadedModelKind }
  }

  private func isModelLoaded(_ modelKind: ModelKind) -> Bool {
    server.performance.loadedModel == modelKind.apiModelID
  }

  private func isModelLoading(_ modelKind: ModelKind) -> Bool {
    server.performance.loadingModel == modelKind.apiModelID
      || server.modelAction == .load(modelKind.apiModelID)
  }

  private var summaryStatusLabel: String {
    return server.state.label
  }

  private var summaryStatusSymbol: String {
    return "circle.fill"
  }

  private var summaryStatusColor: Color {
    return statusColor
  }

  private var summaryDetail: String {
    return configuration.baseURL?.absoluteString ?? L10n.string("Invalid Base URL")
  }

  private var modelFolderSummary: some View {
    HStack(alignment: .center, spacing: 12) {
      Image(systemName: "folder")
        .foregroundStyle(.secondary)
        .frame(width: 18)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 3) {
        Text(L10n.string("Model folder", language: language))
          .font(.caption)
          .foregroundStyle(.secondary)
        Text(modelLibrary.rootURL.path)
          .font(.callout.monospaced())
          .lineLimit(1)
          .truncationMode(.middle)
          .textSelection(.enabled)
          .help(modelLibrary.rootURL.path)
      }
      Spacer(minLength: 16)
      VStack(alignment: .trailing, spacing: 3) {
        Text(L10n.string("Available space", language: language))
          .font(.caption)
          .foregroundStyle(.secondary)
        if let availableBytes = modelLibrary.modelFolderAvailableBytes {
          Text(formattedBytes(availableBytes))
            .font(.callout.weight(.medium).monospacedDigit())
        } else if modelLibrary.isScanning {
          ProgressView()
            .controlSize(.small)
            .accessibilityLabel(L10n.string("Available space", language: language))
        } else {
          Text(L10n.string("Unavailable", language: language))
            .font(.callout.weight(.medium))
        }
      }
    }
    .accessibilityElement(children: .combine)
  }

  private func visibleDownloadReason(
    _ modelKind: ModelKind,
    block: ModelDownloadBlock?
  ) -> String? {
    if case .some(.loadingInstallationPlan) = block { return nil }
    if let block { return block.message }
    let isDisabled = modelDownloadIsDisabled(
      serverIsActive: server.isActive,
      operationIsBusy: modelLibrary.isBusy,
      canStartDownload: modelLibrary.canStartDownload(modelKind),
      hasPartialDownload: modelLibrary.hasPartialDownload,
      targetHasPartialDownload: modelLibrary.hasPartialDownload(for: modelKind)
    )
    guard isDisabled else { return nil }
    return downloadHelp(modelKind, block: nil)
  }

  private func modelDownloadMetadata(_ progress: ModelOperationProgress) -> some View {
    let completed = L10n.string(
      "%@ of %@",
      language: language,
      formattedBytes(progress.completedBytes),
      formattedBytes(progress.totalBytes)
    )
    let speed = progress.bytesPerSecond.flatMap { bytesPerSecond in
      bytesPerSecond > 0
        ? L10n.string("%@/s", language: language, formattedBytes(UInt64(bytesPerSecond)))
        : nil
    }
    let remaining = progress.estimatedSecondsRemaining.flatMap { seconds in
      seconds.isFinite
        ? L10n.string(
          "About %@ remaining", language: language, formattedDuration(seconds))
        : nil
    }

    return HStack(spacing: 14) {
      Text(completed)
      Spacer(minLength: 12)
      if let speed {
        Label(speed, systemImage: "speedometer")
      }
      if let remaining {
        Label(remaining, systemImage: "clock")
      }
    }
    .font(.callout.monospacedDigit())
    .foregroundStyle(.secondary)
  }

  private var statusColor: Color {
    switch server.state {
    case .running: .green
    case .failed: .red
    case .starting, .stopping: .orange
    case .stopped: .secondary
    }
  }

  private func requestDownload(_ modelKind: ModelKind) {
    guard !modelLibrary.isBusy, modelLibrary.canStartDownload(modelKind),
      !modelLibrary.hasPartialDownload || modelLibrary.hasPartialDownload(for: modelKind)
    else { return }
    if modelLibrary.hasPartialDownload(for: modelKind) {
      modelLibrary.startDownload(for: modelKind)
    } else {
      downloadTarget = modelKind
      confirmsDownload = true
    }
  }

  private func requestMTPDownload(_ model: InstalledModelInfo) {
    guard !modelLibrary.isBusy, modelLibrary.mtpDownloadBlock(for: model) == nil else { return }
    if modelLibrary.hasPartialMTPInstallation(for: model) {
      modelLibrary.startMTPInstallation(model)
    } else {
      mtpDownloadTarget = model
      confirmsMTPDownload = true
    }
  }

  private func downloadHelp(
    _ modelKind: ModelKind,
    block: ModelDownloadBlock?
  ) -> String {
    if let block { return block.message }
    if modelLibrary.isBusy {
      return L10n.string(
        "Wait for the current model operation to finish.", language: language)
    }
    if modelLibrary.hasPartialDownload && !modelLibrary.hasPartialDownload(for: modelKind) {
      return L10n.string(
        "Finish the current model download before downloading another model.",
        language: language
      )
    }
    if !modelLibrary.canStartDownload(modelKind) {
      return L10n.string(
        "A model already exists in this location. Verify and repair the existing model first.",
        language: language
      )
    }
    return L10n.string(
      modelLibrary.hasPartialDownload(for: modelKind) ? "Resume Download" : "Download Model",
      language: language
    )
  }

  private func chooseModelDirectory() {
    let panel = NSOpenPanel()
    panel.title = L10n.string("Select Model Folder")
    panel.prompt = L10n.string("Select Folder")
    panel.canChooseDirectories = true
    panel.canChooseFiles = false
    panel.allowsMultipleSelection = false
    if panel.runModal() == .OK, let url = panel.url {
      Task {
        await modelLibrary.setRoot(url)
        if modelLibrary.usableModel(for: modelLibrary.selectedModelKind) == nil,
          let firstModel = modelLibrary.usableModels.first
        {
          modelLibrary.selectedModelKind = firstModel.modelKind
        }
      }
    }
  }

  private func formattedBytes(_ bytes: UInt64) -> String {
    if bytes == 0 { return "0 KB" }
    return ByteCountFormatter.string(
      fromByteCount: Int64(clamping: bytes), countStyle: .file)
  }

  private func formattedDuration(_ seconds: Double) -> String {
    let totalMinutes = max(1, Int(seconds / 60))
    if totalMinutes < 60 { return L10n.string("%lld min", Int64(totalMinutes)) }
    return L10n.string(
      "%lld hr %lld min", Int64(totalMinutes / 60), Int64(totalMinutes % 60))
  }

  private func formattedProgress(_ fraction: Double) -> String {
    if fraction > 0, fraction < 0.001 {
      return fraction.formatted(.percent.precision(.fractionLength(2)))
    }
    if fraction < 0.01 {
      return fraction.formatted(.percent.precision(.fractionLength(1)))
    }
    return fraction.formatted(.percent.precision(.fractionLength(0)))
  }

  private func preflightSymbol(_ status: PreflightStatus) -> String {
    switch status {
    case .passed: "checkmark.circle.fill"
    case .warning: "exclamationmark.triangle.fill"
    case .failed: "xmark.circle.fill"
    }
  }

  private func preflightCheckSymbol(_ id: String) -> String {
    switch id {
    case "architecture": "apple.logo"
    case "memory": "memorychip"
    case "ssd": "externaldrive.fill"
    default: "checklist"
    }
  }

  private func preflightAccessibilityLabel(_ check: PreflightCheck) -> String {
    let status: String
    switch check.status {
    case .passed: status = L10n.string("Passed", language: language)
    case .warning: status = L10n.string("Review", language: language)
    case .failed: status = L10n.string("Action required", language: language)
    }
    return "\(check.title). \(status). \(check.detail)"
  }

  private func preflightColor(_ status: PreflightStatus) -> Color {
    switch status {
    case .passed: .green
    case .warning: .orange
    case .failed: .red
    }
  }
}

func shouldShowModelDownloadReason(
  modelIsInstalled: Bool,
  modelIsDownloading: Bool,
  hasReason: Bool
) -> Bool {
  !modelIsInstalled && !modelIsDownloading && hasReason
}

func shouldShowMTPDownloadButton(_ model: InstalledModelInfo?) -> Bool {
  model?.modelKind.descriptor.supports("mtp") == true && model?.hasMTP == false
}

func modelAdvancedSettingsAreLocked(
  modelID: String,
  loadedModel: String?,
  loadingModel: String?,
  modelActionID: String?
) -> Bool {
  modelID == loadedModel || modelID == loadingModel || modelID == modelActionID
}

func modelDownloadProgressExtraHeight(hasProgressFraction: Bool) -> CGFloat {
  hasProgressFraction ? 72 : 40
}

func modelDownloadIsDisabled(
  serverIsActive _: Bool,
  operationIsBusy: Bool,
  canStartDownload: Bool,
  hasPartialDownload: Bool,
  targetHasPartialDownload: Bool
) -> Bool {
  operationIsBusy || !canStartDownload
    || (hasPartialDownload && !targetHasPartialDownload)
}

func modelSelectionIsLocked(
  serverIsActive _: Bool,
  operationIsBusy: Bool,
  downloadIsActive: Bool,
  hasPartialDownload _: Bool
) -> Bool {
  operationIsBusy && !downloadIsActive
}

@MainActor
func modelKind(withAPIModelID modelID: String?) -> ModelKind? {
  guard let modelID else { return nil }
  return ModelLibrary.supportedModelKinds.first { $0.apiModelID == modelID }
}

private struct AdvancedView: View {
  @Binding var configuration: ServerConfiguration
  let serverActive: Bool
  let language: AppLanguage

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        SectionHeader(title: L10n.string("Power Saving Mode", language: language))
        powerSavingPanel
          .disabled(serverActive)
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }

  private var powerSavingPanel: some View {
    VStack(alignment: .leading, spacing: 14) {
      HStack(alignment: .firstTextBaseline) {
        SettingLabel(
          "SSD read limit",
          hint: "A lower SSD read limit can reduce generation speed.",
          language: language
        )
        Spacer()
        Text(powerSavingLimitLabel(configuration.powerSavingLimitGBps))
          .font(.body.weight(.semibold).monospacedDigit())
      }

      VStack(spacing: 6) {
        HStack {
          Text(L10n.string("Power saving", language: language))
          Spacer()
          Text(L10n.string("Performance", language: language))
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(.secondary)

        Slider(
          value: powerSavingSelection,
          in: 0...Double(ServerConfiguration.powerSavingLimitOptionsGBps.count - 1),
          step: 1
        )
        .accessibilityLabel(L10n.string("SSD read limit", language: language))
        .accessibilityValue(powerSavingLimitLabel(configuration.powerSavingLimitGBps))

        GeometryReader { geometry in
          ZStack(alignment: .topLeading) {
            ForEach(
              Array(ServerConfiguration.powerSavingLimitOptionsGBps.enumerated()),
              id: \.offset
            ) { index, limit in
              let frame = powerSavingLegendFrame(
                index: index,
                count: ServerConfiguration.powerSavingLimitOptionsGBps.count,
                totalWidth: geometry.size.width
              )
              Text(powerSavingLimitLabel(limit))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(
                  width: frame.width,
                  alignment: index == 0
                    ? .leading
                    : index == ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
                      ? .trailing : .center
                )
                .offset(x: frame.minX)
            }
          }
        }
        .frame(height: 16)
        .accessibilityHidden(true)
      }
    }
    .appCard()
  }

  private var powerSavingSelection: Binding<Double> {
    Binding(
      get: {
        Double(
          ServerConfiguration.powerSavingLimitOptionsGBps.firstIndex {
            $0 == configuration.powerSavingLimitGBps
          } ?? ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
        )
      },
      set: { value in
        let index = min(
          max(Int(value.rounded()), 0),
          ServerConfiguration.powerSavingLimitOptionsGBps.count - 1
        )
        configuration.powerSavingLimitGBps =
          ServerConfiguration.powerSavingLimitOptionsGBps[index]
      }
    )
  }

  private func powerSavingLimitLabel(_ limit: Double?) -> String {
    guard let limit else { return L10n.string("Unlimited", language: language) }
    if limit == 0.5 { return L10n.string("500 MB/s", language: language) }
    return L10n.string("%lld GB/s", language: language, Int64(limit))
  }
}

struct ModelAdvancedView: View {
  @AppStorage(ExpertCacheControl.slotsPreferenceKey) private var editCachesInSlots = true
  @Binding var settings: ModelAdvancedSettings
  @Binding var alias: String
  let aliasError: String?
  let settingsLocked: Bool
  let modelURL: URL?
  @State var memoryProfile: MemoryPlanningProfile?
  let mtpAvailable: Bool
  let dsparkAvailable: Bool
  let modelKind: ModelKind
  let language: AppLanguage
  var onBack: () -> Void = {}

  var body: some View {
    VStack(spacing: 0) {
      ViewThatFits(in: .horizontal) {
        HStack(spacing: 24) {
          modelHeading.fixedSize()
          Spacer(minLength: 16)
          memoryOverview.fixedSize()
        }
        VStack(alignment: .leading, spacing: 12) {
          modelHeading
          memoryOverview.frame(maxWidth: .infinity, alignment: .trailing)
        }
      }
      .padding(.horizontal, 40)
      .padding(.vertical, 12)
      Divider()
      settingsForm
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
    .task(id: modelURL) {
      memoryProfile = modelURL.flatMap { MemoryPlanningProfile.load(at: $0, kind: modelKind) }
    }
  }

  private var modelHeading: some View {
    HStack(spacing: 12) {
      Button(action: onBack) {
        Label(L10n.string("Back to Model", language: language), systemImage: "chevron.backward")
      }
      .buttonStyle(TertiaryIconButtonStyle())
      .accessibilityLabel(L10n.string("Back to Model", language: language))
      .help(L10n.string("Back to Model", language: language))
      Text(modelKind.displayName)
        .font(.title2.bold())
        .accessibilityAddTraits(.isHeader)
    }
  }

  private var settingsForm: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        SectionHeader(title: L10n.string("Model", language: language))
        VStack(spacing: 0) {
          SettingRow(
            "Alias",
            hint: impactHint("Alias", "Optional request name for this model. Changes are saved automatically."),
            language: language
          ) {
            VStack(alignment: .trailing, spacing: 6) {
              TextField(
                L10n.string("Optional Alias", language: language),
                text: $alias
              )
              .appInput(width: 340)
              .accessibilityLabel(
                L10n.string("Alias for %@", language: language, modelKind.displayName))

              if let aliasError {
                Label(aliasError, systemImage: "exclamationmark.triangle.fill")
                  .font(.caption)
                  .foregroundStyle(.red)
                  .accessibilityLabel(
                    L10n.string("Error: %@", language: language, aliasError))
              }
            }
          }
        }
        .appCard()
        .disabled(settingsLocked)

        SectionHeader(title: L10n.string("Generate", language: language))
          .padding(.top, 12)
        VStack(spacing: 0) {
          integerField(
            "Max tokens",
            hint: "Default token limit for each request.",
            value: $settings.defaultMaxTokens
          )
          if modelKind.descriptor.supports("adaptiveSampling") {
            Divider()
            toggleField(
              "Use adaptive sampling",
              hint: "Chooses sampling values for chat or thinking. Turn off to use the values below.",
              value: qwenAdaptiveSampling
            )
          }
          if modelKind.descriptor.editableSettings.contains("sampling") {
            Group {
              Divider()
              doubleField(
                "Temperature",
                hint: "A higher value increases output variation.",
                value: $settings.defaultTemperature
              )
              Divider()
              doubleField(
                "Top P",
                hint: "A lower value reduces the candidate token range.",
                value: $settings.defaultTopP
              )
              Divider()
              integerField(
                "Top K",
                hint: "0 disables Top K.",
                value: $settings.defaultTopK
              )
            }
            .disabled(modelKind.descriptor.supports("adaptiveSampling") && qwenAdaptiveSampling.wrappedValue)
          }
          if modelKind.descriptor.supports("approximation") {
            Divider()
            toggleField(
              "Use approximate mode",
              hint: "Off uses all selected experts. On computes one fewer expert and may reduce quality. Unavailable with DSpark or MTP.",
              value: approximationEnabled
            )
            .disabled(settings.dsparkEnabled || mtpEnabled.wrappedValue)
          }
        }
        .appCard()
        .disabled(settingsLocked)

        SectionHeader(title: L10n.string("Runtime", language: language))
          .padding(.top, 12)
        VStack(spacing: 0) {
          ModelSettingsResetRow(settings: $settings, modelKind: modelKind,
            settingsLocked: settingsLocked, language: language)
          Divider()
          cacheMemoryField(.expert, minimum: modelKind == .qwen3_8FlashNext ? 10 : 6)
          Divider()
          SettingRow(
            "Expert cache eviction",
            hint: impactHint("Expert cache eviction", "Route-aware cache remembers expert usage and adjusts memory across layers."),
            language: language
          ) {
            Picker(L10n.string("Expert cache eviction", language: language), selection: expertEvictionPolicy) {
              Text(L10n.string("LRU (recently used)", language: language)).tag("lru")
              Text(L10n.string("LFU (frequently used)", language: language)).tag("lfu")
              Text(L10n.string("Route-aware", language: language)).tag("route")
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(width: 240)
          }
          Divider()
          integerField(
            "Read workers",
            hint:
              "Number of workers that read expert blobs at the same time. The recommended value is 4.",
            value: $settings.readWorkers
          )
          Divider()
          integerField(
            "Prefetch read workers",
            hint: "Number of workers that read expert data ahead of time. The default is 2.",
            value: prefetchReadWorkers
          )
          Divider()
          integerField(
            "MLX memory guideline GiB",
            hint: "0 selects the automatic guideline. This does not reserve memory.",
            value: $settings.memoryLimitGiB
          )
          Divider()
          integerField(
            "Prefill step size",
            hint: "0 selects 128, 256, or 1024 based on the prompt length.",
            value: $settings.prefillStepSize
          )
          Divider()
          integerField(
            "MoE prefill step size",
            hint: "Number of input tokens processed together by MoE. 0 selects automatically.",
            value: moePrefillStepSize
          )
          if modelKind.descriptor.supports("anePrefill") || modelKind.descriptor.supports("deepseekANEPrefill") {
            Divider()
            doubleField(
              "ANE Prefill share",
              hint:
                "Share of query projection output channels assigned to ANE. Use 0 for GPU only and 1 for ANE only. The default and recommended value is 0.",
              value: anePrefillRatio
            )
          }
          if modelKind.descriptor.supports("layerMajorPrefill") {
            Divider()
            toggleField(
              "Use layer-major prefill",
              hint: "Loads routed experts by layer during prefill.",
              value: $settings.layerMajorPrefill
            )
          }
          if modelKind.descriptor.supports("readyExpertDecode") {
            Divider()
            toggleField("Compute experts as they load",
              hint: "Starts available expert calculations while other experts are still loading.",
              value: optionalToggle(\.readyExpertDecode, defaultValue: true))

          }
          if modelKind.descriptor.supports("batchedExpertPrefill") {
            Divider()
            toggleField("Batch expert calculations",
              hint: "Processes the experts for an input batch together.",
              value: optionalToggle(\.batchedExpertPrefill, defaultValue: true,
                suppressed: qwenFlashWavesActive))
            .disabled(!settings.layerMajorPrefill || qwenFlashWavesActive)
          }
          if modelKind.descriptor.supports("nextLayerPrefetch") {
            Divider()
            toggleField("Read the next expert layer ahead",
              hint: "Reads the next layer while the current layer runs. Uses extra memory.",
              value: optionalToggle(\.nextLayerPrefetch, defaultValue: true,
                suppressed: qwenFlashWavesActive))
            .disabled(!settings.layerMajorPrefill || settings.batchedExpertPrefill == false || qwenFlashWavesActive)
          }
          if modelKind.descriptor.supports("packedKVCache") {
            Divider()
            toggleField("Compress attention cache",
              hint: "Reduces attention cache memory. Qwen uses 8-bit storage and may produce different output.",
              value: optionalToggle(\.packedKVCache, defaultValue: false))

          }
          if modelKind.descriptor.supports("packedIndexCache") {
            Divider()
            toggleField("Compress attention index",
              hint: "Uses 4-bit index storage. Qwen may select different attention positions.",
              value: optionalToggle(\.packedIndexCache, defaultValue: false))

          }
          if modelKind.descriptor.supports("candidateIndex") {
            Divider()
            toggleField("Search candidate positions only",
              hint: "Limits later attention searches to the candidates selected by the first indexer.",
              value: optionalToggle(\.candidateIndex, defaultValue: false))

          }
          if modelKind.descriptor.supports("cedPrefill") {
            Divider()
            toggleField("Reduce decoder prefill work",
              hint: "Processes the decoder tail needed to rebuild its attention windows.",
              value: optionalToggle(\.cedPrefill, defaultValue: false))
            .disabled(!settings.layerMajorPrefill || settings.dsparkEnabled)
          }
          if modelKind.descriptor.supports("deepseekANEPrefill") {
            Divider()
            toggleField("Use ANE for prefill",
              hint: "Shares query projection work with ANE. Falls back to GPU when unavailable. May change rounding.",
              value: optionalToggle(\.deepSeekANEPrefill, defaultValue: false))

          }
          if modelKind.descriptor.supports("groupedExperts") {
            Divider()
            toggleField(
              "Prefill acceleration",
              hint:
                "Speeds up prompt processing. Requires layer-major prefill with MTP off. Changes apply on next load.",
              value: qwenGroupedExperts
            )
            .disabled(!settings.layerMajorPrefill || mtpEnabled.wrappedValue || qwenFlashWavesActive)
          }
          if modelKind.descriptor.editableSettings.contains("prefillThreshold") {
            Divider()
            integerField(
              "Layer-major prefill threshold",
              hint:
                "Minimum uncached prompt tokens required for layer-major prefill. The default is 1024.",
              value: layerMajorPrefillThreshold
            )
            .disabled(!settings.layerMajorPrefill)
          }
          if modelKind.descriptor.supports("promptCache") {
            Divider()
            SettingRow(
              "Prompt cache",
              hint: impactHint("Prompt cache", "Memory reuses prompts until the model unloads. Disk also keeps them after restart. Off processes each prompt again."),
              language: language
            ) {
              Picker(L10n.string("Prompt cache", language: language), selection: promptCacheMode) {
                ForEach(PromptCacheMode.allCases) { mode in
                  Text(L10n.string(mode.localizationKey, language: language)).tag(mode)
                }
              }
              .labelsHidden()
              .pickerStyle(.segmented)
              .frame(width: 260)
            }
            if promptCacheMode.wrappedValue != .off {
              Divider()
              integerField(
                "Prompt cache entries",
                hint: "Number of linear conversations to keep. The recommended value is 2.",
                value: $settings.promptCacheEntries
              )
              Divider()
              integerField(
                "Prompt cache GiB",
                hint: "Memory limit for all prompt caches. The recommended value is 8.",
                value: $settings.promptCacheMemoryGiB
              )
            }
          }
          Divider()
          SettingRow(
            "Warmup prompt",
            hint: impactHint("Warmup prompt", "Optional UTF-8 prompt file path"),
            language: language
          ) {
            TextField(
              L10n.string("Optional UTF-8 prompt file path", language: language),
              text: $settings.warmupPromptPath
            )
            .appInput(width: 340)
          }
          if modelKind == .qwen3_8FlashNext {
            ForEach(QwenOptimization.allCases.filter { $0 != .mtpPolicy }) { feature in
              Divider()
              toggleField(feature.title, hint: feature.hint,
                value: optionalToggle(feature.keyPath, defaultValue: false))
            }
          }
          if modelKind.descriptor.supports("mtp") {
            Divider()
            toggleField(
              "Use MTP",
              hint: mtpAvailable
                ? "MTP uses speculative decoding. Text may arrive in short bursts."
                : "Install the MTP files before enabling this setting.",
              value: mtpEnabled
            )
            .disabled(!mtpAvailable)
            Divider()
            cacheMemoryField(.mtp, minimum: 10)
            .disabled(!mtpEnabled.wrappedValue || !mtpAvailable)
            Divider()
            toggleField(QwenOptimization.mtpPolicy.title, hint: QwenOptimization.mtpPolicy.hint,
              value: optionalToggle(\.qwenMTPPolicy, defaultValue: false))
              .disabled(!mtpEnabled.wrappedValue || !mtpAvailable)
            if settings.qwenMTPPolicy == true {
              Divider()
              qwenIntegerChoice("MTP draft tokens", range: 1...5, key: \.qwenMTPDraftTokens)
              Divider()
              qwenIntegerChoice("Zero-acceptance rounds before stopping MTP", range: 1...32,
                key: \.qwenMTPZeroAcceptanceLimit)
            }
          }
          if modelKind.descriptor.editableSettings.contains("kvCachePrecision") {
            Divider()
            toggleField(
              "Use BF16 KV cache",
              hint: "Stores the KV cache in BF16 format.",
              value: $settings.bf16KVCache
            )
          }
          if modelKind.descriptor.supports("dspark") {
            Divider()
            toggleField(
              "Use DSpark",
              hint: "Uses DSpark speculative decoding when it is installed.",
              value: $settings.dsparkEnabled
            )
            .disabled(!dsparkAvailable)
            Divider()
            cacheMemoryField(.dspark, minimum: 30)
            .disabled(!settings.dsparkEnabled || !dsparkAvailable)
            Divider()
            doubleField(
              "DSpark confidence threshold",
              hint:
                "0 keeps all draft tokens. A higher value rejects low-confidence draft tokens early.",
              value: $settings.dsparkConfidenceThreshold
            )
            .disabled(!settings.dsparkEnabled || !dsparkAvailable)
          }
        }
        .appCard()
        .disabled(settingsLocked)
        QwenFlashSettingsSection(settings: $settings, modelKind: modelKind,
          settingsLocked: settingsLocked, language: language)
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
  }


  private func qwenIntegerChoice(_ label: String, range: ClosedRange<Int>,
                                 key: WritableKeyPath<ModelAdvancedSettings, Int?>) -> some View {
    SettingRow(label, hint: L10n.string(QwenOptimization.mtpPolicy.hint, language: language), language: language) {
      Picker(L10n.string(label, language: language), selection: Binding(
        get: { settings[keyPath: key] ?? 2 }, set: { settings[keyPath: key] = $0 }
      )) {
        ForEach(Array(range), id: \.self) { Text(String($0)).tag($0) }
      }
      .labelsHidden()
      .frame(width: 100)
    }
    .disabled(!mtpEnabled.wrappedValue || !mtpAvailable)
  }

  private var qwenFlashWavesActive: Bool {
    modelKind == .qwen3_8FlashNext && settings.qwenFlashWavesEnabled
  }

  private func impactHint(_ label: String, _ hint: String) -> String {
    if qwenFlashWavesActive && ["Batch expert calculations", "Read the next expert layer ahead",
      "Prefill acceleration"].contains(label) {
      return L10n.string(QwenFlashCopy.wavesConflict, language: language)
    }
    if label == "Use MTP" && !mtpAvailable {
      return L10n.string("Install the MTP files before enabling this setting.", language: language)
    }
    if label == "Use DSpark" && !dsparkAvailable {
      return L10n.string("Install the DSpark files before enabling this setting.", language: language)
    }
    let key = AdvancedSettingImpact.key(for: label, modelKind: modelKind)
    if label == "Prompt cache entries" {
      return L10n.string(key, language: language, Int64(modelKind.descriptor.defaults.promptCacheEntries))
    }
    let recommendedSlots: Int? = switch label {
    case "Expert cache GiB": modelKind.descriptor.defaults.slots
    case "MTP expert cache GiB": 32
    case "DSpark expert cache GiB": 768
    default: nil
    }
    if let recommendedSlots {
      let gib = ExpertMemory.defaultGiB(slots: recommendedSlots, blobBytes: blobBytes)
      let formatted = String(format: "%.1f", locale: language.locale, gib)
      return L10n.string(key, language: language, formatted)
    }
    return L10n.string(key.isEmpty ? hint : key, language: language)
  }

  private var blobBytes: UInt64 {
    memoryProfile?.manifest.expertBlobSize ?? ExpertMemory.blobBytes(for: modelKind)
  }

  @ViewBuilder
  private func cacheMemoryField(_ control: ExpertCacheControl, minimum: Int) -> some View {
    let label = control.title
    let value = settings[keyPath: control.budgetKey]
      ?? ExpertMemory.legacyGiB(slots: control.legacySlots(in: settings), blobBytes: blobBytes)
    let capacity = try? ExpertMemory.capacity(gib: value, blobBytes: blobBytes, minimum: minimum)
    let capacityText = capacity.map { L10n.string("Capacity: %lld experts", language: language, Int64($0)) }
      ?? L10n.string("Expert cache memory is too small or invalid.", language: language)
    if !editCachesInSlots {
      let range = ExpertMemory.sliderRange(physicalMemory: ProcessInfo.processInfo.physicalMemory)
      VStack(alignment: .leading, spacing: 14) {
        HStack(alignment: .firstTextBaseline) {
          SettingLabel(label,
            hint: impactHint(label, "Expert blob capacity only; excludes common weights and temporary buffers. Rounded down to whole experts. Applies on next model load."),
            language: language)
          Spacer()
          Text(String(format: "%.1f GiB", locale: language.locale, value))
            .font(.body.weight(.semibold).monospacedDigit())
            .fixedSize()
        }
        VStack(spacing: 6) {
          CacheBudgetSlider(value: Binding(
            get: { ExpertMemory.sliderValue(value, in: range) },
            set: { control.setGiB($0, in: &settings, blobBytes: blobBytes, range: range) }
          ), range: range)
          .accessibilityLabel(L10n.string(label, language: language))
          .accessibilityValue(String(format: "%.1f GiB", locale: language.locale, value))
          .accessibilityHint(capacityText)
          Text(capacityText)
            .font(.caption)
            .foregroundStyle(capacity == nil ? .red : .secondary)
            .frame(maxWidth: .infinity, alignment: .trailing)
          Text(L10n.string("The maximum is this Mac's physical memory, not a safe allocation limit. Leave room for the system and other model memory.", language: language))
            .font(.caption).foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
      }
      .padding(.vertical, 8)
    } else {
      SettingRow(control.slotsTitle,
        hint: L10n.string("Enter the number of experts to retain. Applies on next model load.", language: language),
        language: language
      ) {
        VStack(alignment: .trailing, spacing: 6) {
          TextField(L10n.string(control.slotsTitle, language: language), value: Binding(
            get: { control.slots(in: settings, blobBytes: blobBytes) },
            set: { control.setSlots($0, in: &settings) }
          ), format: .number.grouping(.never))
            .labelsHidden()
            .appInput(width: 120)
            .accessibilityLabel(L10n.string(control.slotsTitle, language: language))
            .accessibilityHint(capacityText)
          Text(capacityText)
            .font(.caption)
            .foregroundStyle(capacity == nil ? .red : .secondary)
            .multilineTextAlignment(.trailing)
            .fixedSize(horizontal: false, vertical: true)
        }
      }
    }
  }

  private var memoryOverview: some View {
    HStack(spacing: 20) {
      Text(L10n.string("Estimated peak memory", language: language))
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
      ForEach([65_536, 131_072], id: \.self) { tokens in
        let estimate = memoryProfile?.estimate(settings, mtpAvailable: mtpAvailable,
          dsparkAvailable: dsparkAvailable, contextTokens: tokens)
        let value = estimate.map { String(format: "%.1f GiB", locale: language.locale,
          $0.total / ExpertMemory.gib) } ?? "— GiB"
        VStack(alignment: .trailing, spacing: 3) {
          Text(tokens == 65_536 ? "64K" : "128K")
            .font(.caption)
            .foregroundStyle(.secondary)
          Text(value)
            .font(.system(size: 16, weight: .bold))
            .monospacedDigit()
        }
        .fixedSize()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(L10n.string("Estimated peak memory for %lld tokens: %@",
          language: language, Int64(tokens), value))
      }
    }
    .help(L10n.string("Capacity estimate for an input-heavy context, including retained caches. 128K is extrapolated; actual usage may vary.", language: language))
  }

  private var layerMajorPrefillThreshold: Binding<Int> {
    Binding(
      get: { settings.layerMajorPrefillThreshold ?? 1_024 },
      set: { settings.layerMajorPrefillThreshold = $0 }
    )
  }

  private var anePrefillRatio: Binding<Double> {
    Binding(
      get: { settings.anePrefillRatio ?? 0 },
      set: { settings.anePrefillRatio = $0 }
    )
  }


  private var prefetchReadWorkers: Binding<Int> {
    Binding(get: { settings.prefetchReadWorkers ?? 2 }, set: { settings.prefetchReadWorkers = $0 })
  }

  private var moePrefillStepSize: Binding<Int> {
    Binding(get: { settings.moePrefillStepSize ?? 0 }, set: { settings.moePrefillStepSize = $0 })
  }

  private func optionalToggle(_ keyPath: WritableKeyPath<ModelAdvancedSettings, Bool?>,
                              defaultValue: Bool, suppressed: Bool = false) -> Binding<Bool> {
    Binding(get: { !suppressed && (settings[keyPath: keyPath] ?? defaultValue) },
            set: { if !suppressed { settings[keyPath: keyPath] = $0 } })
  }

  private var approximationEnabled: Binding<Bool> {
    Binding(get: { settings.approximationEnabled ?? false }, set: { settings.approximationEnabled = $0 })
  }

  private var qwenAdaptiveSampling: Binding<Bool> {
    Binding(get: { settings.qwenAdaptiveSampling ?? true }, set: { settings.qwenAdaptiveSampling = $0 })
  }

  private var expertEvictionPolicy: Binding<String> {
    Binding(get: {
      settings.routeAwareExpertCache == true ? "route" : (settings.recentExpertCache ?? true ? "lru" : "lfu")
    }, set: { policy in
      settings.routeAwareExpertCache = policy == "route"
      if policy != "route" { settings.recentExpertCache = policy == "lru" }
    })
  }

  private var promptCacheMode: Binding<PromptCacheMode> {
    Binding(
      get: { settings.promptCacheMode ?? .memory },
      set: { settings.promptCacheMode = $0 }
    )
  }


  private var qwenGroupedExperts: Binding<Bool> {
    Binding(
      get: { !qwenFlashWavesActive && (settings.qwenGroupedExperts ?? true) },
      set: { if !qwenFlashWavesActive { settings.qwenGroupedExperts = $0 } }
    )
  }

  private var mtpEnabled: Binding<Bool> {
    Binding(
      get: { mtpAvailable && settings.mtpEnabled == true },
      set: { enabled in
        settings.mtpEnabled = enabled
      }
    )
  }

  private var mtpSlots: Binding<Int> {
    Binding(
      get: { settings.mtpSlots ?? 32 },
      set: { settings.mtpSlots = $0 }
    )
  }

  private func integerField(_ label: String, hint: String, value: Binding<Int>) -> some View {
    SettingRow(label, hint: impactHint(label, hint), language: language) {
      TextField(
        L10n.string(label, language: language),
        value: value,
        format: .number.grouping(.never)
      )
      .labelsHidden()
      .appInput(width: 120)
    }
  }

  private func doubleField(_ label: String, hint: String, value: Binding<Double>) -> some View {
    SettingRow(label, hint: impactHint(label, hint), language: language) {
      TextField(
        L10n.string(label, language: language),
        value: value,
        format: .number.precision(.fractionLength(0...6))
      )
      .labelsHidden()
      .appInput(width: 120)
    }
  }

  private func toggleField(_ label: String, hint: String, value: Binding<Bool>) -> some View {
    SettingRow(label, hint: impactHint(label, hint), language: language) {
      Toggle(L10n.string(label, language: language), isOn: value)
        .labelsHidden()
        .accessibilityLabel(L10n.string(label, language: language))
    }
  }
}

func powerSavingLegendFrame(index: Int, count: Int, totalWidth: CGFloat) -> CGRect {
  let width = totalWidth / CGFloat(count - 1)
  let nodeX = CGFloat(index) * width
  let minX = min(max(nodeX - width / 2, 0), totalWidth - width)
  return CGRect(x: minX, y: 0, width: width, height: 0)
}

private struct LogsView: View {
  @ObservedObject var server: ServerController
  @Binding var configuration: ServerConfiguration
  let language: AppLanguage

  var body: some View {
    VStack(alignment: .leading, spacing: 14) {
      SettingRow(
        "Log level",
        hint:
          "Applies the next time the server starts. Debug logs the full JSON body of every request and may contain sensitive content.",
        language: language
      ) {
        Picker(
          L10n.string("Log level", language: language),
          selection: $configuration.logLevel
        ) {
          ForEach(ServerLogLevel.allCases) { level in
            Text(L10n.string(level.localizationKey, language: language)).tag(level)
          }
        }
        .labelsHidden()
        .pickerStyle(.segmented)
        .frame(width: 240, alignment: .trailing)
      }
      .disabled(server.isActive)

      SectionHeader(title: L10n.string("Server log", language: language))
      ScrollView {
        Text(
          server.log.isEmpty
            ? L10n.string(
              "The log will appear here after the server starts.", language: language)
            : server.log
        )
        .font(.system(.callout, design: .monospaced))
        .foregroundStyle(server.log.isEmpty ? .secondary : .primary)
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .topLeading)
      }
      .frame(maxWidth: .infinity, maxHeight: .infinity)
      .appCard()
      .accessibilityLabel(L10n.string("Server log", language: language))
    }
    .frame(maxWidth: AppLayout.contentWidth)
    .frame(maxWidth: .infinity, maxHeight: .infinity)
    .padding(.horizontal, 40)
    .padding(.vertical, 24)
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }
}

struct SettingsView: View {
  @AppStorage(ExpertCacheControl.slotsPreferenceKey) private var editCachesInSlots = true
  @Binding var languageCode: String
  let language: AppLanguage
  @ObservedObject var appUpdater: AppUpdater

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        SectionHeader(title: L10n.string("Preferences", language: language))
          .padding(.top, 12)

        VStack(alignment: .leading, spacing: 0) {
          SettingRow(
            "Language",
            hint: "Select the language used by the app.",
            language: language
          ) {
            Picker(L10n.string("Language", language: language), selection: $languageCode) {
              ForEach(AppLanguage.allCases) { option in
                Text(option.displayName(language: language)).tag(option.rawValue)
              }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .frame(minWidth: 180, alignment: .trailing)
          }
          Divider()
          SettingRow("Edit expert caches in slots",
            hint: "Use integer slot inputs instead of GiB sliders for expert, MTP, and DSpark caches. Switching does not change saved capacity.",
            language: language
          ) {
            Toggle(L10n.string("Edit expert caches in slots", language: language), isOn: $editCachesInSlots)
              .labelsHidden()
              .toggleStyle(.switch)
              .accessibilityLabel(L10n.string("Edit expert caches in slots", language: language))
          }

        }
        .appCard()

        SectionHeader(title: L10n.string("Updates", language: language))
          .padding(.top, 12)

        VStack(alignment: .leading, spacing: 0) {
          SettingRow(
            "Software Updates",
            hint: "Check GitHub Releases for a newer app version.",
            language: language
          ) {
            Button(action: appUpdater.checkForUpdates) {
              Label(
                L10n.string("Check for Updates…", language: language),
                systemImage: "arrow.clockwise"
              )
            }
            .disabled(!appUpdater.canCheckForUpdates)
          }

          Divider()
          SettingRow(
            "Update source",
            hint: "Stable includes official releases. Dev also includes test versions.",
            language: language
          ) {
            Picker(L10n.string("Update source", language: language), selection: $appUpdater.channel) {
              ForEach(UpdateChannel.allCases) { channel in
                Text(channel.rawValue).tag(channel)
              }
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            .frame(width: 220, alignment: .trailing)
            .disabled(!appUpdater.canCheckForUpdates)
          }

          Divider()
          SettingRow(
            "Automatically check for updates",
            hint: "Check for new versions in the background.",
            language: language
          ) {
            Toggle(
              L10n.string("Automatically check for updates", language: language),
              isOn: Binding(
                get: { appUpdater.automaticallyChecksForUpdates },
                set: { appUpdater.setAutomaticallyChecksForUpdates($0) })
            )
            .labelsHidden()
            .toggleStyle(.switch)
          }
        }
        .appCard()
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }
}

struct AboutView: View {
  let language: AppLanguage

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 14) {
        HStack(spacing: 20) {
          Image(nsImage: NSImage(named: NSImage.applicationIconName) ?? NSImage())
            .resizable()
            .interpolation(.high)
            .frame(width: 76, height: 76)
            .accessibilityHidden(true)
          VStack(alignment: .leading, spacing: 5) {
            Text("Whallm")
              .font(.title.bold())
            Text(L10n.string("Local DeepSeek inference from SSD.", language: language))
              .font(.title3)
              .foregroundStyle(.secondary)
            Text(
              L10n.string(
                "Version %@ · build %@",
                language: language,
                appVersion,
                buildVersion
              )
            )
            .font(.callout.monospacedDigit())
            .foregroundStyle(.tertiary)
            .textSelection(.enabled)
          }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCard(padding: 22)

        SectionHeader(title: L10n.string("Project", language: language))
          .padding(.top, 12)

        VStack(spacing: 0) {
          projectLink(
            title: "GitHub Repository",
            note: "Source, issues, and roadmap",
            icon: "chevron.left.forwardslash.chevron.right",
            url: "https://github.com/yanun0323/Whallm"
          )
          Divider()
          projectLink(
            title: "Releases",
            note: "Download the latest macOS app",
            icon: "shippingbox",
            url: "https://github.com/yanun0323/Whallm/releases"
          )
          Divider()
          projectLink(
            title: "Documentation",
            note: "Setup, model management, and API usage",
            icon: "book.closed",
            url: "https://github.com/yanun0323/Whallm#readme"
          )
          Divider()
          projectLink(
            title: "Report an Issue",
            note: "Report bugs and request features on GitHub",
            icon: "exclamationmark.bubble",
            url: "https://github.com/yanun0323/Whallm/issues"
          )
        }
        .appCard()

        SectionHeader(title: L10n.string("License", language: language))
          .padding(.top, 12)

        VStack(alignment: .leading, spacing: 6) {
          Label(
            L10n.string("MIT License", language: language),
            systemImage: "point.3.connected.trianglepath.dotted"
          )
          .font(.headline)
          Text(
            L10n.string(
              "Copyright © 2026 Yanun. See the LICENSE file in the repository for the full text.",
              language: language
            )
          )
          .font(.callout)
          .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCard()
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
  }

  private var appVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "-"
  }

  private var buildVersion: String {
    Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "-"
  }

  private func projectLink(title: String, note: String, icon: String, url: String) -> some View {
    Link(destination: URL(string: url)!) {
      HStack(spacing: 14) {
        Image(systemName: icon)
          .font(.title3)
          .foregroundStyle(.secondary)
          .frame(width: 28)
          .accessibilityHidden(true)
        VStack(alignment: .leading, spacing: 2) {
          Text(L10n.string(title, language: language))
            .font(.body.weight(.semibold))
          Text(L10n.string(note, language: language))
            .font(.callout)
            .foregroundStyle(.secondary)
        }
        Spacer()
        Image(systemName: "arrow.up.right.square")
          .foregroundStyle(.secondary)
          .accessibilityHidden(true)
      }
      .padding(.vertical, 9)
      .contentShape(Rectangle())
    }
    .buttonStyle(.plain)
    .accessibilityLabel(L10n.string(title, language: language))
    .accessibilityHint(L10n.string(note, language: language))
  }
}

private struct MetricView: View {
  let state: ServerController.State
  let performance: LivePerformance
  let history: PerformanceHistory
  let language: AppLanguage
  let clearHistory: () -> Void

  var body: some View {
    ScrollView {
      VStack(alignment: .leading, spacing: 18) {
        SectionHeader(title: localized("Active Now"))
        activeCard

        HStack(alignment: .center) {
          SectionHeader(title: localized("Serving Stats"))
          Spacer()
          Button(action: clearHistory) {
            Label(localized("Clear metric history"), systemImage: "trash")
          }
          .buttonStyle(TertiaryIconButtonStyle(color: .red))
          .disabled(history.isEmpty && performance.appMemory == nil && !performance.memoryResetFailed)
          .help(localized("Clear metric history"))
          .accessibilityLabel(localized("Clear metric history"))
        }
        .padding(.top, 10)

        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 3),
          spacing: 14
        ) {
          metricCard(.inputTokens)
          metricCard(.outputTokens)
          metricCard(.cacheHitRate)
        }

        SectionHeader(title: localized("Average Speed"))
          .padding(.top, 10)

        LazyVGrid(
          columns: Array(repeating: GridItem(.flexible(), spacing: 14), count: 2),
          spacing: 14
        ) {
          metricCard(.prefillTokensPerSecond)
          metricCard(.decodeTokensPerSecond)
        }

        SectionHeader(title: localized("System"))
          .padding(.top, 10)
        systemCard
      }
      .frame(maxWidth: AppLayout.contentWidth)
      .frame(maxWidth: .infinity)
      .padding(.horizontal, 40)
      .padding(.vertical, 24)
    }
    .background(AppTheme.pageBackground)
    .accessibilityElement(children: .contain)
    .environment(\.locale, language.locale)
  }

  @ViewBuilder
  private func metricCard(_ metric: PerformanceMetric) -> some View {
    switch metric {
    case .inputTokens:
      singleValueMetricCard(
        metric,
        label: localized("Live"),
        value: formattedValue(performance.snapshot[metric], for: metric, live: true)
      )
    case .outputTokens:
      singleValueMetricCard(
        metric,
        label: localized("Accumulate"),
        value: performance.hasStatus ? performance.accumulatedOutputTokens.formatted() : "-"
      )
    default:
      historicalMetricCard(metric)
    }
  }

  private func singleValueMetricCard(
    _ metric: PerformanceMetric,
    label: String,
    value: String
  ) -> some View {
    VStack(spacing: 14) {
      Text(localized(metricTitle(metric)))
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
      VStack(spacing: 2) {
        Text(label)
          .font(.caption)
          .foregroundStyle(.secondary)
        Text(value)
          .font(.title2.weight(.semibold).monospacedDigit())
          .lineLimit(1)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 126)
    .appCard(padding: 18)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel("\(localized(metricTitle(metric)))。\(label)：\(value)")
  }

  private func historicalMetricCard(_ metric: PerformanceMetric) -> some View {
    let live = formattedValue(performance.snapshot[metric], for: metric, live: true)
    let statistics = history[metric]
    let maximum = statistics.map { formattedValue($0.maximum, for: metric) } ?? "-"
    let p95 = statistics.map { formattedValue($0.p95, for: metric) } ?? "-"
    return VStack(spacing: 14) {
      Text(localized(metricTitle(metric)))
        .font(.callout.weight(.semibold))
        .foregroundStyle(.secondary)
        .multilineTextAlignment(.center)
      Text(live)
        .font(.title2.weight(.semibold).monospacedDigit())
        .lineLimit(1)
      HStack(spacing: 16) {
        statisticLabel(localized("Maximum"), value: maximum)
        Divider().frame(height: 28)
        statisticLabel("P95", value: p95)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 126)
    .appCard(padding: 18)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      "\(localized(metricTitle(metric)))。\(localized("Live"))：\(live)。"
        + "\(localized("Maximum"))：\(maximum)。P95：\(p95)"
    )
  }

  private func statisticLabel(_ title: String, value: String) -> some View {
    VStack(spacing: 2) {
      Text(title)
        .font(.caption)
        .foregroundStyle(.secondary)
      Text(value)
        .font(.callout.monospacedDigit())
        .lineLimit(1)
    }
    .frame(maxWidth: .infinity)
  }

  private var activeCard: some View {
    HStack(spacing: 12) {
      Circle()
        .fill(statusColor)
        .frame(width: 9, height: 9)
        .accessibilityHidden(true)
      VStack(alignment: .leading, spacing: 3) {
        Text(activeModelLabel)
          .font(.headline)
          .lineLimit(1)
        Text(localizedStateLabel)
          .font(.callout)
          .foregroundStyle(.secondary)
      }
      Spacer()
      if performance.dsparkEnabled {
        Text(
          L10n.string(
            "DSpark · %@ accepted · %@ tokens per round",
            language: language,
            performance.dsparkAcceptanceRate.formatted(
              .percent.precision(.fractionLength(1))),
            performance.dsparkAverageAcceptedLength.formatted(
              .number.precision(.fractionLength(1)))
          )
        )
        .font(.callout.monospacedDigit())
        .foregroundStyle(.secondary)
      }
    }
    .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
    .appCard()
    .accessibilityElement(children: .combine)
  }

  private var systemCard: some View {
    VStack(spacing: 0) {
      HStack {
        Text(localized("Metric"))
        Spacer()
        Text(localized("Live")).frame(width: 120, alignment: .trailing)
        Text(localized("Maximum")).frame(width: 120, alignment: .trailing)
        Text("P95").frame(width: 120, alignment: .trailing)
      }
      .font(.callout.weight(.semibold))
      .foregroundStyle(.secondary)
      .padding(.bottom, 12)

      ForEach(
        [
          PerformanceMetric.memoryUsage,
          .ssdReadSpeed,
          .firstTokenWaitTime,
          .completionTime,
        ]
      ) { metric in
        Divider()
        metricRow(metric)
      }
      Text(localized("Memory uses App + inference process physical footprint (inference process only for standalone servers): every 10 ms during model work, every second while idle. Maximum is retained since server start or clearing metric history, including loading and idle time. Memory P95 uses the slower Status refresh samples. Brief peaks may be missed; — means unavailable."))
        .font(.caption).foregroundStyle(.secondary)
        .fixedSize(horizontal: false, vertical: true)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.top, 12)
      if performance.memoryResetFailed {
        Text(localized("Unable to reset the memory maximum. Try clearing metric history again."))
          .font(.caption).foregroundStyle(.red)
          .frame(maxWidth: .infinity, alignment: .leading)
      }
    }
    .appCard()
  }

  private func metricRow(_ metric: PerformanceMetric) -> some View {
    let value =
      metric == .firstTokenWaitTime
      ? performance.liveFirstTokenWaitTime : performance.snapshot[metric]
    let live = formattedValue(value, for: metric, live: true)
    let statistics = history[metric]
    let maximum = metric == .memoryUsage
      ? performance.memoryMaximum.map { formattedValue($0, for: metric) } ?? "—"
      : statistics.map { formattedValue($0.maximum, for: metric) } ?? "-"
    let p95 = statistics.map { formattedValue($0.p95, for: metric) } ?? "-"
    return HStack {
      Text(localized(metricTitle(metric)))
        .font(.body.weight(.medium))
      Spacer()
      Text(live).frame(width: 120, alignment: .trailing)
      Text(maximum).frame(width: 120, alignment: .trailing)
      Text(p95).frame(width: 120, alignment: .trailing)
    }
    .font(.body.monospacedDigit())
    .padding(.vertical, 10)
    .accessibilityElement(children: .ignore)
    .accessibilityLabel(
      "\(localized(metricTitle(metric)))。\(localized("Live"))：\(live)。"
        + "\(localized("Maximum"))：\(maximum)。P95：\(p95)"
    )
  }

  private func localized(_ key: String) -> String {
    L10n.string(key, language: language)
  }

  private var localizedStateLabel: String {
    switch state {
    case .stopped: localized("Stopped")
    case .starting: localized("Starting")
    case .running: localized("Running")
    case .stopping: localized("Stopping")
    case .failed: localized("Start failed")
    }
  }

  private var activeModelLabel: String {
    if let loadedModel = performance.loadedModel { return loadedModel }
    if let loadingModel = performance.loadingModel {
      return L10n.string("Loading %@", language: language, loadingModel)
    }
    return localized("No model loaded")
  }

  private func metricTitle(_ metric: PerformanceMetric) -> String {
    switch metric {
    case .prefillTokensPerSecond: "Prefill Tok/s"
    case .decodeTokensPerSecond: "Decode Tok/s"
    case .inputTokens: "Input Tokens"
    case .outputTokens: "Output Tokens"
    case .memoryUsage: "Memory usage"
    case .ssdReadSpeed: "SSD read speed"
    case .cacheHitRate: "Expert cache hit rate"
    case .firstTokenWaitTime: "First Token wait time"
    case .completionTime: "Completion time"
    }
  }

  private func formattedValue(
    _ value: Double,
    for metric: PerformanceMetric,
    live: Bool = false
  ) -> String {
    if !value.isFinite { return "—" }
    if value == 0 { return "-" }
    if live && !hasLiveValue(metric, value: value) { return "-" }
    switch metric {
    case .prefillTokensPerSecond, .decodeTokensPerSecond:
      return value.formatted(.number.precision(.fractionLength(1)))
    case .inputTokens, .outputTokens:
      return Int64(value.rounded()).formatted()
    case .memoryUsage:
      return ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .memory)
    case .ssdReadSpeed:
      return "\(ByteCountFormatter.string(fromByteCount: Int64(value), countStyle: .file))/s"
    case .cacheHitRate:
      return value.formatted(.percent.precision(.fractionLength(1)))
    case .firstTokenWaitTime, .completionTime:
      if value < 1 {
        return "\((value * 1_000).formatted(.number.precision(.fractionLength(0)))) ms"
      }
      return "\(value.formatted(.number.precision(.fractionLength(2)))) s"
    }
  }

  private func hasLiveValue(_ metric: PerformanceMetric, value: Double) -> Bool {
    guard performance.hasStatus else { return false }
    switch metric {
    case .memoryUsage:
      return value > 0
    case .prefillTokensPerSecond, .firstTokenWaitTime:
      return value > 0
    case .inputTokens, .outputTokens, .decodeTokensPerSecond, .completionTime:
      return performance.snapshot.inputTokens > 0
    case .ssdReadSpeed, .cacheHitRate:
      return true
    }
  }

  private var statusColor: Color {
    switch state {
    case .running: .green
    case .failed: .red
    case .starting, .stopping: .orange
    case .stopped: .secondary
    }
  }

}

private struct SettingLabel: View {
  let title: String
  let hint: String
  let language: AppLanguage

  init(_ title: String, hint: String, language: AppLanguage) {
    self.title = title
    self.hint = hint
    self.language = language
  }

  var body: some View {
    VStack(alignment: .leading, spacing: 2) {
      Text(L10n.string(title, language: language))
        .font(.body.weight(.medium))
      Text(L10n.string(hint, language: language))
        .font(.callout)
        .foregroundStyle(.secondary)
    }
    .help(L10n.string(hint, language: language))
    .accessibilityElement(children: .combine)
  }
}

struct SettingRow<Value: View>: View {
  let title: String
  let hint: String
  let language: AppLanguage
  let value: Value

  init(
    _ title: String,
    hint: String,
    language: AppLanguage,
    @ViewBuilder value: () -> Value
  ) {
    self.title = title
    self.hint = hint
    self.language = language
    self.value = value()
  }

  var body: some View {
    HStack(alignment: .center, spacing: 32) {
      SettingLabel(title, hint: hint, language: language)
        .frame(maxWidth: .infinity, alignment: .leading)

      value
        .fixedSize(horizontal: true, vertical: false)
    }
    .frame(maxWidth: .infinity)
    .padding(.vertical, 8)
  }
}

func resolvedChatModelName(savedName: String, models: [CatalogModel]) -> String? {
  guard !models.isEmpty else { return nil }
  return models.first {
    $0.requestName == savedName || $0.id == savedName
  }?.requestName ?? models.first?.requestName
}

@MainActor
final class ChatSession: ObservableObject {
  typealias Stream = (
    _ messages: [ChatMessage],
    _ baseURL: URL,
    _ apiKey: String,
    _ model: String,
    _ thinkingMode: String,
    _ seed: UInt32?,
    _ receive: @MainActor @escaping (ChatDelta) -> Void
  ) async throws -> Void

  @Published private(set) var messages: [ChatMessage]
  @Published private(set) var isSending = false
  @Published private(set) var errorMessage: String?

  private let defaults: UserDefaults
  private let stream: Stream
  private var generationTask: Task<Void, Never>?
  private var pendingDeltas: [ChatDelta] = []
  private var deltaFlushTask: Task<Void, Never>?

  init(
    defaults: UserDefaults = .standard,
    stream: @escaping Stream = { messages, baseURL, apiKey, model, thinkingMode, seed, receive in
      _ = try await ChatClient.stream(
        messages: messages,
        baseURL: baseURL,
        apiKey: apiKey,
        model: model,
        thinkingMode: thinkingMode,
        seed: seed,
        enableTestTool: false,
        receive: receive
      )
    }
  ) {
    self.defaults = defaults
    self.stream = stream
    messages = ChatHistory.load(defaults: defaults)
  }

  @discardableResult
  func send(
    text: String,
    configuration: ServerConfiguration,
    model: String,
    thinkingMode: String,
    language: AppLanguage,
    seed: UInt32? = nil
  ) -> Bool {
    let text = text.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty, !isSending, let baseURL = configuration.baseURL else { return false }

    let userMessage = ChatMessage(role: "user", content: text)
    messages.append(userMessage)
    errorMessage = nil
    isSending = true
    let requestMessages = messages
    let assistantID = UUID()
    messages.append(
      ChatMessage(id: assistantID, role: "assistant", content: "", modelName: model))
    save()

    generationTask = Task {
      defer {
        flushPendingDeltas(for: assistantID)
        save()
        isSending = false
        generationTask = nil
      }
      do {
        try await stream(
          requestMessages,
          baseURL,
          configuration.apiKey,
          model,
          thinkingMode,
          seed
        ) { delta in
          self.enqueue(delta, for: assistantID)
        }
      } catch {
        flushPendingDeltas(for: assistantID)
        if Task.isCancelled {
          removeEmptyAssistantMessage(id: assistantID)
          return
        }
        if let index = messages.firstIndex(where: { $0.id == assistantID }),
          messages[index].content.isEmpty,
          messages[index].reasoningContent.isEmpty,
          messages[index].toolCalls.isEmpty
        {
          messages.remove(at: index)
        }
        errorMessage = L10n.string(
          "Could not get a response. %@", language: language, error.localizedDescription)
      }
    }
    return true
  }

  func stopGenerating() {
    generationTask?.cancel()
  }

  func clear() {
    guard !isSending else { return }
    messages.removeAll()
    errorMessage = nil
    save()
  }

  private func save() {
    ChatHistory.save(messages, defaults: defaults)
  }

  private func enqueue(_ delta: ChatDelta, for assistantID: UUID) {
    pendingDeltas.append(delta)
    guard deltaFlushTask == nil else { return }
    deltaFlushTask = Task { [weak self] in
      do {
        try await Task.sleep(for: .milliseconds(50))
      } catch {
        return
      }
      self?.deltaFlushTask = nil
      self?.flushPendingDeltas(for: assistantID)
    }
  }

  private func flushPendingDeltas(for assistantID: UUID) {
    deltaFlushTask?.cancel()
    deltaFlushTask = nil
    guard !pendingDeltas.isEmpty,
      let index = messages.firstIndex(where: { $0.id == assistantID })
    else {
      pendingDeltas.removeAll(keepingCapacity: true)
      return
    }
    var message = messages[index]
    for delta in pendingDeltas {
      message.append(delta)
    }
    pendingDeltas.removeAll(keepingCapacity: true)
    messages[index] = message
  }

  private func removeEmptyAssistantMessage(id: UUID) {
    guard let index = messages.firstIndex(where: { $0.id == id }) else { return }
    let message = messages[index]
    if message.content.isEmpty && message.reasoningContent.isEmpty && message.toolCalls.isEmpty {
      messages.remove(at: index)
    }
  }
}

struct ChatView: View {
  let configuration: ServerConfiguration
  @ObservedObject var server: ServerController
  @ObservedObject var session: ChatSession
  let language: AppLanguage
  @AppStorage("chatDraft") private var input = ""
  @AppStorage("chatThinkingMode") private var thinkingMode = "chat"
  @AppStorage("chatModel") private var selectedModelName = ""
  @State private var seedText = ""
  @State private var seedError: String?
  @FocusState private var seedFocused: Bool
  @State private var showingClearConfirmation = false

  var body: some View {
    // Do not propagate the transcript's ideal height to the window. The page
    // owns a viewport; only the transcript scrolls as streamed content grows.
    GeometryReader { geometry in
      chatContent
        .frame(width: geometry.size.width, height: geometry.size.height)
    }
  }

  private var chatContent: some View {
    VStack(alignment: .leading, spacing: 16) {
      HStack(spacing: 12) {
        HStack(spacing: 8) {
          Text(localized("Decode Tok/s"))
            .foregroundStyle(.secondary)
          Text(liveDecodeRate)
            .font(.headline.monospacedDigit())
        }
        .accessibilityElement(children: .combine)
        Spacer()
        Picker(localized("Model"), selection: $selectedModelName) {
          ForEach(server.catalogModels) { model in
            Text(modelPickerLabel(model)).tag(model.requestName)
          }
        }
        .pickerStyle(.menu)
        .frame(width: 300)
        .disabled(server.catalogModels.isEmpty)
        .accessibilityLabel(localized("Model"))
        Picker(localized("Mode"), selection: $thinkingMode) {
          Text(localized("Chat")).tag("chat")
          Text(localized("Thinking")).tag("thinking")
        }
        .pickerStyle(.segmented)
        .tint(.blue)
        .frame(width: 220)
      }
      .appCard(padding: 16)

      VStack(spacing: 0) {
        ScrollViewReader { scroll in
          ScrollView {
            LazyVStack(alignment: .leading, spacing: 16) {
              if messages.isEmpty {
                ContentUnavailableView(
                  localized("No Test Messages"),
                  systemImage: "bubble.left",
                  description: Text(
                    L10n.string(
                      "Start the server. Then send a message to %@.",
                      language: language,
                      selectedCatalogModel?.requestName ?? localized("Assistant")
                    )
                  )
                )
                .frame(maxWidth: .infinity, minHeight: 280)
              } else {
                ForEach(messages) { message in
                  VStack(alignment: .leading, spacing: 6) {
                    Text(
                      message.role == "user"
                        ? localized("You") : message.modelName ?? localized("Assistant")
                    )
                      .font(.callout.bold())
                      .foregroundStyle(.secondary)
                    if !message.reasoningContent.isEmpty {
                      VStack(alignment: .leading, spacing: 4) {
                        Text(localized("Reasoning"))
                          .font(.callout.bold())
                          .foregroundStyle(.secondary)
                        Text(message.reasoningContent)
                          .foregroundStyle(.secondary)
                          .textSelection(.enabled)
                      }
                    }
                    if !message.content.isEmpty {
                      Text(message.content)
                        .textSelection(.enabled)
                    }
                    ForEach(message.toolCalls) { toolCall in
                      GroupBox(localized("Tool call")) {
                        VStack(alignment: .leading, spacing: 8) {
                          LabeledContent(
                            localized("Function"), value: toolCall.function.name)
                          VStack(alignment: .leading, spacing: 3) {
                            Text(localized("Arguments"))
                              .foregroundStyle(.secondary)
                            Text(toolCall.function.arguments)
                              .font(.system(.body, design: .monospaced))
                              .textSelection(.enabled)
                          }
                          Text(localized("The app does not run this tool."))
                            .font(.callout)
                            .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                      }
                    }
                    if message.role == "assistant" && message.content.isEmpty
                      && message.reasoningContent.isEmpty && message.toolCalls.isEmpty
                    {
                      ProgressView(localized("Generating"))
                        .controlSize(.small)
                    }
                  }
                  .padding(14)
                  .frame(maxWidth: .infinity, alignment: .leading)
                  .background(
                    message.role == "user"
                      ? Color.accentColor.opacity(0.12) : Color.secondary.opacity(0.08),
                    in: RoundedRectangle(cornerRadius: 10)
                  )
                  .accessibilityElement(children: .combine)
                }
              }
              Color.clear.frame(height: 1).id("chat-bottom")
            }
            .padding(8)
          }
          .onChange(of: streamedCharacterCount) {
            scroll.scrollTo("chat-bottom", anchor: .bottom)
          }
        }
      }
      .frame(minHeight: 0, maxHeight: .infinity)
      .layoutPriority(1)
      .appCard(padding: 10)

      if let errorMessage {
        Label(errorMessage, systemImage: "exclamationmark.triangle.fill")
          .foregroundStyle(.red)
          .accessibilityLabel(L10n.string("Error: %@", language: language, errorMessage))
      }

      VStack(alignment: .leading, spacing: 10) {
        HStack(spacing: 10) {
          Text(localized("Seed"))
          TextField(localized("Automatic"), text: $seedText)
            .textFieldStyle(.roundedBorder)
            .frame(width: 150)
            .accessibilityLabel(localized("Seed"))
            .accessibilityHint(seedError ?? localized("Applies to the next message only. Blank uses a random seed."))
            .focused($seedFocused)
            .disabled(isSending)
            .onChange(of: seedText) { seedError = nil }
          Text(localized("Applies to the next message only. Blank uses a random seed."))
            .font(.callout)
            .foregroundStyle(.secondary)
            .fixedSize(horizontal: false, vertical: true)
        }
        if let seedError {
          Label(seedError, systemImage: "exclamationmark.triangle.fill")
            .foregroundStyle(.red)
            .accessibilityLabel(L10n.string("Error: %@", language: language, seedError))
        }
        ZStack(alignment: .topLeading) {
          if input.isEmpty {
            Text(localized("Enter a message…"))
              .foregroundStyle(.tertiary)
              .padding(.horizontal, 5)
              .padding(.vertical, 8)
              .allowsHitTesting(false)
          }
          TextEditor(text: $input)
            .font(.body)
            .scrollContentBackground(.hidden)
            .padding(8)
            .frame(minHeight: 90, maxHeight: 180)
            .background(
              AppTheme.fieldBackground,
              in: RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
            )
            .overlay(
              RoundedRectangle(cornerRadius: AppTheme.fieldRadius)
                .stroke(Color.primary.opacity(0.12))
            )
            .accessibilityLabel(localized("Test message"))
        }

        HStack(spacing: 10) {
          if server.state != .running {
            Label(localized("Start the server first"), systemImage: "server.rack")
              .font(.callout)
              .foregroundStyle(.secondary)
          }
          Spacer()
          Button {
            showingClearConfirmation = true
          } label: {
            Label(localized("Clear Chat"), systemImage: "trash")
          }
          .disabled(messages.isEmpty || isSending)
          if isSending {
            Button(action: stopGenerating) {
              Label(localized("Stop Generating"), systemImage: "stop.fill")
            }
            .controlSize(.large)
          } else {
            Button {
              send()
            } label: {
              Label(localized("Generate"), systemImage: "arrow.up")
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
            .controlSize(.large)
            .disabled(server.state != .running || selectedCatalogModel == nil)
            .keyboardShortcut(.return, modifiers: .command)
          }
        }
      }
      .appCard(padding: 16)
    }
    .frame(maxWidth: AppLayout.contentWidth)
    .frame(maxWidth: .infinity)
    .padding(.horizontal, 40)
    .padding(.vertical, 24)
    .background(AppTheme.pageBackground)
    .environment(\.locale, language.locale)
    .onChange(of: server.catalogModels) { selectAvailableModel() }
    .onAppear { selectAvailableModel() }
    .confirmationDialog(
      localized("Clear the test chat?"),
      isPresented: $showingClearConfirmation,
      titleVisibility: .visible
    ) {
      Button(localized("Clear Chat"), role: .destructive) {
        session.clear()
        input = ""
        seedText = ""
        seedError = nil
      }
      Button(localized("Cancel"), role: .cancel) {}
    } message: {
      Text(
        localized(
          "The app will clear only the local test chat. The server and other API clients are not affected."
        ))
    }
  }

  private var streamedCharacterCount: Int {
    guard let message = messages.last else { return 0 }
    return message.content.count + message.reasoningContent.count
      + message.toolCalls.reduce(0) {
        $0 + $1.function.name.count + $1.function.arguments.count
      }
  }

  private var liveDecodeRate: String {
    let value = server.performance.snapshot.decodeTokensPerSecond
    guard server.performance.generating, value > 0 else { return "-" }
    return value.formatted(.number.precision(.fractionLength(1)))
  }

  private func localized(_ key: String) -> String {
    L10n.string(key, language: language)
  }

  private var messages: [ChatMessage] { session.messages }

  private var isSending: Bool { session.isSending }

  private var errorMessage: String? { session.errorMessage }

  private var selectedCatalogModel: CatalogModel? {
    server.catalogModels.first {
      $0.requestName == selectedModelName || $0.id == selectedModelName
    }
  }

  private func modelPickerLabel(_ model: CatalogModel) -> String {
    guard let alias = model.alias, alias != model.id else { return model.id }
    return "\(alias) — \(model.id)"
  }

  private func selectAvailableModel() {
    guard let resolved = resolvedChatModelName(
      savedName: selectedModelName,
      models: server.catalogModels
    ) else { return }
    selectedModelName = resolved
  }

  private func send() {
    guard let model = selectedCatalogModel?.requestName else { return }
    let seed: UInt32?
    do {
      seed = try ChatClient.parseSeed(seedText, language: language)
    } catch {
      seedError = error.localizedDescription
      seedFocused = true
      return
    }
    if session.send(
      text: input,
      configuration: configuration,
      model: model,
      thinkingMode: thinkingMode,
      language: language,
      seed: seed
    ) {
      input = ""
      seedText = ""
      seedError = nil
    }
  }

  private func stopGenerating() {
    session.stopGenerating()
  }
}
