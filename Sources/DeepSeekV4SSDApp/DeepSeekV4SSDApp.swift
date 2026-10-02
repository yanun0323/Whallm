import AppKit
import DeepSeekRepack
import SwiftUI

@main
struct DeepSeekV4SSDApp: App {
  @StateObject private var server = ServerController()
  @AppStorage(L10n.preferenceKey) private var languageCode = AppLanguage.appDefault.rawValue
  @StateObject private var appUpdater: AppUpdater
  private let verifiesLocalizations: Bool

  init() {
    let verifiesLocalizations = CommandLine.arguments.contains("--verify-localizations")
    self.verifiesLocalizations = verifiesLocalizations
    _appUpdater = StateObject(wrappedValue: AppUpdater(startingUpdater: !verifiesLocalizations))
    if verifiesLocalizations {
      // A directly launched secondary process may never present a window.
      // Complete the real lookup in App initialization, not in a view callback.
      let language = L10n.selectedLanguage
      _ = L10n.string("Server", language: language)
      let marker = "WHALLM_LOCALIZATION_READY:\(language.rawValue)\n"
      FileHandle.standardOutput.write(Data(marker.utf8))
      let localBuildMarker = "WHALLM_LOCAL_BUILD:\(ThroughputSession.dryRunAvailable ? 1 : 0)\n"
      FileHandle.standardOutput.write(Data(localBuildMarker.utf8))
      let packages = ModelPackages.descriptors
      for descriptor in packages {
        _ = ModelPackages.package(for: ModelKind(rawValue: descriptor.kind)!)
      }
      let packageMarker = "WHALLM_MODEL_PACKAGES_READY:\(packages.count)\n"
      FileHandle.standardOutput.write(Data(packageMarker.utf8))
      let visionMarker = "WHALLM_VISION_PACKAGES_READY:\(QwenVisionArtifact.modelKinds.count)\n"
      FileHandle.standardOutput.write(Data(visionMarker.utf8))
    }
    NSApplication.shared.setActivationPolicy(.regular)
  }

  var body: some Scene {
    WindowGroup {
      Group {
        if verifiesLocalizations {
          // Exercise real L10n lookup without constructing views that read Keychain.
          Text(L10n.string("Server", language: selectedLanguage))
        } else {
          ContentView(server: server, appUpdater: appUpdater)
        }
      }
      .font(.body)
      .dynamicTypeSize(.xLarge ... .accessibility5)
      .controlSize(.large)
      .frame(minWidth: 640, minHeight: 320)
      .environment(\.locale, selectedLanguage.locale)
      .onAppear { NSApplication.shared.activate() }
      .onDisappear { server.stop() }
    }
    .defaultSize(width: 1_440, height: 900)
    .windowStyle(.hiddenTitleBar)
    .commands {
      CommandGroup(after: .appInfo) {
        Button(L10n.string("Check for Updates…")) {
          appUpdater.checkForUpdates()
        }
        .disabled(!appUpdater.canCheckForUpdates)
      }
    }
  }

  private var selectedLanguage: AppLanguage {
    (AppLanguage(rawValue: languageCode) ?? .appDefault).resolved
  }
}
