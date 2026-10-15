import AppKit
import SwiftUI
import XCTest

@testable import DeepSeekV4SSDApp

final class MainWindowTests: XCTestCase {
  @MainActor
  func testMainWindowRendersNavigationAndContent() throws {
    let suite = "MainWindowTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }

    for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
      for page in ["server", "logs"] {
        defaults.set(language.rawValue, forKey: L10n.preferenceKey)
        defaults.set(page, forKey: "selectedAppPage")
        let host = NSHostingView(rootView: ContentView(server: ServerController(),
          appUpdater: AppUpdater(startingUpdater: false))
          .defaultAppStorage(defaults)
          .font(.body).dynamicTypeSize(.xLarge ... .accessibility5).controlSize(.large)
          .frame(minWidth: 640, minHeight: 320))
        let window = NSWindow(contentRect: NSRect(x: 100, y: 100, width: 1200, height: 750),
          styleMask: [.titled, .closable, .resizable, .fullSizeContentView],
          backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.titleVisibility = .hidden
        window.titlebarAppearsTransparent = true
        window.contentView = host
        window.orderBack(nil)
        defer { window.close() }

        for size in [NSSize(width: 1200, height: 750), NSSize(width: 860, height: 480)] {
          window.setContentSize(size)
          RunLoop.main.run(until: Date().addingTimeInterval(0.3))
          host.layoutSubtreeIfNeeded()
          let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
          host.cacheDisplay(in: host.bounds, to: bitmap)
          let label = "\(language.rawValue)-\(page)-\(Int(size.width))"
          if let directory = ProcessInfo.processInfo.environment["WHALLM_WINDOW_PREVIEWS"] {
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
              .write(to: URL(fileURLWithPath: directory).appendingPathComponent("\(label).png"))
          }
          let scale = CGFloat(bitmap.pixelsWide) / host.bounds.width
          // Inspect below the titlebar: window chrome alone must not pass.
          // The app is dark, so light pixels here come from labels and controls.
          for (name, region) in [
            ("sidebar", CGRect(x: 15, y: 80, width: 160, height: 180)),
            ("content", CGRect(x: 300, y: 80, width: 450, height: 220)),
          ] {
            var brightPixels = 0
            for y in Int(region.minY * scale)..<Int(region.maxY * scale) {
              for x in Int(region.minX * scale)..<Int(region.maxX * scale) {
                if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                   color.redComponent > 0.5, color.greenComponent > 0.5, color.blueComponent > 0.5 {
                  brightPixels += 1
                }
              }
            }
            XCTAssertGreaterThan(brightPixels, 100, "Blank \(name): \(label)")
          }
        }
      }
    }
  }
}
