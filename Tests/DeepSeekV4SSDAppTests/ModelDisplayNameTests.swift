import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

final class ModelDisplayNameTests: XCTestCase {
  func testAllPickerAndMessageNamesMatchModelPageWithoutChangingRequestNames() throws {
    for descriptor in ModelPackages.descriptors {
      let kind = try XCTUnwrap(ModelKind(rawValue: descriptor.kind))
      for alias in [nil, "My model"] as [String?] {
        let model = CatalogModel(id: kind.apiModelID, alias: alias)
        XCTAssertEqual(model.displayName, kind.displayName)
        XCTAssertEqual(model.id, kind.apiModelID)
        XCTAssertEqual(model.requestName, alias ?? kind.apiModelID)
        XCTAssertEqual(ModelDisplayName.resolve(model.requestName, models: [model]), kind.displayName)
        let message = ChatMessage(role: "assistant", content: "answer", modelName: model.requestName)
        XCTAssertEqual(message.modelDisplayName(models: [model]), kind.displayName)
      }
    }
  }

  func testExactCatalogIDWinsOverAnotherModelsAlias() {
    let qwen = ModelKind.qwen3_8FlashNext
    let swift = ModelKind.swift1_5Qwen3_8FlashNext
    let models = [CatalogModel(id: swift.apiModelID, alias: qwen.apiModelID),
      CatalogModel(id: qwen.apiModelID, alias: "Qwen")]
    XCTAssertEqual(ModelDisplayName.resolve(qwen.apiModelID, models: models), qwen.displayName)
  }

  func testUnknownNamesAndUnresolvedLegacyAliasesRemainUnchanged() {
    let models = [CatalogModel(id: "unknown-model", alias: "Custom")]
    XCTAssertEqual(models[0].displayName, "unknown-model")
    XCTAssertEqual(ModelDisplayName.resolve("unknown-model"), "unknown-model")
    XCTAssertEqual(ModelDisplayName.resolve("Custom", models: models), "Custom")
    XCTAssertEqual(ModelDisplayName.resolve("removed-alias"), "removed-alias")
    XCTAssertEqual(ModelDisplayName.resolve("QWEN3.8-FLASH-NEXT-FP8"), "QWEN3.8-FLASH-NEXT-FP8")
    XCTAssertEqual(ModelDisplayName.resolve("Swift1.5-Qwen3.8-Flash-Next"), "Swift1.5-Qwen3.8-Flash-Next")
    XCTAssertNil(ChatMessage(role: "assistant", content: "old answer").modelDisplayName())
  }

  func testStatusUsesTheSameNameInEveryLanguageWithoutChangingStatusIDs() {
    for descriptor in ModelPackages.descriptors {
      for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
        var performance = LivePerformance(loadingModel: descriptor.apiModelID)
        XCTAssertEqual(performance.modelDisplayLabel(language: language),
          L10n.string("Loading %@", language: language, descriptor.displayName))
        XCTAssertEqual(performance.loadingModel, descriptor.apiModelID)
        performance.loadedModel = descriptor.apiModelID
        XCTAssertEqual(performance.modelDisplayLabel(language: language), descriptor.displayName)
        XCTAssertEqual(performance.loadedModel, descriptor.apiModelID)
        performance.loadedModel = nil
        performance.loadingModel = nil
        XCTAssertEqual(performance.modelDisplayLabel(language: language),
          L10n.string("No model loaded", language: language))
      }
    }
  }

  func testThroughputTablesUseDisplayNamesWhileJSONPreservesAPIIdentity() throws {
    for descriptor in ModelPackages.descriptors {
      let result = Self.result(model: descriptor.apiModelID)
      XCTAssertEqual(result.displayName, descriptor.displayName)
      XCTAssertEqual(result.model, descriptor.apiModelID)
      for format in [ThroughputOutputFormat.plainText, .markdown] {
        let rendered = try format.render([result])
        XCTAssertTrue(rendered.contains(descriptor.displayName))
        XCTAssertFalse(rendered.contains(descriptor.apiModelID))
        XCTAssertTrue(rendered.contains("4096 / 128"))
      }
      let json = try ThroughputOutputFormat.json.render([result])
      let rows = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
      XCTAssertEqual(rows[0]["model"] as? String, descriptor.apiModelID)
      XCTAssertNil(rows[0]["display_name"], "The JSON data contract must not change")
      let decoder = JSONDecoder()
      decoder.keyDecodingStrategy = .convertFromSnakeCase
      let restored = try XCTUnwrap(decoder.decode([ThroughputResult].self, from: Data(json.utf8)).first)
      XCTAssertEqual(restored.model, descriptor.apiModelID)
      XCTAssertEqual(restored.displayName, descriptor.displayName)
      XCTAssertEqual(restored.outputTokenSha256, result.outputTokenSha256)
    }
  }

  @MainActor
  func testThroughputHeadingUsesCompletedResultRatherThanAliasOrNewSelection() async {
    let kind = ModelKind.swift1_5Qwen3_8FlashNext
    let session = ThroughputSession()
    session.model = "My model"
    session.contextLengths = [4096]
    session.start(prepare: {}, run: { _, _ in Self.result(model: kind.apiModelID) }, unload: {})
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while session.isRunning && ContinuousClock.now < deadline { await Task.yield() }
    XCTAssertFalse(session.isRunning)
    XCTAssertNil(session.error)
    XCTAssertEqual(session.resultDisplayName, kind.displayName)
    XCTAssertEqual(session.results.first?.model, kind.apiModelID)
    XCTAssertEqual(session.model, "My model")
    session.model = ModelKind.qwen3_8FlashNext.apiModelID
    XCTAssertEqual(session.resultDisplayName, kind.displayName)
  }

  @MainActor
  func testDryRunRetainsItsSimulatedHeading() {
    #if WHALLM_LOCAL_BUILD
    let session = ThroughputSession()
    session.model = ThroughputSession.dryRunModel
    session.runDryRun()
    XCTAssertEqual(session.resultDisplayName, "Dry run (simulated)")
    XCTAssertEqual(session.results.first?.model, "dry-run")
    #endif
  }

  @MainActor
  func testChatSendsAliasButPersistsCanonicalIdentityAcrossAliasChanges() async throws {
    let suite = "ModelDisplayNameTests.\(UUID().uuidString)"
    let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
    defer { defaults.removePersistentDomain(forName: suite) }
    let kind = ModelKind.swift1_5Qwen3_8FlashNext
    let selected = CatalogModel(id: kind.apiModelID, alias: "My model")
    var requestedModel: String?
    let session = ChatSession(defaults: defaults, stream: { _, _, _, model, _, _, _, receive in
      requestedModel = model
      receive(ChatDelta(content: "answer", reasoningContent: ""))
    })
    XCTAssertTrue(session.send(text: "Hello", configuration: .localDefault,
      model: selected.requestName, modelID: selected.id, thinkingMode: "chat", language: .english))
    let deadline = ContinuousClock.now.advanced(by: .seconds(2))
    while session.isSending && ContinuousClock.now < deadline { await Task.yield() }
    XCTAssertFalse(session.isSending)
    XCTAssertEqual(requestedModel, "My model")
    let message = try XCTUnwrap(ChatHistory.load(defaults: defaults).last)
    XCTAssertEqual(message.content, "answer")
    XCTAssertEqual(message.modelName, kind.apiModelID)
    XCTAssertEqual(message.modelDisplayName(), kind.displayName)
    let changedAliases = [CatalogModel(id: kind.apiModelID, alias: "New alias"),
      CatalogModel(id: ModelKind.qwen3_8FlashNext.apiModelID, alias: "My model")]
    XCTAssertEqual(message.modelDisplayName(models: changedAliases), kind.displayName)
    let encoded = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(message)) as? [String: Any])
    XCTAssertEqual(Set(encoded.keys), ["role", "content"])
    XCTAssertNil(encoded["modelName"], "UI metadata must not be sent as a chat message field")
  }

  private static func result(model: String) -> ThroughputResult {
    ThroughputResult(model: model, slots: 3072, benchmarkContext: .code,
      corpusSha256: "fixture", contextTokens: 4096, generationTokens: 128, generationLimit: 128,
      ttftMs: 100, tpotMs: 10, prefillTps: 100, decodeTps: 10, elapsedSeconds: 12,
      throughputTps: 100, peakAppMemoryBytes: 1024, memoryScope: "app",
      outputTokenSha256: "fixture-output", promptCacheReusedTokens: 0, finishReason: "length")
  }
}
