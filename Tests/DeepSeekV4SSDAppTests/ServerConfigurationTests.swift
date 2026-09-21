import Foundation
import DeepSeekRepack
import XCTest

@testable import DeepSeekV4SSDApp

final class ServerConfigurationTests: XCTestCase {
  @MainActor
  func testAccelerationDefaultsAndOptOutsReachSerializedCatalogForEveryModel() throws {
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      for enabled in [true, false] {
        var settings = ModelAdvancedSettings.defaults(for: kind)
        if !enabled {
          settings.layerMajorPrefill = false
          settings.readyExpertDecode = false
          settings.batchedExpertPrefill = false
          settings.nextLayerPrefetch = false
          settings.qwenGroupedExperts = false
        }
        let catalog = try ModelLibrary.makeServerCatalog(models: [installedModel(kind)],
          aliases: [:], settings: [kind: settings], powerSavingLimitGBps: nil)
        let runtime = try XCTUnwrap(catalog.models.first).runtime
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(runtime)) as? [String: Any])
        for key in ["layer_major_prefill", "ready_expert_decode", "batched_expert_prefill"] {
          XCTAssertEqual(json[key] as? Bool, enabled, "\(kind): \(key)")
        }
        XCTAssertEqual(json["qwen_next_layer_prefetch"] as? Bool, enabled && kind == .qwen3_8FlashNext)
        XCTAssertEqual(json["v41_next_layer_prefetch"] as? Bool, enabled && kind == .deepSeekV41)
        XCTAssertEqual(json["qwen_grouped_experts"] as? Bool, enabled && kind == .qwen3_8FlashNext)
      }
    }
  }

  @MainActor
  func testQwenSpeedDefaultsReachRuntime() throws {
    let settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [installedModel(.qwen3_8FlashNext, hasMTP: true)],
      aliases: [:], settings: [.qwen3_8FlashNext: settings], powerSavingLimitGBps: nil)
    let runtime = try XCTUnwrap(catalog.models.first).runtime
    XCTAssertEqual(runtime.readWorkers, 16)
    XCTAssertEqual(runtime.expertEvictionPolicy, "lru")
    XCTAssertTrue(runtime.readyExpertDecode)
    XCTAssertTrue(runtime.layerMajorPrefill)
    XCTAssertEqual(runtime.prefillStepSize, 1024)
    XCTAssertEqual(runtime.memoryLimitGiB, 30)
    XCTAssertTrue(runtime.batchedExpertPrefill)
    XCTAssertTrue(runtime.qwenNextLayerPrefetch)
    XCTAssertTrue(runtime.qwenGroupedExperts)
    XCTAssertTrue(runtime.qwenPooledIndexCache)
    XCTAssertTrue(runtime.qwenNgramLookupOptimized)
    XCTAssertTrue(runtime.qwenCompileTensorOps)
    XCTAssertTrue(runtime.qwenPhaseMemory)
    XCTAssertFalse(runtime.qwenQuantizedKV)
    XCTAssertFalse(runtime.qwenQuantizedIndex)
    XCTAssertFalse(runtime.mtpEnabled)
    XCTAssertEqual(runtime.anePrefillRatio, 0)
    XCTAssertEqual(settings.approximationEnabled, false)
    XCTAssertEqual(settings.slots, 3072)
    XCTAssertEqual(settings.expertCacheGiB, 7.5)
    XCTAssertEqual(settings.promptCacheMode, .memory)
    XCTAssertEqual(settings.defaultMaxTokens, 8192)
  }

  @MainActor
  func testQwenOptimizationSwitchesPersistAndReachRuntimeOnlyForQwen() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    for kind in [ModelKind.qwen3_8FlashNext, .deepSeekV4, .deepSeekV41] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      for feature in QwenOptimization.allCases {
        XCTAssertEqual(settings[keyPath: feature.keyPath] == true,
                       kind == .qwen3_8FlashNext && feature.keyPath != \ModelAdvancedSettings.qwenMTPPolicy)
        settings[keyPath: feature.keyPath] = true
      }
      settings.qwenMTPDraftTokens = 3
      settings.qwenMTPZeroAcceptanceLimit = 4
      settings.save(for: kind, defaults: isolated.defaults)
      let restored = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
      for feature in QwenOptimization.allCases {
        XCTAssertEqual(restored[keyPath: feature.keyPath], true)
      }
      let catalog = try ModelLibrary.makeServerCatalog(models: [installedModel(kind, hasMTP: true)],
        aliases: [:], settings: [kind: restored], powerSavingLimitGBps: nil)
      let runtime = try XCTUnwrap(catalog.models.first).runtime
      let enabled = kind == .qwen3_8FlashNext
      XCTAssertEqual(runtime.qwenPooledIndexCache, enabled)
      XCTAssertEqual(runtime.qwenNgramLookupOptimized, enabled)
      XCTAssertEqual(runtime.qwenCompileTensorOps, enabled)
      XCTAssertEqual(runtime.qwenPhaseMemory, enabled)
      XCTAssertEqual(runtime.qwenMTPDraftTokens, enabled ? 3 : 5)
      XCTAssertEqual(runtime.qwenMTPZeroAcceptanceLimit, enabled ? 4 : 1)
      let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(runtime)) as? [String: Any])
      XCTAssertEqual(json["qwen_phase_memory"] as? Bool, enabled)
      XCTAssertEqual(json["qwen_mtp_draft_tokens"] as? Int, enabled ? 3 : 5)
    }
  }

  func testOldQwenSettingsAndMTPPolicyOffPreserveChoices() throws {
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    var old = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
    for key in ["qwenPooledIndexCache", "qwenNgramLookupOptimized", "qwenCompileTensorOps",
                "qwenPhaseMemory", "qwenMTPPolicy", "qwenMTPDraftTokens", "qwenMTPZeroAcceptanceLimit"] {
      old.removeValue(forKey: key)
    }
    let legacy = try JSONDecoder().decode(ModelAdvancedSettings.self, from: JSONSerialization.data(withJSONObject: old))
    for feature in QwenOptimization.allCases { XCTAssertFalse(legacy[keyPath: feature.keyPath] == true) }
    XCTAssertEqual(legacy.effectiveQwenMTPDraftTokens, 5)
    XCTAssertEqual(legacy.effectiveQwenMTPZeroAcceptanceLimit, 1)
    settings.qwenMTPPolicy = true
    settings.qwenMTPDraftTokens = 3
    settings.qwenMTPZeroAcceptanceLimit = 4
    settings.qwenMTPPolicy = false
    XCTAssertEqual(settings.effectiveQwenMTPDraftTokens, 5)
    XCTAssertEqual(settings.effectiveQwenMTPZeroAcceptanceLimit, 1)
    settings.qwenMTPPolicy = true
    XCTAssertEqual(settings.effectiveQwenMTPDraftTokens, 3)
    XCTAssertEqual(settings.effectiveQwenMTPZeroAcceptanceLimit, 4)
    settings.qwenMTPDraftTokens = 6
    XCTAssertThrowsError(try settings.validate(for: .qwen3_8FlashNext))
    settings.qwenMTPDraftTokens = 2
    settings.qwenMTPZeroAcceptanceLimit = 33
    XCTAssertThrowsError(try settings.validate(for: .qwen3_8FlashNext))
  }

  func testQwenOptimizationCopyIsLocalized() {
    let extra = ["Speed improvements are not yet verified. Changes apply on next model load.",
                 "MTP draft tokens", "Zero-acceptance rounds before stopping MTP",
                 "Choose 1–5 MTP draft tokens and 1–32 zero-acceptance rounds."]
    for language in [AppLanguage.traditionalChinese, .simplifiedChinese] {
      for key in QwenOptimization.allCases.flatMap({ [$0.title, $0.hint] }) + extra {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }

  @MainActor
  func testRouteAwareCacheRoundTripsAndReachesCatalog() async throws {
    for descriptor in ModelPackages.descriptors {
      let kind = try XCTUnwrap(ModelKind(rawValue: descriptor.kind))
      var settings = ModelAdvancedSettings.defaults(for: kind)
      settings.routeAwareExpertCache = true
      let restored = try JSONDecoder().decode(ModelAdvancedSettings.self, from: JSONEncoder().encode(settings))
      XCTAssertEqual(restored.routeAwareExpertCache, true)
      let catalog = try ModelLibrary.makeServerCatalog(
        models: [installedModel(kind)], aliases: [:], settings: [kind: restored],
        powerSavingLimitGBps: nil)
      XCTAssertEqual(try XCTUnwrap(catalog.models.first).runtime.expertEvictionPolicy, "route")
    }
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      XCTAssertNotEqual(L10n.string("Route-aware", language: language), "Route-aware")
    }
  }

  func testAllModelsDefaultTo8192OutputTokensAndPreserveSavedValues() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    for descriptor in ModelPackages.descriptors {
      let kind = try XCTUnwrap(ModelKind(rawValue: descriptor.kind))
      XCTAssertEqual(ModelAdvancedSettings.defaults(for: kind).defaultMaxTokens, 8_192)
      XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults).defaultMaxTokens, 8_192)
      var custom = ModelAdvancedSettings.defaults(for: kind)
      custom.defaultMaxTokens = 4_096
      custom.save(for: kind, defaults: isolated.defaults)
      XCTAssertEqual(ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults).defaultMaxTokens, 4_096)
    }
  }

  @MainActor
  func testSSDSettingsDefaultsMigrateAndPreserveExplicitChoices() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(settings.recentExpertCache, true)
      settings.slots = 900
      var old = try XCTUnwrap(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
      old.removeValue(forKey: "recentExpertCache")
      let decoded = try JSONDecoder().decode(
        ModelAdvancedSettings.self, from: JSONSerialization.data(withJSONObject: old))
      decoded.save(for: kind, defaults: isolated.defaults)
      settings = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
      XCTAssertEqual(settings.slots, 900)
      XCTAssertEqual(settings.recentExpertCache, true)
      for enabled in [false, true] {
        settings.recentExpertCache = enabled
        settings.save(for: kind, defaults: isolated.defaults)
        let restored = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
        XCTAssertEqual(restored.recentExpertCache, enabled)
        let catalog = try ModelLibrary.makeServerCatalog(
          models: [installedModel(kind)], aliases: [:], settings: [kind: restored],
          powerSavingLimitGBps: nil)
        let runtime = try XCTUnwrap(catalog.models.first).runtime
        XCTAssertEqual(runtime.expertEvictionPolicy, enabled ? "lru" : "lfu")
      }
    }
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      for label in ["Keep recently used experts"] {
        XCTAssertNotEqual(L10n.string(label, language: language), label)
      }
    }
  }

  @MainActor
  func testPromptCacheModesMigrateSaveAndReachRuntime() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      XCTAssertEqual(settings.promptCacheMode, .memory)
      var old = try XCTUnwrap(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
      old.removeValue(forKey: "promptCacheMode")
      isolated.defaults.set(try JSONSerialization.data(withJSONObject: old),
        forKey: "modelAdvancedSettings.\(kind.rawValue)")
      settings = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
      XCTAssertEqual(settings.promptCacheMode, .memory)
      for mode in PromptCacheMode.allCases {
        settings.promptCacheMode = mode
        settings.save(for: kind, defaults: isolated.defaults)
        let restored = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
        XCTAssertEqual(restored.promptCacheMode, mode)
        let catalog = try ModelLibrary.makeServerCatalog(
          models: [installedModel(kind)], aliases: [:], settings: [kind: restored],
          powerSavingLimitGBps: nil)
        let runtime = try XCTUnwrap(catalog.models.first).runtime
        XCTAssertEqual(runtime.promptCacheEntries, mode == .off ? 0 : settings.promptCacheEntries)
        XCTAssertEqual(runtime.persistentPromptCache, mode == .disk)
      }
    }
    XCTAssertEqual(ModelAdvancedSettings.defaults(for: .deepSeekV41).promptCacheMode, .memory)
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      for key in ["Prompt cache", "Off", "Memory", "Disk"] {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }

  @MainActor
  func testNewAdvancedControlsMigratePersistAndReachCatalog() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      var settings = ModelAdvancedSettings.defaults(for: kind)
      var old = try XCTUnwrap(
        JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
      for key in ["prefetchReadWorkers", "moePrefillStepSize", "approximationEnabled", "qwenAdaptiveSampling"] {
        old.removeValue(forKey: key)
      }
      old["recentExpertCache"] = false
      let decoded = try JSONDecoder().decode(ModelAdvancedSettings.self,
        from: JSONSerialization.data(withJSONObject: old))
      decoded.save(for: kind, defaults: isolated.defaults)
      settings = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
      XCTAssertEqual(settings.prefetchReadWorkers, 2)
      XCTAssertEqual(settings.moePrefillStepSize, 0)
      XCTAssertEqual(settings.approximationEnabled, false)
      XCTAssertEqual(settings.qwenAdaptiveSampling, true)
      XCTAssertEqual(settings.recentExpertCache, false)
      settings.prefetchReadWorkers = 3
      settings.moePrefillStepSize = 64
      settings.approximationEnabled = true
      settings.defaultTemperature = 0.4
      settings.defaultTopP = 0.6
      settings.defaultTopK = 7
      for adaptive in [false, true] {
        settings.qwenAdaptiveSampling = adaptive
        settings.save(for: kind, defaults: isolated.defaults)
        let restored = ModelAdvancedSettings.loadOrDefault(for: kind, defaults: isolated.defaults)
        XCTAssertEqual(restored.defaultTemperature, 0.4)
        XCTAssertEqual(restored.defaultTopP, 0.6)
        XCTAssertEqual(restored.defaultTopK, 7)
        let catalog = try ModelLibrary.makeServerCatalog(
          models: [installedModel(kind)], aliases: [:], settings: [kind: restored],
          powerSavingLimitGBps: nil)
        let entry = try XCTUnwrap(catalog.models.first)
        XCTAssertEqual(entry.runtime.prefetchReadWorkers, 3)
        XCTAssertEqual(entry.runtime.moePrefillStepSize, 64)
        XCTAssertEqual(entry.runtime.expertEvictionPolicy, "lfu")
        XCTAssertEqual(entry.defaults.qwenAdaptiveSampling, adaptive)
        XCTAssertEqual(entry.defaults.approximationMode,
          "learned-route-drop-lowest-1")
      }
      settings.prefetchReadWorkers = 0
      XCTAssertThrowsError(try settings.validate(for: kind))
      settings.prefetchReadWorkers = 2
      settings.moePrefillStepSize = -1
      XCTAssertThrowsError(try settings.validate(for: kind))
    }
    var deepSeek = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    deepSeek.approximationEnabled = true
    deepSeek.dsparkEnabled = true
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [installedModel(.deepSeekV4, hasDSpark: true)], aliases: [:],
      settings: [.deepSeekV4: deepSeek], powerSavingLimitGBps: nil)
    XCTAssertEqual(catalog.models.first?.defaults.approximationMode, "exact")
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      for label in ["Use adaptive sampling", "Use approximate mode", "Expert cache eviction", "Prefetch read workers", "MoE prefill step size"] {
        XCTAssertNotEqual(L10n.string(label, language: language), label)
      }
    }
  }

  func testAdvancedSettingsLockOnlyForTheActiveModel() {
    let qwen = "qwen3.8-flash-next-fp8"
    XCTAssertFalse(
      modelAdvancedSettingsAreLocked(
        modelID: qwen,
        loadedModel: "deepseek-v4-flash-0731",
        loadingModel: nil,
        modelActionID: nil
      ))
    XCTAssertTrue(
      modelAdvancedSettingsAreLocked(
        modelID: qwen,
        loadedModel: qwen,
        loadingModel: nil,
        modelActionID: nil
      ))
    XCTAssertTrue(
      modelAdvancedSettingsAreLocked(
        modelID: qwen,
        loadedModel: nil,
        loadingModel: qwen,
        modelActionID: nil
      ))
    XCTAssertTrue(
      modelAdvancedSettingsAreLocked(
        modelID: qwen,
        loadedModel: nil,
        loadingModel: nil,
        modelActionID: qwen
      ))
  }

  func testMTPDownloadButtonAppearsOnlyForQwenWithoutMTP() {
    XCTAssertTrue(shouldShowMTPDownloadButton(installedModel(.qwen3_8FlashNext)))
    XCTAssertFalse(
      shouldShowMTPDownloadButton(installedModel(.qwen3_8FlashNext, hasMTP: true)))
    XCTAssertFalse(shouldShowMTPDownloadButton(installedModel(.deepSeekV4)))
    XCTAssertFalse(shouldShowMTPDownloadButton(nil))
  }

  func testActiveDownloadDoesNotReserveExtraModelListHeight() {
    XCTAssertFalse(
      shouldShowModelDownloadReason(
        modelIsInstalled: false,
        modelIsDownloading: true,
        hasReason: true
      )
    )
    XCTAssertTrue(
      shouldShowModelDownloadReason(
        modelIsInstalled: false,
        modelIsDownloading: false,
        hasReason: true
      )
    )
  }

  func testDownloadProgressHeightIncludesBottomPadding() {
    XCTAssertEqual(modelDownloadProgressExtraHeight(hasProgressFraction: false), 40)
    XCTAssertEqual(modelDownloadProgressExtraHeight(hasProgressFraction: true), 72)
  }

  @MainActor
  func testLoadedModelIDMapsToItsModelKind() {
    XCTAssertEqual(modelKind(withAPIModelID: "deepseek-v4-flash-0731"), .deepSeekV4)
    XCTAssertEqual(modelKind(withAPIModelID: "qwen3.8-flash-next-fp8"), .qwen3_8FlashNext)
    XCTAssertNil(modelKind(withAPIModelID: "unknown"))
    XCTAssertNil(modelKind(withAPIModelID: nil))
  }

  func testServerAndModelControlsRemainIndependent() {
    XCTAssertFalse(
      modelDownloadIsDisabled(
        serverIsActive: true,
        operationIsBusy: false,
        canStartDownload: true,
        hasPartialDownload: false,
        targetHasPartialDownload: false
      )
    )
    XCTAssertFalse(
      modelSelectionIsLocked(
        serverIsActive: true,
        operationIsBusy: true,
        downloadIsActive: true,
        hasPartialDownload: true
      )
    )
    XCTAssertTrue(
      modelSelectionIsLocked(
        serverIsActive: false,
        operationIsBusy: true,
        downloadIsActive: false,
        hasPartialDownload: false
      )
    )
  }

  func testPowerSavingLegendAnchorsAlignWithSliderNodes() {
    let count = ServerConfiguration.powerSavingLimitOptionsGBps.count
    let totalWidth = CGFloat(700)
    let nodeSpacing = totalWidth / CGFloat(count - 1)

    for index in 1..<(count - 1) {
      let frame = powerSavingLegendFrame(index: index, count: count, totalWidth: totalWidth)
      XCTAssertEqual(frame.midX, CGFloat(index) * nodeSpacing, accuracy: 0.001)
    }
    XCTAssertEqual(
      powerSavingLegendFrame(index: 0, count: count, totalWidth: totalWidth).minX,
      0,
      accuracy: 0.001
    )
    XCTAssertEqual(
      powerSavingLegendFrame(index: count - 1, count: count, totalWidth: totalWidth).maxX,
      totalWidth,
      accuracy: 0.001
    )
  }

  func testServerConfigurationContainsOnlyServerArguments() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    var configuration = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "secret")
    configuration.host = "0.0.0.0"
    configuration.port = 9_000
    configuration.logLevel = .debug
    configuration.powerSavingLimitGBps = 2

    let arguments = configuration.arguments(modelCatalogPath: "/tmp/catalog.json")

    XCTAssertEqual(configuration.baseURL?.absoluteString, "http://127.0.0.1:9000")
    XCTAssertEqual(
      arguments,
      [
        "-m", "deepseek_v4_ssd.server",
        "--model-catalog", "/tmp/catalog.json",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--log-level", "debug",
      ]
    )
    XCTAssertFalse(arguments.contains("secret"))
    XCTAssertFalse(arguments.contains("--model"))
    XCTAssertFalse(arguments.contains("--public-model"))
  }

  func testServerConfigurationPersistenceDoesNotStoreAPIKey() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    var configuration = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "")
    configuration.host = "0.0.0.0"
    configuration.port = 9_000
    configuration.logLevel = .error
    configuration.apiKey = "secret"
    configuration.powerSavingLimitGBps = 0.5

    configuration.save(defaults: isolated.defaults)
    let restored = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "secret")

    XCTAssertEqual(restored, configuration)
    let storedData = try XCTUnwrap(
      isolated.defaults.data(forKey: ServerConfiguration.preferenceKey))
    XCTAssertFalse(String(decoding: storedData, as: UTF8.self).contains("secret"))
  }

  func testSavedServerConfigurationWithoutLogLevelDefaultsToInfo() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let oldConfiguration: [String: Any] = [
      "runtimeDirectory": "",
      "pythonExecutable": "",
      "host": "127.0.0.1",
      "port": 11_434,
      "apiKey": "",
    ]
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: oldConfiguration),
      forKey: ServerConfiguration.preferenceKey
    )

    let restored = ServerConfiguration.load(defaults: isolated.defaults, apiKey: "")

    XCTAssertEqual(restored.logLevel, .info)
  }

  func testAdvancedSettingsUsePerModelDefaultsAndNormalization() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }

    var deepSeek = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    deepSeek.slots = 700
    deepSeek.bf16KVCache = true
    deepSeek.mtpEnabled = true
    deepSeek.dsparkEnabled = true
    deepSeek.save(for: .deepSeekV4, defaults: isolated.defaults)

    var qwen = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    XCTAssertEqual(qwen.slots, 3_072)
    for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
      let hint = L10n.string(
        "Number of routed experts in the Active Parameters Cache. The recommended value is %lld.",
        language: language, Int64(ModelKind.qwen3_8FlashNext.descriptor.defaults.slots))
      XCTAssertEqual(hint.filter(\.isNumber), "3072", hint)
    }
    XCTAssertFalse(qwen.mtpEnabled ?? true)
    XCTAssertEqual(qwen.mtpSlots, 32)
    XCTAssertEqual(qwen.anePrefillRatio, 0)
    qwen.slots = 900
    qwen.bf16KVCache = true
    qwen.mtpEnabled = true
    qwen.mtpSlots = 512
    qwen.anePrefillRatio = 0.5
    qwen.dsparkEnabled = true
    qwen.defaultTemperature = 1.0
    qwen.defaultTopP = 0.95
    qwen.defaultTopK = 3
    isolated.defaults.set(
      try JSONEncoder().encode(qwen),
      forKey: "modelAdvancedSettings.qwen3.8-flash-next"
    )

    let restoredDeepSeek = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4, defaults: isolated.defaults)
    let restoredQwen = ModelAdvancedSettings.loadOrDefault(
      for: .qwen3_8FlashNext, defaults: isolated.defaults)

    XCTAssertEqual(restoredDeepSeek.slots, 700)
    XCTAssertTrue(restoredDeepSeek.bf16KVCache)
    XCTAssertFalse(restoredDeepSeek.mtpEnabled ?? true)
    XCTAssertTrue(restoredDeepSeek.dsparkEnabled)
    XCTAssertEqual(restoredDeepSeek.layerMajorPrefillThreshold, 1_024)
    XCTAssertEqual(restoredQwen.slots, 900)
    XCTAssertEqual(restoredQwen.defaultMaxTokens, 8_192)
    XCTAssertEqual(restoredQwen.defaultTemperature, 1.0)
    XCTAssertEqual(restoredQwen.defaultTopP, 0.95)
    XCTAssertEqual(restoredQwen.defaultTopK, 3)
    XCTAssertFalse(restoredQwen.bf16KVCache)
    XCTAssertTrue(restoredQwen.mtpEnabled == true)
    XCTAssertEqual(restoredQwen.mtpSlots, 512)
    XCTAssertEqual(restoredQwen.anePrefillRatio, 0.5)
    XCTAssertFalse(restoredQwen.dsparkEnabled)
    XCTAssertEqual(restoredQwen.layerMajorPrefillThreshold, 1_024)
  }

  @MainActor
  func testPrefillAccelerationMigratesPersistsAndReachesCatalog() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    XCTAssertEqual(settings.qwenGroupedExperts, true)
    XCTAssertEqual(ModelAdvancedSettings.defaults(for: .deepSeekV4).qwenGroupedExperts, false)
    settings.slots = 5_000
    var old = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(settings)) as? [String: Any])
    old.removeValue(forKey: "qwenGroupedExperts")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: old),
      forKey: "modelAdvancedSettings.qwen3.8-flash-next")
    settings = ModelAdvancedSettings.loadOrDefault(for: .qwen3_8FlashNext, defaults: isolated.defaults)
    XCTAssertEqual(settings.qwenGroupedExperts, true)
    XCTAssertEqual(settings.slots, 5_000)

    for enabled in [false, true] {
      settings.qwenGroupedExperts = enabled
      settings.save(for: .qwen3_8FlashNext, defaults: isolated.defaults)
      let restored = ModelAdvancedSettings.loadOrDefault(for: .qwen3_8FlashNext, defaults: isolated.defaults)
      XCTAssertEqual(restored.qwenGroupedExperts, enabled)
      let catalog = try ModelLibrary.makeServerCatalog(
        models: [installedModel(.qwen3_8FlashNext, hasMTP: true)],
        aliases: [:], settings: [.qwen3_8FlashNext: restored], powerSavingLimitGBps: nil)
      let object = try XCTUnwrap(
        try XCTUnwrap(catalog.models.first).jsonObject() as? [String: Any])
      let runtime = try XCTUnwrap(object["runtime"] as? [String: Any])
      XCTAssertEqual(runtime["qwen_grouped_experts"] as? Bool, enabled)
      XCTAssertEqual(runtime["mtp_enabled"] as? Bool, false)
    }
    XCTAssertEqual(settings.normalized(for: .deepSeekV4).qwenGroupedExperts, false)
    for language in [AppLanguage.english, .simplifiedChinese, .traditionalChinese] {
      let label = L10n.string("Prefill acceleration", language: language)
      XCTAssertEqual(label, language == .english ? "Prefill acceleration" : "Prefill 加速")
    }
  }

  func testSavedAdvancedSettingsWithoutNewFieldsUseTheNewDefaults() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let encoded = try JSONEncoder().encode(
      ModelAdvancedSettings.defaults(for: .deepSeekV4))
    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: encoded) as? [String: Any])
    object.removeValue(forKey: "layerMajorPrefillThreshold")
    object.removeValue(forKey: "mtpEnabled")
    object.removeValue(forKey: "mtpSlots")
    object.removeValue(forKey: "anePrefillRatio")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: object),
      forKey: "modelAdvancedSettings.deepseek-v4"
    )

    let restored = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4,
      defaults: isolated.defaults
    )

    XCTAssertEqual(restored.layerMajorPrefillThreshold, 1_024)
    XCTAssertFalse(restored.mtpEnabled ?? true)
    XCTAssertEqual(restored.mtpSlots, 32)
    XCTAssertEqual(restored.anePrefillRatio, 0)
  }

  func testLegacyAdvancedSettingsMigrateOnlyToTheCurrentModel() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("deepseek-v4", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: legacyConfiguration(publicModel: "custom")),
      forKey: ServerConfiguration.preferenceKey
    )

    let deepSeek = ModelAdvancedSettings.loadOrDefault(
      for: .deepSeekV4, defaults: isolated.defaults)
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    let qwen = ModelAdvancedSettings.loadOrDefault(
      for: .qwen3_8FlashNext, defaults: isolated.defaults)

    XCTAssertEqual(deepSeek.slots, 640)
    XCTAssertEqual(deepSeek.defaultTemperature, 0.7)
    XCTAssertEqual(deepSeek.layerMajorPrefillThreshold, 1_024)
    XCTAssertEqual(qwen.slots, 3_072)
    XCTAssertEqual(qwen.defaultTemperature, 0.7)
  }

  @MainActor
  func testAliasCanBeSavedTrimmedClearedAndSetBeforeInstall() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(
      try library.saveAlias("  work-model  ", for: .deepSeekV4),
      "work-model"
    )
    XCTAssertEqual(library.alias(for: .deepSeekV4), "work-model")
    XCTAssertNil(library.usableModel(for: .deepSeekV4))

    XCTAssertEqual(try library.saveAlias("   ", for: .deepSeekV4), "")
    XCTAssertEqual(library.alias(for: .deepSeekV4), "")
    XCTAssertNil(
      isolated.defaults.string(forKey: ModelLibrary.aliasPreferenceKey(for: .deepSeekV4)))
  }

  @MainActor
  func testAliasValidationIsCaseSensitiveAndRejectsOtherNames() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertNoThrow(
      try library.saveAlias("deepseek-v4-flash-0731", for: .deepSeekV4))
    XCTAssertNoThrow(try library.saveAlias("Work", for: .deepSeekV4))
    XCTAssertNoThrow(try library.saveAlias("work", for: .qwen3_8FlashNext))
    XCTAssertThrowsError(try library.saveAlias("Work", for: .qwen3_8FlashNext))
    XCTAssertThrowsError(
      try library.saveAlias("deepseek-v4-flash-0731", for: .qwen3_8FlashNext))
  }

  @MainActor
  func testLegacyCustomPublicModelMigratesToCurrentAlias() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(withJSONObject: ["publicModel": "  old-alias  "]),
      forKey: ServerConfiguration.preferenceKey
    )

    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(library.alias(for: .qwen3_8FlashNext), "old-alias")
  }

  @MainActor
  func testLegacyQwenDefaultDoesNotMigrateToAlias() throws {
    let isolated = try isolatedDefaults()
    defer { isolated.defaults.removePersistentDomain(forName: isolated.suite) }
    isolated.defaults.set("qwen3.8-flash-next", forKey: "selectedInstallModelKind")
    isolated.defaults.set(
      try JSONSerialization.data(
        withJSONObject: ["publicModel": "Qwen/Qwen3.8-Flash-Next-FP8"]),
      forKey: ServerConfiguration.preferenceKey
    )

    let library = ModelLibrary(defaults: isolated.defaults)

    XCTAssertEqual(library.alias(for: .qwen3_8FlashNext), "")
  }

  @MainActor
  func testCatalogUsesFixedIDsAliasesAndCompleteSnakeCaseRuntime() throws {
    var deepSeek = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    deepSeek.slots = 700
    deepSeek.defaultTemperature = 0.4
    deepSeek.dsparkEnabled = true
    let deepSeekV41 = ModelAdvancedSettings.defaults(for: .deepSeekV41)
    var qwen = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    qwen.slots = 900
    qwen.mtpEnabled = true
    qwen.mtpSlots = 512
    qwen.anePrefillRatio = 0.5
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [
        installedModel(
          .deepSeekV4,
          issues: [InstalledFileIssue(path: "common.bin", kind: .checksumMismatch)]
        ),
        installedModel(.deepSeekV41),
        installedModel(.qwen3_8FlashNext, hasMTP: true),
        installedModel(.deepSeekV4, hasDSpark: true),
      ],
      aliases: [.deepSeekV4: "work-model"],
      settings: [
        .deepSeekV4: deepSeek,
        .deepSeekV41: deepSeekV41,
        .qwen3_8FlashNext: qwen,
      ],
      powerSavingLimitGBps: 2
    )

    XCTAssertEqual(
      catalog.models.map(\.id),
      ["deepseek-v4-flash-0731", "deepseek-v4.1-flash", "qwen3.8-flash-next-fp8"]
    )
    XCTAssertEqual(catalog.models[0].alias, "work-model")
    XCTAssertTrue(catalog.models[0].runtime.dsparkEnabled)
    XCTAssertEqual(catalog.models[0].runtime.powerSavingLimitGBps, 2)
    XCTAssertTrue(catalog.models[1].runtime.layerMajorPrefill)
    XCTAssertTrue(catalog.models[1].runtime.batchedExpertPrefill)
    XCTAssertTrue(catalog.models[1].runtime.v41NextLayerPrefetch)
    XCTAssertFalse(catalog.models[1].runtime.fp8KVCache)
    XCTAssertEqual(catalog.models[1].runtime.promptCacheEntries, 1)
    XCTAssertFalse(catalog.models[1].runtime.persistentPromptCache)
    XCTAssertFalse(catalog.models[1].runtime.dsparkEnabled)
    XCTAssertFalse(catalog.models[1].runtime.mtpEnabled)
    XCTAssertTrue(catalog.models[2].runtime.mtpEnabled)
    XCTAssertEqual(catalog.models[2].runtime.mtpSlots, 512)
    XCTAssertEqual(catalog.models[2].defaults.temperature, 0.7)
    XCTAssertEqual(catalog.models[2].defaults.topP, 0.8)
    XCTAssertEqual(catalog.models[2].defaults.topK, 20)
    XCTAssertEqual(catalog.availableModels[0].requestName, "work-model")

    let object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: catalog.encoded()) as? [String: Any])
    let models = try XCTUnwrap(object["models"] as? [[String: Any]])
    let runtime = try XCTUnwrap(models[0]["runtime"] as? [String: Any])
    XCTAssertEqual(
      Set(runtime.keys),
      [
        "slots", "expert_cache_bytes", "dspark_cache_bytes", "read_workers", "prefetch_read_workers", "prefill_step_size",
        "fp8_kv_cache", "memory_limit_gib", "layer_major_prefill",
        "layer_major_prefill_threshold",
        "prompt_cache_entries", "prompt_cache_memory_gib", "persistent_prompt_cache",
        "persistent_prompt_cache_entries", "prompt_cache_directory",
        "moe_prefill_step_size", "batched_expert_prefill", "ane_prefill",
        "ane_prefill_ratio",
        "fp4_index_cache",
        "mtp_enabled", "mtp_slots",
        "dspark_enabled", "dspark_prompt_cache", "dspark_confidence_threshold",
        "dspark_slots",
        "dspark_fallback_enabled", "dspark_sequential_verification",
        "expert_route_trace", "expert_page_cache_probe",
        "separate_prefill_io", "expert_file_cache_policy", "ready_expert_decode", "staged_expert_streaming",
        "qwen_next_layer_prefetch",
        "qwen_expert_wave_slots", "qwen_ngram_io", "qwen_ngram_cache_bytes", "qwen_sparse_sdpa", "qwen_qsa_query_chunk", "qwen_qsa_indexed", "qwen_prefill_read_experts", "qwen_prefill_seed_experts", "qwen_shared_expert_overlap",
        "qwen_quantized_kv", "qwen_quantized_index", "qwen_pooled_index_cache", "qwen_ngram_lookup_optimized",
        "qwen_compile_tensor_ops", "qwen_phase_memory", "qwen_mtp_draft_tokens", "qwen_mtp_zero_acceptance_limit", "v41_packed_kv", "v41_packed_index",
        "v41_candidate_index", "v41_ced_prefill", "v41_next_layer_prefetch", "deepseek_ane_prefill", "v41_layer_major_prefill",
        "qwen_grouped_experts", "expert_eviction_policy",
        "power_saving_limit_gbps",
      ]
    )
    for model in models {
      let config = try XCTUnwrap(model["runtime"] as? [String: Any])
      XCTAssertEqual(config["separate_prefill_io"] as? Bool, true)
      XCTAssertNil(config["dspark_hash_prefetch"])
      XCTAssertNil(config["adaptive_expert_prefill_threshold"])
    }
    XCTAssertEqual(runtime["layer_major_prefill_threshold"] as? Int, 1_024)
    XCTAssertEqual(runtime["qwen_next_layer_prefetch"] as? Bool, false)
    XCTAssertNil(runtime["qwen_grouped_decode"])
    XCTAssertEqual(runtime["expert_eviction_policy"] as? String, "lru")
    XCTAssertNil(runtime["qwen_short_block"])
    XCTAssertEqual(runtime["qwen_grouped_experts"] as? Bool, false)
    XCTAssertEqual(runtime["ane_prefill"] as? Bool, false)
    let v41Runtime = try XCTUnwrap(models[1]["runtime"] as? [String: Any])
    XCTAssertEqual(v41Runtime["layer_major_prefill"] as? Bool, true)
    XCTAssertEqual(v41Runtime["fp8_kv_cache"] as? Bool, false)
    XCTAssertEqual(v41Runtime["prompt_cache_entries"] as? Int, 1)
    XCTAssertEqual(v41Runtime["persistent_prompt_cache"] as? Bool, false)
    XCTAssertEqual(models[1]["model_kind"] as? String, "deepseek-v4.1")
    let qwenRuntime = try XCTUnwrap(models[2]["runtime"] as? [String: Any])
    XCTAssertNil(qwenRuntime["qwen_grouped_decode"])
    XCTAssertEqual(qwenRuntime["ane_prefill"] as? Bool, true)
    XCTAssertNil(qwenRuntime["qwen_short_block"])
    XCTAssertEqual(qwenRuntime["qwen_grouped_experts"] as? Bool, true)
    XCTAssertEqual(qwenRuntime["ane_prefill_ratio"] as? Double, 0.5)
    XCTAssertEqual(models[0]["model_kind"] as? String, "deepseek-v4")
    XCTAssertTrue(models[0]["warmup_prompt_path"] is NSNull)
  }

  @MainActor
  func testNewPerformanceSettingsReachOnlyTheirModelAndDisableConflictingPrefill() throws {
    var settings = ModelAdvancedSettings.defaults(for: .deepSeekV41)
    XCTAssertFalse(settings.approximationEnabled == true)
    XCTAssertFalse(settings.packedKVCache == true)
    XCTAssertFalse(settings.cedPrefill == true)
    settings.layerMajorPrefill = true
    settings.packedKVCache = true
    settings.packedIndexCache = true
    settings.candidateIndex = true
    settings.cedPrefill = true
    settings.nextLayerPrefetch = true
    settings.batchedExpertPrefill = true
    settings.deepSeekANEPrefill = true
    settings.readyExpertDecode = false
    settings.approximationEnabled = true
    func catalog(_ value: ModelAdvancedSettings) throws -> ModelCatalog {
      try ModelLibrary.makeServerCatalog(models: [installedModel(.deepSeekV41, hasDSpark: true)],
        aliases: [:], settings: [.deepSeekV41: value], powerSavingLimitGBps: nil)
    }
    let enabled = try XCTUnwrap(catalog(settings).models.first)
    XCTAssertTrue(enabled.runtime.v41LayerMajorPrefill)
    XCTAssertTrue(enabled.runtime.v41PackedKV)
    XCTAssertTrue(enabled.runtime.v41PackedIndex)
    XCTAssertTrue(enabled.runtime.v41CandidateIndex)
    XCTAssertTrue(enabled.runtime.v41CEDPrefill)
    XCTAssertTrue(enabled.runtime.v41NextLayerPrefetch)
    XCTAssertTrue(enabled.runtime.deepseekANEPrefill)
    XCTAssertFalse(enabled.runtime.readyExpertDecode)
    XCTAssertFalse(enabled.runtime.qwenQuantizedKV)
    settings.dsparkEnabled = true
    let speculative = try XCTUnwrap(catalog(settings).models.first)
    XCTAssertTrue(speculative.runtime.dsparkEnabled)
    XCTAssertTrue(speculative.runtime.v41LayerMajorPrefill)
    XCTAssertFalse(speculative.runtime.v41CEDPrefill)
    XCTAssertTrue(speculative.runtime.v41NextLayerPrefetch)
    XCTAssertEqual(speculative.defaults.approximationMode, "exact")
  }

  @MainActor
  func testCatalogDisablesMTPWhenTheInstalledModelHasNoMTPFiles() throws {
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    settings.mtpEnabled = true
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [installedModel(.qwen3_8FlashNext)],
      aliases: [:],
      settings: [.qwen3_8FlashNext: settings],
      powerSavingLimitGBps: nil
    )

    XCTAssertFalse(try XCTUnwrap(catalog.models.first).runtime.mtpEnabled)
  }

  @MainActor
  func testEmptyCatalogAndTemporaryFileCleanup() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let catalog = try ModelLibrary.makeServerCatalog(
      models: [], aliases: [:], settings: [:], powerSavingLimitGBps: nil)
    let temporary = try TemporaryModelCatalog(catalog: catalog, directory: root)

    XCTAssertTrue(catalog.models.isEmpty)
    XCTAssertTrue(FileManager.default.fileExists(atPath: temporary.url.path))
    let attributes = try FileManager.default.attributesOfItem(atPath: temporary.url.path)
    XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
    temporary.remove()
    XCTAssertFalse(FileManager.default.fileExists(atPath: temporary.url.path))
  }

  @MainActor
  func testServerStartFailureRemovesTheTemporaryCatalog() throws {
    let root = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    let runtime = root.appending(path: "runtime")
    let package = runtime.appending(path: "deepseek_v4_ssd")
    let executable = root.appending(path: "invalid-python")
    try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
    try Data().write(to: package.appending(path: "server.py"))
    try Data("#!/missing/whallm-python\n".utf8).write(to: executable)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o700],
      ofItemAtPath: executable.path
    )
    defer { try? FileManager.default.removeItem(at: root) }

    let configuration = ServerConfiguration(
      runtimeDirectory: runtime.path,
      pythonExecutable: executable.path,
      pythonHome: nil,
      sitePackages: nil,
      host: "127.0.0.1",
      port: 11_434,
      logLevel: .info,
      apiKey: "",
      powerSavingLimitGBps: nil
    )
    let before = temporaryCatalogNames()
    let controller = ServerController()

    controller.start(configuration, catalog: ModelCatalog(models: []))

    guard case .failed = controller.state else {
      return XCTFail("Expected server startup to fail.")
    }
    XCTAssertEqual(temporaryCatalogNames(), before)
  }

  func testLocalizationSupportsAllSelectableLanguages() {
    XCTAssertEqual(AppLanguage.appDefault, .system)
    XCTAssertEqual(L10n.string("Stopped", language: .english), "Stopped")
    XCTAssertEqual(L10n.string("Language", language: .simplifiedChinese), "语言")
    XCTAssertEqual(L10n.string("Language", language: .traditionalChinese), "語言")
    XCTAssertEqual(L10n.string("Model", language: .traditionalChinese), "模型")
    XCTAssertEqual(L10n.string("Alias", language: .traditionalChinese), "Alias")
    XCTAssertEqual(
      L10n.string(
        "Optional request name for this model. Changes are saved automatically.",
        language: .traditionalChinese
      ),
      "此模型的選用 request 名稱。變更會自動儲存。"
    )
    XCTAssertEqual(L10n.string("Assistant", language: .traditionalChinese), "助理")
    XCTAssertEqual(L10n.string("Use MTP", language: .traditionalChinese), "使用 MTP")
  }

  func testSystemLanguageUsesSupportedLanguageOrFallsBackToEnglish() {
    XCTAssertEqual(
      AppLanguage.systemDefault(preferredLanguages: ["zh-Hant-TW"]), .traditionalChinese)
    XCTAssertEqual(
      AppLanguage.systemDefault(preferredLanguages: ["zh-Hans-CN"]), .simplifiedChinese)
    XCTAssertEqual(AppLanguage.systemDefault(preferredLanguages: ["en-US"]), .english)
    XCTAssertEqual(AppLanguage.systemDefault(preferredLanguages: ["ja-JP"]), .english)
  }

  func testAPIKeyRoundTripsThroughIsolatedKeychainItem() {
    let service = "ServerConfigurationTests.\(UUID().uuidString)"
    let account = "api-key"
    defer { AppKeychain.saveAPIKey("", service: service, account: account) }

    AppKeychain.saveAPIKey("secret", service: service, account: account)

    XCTAssertEqual(AppKeychain.readAPIKey(service: service, account: account), "secret")
  }
}

private func isolatedDefaults() throws -> (defaults: UserDefaults, suite: String) {
  let suite = "ServerConfigurationTests.\(UUID().uuidString)"
  return (try XCTUnwrap(UserDefaults(suiteName: suite)), suite)
}

private func temporaryCatalogNames() -> Set<String> {
  let names =
    (try? FileManager.default.contentsOfDirectory(
      atPath: FileManager.default.temporaryDirectory.path)) ?? []
  return Set(names.filter { $0.hasPrefix("whallm-model-catalog-") })
}

private func installedModel(
  _ modelKind: ModelKind,
  hasMTP: Bool = false,
  hasDSpark: Bool = false,
  issues: [InstalledFileIssue] = []
) -> InstalledModelInfo {
  let modelID = modelKind.descriptor.checkpointModelID
  return InstalledModelInfo(
    url: URL(fileURLWithPath: "/tmp/\(modelKind.rawValue).dsv4"),
    size: 1,
    quickIssues: issues,
    hasMTP: hasMTP,
    hasDSpark: hasDSpark,
    modelKind: modelKind,
    modelID: modelID
  )
}

private func legacyConfiguration(publicModel: String) -> [String: Any] {
  [
    "publicModel": publicModel,
    "slots": 640,
    "readWorkers": 3,
    "memoryLimitGiB": 0,
    "prefillStepSize": 0,
    "layerMajorPrefill": true,
    "promptCacheEntries": 2,
    "promptCacheMemoryGiB": 8,
    "warmupPromptPath": "",
    "bf16KVCache": false,
    "dsparkEnabled": false,
    "dsparkSlots": 768,
    "dsparkConfidenceThreshold": 0.6,
    "defaultMaxTokens": 4_096,
    "defaultTemperature": 0.7,
    "defaultTopP": 0.9,
    "defaultTopK": 0,
  ]
}
