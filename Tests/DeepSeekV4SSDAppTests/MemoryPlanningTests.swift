import AppKit
import SwiftUI
import DeepSeekRepack
import Foundation
import XCTest
@testable import DeepSeekV4SSDApp

final class MemoryPlanningTests: XCTestCase {
  func testBudgetRoundsDownAndPreservesLegacyCapacity() throws {
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      let blob = ExpertMemory.blobBytes(for: kind)
      for slots in [10, 768, 900, 1_152, 3_072] {
        let gib = ExpertMemory.legacyGiB(slots: slots, blobBytes: blob)
        XCTAssertEqual(try ExpertMemory.capacity(gib: gib, blobBytes: blob, minimum: 6), slots)
      }
      let capacity = try ExpertMemory.capacity(gib: 8, blobBytes: blob, minimum: 6)
      XCTAssertLessThanOrEqual(UInt64(capacity) * blob, 8 * 1_073_741_824)
      XCTAssertGreaterThan(UInt64(capacity + 1) * blob, 8 * 1_073_741_824)
    }
    for value in [0.0, -1, .nan, .infinity, 1_025] {
      XCTAssertThrowsError(try ExpertMemory.bytes(gib: value))
    }
    XCTAssertThrowsError(try ExpertMemory.capacity(gib: 0.001, blobBytes: 2_611_200, minimum: 10))
  }

  @MainActor
  func testMemorySettingsPersistAndReachCatalogWithoutChangingLegacySlots() throws {
    let suite = "MemoryPlanningTests.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    s.slots = 900
    s.expertCacheGiB = 8.25
    s.mtpCacheGiB = 0.5
    s.estimateInputTokens = 16_384
    s.save(for: .qwen3_8FlashNext, defaults: defaults)
    let restored = ModelAdvancedSettings.loadOrDefault(for: .qwen3_8FlashNext, defaults: defaults)
    XCTAssertEqual(restored.expertCacheGiB, 8.25)
    XCTAssertEqual(restored.slots, 900)
    XCTAssertEqual(restored.estimateInputTokens, 16_384)
    let model = InstalledModelInfo(url: URL(fileURLWithPath: "/tmp/memory-test"), size: 0,
      quickIssues: [], hasMTP: true, hasDSpark: false, modelKind: .qwen3_8FlashNext, modelID: "fixture")
    let catalog = try ModelLibrary.makeServerCatalog(models: [model], aliases: [:],
      settings: [.qwen3_8FlashNext: restored], powerSavingLimitGBps: nil)
    let json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(catalog)) as? [String: Any])
    let models = try XCTUnwrap(json["models"] as? [[String: Any]])
    let runtime = try XCTUnwrap(models[0]["runtime"] as? [String: Any])
    XCTAssertEqual((runtime["expert_cache_bytes"] as? NSNumber)?.uint64Value, 8_858_370_048)
    XCTAssertEqual((runtime["mtp_cache_bytes"] as? NSNumber)?.uint64Value, 536_870_912)
    XCTAssertNil(runtime["dspark_cache_bytes"])
    // Old preferences without new fields still decode and preserve exact slots.
    var old = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(s)) as? [String: Any])
    for key in ["expertCacheGiB", "mtpCacheGiB", "dsparkCacheGiB", "estimateInputTokens"] { old.removeValue(forKey: key) }
    let legacy = try JSONDecoder().decode(ModelAdvancedSettings.self, from: JSONSerialization.data(withJSONObject: old))
    XCTAssertNil(legacy.expertCacheGiB)
    XCTAssertEqual(legacy.slots, 900)
  }

  func testEstimateRespondsToMTPContextCompressionAndCacheBudget() throws {
    let p = try profile()
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    func estimate(_ settings: ModelAdvancedSettings, available: Bool = true) throws -> MemoryEstimate {
      try XCTUnwrap(p.estimate(settings, mtpAvailable: available, dsparkAvailable: false))
    }
    let baseline = try estimate(s)
    let fixedContext = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: false,
      contextTokens: 262_144))
    XCTAssertEqual(fixedContext.inputTokens + fixedContext.outputTokens, 262_144)
    XCTAssertGreaterThan(fixedContext.total, baseline.total)
    var changedLength = s
    changedLength.defaultMaxTokens = 4_096
    changedLength.estimateInputTokens = 16_384
    XCTAssertEqual(p.estimate(changedLength, mtpAvailable: true, dsparkAvailable: false,
      contextTokens: 262_144)?.total, fixedContext.total)
    XCTAssertNil(p.estimate(s, mtpAvailable: true, dsparkAvailable: false, contextTokens: 262_145))
    s.mtpEnabled = true
    let mtp = try estimate(s)
    // MTP disables ordinary Prompt Cache retention, so the final total need not
    // always increase. Its own weights/state and verification allocations must.
    XCTAssertGreaterThan(mtp.decoding.auxiliary, baseline.decoding.auxiliary)
    XCTAssertGreaterThan(mtp.auxiliary, 0)
    XCTAssertEqual(try estimate(s, available: false).total, baseline.total)
    s.mtpEnabled = false
    s.estimateInputTokens = 32_768
    let long = try estimate(s)
    XCTAssertGreaterThan(long.conversation, baseline.conversation)
    s.packedKVCache = true
    s.packedIndexCache = true
    XCTAssertLessThan(try estimate(s).conversation, long.conversation)
    s.expertCacheGiB = 16
    XCTAssertGreaterThan(try estimate(s).model, baseline.model)
    s.promptCacheMode = .off
    let off = try estimate(s)
    s.promptCacheMode = .memory
    XCTAssertEqual(try estimate(s).conversation - off.conversation,
      min(3 * Double(s.promptCacheEntries) * off.conversation,
        max(off.conversation, 8 * ExpertMemory.gib)), accuracy: 1)
    s.estimateInputTokens = 262_144
    XCTAssertNil(p.estimate(s, mtpAvailable: true, dsparkAvailable: false))
    s.estimateInputTokens = 4_096
    s.expertCacheGiB = -1
    XCTAssertNil(p.estimate(s, mtpAvailable: true, dsparkAvailable: false))
  }

  func testDSparkEstimateAndDisabledSidecarBudgets() throws {
    let descriptor = try JSONDecoder().decode(DSparkDescriptor.self,
      from: JSONSerialization.data(withJSONObject: ["layerCount": 3, "blockSize": 5,
        "noiseTokenID": 128799, "targetLayerIDs": [40, 41, 42], "markovRank": 256, "commonTensors": []]))
    let files = try JSONDecoder().decode([InstalledFile].self,
      from: JSONSerialization.data(withJSONObject: [
        ["path": "common.bin", "size": 8 * 1_073_741_824, "sha256": "fixture"],
        ["path": "dspark/common.bin", "size": 500_000_000, "sha256": "fixture"]]))
    let manifest = InstalledManifest(formatVersion: 1, modelID: "fixture", revision: "fixture",
      layerCount: 43, expertCount: 256, selectedExpertCount: 6, expertBlobSize: 13_369_344,
      files: files, commonTensors: [], expertRegions: [], dspark: descriptor, maximumContext: 1_048_576)
    let p = MemoryPlanningProfile(kind: .deepSeekV4, manifest: manifest,
      config: ["hidden_size": 4_096, "head_dim": 512, "num_attention_heads": 64])
    var s = ModelAdvancedSettings.defaults(for: .deepSeekV4)
    let baseline = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: true))
    s.dsparkCacheGiB = 16
    XCTAssertEqual(p.estimate(s, mtpAvailable: false, dsparkAvailable: true)?.total, baseline.total)
    s.dsparkEnabled = true
    let enabled = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: true))
    XCTAssertGreaterThan(enabled.total, baseline.total)
    XCTAssertGreaterThan(enabled.auxiliary, 16 * ExpertMemory.gib)
    s.dsparkCacheGiB = 0.01
    XCTAssertNil(p.estimate(s, mtpAvailable: false, dsparkAvailable: true))
  }

  func testAllImpactTextsAreLocalized() {
    let labels = ["MTP expert cache GiB", "DSpark expert cache GiB", "Prefetch read workers",
      "ANE Prefill share", "Prefill step size", "MoE prefill step size", "Prompt cache entries",
      "Prompt cache GiB", "Expert cache GiB", "Use MTP", "Use approximate mode", "Compress attention cache",
      "Use ANE for prefill", "Use layer-major prefill", "Read workers", "Prompt cache", "Expert cache eviction",
      "Search candidate positions only", "Reduce decoder prefill work", "Max tokens", "Temperature",
      "Memory limit GiB", "Warmup prompt", "Top K", "Top P", "Use adaptive sampling",
      "Layer-major prefill threshold", "DSpark confidence threshold"]
    for language in [AppLanguage.simplifiedChinese, .traditionalChinese] {
      for label in labels {
        let key = AdvancedSettingImpact.key(for: label)
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
      for key in ["Playground", "Status", "About", "Estimated peak memory",
                  "Estimated peak memory for %lld tokens: %@",
                  "Capacity estimate for an input-heavy context, including retained caches. 128K is extrapolated; actual usage may vary."] {
        XCTAssertNotEqual(L10n.string(key, language: language), key)
      }
    }
  }

  func testPeakUsesLargestPhaseAndDoesNotStackDecodeSlotsOnBatchedPrefill() throws {
    for kind in [ModelKind.qwen3_8FlashNext, .deepSeekV4, .deepSeekV41] {
      let p = try kind == .qwen3_8FlashNext ? profile() : deepSeekProfile(kind)
      var s = ModelAdvancedSettings.defaults(for: kind)
      s.layerMajorPrefill = true
      s.batchedExpertPrefill = true
      s.promptCacheMode = .off
      s.expertCacheGiB = 8
      let a = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 32_768))
      s.expertCacheGiB = 16
      let b = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 32_768))
      XCTAssertEqual(a.prefill.model, b.prefill.model, "\(kind): released slots are not resident during batched prefill")
      XCTAssertGreaterThan(b.decoding.model, a.decoding.model)
      XCTAssertEqual(a.loading.model, 9 * ExpertMemory.gib)
      XCTAssertEqual(a.total, max(a.loading.total, a.prefill.total, a.decoding.total) + a.margin)
      XCTAssertLessThan(a.total, a.loading.total + a.prefill.total + a.decoding.total)
      s.batchedExpertPrefill = false
      let individual = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 32_768))
      XCTAssertEqual(individual.prefill.model, individual.decoding.model)
    }
  }

  func testPromptCacheCountsGroupedSnapshotsAndIsDisabledForMTP() throws {
    let p = try profile()
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    s.promptCacheMode = .off
    let off = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: false, contextTokens: 1_024))
    s.promptCacheMode = .memory
    s.promptCacheEntries = 1
    let one = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: false, contextTokens: 1_024))
    XCTAssertEqual(one.decoding.conversation, off.decoding.conversation * 4, accuracy: 1)
    s.promptCacheMemoryGiB = 16
    XCTAssertEqual(p.estimate(s, mtpAvailable: true, dsparkAvailable: false, contextTokens: 1_024)?.total, one.total)
    s.mtpEnabled = true
    let mtp = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: false, contextTokens: 1_024))
    XCTAssertEqual(mtp.decoding.conversation, off.decoding.conversation)
    XCTAssertGreaterThan(mtp.decoding.auxiliary, mtp.prefill.auxiliary)
    XCTAssertEqual(mtp.decoding.auxiliary - mtp.prefill.auxiliary, off.decoding.conversation, accuracy: 1)
  }

  func testGroupedPromptCacheBudgetAndInFlightSnapshots() throws {
    for kind in [ModelKind.deepSeekV4, .deepSeekV41, .qwen3_8FlashNext] {
      let p = try kind == .qwen3_8FlashNext ? profile() : deepSeekProfile(kind)
      for layerMajor in [false, true] {
        var s = ModelAdvancedSettings.defaults(for: kind)
        s.layerMajorPrefill = layerMajor
        s.promptCacheMode = .off
        let off = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 1_024))
        let state = off.decoding.conversation
        s.promptCacheMode = .memory
        s.promptCacheEntries = 2
        s.promptCacheMemoryGiB = 16
        let enabled = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 1_024))
        XCTAssertEqual(enabled.decoding.conversation - state, 6 * state, accuracy: 1)
        // A short suffix after a prefix hit can use chunks even with layer-major enabled.
        // Its two in-flight checkpoints are separate from the retained-cache budget.
        XCTAssertEqual(enabled.prefill.temporary - off.prefill.temporary, 2 * state, accuracy: 1)
        XCTAssertEqual(enabled.decoding.temporary - off.decoding.temporary, 2 * state, accuracy: 1)
        s.promptCacheMode = .disk
        XCTAssertEqual(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 1_024)?.total,
          enabled.total)
        s.promptCacheEntries = 128
        s.promptCacheMemoryGiB = 1
        let capped = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 32_768))
        s.promptCacheMode = .off
        let uncached = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 32_768))
        XCTAssertEqual(capped.decoding.conversation - uncached.decoding.conversation,
          max(uncached.decoding.conversation, ExpertMemory.gib), accuracy: 1)
      }
    }
    let p = try profile()
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    s.promptCacheMode = .off
    let off = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    XCTAssertGreaterThan(off.decoding.conversation, ExpertMemory.gib)
    s.promptCacheMode = .memory
    s.promptCacheMemoryGiB = 1
    let overBudget = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    // Runtime keeps one newest state even when it exceeds the memory budget.
    XCTAssertEqual(overBudget.decoding.conversation, 2 * off.decoding.conversation, accuracy: 1)
  }

  func testV41CountsCacheOwnersAndNativePackedByteWidths() throws {
    let p = try deepSeekProfile(.deepSeekV41)
    var s = ModelAdvancedSettings.defaults(for: .deepSeekV41)
    s.promptCacheMode = .off
    let raw = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 16_384))
    // Eight window rings; two KV owners (ratio 2 and 1); index layer 6 is a
    // consumer and must not add a third history. FP32 partial compressor state.
    let partial = 2.0 * 64 * 8
    let history = 16_384.0 * 8
    let rawRecent = 8.0 * 128.0 * 64.0 * 2.0
    let rawCompressed = (8_192.0 + 16_384.0) * (64.0 + 32.0) * 2.0
    let rawBytes = rawRecent + rawCompressed + partial + history
    XCTAssertEqual(raw.decoding.conversation, rawBytes, accuracy: 1)
    s.packedKVCache = true
    s.packedIndexCache = true
    let packed = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 16_384))
    let packedRecent = 8.0 * 128.0 * 64.0 * (1.0 + 1.0 / 32.0)
    let packedKVPerCell = 64.0 * (0.5 + 1.0 / 16.0)
    let packedIndexPerCell = 32.0 * (0.5 + 1.0 / 32.0)
    let packedCompressed = (8_192.0 + 16_384.0) * (packedKVPerCell + packedIndexPerCell)
    let packedBytes = packedRecent + packedCompressed + partial + history
    XCTAssertEqual(packed.decoding.conversation, packedBytes, accuracy: 1)
    XCTAssertLessThan(packed.total, raw.total)
    // The inference-style configuration aliases must give the same result.
    var flat = p.config
    for (hub, native) in [("hidden_size", "dim"), ("num_attention_heads", "n_heads"),
                         ("kv_source_layer_ids", "kv_source_layers"), ("index_source_layer_ids", "index_source_layers")] {
      flat[native] = flat.removeValue(forKey: hub)
    }
    let equivalent = MemoryPlanningProfile(kind: .deepSeekV41, manifest: p.manifest, config: flat)
    XCTAssertEqual(equivalent.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 16_384)?.total, packed.total)
  }

  func testFullContextDoesNotReserveFullPromptArraysForChunkedPrefill() throws {
    let p = try profile()
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    s.promptCacheMode = .off
    s.layerMajorPrefill = false
    let chunked = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    s.layerMajorPrefill = true
    let layered = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    XCTAssertGreaterThan(layered.prefill.temporary - chunked.prefill.temporary, 10 * ExpertMemory.gib)
    XCTAssertEqual(layered.decoding.conversation, chunked.decoding.conversation)
    XCTAssertLessThan(layered.total, 60 * ExpertMemory.gib)
    s.anePrefillRatio = .nan
    XCTAssertNil(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
  }

  func testDSparkPlansLayerMajorPrefillOnlyForV41() throws {
    for kind in [ModelKind.deepSeekV4, .deepSeekV41] {
      let p = try deepSeekProfile(kind)
      let descriptor = try JSONDecoder().decode(DSparkDescriptor.self,
        from: JSONSerialization.data(withJSONObject: ["layerCount": 3, "blockSize": 5,
          "noiseTokenID": 128799, "targetLayerIDs": [5, 6, 7], "markovRank": 256, "commonTensors": []]))
      let file = try JSONDecoder().decode(InstalledFile.self,
        from: JSONSerialization.data(withJSONObject: ["path": "dspark/common.bin", "size": 200_000_000, "sha256": "fixture"]))
      let m = p.manifest
      let manifest = InstalledManifest(formatVersion: 1, modelID: "fixture", revision: "fixture",
        layerCount: m.layerCount, expertCount: m.expertCount, selectedExpertCount: m.selectedExpertCount,
        expertBlobSize: m.expertBlobSize, files: m.files + [file], commonTensors: [], expertRegions: [],
        dspark: descriptor, maximumContext: 1_048_576)
      let dspark = MemoryPlanningProfile(kind: kind, manifest: manifest, config: p.config)
      var s = ModelAdvancedSettings.defaults(for: kind)
      s.dsparkEnabled = true
      s.layerMajorPrefill = true
      let a = try XCTUnwrap(dspark.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 32_768))
      s.layerMajorPrefill = false
      let b = try XCTUnwrap(dspark.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 32_768))
      if kind == .deepSeekV41 {
        XCTAssertNotEqual(a.prefill.temporary, b.prefill.temporary)
        XCTAssertEqual(a.decoding.total, b.decoding.total)
      } else {
        XCTAssertEqual(a.total, b.total)
      }
      XCTAssertGreaterThan(a.decoding.auxiliary, a.prefill.auxiliary)
      s.promptCacheMode = .off
      let off = try XCTUnwrap(dspark.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 32_768))
      if kind == .deepSeekV41 {
        let draftState = 3.0 * 128 * 64 * 4
        let bundle = off.decoding.conversation + draftState
        XCTAssertEqual(b.decoding.conversation - off.decoding.conversation,
          Double(s.promptCacheEntries) * bundle, accuracy: 1)
        XCTAssertEqual(b.prefill.temporary, off.prefill.temporary)
        XCTAssertEqual(b.decoding.temporary, off.decoding.temporary)
        s.promptCacheMode = .disk
        XCTAssertEqual(dspark.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 32_768)?.total,
          b.total)
        s.promptCacheEntries = 128
        s.promptCacheMemoryGiB = 1
        let capped = try XCTUnwrap(dspark.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 32_768))
        XCTAssertEqual(capped.decoding.conversation - off.decoding.conversation, ExpertMemory.gib, accuracy: 1)
        var largeConfig = p.config
        largeConfig["head_dim"] = 512
        let large = MemoryPlanningProfile(kind: kind, manifest: manifest, config: largeConfig)
        s.promptCacheMode = .off
        let largeOff = try XCTUnwrap(large.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 1_048_576))
        let largeBundle = largeOff.decoding.conversation + 3 * 128 * 512 * 4
        XCTAssertGreaterThan(largeBundle, ExpertMemory.gib)
        s.promptCacheMode = .memory
        let overBudget = try XCTUnwrap(large.estimate(s, mtpAvailable: false, dsparkAvailable: true, contextTokens: 1_048_576))
        XCTAssertEqual(overBudget.decoding.conversation - largeOff.decoding.conversation, largeBundle, accuracy: 1)
      } else {
        XCTAssertEqual(off.total, b.total)
      }
    }
  }

  func testPrefillDominatedPeakDoesNotAddReleasedExpertCache() throws {
    let p = try profile()
    var s = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    s.layerMajorPrefill = true
    s.batchedExpertPrefill = true
    s.expertCacheGiB = 8
    let small = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    s.expertCacheGiB = 16
    let large = try XCTUnwrap(p.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 262_144))
    XCTAssertGreaterThan(small.prefill.total, large.decoding.total)
    XCTAssertEqual(small.total, large.total)
    XCTAssertGreaterThan(large.decoding.model, small.decoding.model)
  }

  func testV41CandidateStorageUsesEachChunksVisibleHistoryAndInt64Positions() throws {
    let base = try deepSeekProfile(.deepSeekV41)
    var config = base.config
    config["candidate_source_layer_id"] = 4
    config["candidate_topk_blocks"] = 2
    config["candidate_block_size"] = 8
    let candidates = MemoryPlanningProfile(kind: base.kind, manifest: base.manifest, config: config)
    var s = ModelAdvancedSettings.defaults(for: .deepSeekV41)
    s.layerMajorPrefill = true
    s.prefillStepSize = 256
    s.promptCacheMode = .off
    let without = try XCTUnwrap(base.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 1_024))
    let with = try XCTUnwrap(candidates.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: 1_024))
    // Candidate masks are bool; production _candidate_blocks promotes positions
    // to int64. Earlier chunks retain their own history width, not cache capacity.
    let historyWidths: [Double] = [256, 512, 768, 1_024]
    let expected = historyWidths.reduce(0.0) { total, width in
      total + 256.0 * (width + 16.0 * 8.0)
    }
    XCTAssertEqual(with.prefill.temporary - without.prefill.temporary, expected, accuracy: 1)
  }

  func testInstalledMemoryEstimateAudit() throws {
    guard ProcessInfo.processInfo.environment["WHALLM_MEMORY_AUDIT"] != nil else { return }
    for (kind, name) in [(ModelKind.qwen3_8FlashNext, "qwen3.8-flash-next.dsv4"),
                         (.deepSeekV41, "deepseek-v4.1-flash.dsv4")] {
      let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".dsmodel/" + name)
      let p = try XCTUnwrap(MemoryPlanningProfile.load(at: url, kind: kind))
      let saved = (ModelAdvancedSettings.load(for: kind, defaults: UserDefaults(suiteName: "com.deepseekv4ssd.app")!) ?? .defaults(for: kind)).normalized(for: kind)
      for tokens in [32_768, 65_536, 131_072] {
        let e = try XCTUnwrap(p.estimate(saved, mtpAvailable: true, dsparkAvailable: true, contextTokens: tokens))
        print("SAVED \(kind) tokens=\(tokens) slots=\(saved.slots) expertGiB=\(String(describing: saved.expertCacheGiB)) layerMajor=\(saved.layerMajorPrefill) batched=\(String(describing: saved.batchedExpertPrefill)) step=\(saved.prefillStepSize) packedKV=\(String(describing: saved.packedKVCache)) packedIndex=\(String(describing: saved.packedIndexCache)) total=\(e.total / ExpertMemory.gib) prefill=\(e.prefill.total / ExpertMemory.gib) decode=\(e.decoding.total / ExpertMemory.gib) prefillModel=\(e.prefill.model / ExpertMemory.gib) prefillConversation=\(e.prefill.conversation / ExpertMemory.gib) prefillTemp=\(e.prefill.temporary / ExpertMemory.gib) prefetch=\(String(describing: saved.nextLayerPrefetch)) candidateOnly=\(String(describing: saved.candidateIndex))")
      }
      for tokens in [16_384, 32_768, 262_144] {
        var previousDecode = 0.0
        for budget in [8.0, 16.0, 32.0] {
          var s = ModelAdvancedSettings.defaults(for: kind)
          s.expertCacheGiB = budget
          let e = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: true, contextTokens: tokens))
          XCTAssertGreaterThan(e.decoding.model, previousDecode)
          previousDecode = e.decoding.model
          print("AUDIT \(kind) tokens=\(tokens) expert=\(budget) load=\(e.loading.total / ExpertMemory.gib) prefill=\(e.prefill.total / ExpertMemory.gib) decode=\(e.decoding.total / ExpertMemory.gib) total=\(e.total / ExpertMemory.gib) prefillTemp=\(e.prefill.temporary / ExpertMemory.gib)")
        }
      }
    }
  }

  func testV41SplitPrefillUsesStructuralAllowanceUntilRecalibrated() throws {
    let base = try deepSeekProfile(.deepSeekV41)
    let manifest = InstalledManifest(formatVersion: 1, modelID: "fixture", revision: "fixture",
      layerCount: 40, expertCount: 256, selectedExpertCount: 6,
      expertBlobSize: ExpertMemory.blobBytes(for: .deepSeekV41),
      files: base.manifest.files, commonTensors: [], expertRegions: [], maximumContext: 1_048_576)
    let profile = MemoryPlanningProfile(kind: .deepSeekV41, manifest: manifest, config: [
      "hidden_size": 5_120, "head_dim": 512, "num_attention_heads": 64, "hc_mult": 4,
      "index_n_heads": 32, "index_head_dim": 128, "candidate_source_layer_id": 20,
      "candidate_topk_blocks": 64, "candidate_block_size": 64,
      "compress_ratios": [0, 0] + Array(repeating: 2, count: 18) + Array(repeating: 1, count: 20),
      "kv_source_layer_ids": [2, 20], "index_source_layer_ids": [2, 20]])
    var settings = ModelAdvancedSettings.defaults(for: .deepSeekV41)
    settings.prefillStepSize = 1_024
    settings.layerMajorPrefill = true
    settings.batchedExpertPrefill = true
    settings.packedKVCache = true
    settings.packedIndexCache = true
    settings.cedPrefill = true
    settings.candidateIndex = true
    func estimate(_ s: ModelAdvancedSettings, _ n: Int) throws -> MemoryEstimate {
      try XCTUnwrap(profile.estimate(s, mtpAvailable: false, dsparkAvailable: false, contextTokens: n))
    }
    let measured = try estimate(settings, 65_536)
    var fallback = settings
    fallback.candidateIndex = false
    let structural = try estimate(fallback, 65_536)
    XCTAssertEqual(measured.prefill.temporary, structural.prefill.temporary)
    XCTAssertEqual(measured.decoding.total, structural.decoding.total)
    fallback = settings
    fallback.cedPrefill = false
    XCTAssertEqual(try estimate(fallback, 65_536).prefill.temporary, structural.prefill.temporary)
    // A longer context grows the retained source scores faster than linearly.
    let short = try estimate(settings, 32_768)
    let long = try estimate(settings, 131_072)
    XCTAssertGreaterThan(long.prefill.temporary - measured.prefill.temporary,
      2 * (measured.prefill.temporary - short.prefill.temporary))
    // The two header scenarios use their own context, independent of generation defaults.
    settings.defaultMaxTokens = 1_000_000
    settings.estimateInputTokens = 262_144
    XCTAssertEqual(try estimate(settings, 65_536).total, measured.total)
    XCTAssertEqual(try estimate(settings, 131_072).total, long.total)
  }

  private func deepSeekProfile(_ kind: ModelKind) throws -> MemoryPlanningProfile {
    let base = try profile()
    let manifest = InstalledManifest(formatVersion: 1, modelID: "fixture", revision: "fixture",
      layerCount: 8, expertCount: 256, selectedExpertCount: 6, expertBlobSize: ExpertMemory.blobBytes(for: kind),
      files: base.manifest.files, commonTensors: [], expertRegions: [], maximumContext: 1_048_576)
    return MemoryPlanningProfile(kind: kind, manifest: manifest, config: [
      "hidden_size": 512, "head_dim": 64, "num_attention_heads": 8, "index_head_dim": 32,
      "compress_ratios": [0, 0, 2, 2, 1, 1, 1, 1],
      "kv_source_layer_ids": [2, 4], "index_source_layer_ids": [2, 4, 6]])
  }

  @MainActor
  func testAdvancedSettingsLayout() throws {
    guard let directory = ProcessInfo.processInfo.environment["WHALLM_MEMORY_PREVIEWS"] else { return }
    let url = URL(fileURLWithPath: directory)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    let modelURL = FileManager.default.homeDirectoryForCurrentUser
      .appendingPathComponent(".dsmodel/qwen3.8-flash-next.dsv4")
    let installed = try XCTUnwrap(MemoryPlanningProfile.load(at: modelURL, kind: .qwen3_8FlashNext))
    // Optional metadata-only audit alongside the rendered previews; never loads model weights.
    for (kind, name) in [(ModelKind.qwen3_8FlashNext, "qwen3.8-flash-next.dsv4"),
                         (.deepSeekV41, "deepseek-v4.1-flash.dsv4")] {
      let path = modelURL.deletingLastPathComponent().appendingPathComponent(name)
      guard let p = MemoryPlanningProfile.load(at: path, kind: kind) else { continue }
      for tokens in [16_384, 32_768, 262_144] {
        let s = ModelAdvancedSettings.defaults(for: kind)
        let e = try XCTUnwrap(p.estimate(s, mtpAvailable: true, dsparkAvailable: true, contextTokens: tokens))
        print("MEMORY_ESTIMATE \(kind) tokens=\(tokens) loading=\(e.loading.total / ExpertMemory.gib) prefill=\(e.prefill.total / ExpertMemory.gib) decode=\(e.decoding.total / ExpertMemory.gib) total=\(e.total / ExpertMemory.gib) GiB")
      }
    }
    for language in [AppLanguage.english, .traditionalChinese, .simplifiedChinese] {
      var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
      settings.mtpEnabled = true
      let view = ModelAdvancedView(settings: .constant(settings), alias: .constant(""),
        aliasError: nil, settingsLocked: false, modelURL: modelURL, memoryProfile: installed,
        mtpAvailable: true, dsparkAvailable: false, modelKind: .qwen3_8FlashNext, language: language)
      let host = NSHostingView(rootView: view.font(.body)
        .dynamicTypeSize(.xLarge ... .accessibility5).controlSize(.large))
      host.frame = NSRect(x: 0, y: 0, width: 1_100, height: 2_650)
      host.layoutSubtreeIfNeeded()
      let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
      host.cacheDisplay(in: host.bounds, to: bitmap)
      let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
      try data.write(to: url.appendingPathComponent("advanced-\(language.rawValue).png"))
    }

    func render<V: View>(_ view: V, name: String, width: CGFloat = 1_280, scroll: Bool = false) throws {
      let host = NSHostingView(rootView: view.preferredColorScheme(.dark).font(.body)
        .dynamicTypeSize(.xLarge ... .accessibility5).controlSize(.large))
      host.frame = NSRect(x: 0, y: 0, width: width, height: 900)
      let window = NSWindow(contentRect: NSRect(x: -10_000, y: -10_000, width: width, height: 900),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = host
      defer { window.close() }
      window.orderBack(nil)
      RunLoop.main.run(until: Date().addingTimeInterval(0.1))
      host.layoutSubtreeIfNeeded()
      if scroll {
        func descendants(_ view: NSView) -> [NSScrollView] {
          (view as? NSScrollView).map { [$0] } ?? view.subviews.flatMap(descendants)
        }
        let panes = descendants(host).filter { ($0.documentView?.frame.height ?? 0) > $0.bounds.height + 200 }
        XCTAssertFalse(panes.isEmpty, "The settings scroll view must be realized for the scrolled preview")
        for pane in panes {
          pane.contentView.scroll(to: NSPoint(x: 0, y: 650))
          pane.reflectScrolledClipView(pane.contentView)
        }
        RunLoop.main.run(until: Date().addingTimeInterval(0.1))
        host.layoutSubtreeIfNeeded()
      }
      let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
      host.cacheDisplay(in: host.bounds, to: bitmap)
      let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
      try data.write(to: url.appendingPathComponent("\(name).png"))
    }
    let settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    let modelPage = ModelAdvancedView(settings: .constant(settings), alias: .constant(""),
      aliasError: nil, settingsLocked: false, modelURL: modelURL, memoryProfile: installed,
      mtpAvailable: true, dsparkAvailable: false, modelKind: .qwen3_8FlashNext, language: .traditionalChinese)
    let shell = HStack(spacing: 0) {
      AppSidebar(selection: .constant(.model), language: .traditionalChinese).frame(width: 220)
      modelPage
    }
    try render(shell, name: "model-header-top")
    try render(shell, name: "model-header-scrolled", scroll: true)
    try render(shell, name: "model-header-narrow", width: 1_000)
    try render(AboutView(language: .traditionalChinese), name: "about")
    let suite = "SettingsPreview.\(UUID().uuidString)"
    let defaults = UserDefaults(suiteName: suite)!
    defer { defaults.removePersistentDomain(forName: suite) }
    try render(SettingsView(languageCode: .constant("zh-Hant"), language: .traditionalChinese,
      appUpdater: AppUpdater(startingUpdater: false, defaults: defaults)), name: "settings")
  }

  func testQwenFlashEstimateCountsPackedCacheAndDoesNotTreatWaveSizeAsTotalCapacity() throws {
    let p = try profile()
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    func estimate() throws -> MemoryEstimate {
      try XCTUnwrap(p.estimate(settings, mtpAvailable: false, dsparkAvailable: false))
    }
    let baseline = try estimate()
    settings.qwenNgramIO = "pread"
    settings.qwenNgramCacheMiB = 64
    let cached = try estimate()
    XCTAssertEqual(cached.prefill.temporary - baseline.prefill.temporary, 64 * 1_048_576, accuracy: 1)
    XCTAssertEqual(cached.decoding.temporary - baseline.decoding.temporary, 64 * 1_048_576, accuracy: 1)
    settings.qwenNgramIO = "mmap"
    XCTAssertEqual(try estimate().total, baseline.total)
    settings.qwenExpertWaveSlots = 1
    let waves = try estimate()
    settings.qwenExpertWaveSlots = 512
    XCTAssertEqual(try estimate().model, waves.model)
    settings.batchedExpertPrefill = false
    settings.nextLayerPrefetch = false
    XCTAssertEqual(try estimate().prefill.model, waves.prefill.model)
    settings.qwenNgramCacheMiB = Int.max
    XCTAssertNil(p.estimate(settings, mtpAvailable: false, dsparkAvailable: false))
  }

  func testStreamingStagingAndSeedEstimatesAreSeparateFromDecodeCapacity() throws {
    let p = try profile()
    var settings = ModelAdvancedSettings.defaults(for: .qwen3_8FlashNext)
    settings.expertCacheGiB = 8
    settings.readWorkers = 2
    settings.layerMajorPrefill = true
    settings.batchedExpertPrefill = true
    settings.qwenExpertWaveSlots = 0
    func estimate() throws -> MemoryEstimate {
      try XCTUnwrap(p.estimate(settings, mtpAvailable: false, dsparkAvailable: false))
    }
    let baseline = try estimate()
    settings.qwenPrefillReadExperts = 4
    settings.qwenPrefillSeedExperts = 32
    let active = try estimate()
    let scratch = 2_611_200.0 * 3 * 2
    let seeds = 2_611_200.0 * 32 * 48
    XCTAssertEqual(active.prefill.temporary - baseline.prefill.temporary, scratch, accuracy: 1)
    XCTAssertEqual(active.decoding.temporary - baseline.decoding.temporary, scratch, accuracy: 1)
    XCTAssertEqual(active.prefill.model - baseline.prefill.model, seeds, accuracy: 1)
    XCTAssertEqual(active.decoding.model, baseline.decoding.model)
    settings.layerMajorPrefill = false
    let inactive = try estimate()
    settings.qwenPrefillReadExperts = 1
    settings.qwenPrefillSeedExperts = 0
    XCTAssertEqual(try estimate().total, inactive.total)
  }

  private func profile() throws -> MemoryPlanningProfile {
    let file: (String, UInt64) throws -> InstalledFile = { name, size in
      try JSONDecoder().decode(InstalledFile.self, from: JSONSerialization.data(withJSONObject:
        ["path": name, "size": size, "sha256": "fixture"]))
    }
    let manifest = InstalledManifest(formatVersion: 2, modelID: "fixture", revision: "fixture",
      layerCount: 48, expertCount: 512, selectedExpertCount: 10, expertBlobSize: 2_611_200,
      files: [try file("common.bin", 9 * 1_073_741_824), try file("mtp/common.bin", 200_000_000)],
      commonTensors: [], expertRegions: [],
      mtp: MTPDescriptor(layerCount: 1, useDedicatedEmbeddings: false, commonTensors: []),
      modelKind: .qwen3_8FlashNext, maximumContext: 262_144)
    return MemoryPlanningProfile(kind: .qwen3_8FlashNext, manifest: manifest,
      config: ["hidden_size": 2_560, "head_dim": 256, "num_attention_heads": 24])
  }
}
