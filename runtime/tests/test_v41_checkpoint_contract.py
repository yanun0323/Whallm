"""Metadata-only V4.1 checkpoint checks; no weights or GPU operations."""
import json
from pathlib import Path
from unittest.mock import patch
import unittest

from deepseek_v4_ssd.manifest import InstalledModel
from deepseek_v4_ssd.model_support.catalog import BY_KIND, _catalog_path
from deepseek_v4_ssd.model_support.native_deepseek import NativeDeepSeekV41Support

KINDS = ("deepseek-v4.1", "deepseek-v4.1-1017")


def manifest(kind, dspark=False):
    descriptor = BY_KIND[kind]
    tables = [dict(layer=layer, weightFile=f"engram/layer_{layer:02d}.weight.bin",
                   scaleFile=f"engram/layer_{layer:02d}.scale.bin", rows=rows,
                   dimension=256, blockSize=32)
              for layer, rows in ((1, 384006168), (14, 384016682))]
    files = [dict(path=f"experts/layer_{layer:02d}.bin", size=384 * 18800640)
             for layer in range(40)]
    for table in tables:
        files += [dict(path=table["weightFile"], size=table["rows"] * 256),
                  dict(path=table["scaleFile"], size=table["rows"] * 8)]
    raw = dict(formatVersion=3, modelKind=kind, modelID=descriptor.checkpoint_model_id,
               revision=descriptor.checkpoint_revision, layerCount=40, expertCount=384,
               selectedExpertCount=6, expertBlobSize=18800640, maximumContext=1048576,
               engram=dict(tables=tables), files=files)
    if dspark:
        raw["dspark"] = dict(layerCount=3, blockSize=5, noiseTokenID=128799,
                             targetLayerIDs=[37, 38, 39], markovRank=256,
                             commonTensors=[{} for _ in range(97)])
    return raw


class CheckpointContractTests(unittest.TestCase):
    def test_both_checkpoints_keep_their_identity_with_and_without_dspark(self):
        for kind in KINDS:
            for dspark in (False, True):
                with self.subTest(kind=kind, dspark=dspark):
                    contract = NativeDeepSeekV41Support(BY_KIND[kind]).manifest_contract(manifest(kind, dspark))
                    self.assertEqual(contract["model_kind"], kind)
                    self.assertEqual(contract["expert_count"], 384)
                    self.assertEqual("dspark/common.bin" in contract["required"], dspark)
                    self.assertTrue(InstalledModel.is_deepseek_v41.fget(type("Model", (), {"model_kind": kind})()))

    def test_crossed_and_unpinned_checkpoint_identities_are_rejected(self):
        for kind, other in (KINDS, KINDS[::-1]):
            support = NativeDeepSeekV41Support(BY_KIND[kind])
            for field in ("modelKind", "modelID", "revision"):
                invalid = manifest(kind)
                invalid[field] = manifest(other)[field]
                with self.subTest(kind=kind, field=field), self.assertRaisesRegex(ValueError, "pinned V4.1"):
                    support.manifest_contract(invalid)
            invalid = manifest(kind)
            invalid["revision"] = "main"
            with self.assertRaises(ValueError):
                support.manifest_contract(invalid)

    def test_1017_does_not_relax_engram_or_expert_layout(self):
        support = NativeDeepSeekV41Support(BY_KIND[KINDS[1]])
        for target in ("expertCount", "expertBlobSize", "maximumContext"):
            raw = manifest(KINDS[1])
            raw[target] -= 1
            with self.subTest(target=target), self.assertRaises(ValueError):
                support.manifest_contract(raw)
        for target in ("experts/layer_00.bin", "engram/layer_01.weight.bin", "engram/layer_14.scale.bin"):
            raw = manifest(KINDS[1])
            next(item for item in raw["files"] if item["path"] == target)["size"] -= 1
            with self.subTest(target=target), self.assertRaises(ValueError):
                support.manifest_contract(raw)
        raw = manifest(KINDS[1])
        raw["engram"]["tables"][0]["rows"] -= 1
        with self.assertRaises(ValueError):
            support.manifest_contract(raw)

    def test_each_checkpoint_requires_its_own_encoder_checksum(self):
        from deepseek_v4_ssd.tool_codec import (
            DeepSeekV41ToolCodec, V41_ENCODER_SHA256, V41_1017_ENCODER_SHA256)
        self.assertNotEqual(V41_ENCODER_SHA256, V41_1017_ENCODER_SHA256)
        for kind, digest in zip(KINDS, (V41_ENCODER_SHA256, V41_1017_ENCODER_SHA256)):
            with patch.object(DeepSeekV41ToolCodec, "from_encoder") as open_encoder:
                NativeDeepSeekV41Support(BY_KIND[kind]).open_codec(Path("/model"), None)
            open_encoder.assert_called_once_with(Path("/model/encoding/encoding.py"), digest)

    def test_catalog_separates_installations_api_ids_and_cache_identity(self):
        catalog = {item["kind"]: item for item in json.loads(_catalog_path().read_text())["models"]}
        old, new = (catalog[kind] for kind in KINDS)
        for key in ("directoryName", "apiModelID", "checkpointModelID", "checkpointRevision"):
            self.assertNotEqual(old[key], new[key])
        self.assertEqual(old["checkpointRevision"], "dba1be0a40aa45a94ad051997016db3960a90277")
        self.assertEqual(new["checkpointRevision"], "1bae0e1088993b9b82de08f59de5be5a34e42170")
        self.assertIn("nativeEngine", new["features"])
        self.assertEqual(old["requiredPaths"], new["requiredPaths"])


if __name__ == "__main__":
    unittest.main()
