from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from Scripts.validate_mimo_weights import checked_range
from Scripts.audit_mimo_checkpoint import (
    COMPANIONS, EXPERT_BYTES, read_json, safe_shard, validate_contract, validate_header,
)


def fixture():
    config = dict(model_type="mimo_v2", num_hidden_layers=48, hidden_size=4096,
                  n_routed_experts=256, num_experts_per_tok=8, moe_intermediate_size=2048,
                  max_position_embeddings=1048576, attention_projection_layout="fused_qkv",
                  moe_layer_freq=[0] + [1] * 47, quantization_config={"store_dtype": "mxfp4"})
    tensors, weight_map = {}, {}
    offset = 0
    for layer in range(1, 48):
        for expert in range(256):
            for projection, rows, cols in (("gate_proj", 2048, 4096),
                                           ("up_proj", 2048, 4096),
                                           ("down_proj", 4096, 2048)):
                for suffix, width in (("weight", cols // 2), ("weight_scale", cols // 32)):
                    name = f"model.layers.{layer}.mlp.experts.{expert}.{projection}.{suffix}"
                    tensors[name] = dict(dtype="U8", shape=[rows, width],
                                         data_offsets=[offset, offset + rows * width])
                    offset += rows * width
                    weight_map[name] = "model.safetensors"
    for layer in range(3):
        name = f"model.mtp.layers.{layer}.weight"
        tensors[name] = dict(dtype="BF16", shape=[1], data_offsets=[offset, offset + 2])
        offset += 2
        weight_map[name] = "model.safetensors"
    headers = {"model.safetensors": {"tensors": tensors}}
    files = {"model.safetensors": {"size": offset + 16}}
    for name in COMPANIONS:
        headers[name] = {"tensors": {"weight": dict(dtype="F32", shape=[1], data_offsets=[0, 4])}}
        files[name] = {"size": 20}
    return config, {"weight_map": weight_map, "metadata": {"total_size": offset}}, headers, files


class MiMoCheckpointAuditTests(unittest.TestCase):
    def test_header_validation(self):
        tensor = dict(dtype="BF16", shape=[2, 3], data_offsets=[0, 12])
        self.assertEqual(validate_header({"tensors": {"a": tensor}}, 28), {"a": tensor})
        for changes in (dict(dtype="unknown"), dict(shape=[2, True]), dict(shape=[2, -1]),
                        dict(data_offsets=[1, 13]), dict(data_offsets=[0, 14]),
                        dict(data_offsets=[0, 100]), dict(data_offsets=[False, 12])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_header({"tensors": {"a": {**tensor, **changes}}}, 28)
        with self.assertRaises(ValueError):
            validate_header({"tensors": {"a": tensor, "b": tensor}}, 28)
        with self.assertRaises(ValueError):
            validate_header({"tensors": {"a": tensor}}, 29)

    def test_shard_path_validation(self):
        self.assertEqual(safe_shard("audio_tokenizer/model.safetensors"),
                         "audio_tokenizer/model.safetensors")
        for name in ("../model.safetensors", "/model.safetensors", "a/../model.safetensors",
                     "a//model.safetensors", "https://evil/model.safetensors", "x\\m.safetensors", 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_shard(name)

    def test_json_rejects_duplicates_nonfinite_and_trailing_commas(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.json"
            for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}', '{"a":1,}'):
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    read_json(path)

    def test_checkpoint_slices_reject_ignored_or_wrong_ranges_before_reading(self):
        class Response(BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 8-11/100"}

            def read(self, size=-1):
                self.was_read = True
                return super().read(size)

        response = Response(b"test")
        self.assertEqual(checked_range("https://huggingface.co/test", 8, 4,
                                       opener=lambda *a, **k: response), b"test")
        for status, header in ((200, "bytes 8-11/100"), (206, "bytes 0-3/100"),
                               (206, "bytes 8-11/*"), (206, "bytes 8-11/10")):
            response = Response(b"test")
            response.status = status
            response.headers = {"Content-Range": header}
            response.was_read = False
            with self.subTest(status=status, header=header), self.assertRaises(ValueError):
                checked_range("https://huggingface.co/test", 8, 4,
                              opener=lambda *a, **k: response)
            self.assertFalse(response.was_read)
        for data in (b"tes", b"tests"):
            response = Response(data)
            with self.assertRaises(ValueError):
                checked_range("https://huggingface.co/test", 8, 4,
                              opener=lambda *a, **k: response)
        for start, length in ((-1, 4), (0, 0), (0, 65 * 1024**2), (True, 1)):
            with self.assertRaises(ValueError):
                checked_range("https://huggingface.co/test", start, length,
                              opener=lambda *a, **k: self.fail("invalid bounds reached network"))

    def test_complete_contract_and_malformed_cases(self):
        config, index, headers, files = fixture()
        report = validate_contract(config, index, headers, files)
        self.assertEqual(report["routed_experts"], 12032)
        self.assertEqual(report["tensor_bytes_by_category"]["routed_experts"], 12032 * EXPERT_BYTES)
        self.assertEqual(report["mtp_layers"], [0, 1, 2])
        config["moe_layer_freq"][0] = 1
        with self.assertRaisesRegex(ValueError, "moe_layer_freq"):
            validate_contract(config, index, headers, files)
        config["moe_layer_freq"][0] = 0
        index["metadata"]["total_size"] += 1
        with self.assertRaisesRegex(ValueError, "total_size"):
            validate_contract(config, index, headers, files)
        index["metadata"]["total_size"] -= 1
        key = "model.layers.1.mlp.experts.0.gate_proj.weight"
        # Same number of bytes, wrong matrix orientation must still fail.
        tensors = headers["model.safetensors"]["tensors"]
        tensors[key]["shape"] = [1024, 4096]
        with self.assertRaisesRegex(ValueError, "layout mismatch"):
            validate_contract(config, index, headers, files)
        tensors[key]["shape"] = [2048, 2048]
        index["weight_map"][key] = "missing.safetensors"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_contract(config, index, headers, files)
        index["weight_map"][key] = "model.safetensors"
        del index["weight_map"][key]
        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_contract(config, index, headers, files)


if __name__ == "__main__":
    unittest.main()
